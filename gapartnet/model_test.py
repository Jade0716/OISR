import torch
import numpy as np
from network.model import GAPartNet
from os.path import join as pjoin
from misc.info import OBJECT_NAME2ID, PART_ID2NAME, PART_NAME2ID, get_symmetry_matrix
from structure.point_cloud import PointCloud, PointCloudBatch
from typing import Optional, Tuple, Union, List
from epic_ops.voxelize import voxelize
import copy
from misc.visu import visualize_gapartnet
from misc.pose_fitting import estimate_pose_from_npcs
from train import GAPartNetDataset, collate_fn
from network.grouping_utils import (apply_nms, cluster_proposals, compute_ap,
                               compute_npcs_loss, filter_invalid_proposals,
)
import argparse
import os
import open3d as o3d
import torch
import torch.nn as nn
import numpy as np
from structure.point_cloud import PointCloud, PointCloudBatch
from pathlib import Path
from torch.utils.data import ConcatDataset, DataLoader
from typing import Optional, Tuple, Union, List
import matplotlib.pyplot as plt
from misc.info import OBJECT_NAME2ID, PART_ID2NAME
import pandas as pd
def analyze_proposals_sorted_by_score(proposals, top_k=None):
    if proposals is None or not hasattr(proposals, 'proposal_offsets'):
        print("No valid proposals.")
        return

    offsets = proposals.proposal_offsets
    num_proposals = offsets.shape[0] - 1

    pt_sem_classes = proposals.pt_sem_classes.long()
    scores = proposals.score_preds
    ious = proposals.ious
    sem_preds = proposals.sem_preds
    instance_labels = getattr(proposals, 'instance_labels', None)
    instance_sem_labels = getattr(proposals, 'instance_sem_labels', None)
    batch_indices = getattr(proposals, 'batch_indices', None)
    print(f"Analyzing {num_proposals} proposals...")
    results = []

    for i in range(num_proposals):
        start = offsets[i].item()
        end = offsets[i + 1].item()

        pred_class = pt_sem_classes[i].item()
        score = scores[i].item()
        iou = ious[i].max().item()
        num_points = end - start

        # 获取 GT 类别
        gt_sem_class = -1
        if instance_labels is not None and instance_sem_labels is not None:
            inst_id = instance_labels[start].item()
            if inst_id >= 0 and batch_indices is not None:
                batch_idx = batch_indices[start].item()
                if batch_idx < instance_sem_labels.shape[0] and inst_id < instance_sem_labels.shape[1]:
                    gt_sem_class = instance_sem_labels[batch_idx, inst_id].item()

        results.append((score, i, pred_class, gt_sem_class, num_points, iou))

    # 按 score 降序排序
    results.sort(reverse=True, key=lambda x: x[0])

    if top_k:
        results = results[:top_k]

    print(f"{'Idx':<6} {'Score':<8} {'PredCls':<8} {'GTCls':<6} {'Points':<8} {'IoU':<6}")
    for score, i, pred_class, gt_class, num_points, iou in results:
        print(f"{i:<6} {score:<8.4f} {pred_class:<8} {gt_class:<6} {num_points:<8} {iou:<6.4f}")
def FindMaxDis(pointcloud):
    max_xyz = pointcloud.max(0)
    min_xyz = pointcloud.min(0)
    center = (max_xyz + min_xyz) / 2
    max_radius = ((((pointcloud - center)**2).sum(1))**0.5).max()
    return max_radius, center
def WorldSpaceToBallSpace(pointcloud):
    """
    change the raw pointcloud in world space to united vector ball space
    return: max_radius: the max_distance in raw pointcloud to center
            center: [x,y,z] of the raw center
    """
    max_radius, center = FindMaxDis(pointcloud)
    pointcloud_normalized = (pointcloud - center) / max_radius
    return pointcloud_normalized, max_radius, center




def main():
    # CUDA settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    np.random.seed(1234)
    net = GAPartNet()
    # if torch.cuda.device_count() > 1:
    #     net = nn.DataParallel(net).cuda()
    # else:
    #     net = net.to(device)
    net.load_state_dict(torch.load("/home/liuyuyan/GaPartNet/gapartnet/checkpoints/100flowstats0.02flowxyzoffsetslider/models/final_model.pth"), strict=False)
    net.to(device)
    root_dir: str = "/16T/liuyuyan/GAPartNetAllWithFlows"
    max_points: int = 20000
    voxel_size: Tuple[float, float, float] = (1 / 100, 1 / 100, 1 / 100)
    train_batch_size: int = 1
    val_batch_size: int = 32
    test_batch_size: int = 32
    num_workers: int = 16
    pos_jitter: float = 0.
    color_jitter: float = 0.1
    flip_prob: float = 0.
    rotate_prob: float = 0.
    train_few_shot: bool = False
    val_few_shot: bool = False
    intra_few_shot: bool = False
    inter_few_shot: bool = False
    few_shot_num: int = 256
    val_min_points_per_class_use = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    train_data_files = GAPartNetDataset(
        Path(root_dir) / "train" / "pth",
        shuffle=True,
        max_points=max_points,
        augmentation=True,
        voxel_size=voxel_size,
        few_shot=train_few_shot,
        few_shot_num=few_shot_num,
        pos_jitter=pos_jitter,
        color_jitter=color_jitter,
        flip_prob=flip_prob,
        rotate_prob=rotate_prob,
    )
    inter_data_files = GAPartNetDataset(
        Path(root_dir) / "test_inter" / "pth",
        shuffle=True,
        max_points=max_points,
        augmentation=False,
        voxel_size=voxel_size,
        few_shot=inter_few_shot,
        few_shot_num=few_shot_num,
        pos_jitter=pos_jitter,
        color_jitter=color_jitter,
        flip_prob=flip_prob,
        rotate_prob=rotate_prob,
    )
    train_dataloader = DataLoader(train_data_files,
                                  batch_size=train_batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  collate_fn=collate_fn,
                                  pin_memory=True,
                                  drop_last=True,
                                  )
    val_dataloader = DataLoader(inter_data_files,
                                batch_size=1,
                                shuffle=False,
                                num_workers=num_workers,
                                collate_fn=collate_fn,
                                pin_memory=True,
                                drop_last=False
                                )
    for pc in val_dataloader:
        pc = [Point.to('cuda') for Point in pc]  # List["PointCloud"]
        if len(pc)!=0 :#and pc[0].pc_id.startswith("Table_32761"):
            data_batch = PointCloud.collate(pc)  # PointCloudBatch
            net.eval()
            with torch.no_grad():
                pc_ids, sem_seg, proposals, _ = net(data_batch)
            print(f"{pc_ids}真实实例：{data_batch.instance_sem_labels, data_batch.num_points_per_instance}")
            print(f"分割预测：{sem_seg.all_accu,sem_seg.pixel_accu,torch.unique(sem_seg.sem_labels), torch.unique(sem_seg.sem_preds)}")
            sample_ids = range(len(pc_ids))
            sample_id = 0
            batch_id = sample_id // 1
            batch_sample_id = sample_id % 1
            if proposals is not None:
                proposals.pt_sem_classes = proposals.sem_preds[proposals.proposal_offsets[:-1].long()].long()
                print(f"beyond:{proposals.proposal_offsets.shape[0]-1}")
                proposals = filter_invalid_proposals(
                    proposals,
                    score_threshold=net.val_score_threshold,
                    val_min_points_per_class=val_min_points_per_class_use,
                )
                proposals = apply_nms(proposals, net.val_nms_iou_threshold)  # 非极大值抑制（NMS），用来过滤掉重叠太多的重复 proposal
                print(f"after:{proposals.proposal_offsets.shape[0]-1}")

                proposals.pt_sem_classes = proposals.sem_preds[proposals.proposal_offsets[:-1].long()]
                analyze_proposals_sorted_by_score(proposals)
                pt_xyz = proposals.pt_xyz
                batch_indices = proposals.batch_indices
                proposal_offsets = proposals.proposal_offsets
                num_points_per_proposal = proposals.num_points_per_proposal
                num_proposals = num_points_per_proposal.shape[0]
                score_preds = proposals.score_preds
                mask = proposals.valid_mask
                indices = torch.arange(mask.shape[0], dtype=torch.int64, device=sem_seg.sem_preds.device)
                proposal_indices = indices[proposals.valid_mask][proposals.sorted_indices]

                ins_seg_preds = torch.ones(mask.shape[0]) * 0
                for ins_i in range(len(proposal_offsets) - 1):
                    ins_seg_preds[proposal_indices[proposal_offsets[ins_i]:proposal_offsets[ins_i + 1]]] = ins_i + 1

                npcs_maps = torch.ones(proposals.valid_mask.shape[0], 3, device=proposals.valid_mask.device) * 0.0
                valid_index = torch.where(proposals.valid_mask == True)[0][
                    proposals.sorted_indices.long()[torch.where(proposals.npcs_valid_mask == True)]]
                npcs_maps[valid_index] = proposals.npcs_preds
                # bounding box
                bboxes = []
                for proposal_i in range(len(proposal_offsets) - 1):
                    npcs_i = npcs_maps[proposal_indices[proposal_offsets[proposal_i]:proposal_offsets[proposal_i + 1]]]
                    npcs_i = npcs_i - 0.5
                    xyz_i = pt_xyz[proposal_offsets[proposal_i]:proposal_offsets[proposal_i + 1]]
                    # import pdb; pdb.set_trace()
                    if xyz_i.shape[0] < 5:
                        continue
                    bbox_xyz, scale, rotation, translation, out_transform, best_inlier_idx = estimate_pose_from_npcs(
                        xyz_i.cpu().numpy(), npcs_i.cpu().numpy())
                    # import pdb; pdb.set_trace()
                    if scale[0] == None:
                        continue
                    bboxes.append(bbox_xyz.tolist())

                # get the sampled data point
                sample_sem_pred = sem_seg.sem_preds.reshape(-1, 20000)
                sample_ins_seg_pred = ins_seg_preds.reshape(-1, 20000)
                sample_npcs_map = npcs_maps.reshape(-1, 20000, 3)

                visualize_gapartnet(
                    SAVE_ROOT="output/GAPartNetWithFlow_result",
                    RAW_IMG_ROOT="data/image_kuafu",
                    GAPARTNET_DATA_ROOT="/16T/liuyuyan/GAPartNetAllWithFlows",
                    # save_option=["raw", "pc", "sem_pred", "sem_gt", "ins_pred", "ins_gt", "npcs_pred", "npcs_gt", "bbox_gt", "bbox_gt_pure", "bbox_pred", "bbox_pred_pure"],
                    save_option=["raw", "pc", "sem_pred", "ins_pred", "npcs_pred", "bbox_pred", "bbox_pred_pure"],
                    name=pc_ids[sample_id],
                    split="test_inter",
                    sem_preds=sample_sem_pred.cpu().numpy().squeeze(),  # type: ignore
                    ins_preds=sample_ins_seg_pred.cpu().numpy().squeeze(),
                    npcs_preds=sample_npcs_map.cpu().numpy().squeeze(),
                    bboxes=bboxes,
                )
                input("按下回车键继续...")
            # break
if __name__ == '__main__':
    main()
