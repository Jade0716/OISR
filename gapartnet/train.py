from __future__ import print_function

from network.losses import focal_loss, dice_loss, pixel_accuracy, mean_iou
from network.grouping_utils import (apply_nms, cluster_proposals, compute_ap,
                               compute_npcs_loss, filter_invalid_proposals,
)
from structure.instances import Instances
import copy
from scipy.stats import spearmanr
from pathlib import Path
from typing import Optional, Tuple, Union, List
from epic_ops.voxelize import voxelize
from torch.utils.data import Dataset
import random
import open3d as o3d
from glob import glob
from tqdm import tqdm
from structure.point_cloud import PointCloud, PointCloudBatch
from misc.info import OBJECT_NAME2ID, PART_ID2NAME, PART_NAME2ID, get_symmetry_matrix
import argparse
import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import ConcatDataset, DataLoader
from network.model import GAPartNet


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


class IOStream:
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


def _init_(args):
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)
    if not os.path.exists('checkpoints/' + args.exp_name + '/' + 'models'):
        os.makedirs('checkpoints/' + args.exp_name + '/' + 'models')
    os.system('cp train.py checkpoints' + '/' + args.exp_name + '/' + 'train.py.backup')
    os.system('cp network/model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')


def weights_init(m):
    classname=m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.kaiming_normal_(m.weight.data)
    if classname.find('Conv1d') != -1:
        nn.init.kaiming_normal_(m.weight.data)
        
class GAPartNetDataset(Dataset):
    def __init__(
            self,
            root_dir: Union[str, Path, List] = "",
            shuffle: bool = False,
            max_points: int = 20000,
            augmentation: bool = False,
            voxel_size: Tuple[float, float, float] = (1 / 100, 1 / 100, 1 / 100),
            few_shot=False,
            few_shot_num=512,
            pos_jitter: float = 0.,
            color_jitter: float = 0.,
            flip_prob: float = 0.,
            rotate_prob: float = 0.,
            nopart_path: str = "data/nopart.txt",
            no_label=False,
    ):
        if type(root_dir) == list:
            file_paths = []
            for rt in root_dir:
                file_paths += glob(str(rt) + "/*.pth")
        else:
            file_paths = glob(str(root_dir) + "/*.pth")
        # file_paths = [path for path in file_paths if Path(path).stem.startswith("Table")]
        self.nopart_files = open(nopart_path, "r").readlines()[0].split(" ")
        self.nopart_names = [p.split("/")[-1].split(".")[0] for p in self.nopart_files]
        file_paths = [path for path in file_paths
                      if path.split("/")[-1].split(".")[0] not in self.nopart_names]
        if shuffle:
            random.shuffle(file_paths)
        if few_shot:
            file_paths = file_paths[:few_shot_num]
        self.pc_paths = file_paths
        self.no_label = no_label
        self.augmentation = augmentation
        self.pos_jitter = pos_jitter
        self.color_jitter = color_jitter
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob
        self.voxel_size = voxel_size
        self.max_points = max_points

    def __len__(self):
        return len(self.pc_paths)

    def __getitem__(self, idx):
        path = self.pc_paths[idx]
        file = load_data(path, no_label=self.no_label)   #Pointlcould类
        if file is None:
            return None
        if not bool((file.instance_labels != -100).any()):
            # print(path)
            return None  # 直接返回 None，后续 DataLoader 过滤
        file = downsample(file, max_points=self.max_points)
        file = compact_instance_labels(file)   #压缩实例标签 instance_labels 确保实例标签编号连续
        if self.augmentation:    #False
            file = apply_augmentations(file,            #数据增强
                                       pos_jitter=self.pos_jitter,
                                       color_jitter=self.color_jitter,
                                       flip_prob=self.flip_prob,
                                       rotate_prob=self.rotate_prob, )
        file = generate_inst_info(file)    #生成实例信息
        file = file.to_tensor()
        file = apply_voxelization(file, voxel_size=self.voxel_size)

        return file

def collate_fn(batch):
    return [item for item in batch if item is not None]
# def collate_fn(batch):
#     return [item for item in batch]


def apply_augmentations(         #数据增强
        pc: PointCloud,
        *,
        pos_jitter: float = 0.,
        color_jitter: float = 0.,
        flip_prob: float = 0.,
        rotate_prob: float = 0.,
) -> PointCloud:
    pc = copy.copy(pc)

    m = np.eye(3)
    if pos_jitter > 0:
        m += np.random.randn(3, 3) * pos_jitter

    if flip_prob > 0:
        if np.random.rand() < flip_prob:
            m[0, 0] = -m[0, 0]

    if rotate_prob > 0:
        if np.random.rand() < flip_prob:
            theta = np.random.rand() * np.pi * 2
            m = m @ np.asarray([
                [np.cos(theta), np.sin(theta), 0],
                [-np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ])

    pc.points = pc.points.copy()
    pc.points[:, :3] = pc.points[:, :3] @ m

    if color_jitter > 0:
        pc.points[:, 6:] += np.random.randn(
            1, pc.points.shape[1] - 6
        ) * color_jitter

    return pc


def downsample(pc: PointCloud, *, max_points: int = 20000) -> PointCloud:
    pc = copy.copy(pc)

    num_points = pc.points.shape[0]

    if num_points > max_points:
        assert False, (num_points, max_points)

    return pc


def compact_instance_labels(pc: PointCloud) -> PointCloud:         #压缩实例标签
    pc = copy.copy(pc)

    valid_mask = pc.instance_labels >= 0
    instance_labels = pc.instance_labels[valid_mask]
    _, instance_labels = np.unique(instance_labels, return_inverse=True)
    pc.instance_labels[valid_mask] = instance_labels

    return pc


def generate_inst_info(pc: PointCloud) -> PointCloud:          #生成实例信息
    pc = copy.copy(pc)

    num_points = pc.points.shape[0]

    num_instances = int(pc.instance_labels.max()) + 1
    instance_regions = np.zeros((num_points, 9), dtype=np.float32)
    num_points_per_instance = []
    instance_sem_labels = []

    assert num_instances > 0
    for i in range(num_instances):
        indices = np.where(pc.instance_labels == i)[0]

        xyz_i = pc.points[indices, :3]
        min_i = xyz_i.min(0)
        max_i = xyz_i.max(0)
        mean_i = xyz_i.mean(0)
        instance_regions[indices, 0:3] = mean_i
        instance_regions[indices, 3:6] = min_i
        instance_regions[indices, 6:9] = max_i

        num_points_per_instance.append(indices.shape[0])
        instance_sem_labels.append(int(pc.sem_labels[indices[0]]))

    pc.num_instances = num_instances
    pc.instance_regions = instance_regions
    pc.num_points_per_instance = np.asarray(num_points_per_instance, dtype=np.int32)
    pc.instance_sem_labels = np.asarray(instance_sem_labels, dtype=np.int32)
    return pc

def visualize_point_cloud(points):

    pcd0 = o3d.geometry.PointCloud()
    pcd0.points = o3d.utility.Vector3dVector(points[:,:3])
    pcd0.paint_uniform_color([1, 0, 0])  # 红色
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(points[:,6:9])
    pcd1.paint_uniform_color([0, 0, 1])  # 红色

    # 使用Open3D可视化点云
    o3d.visualization.draw_geometries([pcd0, pcd1])
def apply_voxelization(                  #点云体素化
        pc: PointCloud, *, voxel_size: Tuple[float, float, float]
) -> PointCloud:
    pc = copy.copy(pc)
    num_points = pc.points.shape[0]
    pt_xyz = pc.points[:, :3]
    points_range_min = pt_xyz.min(0)[0] - 1e-4
    points_range_max = pt_xyz.max(0)[0] + 1e-4
    voxel_features, voxel_coords, _, pc_voxel_id = voxelize(  #,体素特征(xyz,颜色),体素坐标,_,每个点对应的体素索引
        pt_xyz, pc.points,
        batch_offsets=torch.as_tensor([0, num_points], dtype=torch.int64, device=pt_xyz.device),
        voxel_size=torch.as_tensor(voxel_size, device=pt_xyz.device),
        points_range_min=torch.as_tensor(points_range_min, device=pt_xyz.device),
        points_range_max=torch.as_tensor(points_range_max, device=pt_xyz.device),
        reduction="mean",
    )
    assert (pc_voxel_id >= 0).all()

    voxel_coords_range = (voxel_coords.max(0)[0] + 1).clamp(min=128, max=None)

    pc.voxel_features = voxel_features
    pc.voxel_coords = voxel_coords
    pc.voxel_coords_range = voxel_coords_range.tolist()
    pc.pc_voxel_id = pc_voxel_id

    return pc
def compute_instance_flow_features(pc_id, flow: np.ndarray, instance_labels: np.ndarray, sem_labels: np.ndarray) -> np.ndarray:

    num_points = flow.shape[0]
    flow_lengths = np.linalg.norm(flow, axis=1)
    instance_features = np.zeros((num_points, 3), dtype=np.float32)

    valid_mask = instance_labels >= 0
    valid_instance_labels = instance_labels[valid_mask]
    unique_instances = np.unique(valid_instance_labels)

    for instance_id in unique_instances:
        mask = instance_labels == instance_id
        instance_flow = flow[mask]
        instance_flow_lens = flow_lengths[mask]

        mean_len = instance_flow_lens.mean()
        q1 = np.percentile(instance_flow_lens, 25)
        q3 = np.percentile(instance_flow_lens, 75)
        iqr_len = q3 - q1  # 四分位差

        normed_flow = instance_flow / (np.linalg.norm(instance_flow, axis=1, keepdims=True) + 1e-6)
        mean_dir = normed_flow.mean(axis=0)
        mean_dir /= np.linalg.norm(mean_dir) + 1e-6
        mean_dir_cos_z = np.dot(mean_dir, np.array([0, 0, 1]))

        instance_features[mask] = np.array([mean_len, iqr_len, mean_dir_cos_z], dtype=np.float32)


        sem_ids = sem_labels[mask]
        sem_id = np.bincount(sem_ids).argmax()
        # if sem_id == 5 and pc_id.startswith("Table_32761"):
        #     print(f"{instance_id} [Slider], mean_len={mean_len:.4f}, iqr_len={iqr_len:.4f}, cos_z={mean_dir_cos_z:.4f}")
        # if sem_id == 6:
            # print(f"{pc_id} [Slider], mean_len={mean_len:.4f}, iqr_len={iqr_len:.4f}, cos_z={mean_dir_cos_z:.4f}")
        # if sem_id == 7:
        #     print(f"{pc_id} [Hinge] , mean_len={mean_len:.4f}, iqr_len={iqr_len:.4f}, cos_z={mean_dir_cos_z:.4f}")

    return instance_features


def load_data(file_path: str, no_label: bool = False):
    if no_label:
        raise NotImplementedError

    pc_data = torch.load(file_path)
    pc_id = file_path.split("/")[-1].split(".")[0]
    object_cat = OBJECT_NAME2ID[pc_id.split("_")[0]]

    if pc_data[2].shape != (20000, 3):
        return None
    flow = (pc_data[1] - pc_data[0]).astype(np.float32)  # shape: (N, 3)
    instance_labels = pc_data[4].astype(np.int32)

    flow_stats_per_point = compute_instance_flow_features(pc_id, flow*100, instance_labels, pc_data[3].astype(np.int64))  # shape: (N, 3)
    points = np.concatenate([pc_data[0],pc_data[1],pc_data[2],flow_stats_per_point], axis=-1).astype(np.float32)

    return PointCloud(
        pc_id=pc_id,
        obj_cat=object_cat,
        points=points,  # shape: (N, 12)
        sem_labels=pc_data[3].astype(np.int64),
        instance_labels=instance_labels,
        gt_npcs=pc_data[5].astype(np.float32),
    )





def train(args, epochs, net, train_loader, val_loader, textio):
    val_class_instance_counter = [0] * net.num_part_classes
    val_min_points_per_class = [3] * net.num_part_classes
    val_min_points_per_class_use = [3] * net.num_part_classes
    # [0, 10, 10, 10, 1000, 1000, 1000, 1000, 10, 10]
    learning_rate : float = 1e-3
    model_epoch = 0
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),lr=learning_rate,)
    best_mean_ap50 = 0.0
    splits = ["val", "test_intra", "test_inter"]
    for epoch in tqdm(range(epochs)):
        file_count = 0
        net.train()
        textio.cprint('==epoch: %d, learning rate: %f==' % (epoch, opt.param_groups[0]['lr']))

        total_loss = 0.0
        num_batches = 0

        for pc in tqdm(train_loader, desc=f'Training Epoch {epoch}'):
            file_count += len(pc)  # 统计有效文件数
            pc = [Point.to(net.device) for Point in pc]  # List["PointCloud"]
            data_batch = PointCloud.collate(pc)  # PointCloudBatch
            _, _, _, stats_dict = net(data_batch)

            # 计算 loss
            loss = sum([stats_dict[key] for key in stats_dict.keys() if
                        'loss' in key and isinstance(stats_dict[key], torch.Tensor)])

            total_loss += loss.item()  # 累加 loss 值
            num_batches += 1  # 统计 batch 数量

            opt.zero_grad()
            loss.backward()
            opt.step()
        # 计算并打印本轮平均 loss
        avg_loss = total_loss / num_batches if num_batches > 0 else 0

        textio.cprint(f"Epoch {epoch}: Used Files = {file_count}, Skipped None Items = {len(train_loader.dataset) - file_count }, Avg Loss = {avg_loss:.4f}")
        with torch.no_grad():

            # validation
            val_file_count = 0
            net.eval()  # 切换到评估模式
            num_val_batches = 0
            val_loss = 0.0
            val_results = [[] for _ in range(len(val_loader))]  # 用于存储每个数据加载器的结果，按 dataloader_idx 划分
            

            # 遍历 val_loader 中的每个数据加载器
            for loader_idx, loader in enumerate(val_loader):  # 遍历每个数据加载器
                for pc in tqdm(loader, desc=f'Validation Epoch {splits[loader_idx]}'):
                    val_file_count += len(pc)
                    pc = [Point.to(net.device) for Point in pc]  # 将每个点云转换到正确的设备
                    data_batch = PointCloud.collate(pc)  # PointCloudBatch
                    pc_ids, sem_seg, proposals, stats_dict = net(data_batch)  # 网络输出

                    # 计算损失
                    loss = sum([stats_dict[key] for key in stats_dict.keys() if
                                'loss' in key and isinstance(stats_dict[key], torch.Tensor)])
                    val_loss += loss.item()
                    num_val_batches += 1  # 统计 batch 数量

                    
                    

                    # 处理 proposals
                    if proposals is not None:

                        proposals.pt_sem_classes = proposals.sem_preds[proposals.proposal_offsets[:-1].long()].long()
                        # print(f"beyond:{proposals.proposal_offsets.shape[0]}")
                        proposals = filter_invalid_proposals(
                            proposals,
                            score_threshold=net.val_score_threshold,
                            val_min_points_per_class=val_min_points_per_class_use,
                        )
                        # print(f"after:{proposals.proposal_offsets.shape[0]}")
                        proposals = apply_nms(proposals, net.val_nms_iou_threshold)   #非极大值抑制（NMS），用来过滤掉重叠太多的重复 proposal
                        proposals.pt_sem_classes = proposals.sem_preds[proposals.proposal_offsets[:-1].long()]
                        # analyze_proposals_sorted_by_score(proposals)
                        proposals_ = Instances(
                            score_preds=proposals.score_preds,
                            pt_sem_classes=proposals.pt_sem_classes,
                            batch_indices=proposals.batch_indices,
                            instance_sem_labels=proposals.instance_sem_labels,
                            ious=proposals.ious,
                            proposal_offsets=proposals.proposal_offsets,
                            valid_mask=proposals.valid_mask
                        )
                    else:
                        proposals_ = None
                    # 将当前批次的结果根据 dataloader_idx 存储到 val_results
                    val_results[loader_idx].append((pc_ids, sem_seg, proposals_))
                    del proposals, proposals_
            avg_val_loss = val_loss / num_val_batches if num_val_batches > 0 else 0

            textio.cprint(f"Val Files = {val_file_count}, Skipped None Items = {(sum(len(loader.dataset) for loader in val_loader)) - val_file_count}, Avg Val Loss = {avg_val_loss:.4f}")


            all_accus = []
            pixel_accus = []
            # mious = []
            mean_ap50 = []
            # mAPs = []
            for i_, val_result in enumerate(val_results):
                # textio.cprint(f"[Debug] Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

                split = splits[i_]
                # pc_ids = [i for x in val_result for i in x[0]]
                # batch_size = val_result[0][1].batch_size
                # data_size = sum(x[1].batch_size for x in val_result)
                all_accu = sum(x[1].all_accu for x in val_result) / len(val_result)  #取自sem_seg
                pixel_accu = sum(x[1].pixel_accu for x in val_result) / len(val_result)  #取自sem_seg
                # semantic segmentation
                # sem_preds = torch.cat(
                #     [x[1].sem_preds for x in val_result], dim=0
                # )
                # sem_labels = torch.cat(
                #     [x[1].sem_labels for x in val_result], dim=0
                # )
                # miou = mean_iou(sem_preds, sem_labels, num_classes=net.num_part_classes)
                # instance segmentation
                proposals = [x[2] for x in val_result if x[2] != None]

                # semantic segmentation
                all_accus.append(all_accu)
                # mious.append(miou)
                pixel_accus.append(pixel_accu)


                # instance segmentation

                thes = [0.5 + 0.05 * i for i in range(10)]
                aps = []
                for the in thes: #if self.current_epoch >= self.start_scorenet:
                    ap = compute_ap(proposals, net.num_part_classes, the)
                    aps.append(ap)
                    if the == 0.5:
                        ap50 = ap
                # mAP = np.array(aps).mean()
                # mAPs.append(mAP)

                # if self.current_epoch >= self.start_scorenet:
                # 记录 AP@50
                for class_idx in range(1, net.num_part_classes):
                    partname = PART_ID2NAME[class_idx]
                    textio.cprint(f"Validation {split}/AP@50_{partname}: {np.mean(ap50[class_idx - 1]) * 100:.2f}%")
                mean_ap50.append(np.mean(ap50))
                del  proposals, ap
                del  split, all_accu, pixel_accu
                thes.clear()
                aps.clear()




            textio.cprint("------results of test_inter------")
            textio.cprint(f"mean_all_accu: {(all_accus[0]) * 100.0:.2f}")
            textio.cprint(f"mean_pixel_accu: {(pixel_accus[0]) * 100.0:.2f}")
            # textio.cprint(f"mean_miou: {(mious[0]) * 100.0:.2f}")
            textio.cprint(f"mean_AP@50: {(mean_ap50[0]) * 100.0:.2f}")
            # textio.cprint(f"mean_mAP: {(mAPs[0]) * 100.0:.2f}")
            mean_ap50 = mean_ap50[0]
            if mean_ap50 >= best_mean_ap50:
                torch.save(net.state_dict(), f'checkpoints/{args.exp_name}/models/final_model.pth')
                textio.cprint(f"Save the model from epoch {epoch}")
                best_mean_ap50 = mean_ap50
                model_epoch = epoch
        all_accus.clear()

        pixel_accus.clear()
        # mious.clear()
        del mean_ap50
        # mAPs.clear()
        val_results.clear()
        torch.cuda.empty_cache()
    textio.cprint(f"model from epoch {model_epoch}")
def test(net, test_loader, test_epochs, textio):
    for epoch in tqdm(range(test_epochs)):
        num_proposals = 0
        val_min_points_per_class_use = [3] * net.num_part_classes
        with torch.no_grad():
            net.eval()
            num_test_batches = 0
            test_results = [[] for _ in range(len(test_loader))]
            splits = ["val", "test_intra", "test_inter"]

            for loader_idx, loader in enumerate(test_loader):
                test_loss = 0.0
                test_file_count = 0

                for pc in tqdm(loader, desc=f'Validation Epoch {splits[loader_idx]}'):

                    if len(pc) == 0 :
                        continue  # 跳过空数据

                    test_file_count += len(pc)
                    pc = [Point.to(net.device) for Point in pc]
                    data_batch = PointCloud.collate(pc)
                    pc_ids, sem_seg, proposals, stats_dict = net(data_batch)

                    # 计算损失
                    loss = sum([stats_dict[key] for key in stats_dict.keys() if
                                'loss' in key and isinstance(stats_dict[key], torch.Tensor)])
                    test_loss += loss.item()
                    num_test_batches += 1  # 统计 batch 数量

                    # 处理 proposals
                    if proposals is not None:
                        proposals.pt_sem_classes = proposals.sem_preds[proposals.proposal_offsets[:-1].long()].long()
                        proposals = filter_invalid_proposals(
                            proposals,
                            score_threshold=net.val_score_threshold,
                            val_min_points_per_class=val_min_points_per_class_use,
                        )
                        proposals = apply_nms(proposals, net.val_nms_iou_threshold)  # 非极大值抑制（NMS），用来过滤掉重叠太多的重复 proposal
                        proposals.pt_sem_classes = proposals.sem_preds[proposals.proposal_offsets[:-1].long()]

                        proposals_ = Instances(
                            score_preds=proposals.score_preds,
                            pt_sem_classes=proposals.pt_sem_classes,
                            batch_indices=proposals.batch_indices,
                            instance_sem_labels=proposals.instance_sem_labels,
                            ious=proposals.ious,
                            proposal_offsets=proposals.proposal_offsets,
                            valid_mask=proposals.valid_mask
                        )
                        num_proposals += proposals.proposal_offsets.shape[0] - 1
                    else:
                        proposals_ = None

                    # 将当前批次的结果根据 dataloader_idx 存储到 val_results
                    test_results[loader_idx].append((pc_ids, sem_seg, proposals_))
                avg_test_loss =test_loss / num_test_batches if num_test_batches > 0 else 0
                textio.cprint(f"Val Files = {test_file_count}, Skipped None Items = {(len(loader.dataset)) - test_file_count}, Avg Val Loss = {avg_test_loss:.4f}")

            all_accus = []
            pixel_accus = []
            mious = []
            mean_ap50 = []
            mAPs = []
            category_ap50 = []
            for i_, test_result in enumerate(test_results):

                split = splits[i_]
                pc_ids = [i for x in test_result for i in x[0]]
                batch_size = test_result[0][1].batch_size
                data_size = sum(x[1].batch_size for x in test_result)
                all_accu = sum(x[1].all_accu for x in test_result) / len(test_result)  # 取自sem_seg
                pixel_accu = sum(x[1].pixel_accu for x in test_result) / len(test_result)  # 取自sem_seg
                # semantic segmentation
                sem_preds = torch.cat(
                    [x[1].sem_preds for x in test_result], dim=0
                )
                sem_labels = torch.cat(
                    [x[1].sem_labels for x in test_result], dim=0
                )
                miou = mean_iou(sem_preds, sem_labels, num_classes=net.num_part_classes)
                # instance segmentation
                proposals = [x[2] for x in test_result if x[2] != None]

                # semantic segmentation
                all_accus.append(all_accu)
                mious.append(miou)
                pixel_accus.append(pixel_accu)

                # instance segmentation

                thes = [0.5 + 0.05 * i for i in range(10)]
                aps = []
                for the in thes:  # if self.current_epoch >= self.start_scorenet:
                    ap = compute_ap(proposals, net.num_part_classes, the)
                    aps.append(ap)
                    if the == 0.5:
                        ap50 = ap
                mAP = np.array(aps).mean()
                mAPs.append(mAP)

                # if self.current_epoch >= self.start_scorenet:
                # 记录 AP@50
                for class_idx in range(1, net.num_part_classes):
                    partname = PART_ID2NAME[class_idx]
                    textio.cprint(f"Validation {split}/AP@50_{partname}: {np.mean(ap50[class_idx - 1]) * 100:.2f}%")
                mean_ap50.append(np.mean(ap50))

                del sem_preds, sem_labels, proposals, ap, aps, ap50
                del pc_ids, split, batch_size, data_size, all_accu, pixel_accu, miou
                torch.cuda.empty_cache()
            for i in range(len(all_accus)):
                textio.cprint(f"------results of {splits[i]}------")
                textio.cprint(f"mean_all_accu: {(all_accus[i]) * 100.0:.2f}")
                textio.cprint(f"mean_pixel_accu: {(pixel_accus[i]) * 100.0:.2f}")
                textio.cprint(f"mean_miou: {(mious[i]) * 100.0:.2f}")
                textio.cprint(f"mean_AP@50: {(mean_ap50[i]) * 100.0:.2f}")
                textio.cprint(f"mean_mAP: {(mAPs[i]) * 100.0:.2f}")
        test_results.clear()
        torch.cuda.empty_cache()



def main():
    # CUDA settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    np.random.seed(1234)

    parser = argparse.ArgumentParser(description='Point Cloud Registration')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--test', action='store_true', default=False,
                        help='evaluate the model')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to pretrained model. If None, train from scratch.')
    args = parser.parse_args()
    _init_(args)

    textio = IOStream('checkpoints/' + args.exp_name + '/run.log')
    textio.cprint(str(args))


    net = GAPartNet().cuda()
    if args.model_path:
        textio.cprint(f'Loading pretrained model from {args.model_path}')
        net.load_state_dict(torch.load(args.model_path))
        textio.cprint('Pretrained model loaded successfully, resuming training...')
    else:
        textio.cprint('No pretrained model provided, training from scratch...')
        net.apply(weights_init)
    # if torch.cuda.device_count() > 1:
    #     net = nn.DataParallel(net).cuda()
    # else:
    #     net = net.to(device)

    print("Let's use", torch.cuda.device_count(), net.device, "GPUs!")
    root_dir: str = "/16T/liuyuyan/GAPartNetAllWithFlows"
    max_points: int = 20000
    voxel_size: Tuple[float, float, float] = (1 / 100, 1 / 100, 1 / 100)
    train_batch_size: int = 32#32
    val_batch_size: int = 32
    test_batch_size: int = 32
    num_workers: int = 80
    pos_jitter: float = 0.
    color_jitter: float = 0.1
    flip_prob: float = 0.
    rotate_prob: float = 0.
    train_few_shot: bool = False
    val_few_shot: bool = False
    intra_few_shot: bool = False
    inter_few_shot: bool = False
    few_shot_num: int = 256
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
    val_data_files = GAPartNetDataset(
        Path(root_dir) / "val" / "pth",
        shuffle=True,
        max_points=max_points,
        augmentation=False,
        voxel_size=voxel_size,
        few_shot=val_few_shot,
        few_shot_num=few_shot_num,
        pos_jitter=pos_jitter,
        color_jitter=color_jitter,
        flip_prob=flip_prob,
        rotate_prob=rotate_prob,
    )
    intra_data_files = GAPartNetDataset(
        Path(root_dir) / "test_intra" / "pth",
        shuffle=True,
        max_points=max_points,
        augmentation=False,
        voxel_size=voxel_size,
        few_shot=intra_few_shot,
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

    # 创建 val_dataloader
    val_dataloader = [
        # DataLoader(val_data_files,
        #                         batch_size=val_batch_size,
        #                         shuffle=False,
        #                         num_workers=num_workers,
        #                         collate_fn=collate_fn,
        #                         pin_memory=True,
        #                         drop_last=False
        #                         ),
        # DataLoader(intra_data_files,
        #                          batch_size=val_batch_size,
        #                          shuffle=False,
        #                          num_workers=num_workers,
        #                          collate_fn=collate_fn,
        #                          pin_memory=True,
        #                          drop_last=False
        #                          ),
        DataLoader(inter_data_files,
                                 batch_size=val_batch_size,
                                 shuffle=False,
                                 num_workers=num_workers,
                                 collate_fn=collate_fn,
                                 pin_memory=True,
                                 drop_last=False
                                 ),
                      ]
    test_dataloader = [
        DataLoader(val_data_files,
                                batch_size=test_batch_size,
                                shuffle=False,
                                num_workers=num_workers,
                                collate_fn=collate_fn,
                                pin_memory=True,
                                drop_last=False
                                ),
        DataLoader(intra_data_files,
                                 batch_size=test_batch_size,
                                 shuffle=False,
                                 num_workers=num_workers,
                                 collate_fn=collate_fn,
                                 pin_memory=True,
                                 drop_last=False
                                 ),
        DataLoader(inter_data_files,
                   batch_size=test_batch_size,
                   shuffle=False,
                   num_workers=num_workers,
                   collate_fn=collate_fn,
                   pin_memory=True,
                   drop_last=False
                   ),
    ]
    if not args.test:
        # 训练模型
        train(args, 700, net, train_dataloader, val_dataloader, textio)
    else:
        test(net, test_dataloader, 200, textio)



    print('FINISH')
if __name__ == '__main__':
    main()