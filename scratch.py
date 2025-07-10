from __future__ import print_function
from PIL import Image
import os
from argparse import ArgumentParser

import importlib
import torch
import open3d as o3d
from os.path import join as pjoin

import copy
from pathlib import Path
from typing import Optional, Tuple, Union, List
from epic_ops.voxelize import voxelize
from torch.utils.data import Dataset
import random
import open3d as o3d
from glob import glob
from tqdm import tqdm
from structure.point_cloud import PointCloud
from misc.info import OBJECT_NAME2ID, PART_ID2NAME, PART_NAME2ID, get_symmetry_matrix
import argparse
import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import ConcatDataset, DataLoader
from network.model import GAPartNet

import numpy as np
from numpy.random.mtrand import sample

import torch

CUDA = torch.cuda.is_available()
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
    if pc_data[2].shape != (20000, 3):
        return None
    flow = (pc_data[1] - pc_data[0]).astype(np.float32)  # shape: (N, 3)
    points = np.concatenate([pc_data[0],pc_data[2]], axis=-1).astype(np.float32)
    print(pc_data[0])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_data[0])
    pcd.colors = o3d.utility.Vector3dVector(pc_data[2])
    # 可视化
    o3d.visualization.draw_geometries([pcd])
    return PointCloud(
        pc_id=pc_id,
        obj_cat=None,
        points=points,  # shape: (N, 12)
        sem_labels=None,
        instance_labels=None,
        gt_npcs=None,
    )
def farthest_point_sample(xyz, npoint):
    """
    Copied from CAPTRA

    Input:
        xyz: pointcloud data, [B, N, 3], tensor
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    # return torch.randint(0, N, (B, npoint), dtype=torch.long).to(device)
    # if CUDA:
    #     print('Use pointnet2_cuda!')
    #     idx = futils.furthest_point_sample(xyz, npoint).long()
    #     return idx

    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B, ), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid)**2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def FPS(pcs0, npoint):
    """
    Input:
        pcs0: pointcloud data, [N, 3]
        npoint: number of samples
    Return:
        sampled_pcs0: [npoint, 3]
        fps_idx: sampled pointcloud index, [npoint, ]
    """
    if pcs0.shape[0] < npoint:
        print('Error! shape[0] of point cloud is less than npoint!')
        return None, None

    if pcs0.shape[0] == npoint:
        return pcs0, np.arange(pcs0.shape[0])

    pcs0_tensor = torch.from_numpy(np.expand_dims(pcs0, 0)).float()
    fps_idx_tensor = farthest_point_sample(pcs0_tensor, npoint)
    fps_idx = fps_idx_tensor.cpu().numpy()[0]
    sampled_pcs0 = pcs0[fps_idx]
    return sampled_pcs0, fps_idx

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
def get_point_cloud_from_mask(color_mask, depth_mask):

    # 相机内参（已写死）
    fx, fy = 899.1470336914062, 899.4546508789062
    cx, cy = 665.6021118164062, 365.4790344238281

    # 找出 mask 区域的有效像素索引
    valid_pixels = np.argwhere(depth_mask > 0)

    # 提前分配点云和颜色列表
    point_cloud = []
    per_point_rgb = []
    per_point_idx = []

    for y_, x_ in valid_pixels:
        z_new = float(depth_mask[y_, x_])
        x_new = (x_ - cx) * z_new / fx
        y_new = (y_ - cy) * z_new / fy
        point_cloud.append([x_new, y_new, z_new])
        per_point_rgb.append(color_mask[y_, x_] / 255.0)
        per_point_idx.append([y_, x_])

    return np.array(point_cloud), np.array(per_point_rgb), np.array(per_point_idx)

def sample_and_save(model, filename, save_path, num_points, gt_flow, visualize=False):
    pth_save_path = pjoin(save_path, 'pth')
    os.makedirs(pth_save_path, exist_ok=True)
    meta_save_path = pjoin(save_path, 'meta')
    os.makedirs(meta_save_path, exist_ok=True)
    color_mask_array_0 = np.array(Image.open('color_0_mask.png'))
    depth_mask_array_0 = np.array(Image.open('depth_0_mask.png'))
    color_mask_array_1 = np.array(Image.open('color_1_mask.png'))
    depth_mask_array_1 = np.array(Image.open('depth_1_mask.png'))
    # Get point cloud from back-projection
    pcs0, pcs0_rgb, pcs0_idx = get_point_cloud_from_mask(color_mask_array_0, depth_mask_array_0)
    pcs1, pcs1_rgb, pcs1_idx = get_point_cloud_from_mask(color_mask_array_1, depth_mask_array_1)

    if pcs0.shape[0] < num_points or pcs1.shape[0] < num_points:
        return -1
    else:
        # FPS sampling
        pcs0_sampled, fps_idx0 = FPS(pcs0, num_points)
        pcs1_sampled, fps_idx1 = FPS(pcs1, num_points)
        if pcs0_sampled is None or  pcs1_sampled is None:
            return -1

    pcs0_rgb_sampled = pcs0_rgb[fps_idx0]
    pcs0_idx_sampled = pcs0_idx[fps_idx0].astype(np.int32)
    pcs1_rgb_sampled = pcs0_rgb[fps_idx1]
    pcs1_idx_sampled = pcs0_idx[fps_idx1].astype(np.int32)

    # normalize point cloud
    pcs0_sampled_normalized, max_radius, center = WorldSpaceToBallSpace(pcs0_sampled)
    pcs1_sampled_normalized = (pcs1_sampled - center) / max_radius
    scale_param = np.array([max_radius, center[0], center[1], center[2]])

    points0 = torch.tensor(pcs0_sampled_normalized * 10, dtype=torch.float32).unsqueeze(0).cuda()  # (1, N, 3)
    points1 = torch.tensor(pcs1_sampled_normalized * 10, dtype=torch.float32).unsqueeze(0).cuda()  # (1, N, 3)
    color0 = torch.tensor(pcs0_rgb[fps_idx0], dtype=torch.float32).unsqueeze(0).cuda()  # (1, N, 3)
    color1 = torch.tensor(pcs1_rgb[fps_idx1], dtype=torch.float32).unsqueeze(0).cuda()  # (1, N, 3)
    model = model.eval()
    with torch.no_grad():
        flows, fps_pc1_idxs, _, _, _ = model(points0, points1, color0, color1, gt_flow)
    pcs_flow_sampled = flows[0][3].squeeze(0).detach().cpu().numpy().transpose(1, 0)  # flow_np:(2048,3)
    pcs0_with_flow = pcs0_sampled_normalized + pcs_flow_sampled / 10
    print(pcs_flow_sampled.shape)
    # visualize
    if visualize:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcs0_with_flow)
        pcd.colors = o3d.utility.Vector3dVector(pcs0_rgb_sampled)
        o3d.visualization.draw_geometries([pcd])

    torch.save(
        (pcs0_sampled_normalized.astype(np.float32), pcs0_with_flow.astype(np.float32), pcs0_rgb[fps_idx0].astype(
            np.float32), pcs0_idx_sampled.astype(np.int32)),
        pjoin(pth_save_path, filename + '.pth'))
    np.savetxt(pjoin(meta_save_path, filename + '.txt'), scale_param, delimiter=',')


    return 0
import lightning.pytorch as pl
from network.model import GAPartNet
def inference_single(model_ckpt_path, device='cuda'):
    # 1. 加载模型
    model = GAPartNet.load_from_checkpoint(
    model_ckpt_path,
    num_part_classes=10,   # 必须显式传
    debug=True,
    in_channels=6,
    backbone_type="SparseUNet",
    backbone_cfg={"channels": [16,32,48,64,80,96,112], "block_repeat": 2},
    instance_seg_cfg={
        "ball_query_radius": 0.04,
        "max_num_points_per_query": 50,
        "min_num_points_per_proposal": 5,
        "max_num_points_per_query_shift": 300,
        "score_fullscale": 28,
        "score_scale": 50,
    },
    learning_rate=0.001,
    ignore_sem_label=-100,
    use_sem_focal_loss=True,
    use_sem_dice_loss=True,
    training_schedule=[5, 10],
    val_nms_iou_threshold=0.3,
    val_ap_iou_threshold=0.5,
    symmetry_indices=[0, 1, 3, 3, 2, 0, 3, 2, 4, 1],
    visualize_cfg={
        "visualize": False,
        "visualize_dir": "visu",
        "sample_num": 10,
        "RAW_IMG_ROOT": "data/image_kuafu",
        "GAPARTNET_DATA_ROOT": "data/GAPartNet_All",
        "SAVE_ROOT": "output/GAPartNet_result",
        "save_option": ["raw", "pc", "sem_pred", "sem_gt", "ins_pred", "ins_gt", "npcs_pred", "npcs_gt", "bbox_gt", "bbox_gt_pure", "bbox_pred", "bbox_pred_pure"]
    }
)
    model.eval()
    model.to(device)

    # 2. 加载数据
    file = load_data("/home/liuyuyan/OISR/OISR/dataset/process_tools/output/pth/test.pth", no_label=False)
    # file = load_data("/16T/liuyuyan/GAPartNetAllWithFlows/test_inter/pth/Safe_102381_0_14.pth", no_label=False)
    
    file = file.to_tensor()
    pc = apply_voxelization(file, voxel_size=(1 / 100, 1 / 100, 1 / 100))
    if pc is None:
        print("Data loading failed or point cloud format is invalid.")
        return
    pc = [pc.to(model.device)]
    data_batch = PointCloud.collate(pc)  # PointCloudBatch

    with torch.no_grad():
        pc_ids, sem_seg, proposals, stats_dict = model(data_batch, 0)

    # 4. 输出结果
    print("pc_ids:", pc_ids)
    print("================================================")
    print("sem_seg:", sem_seg)
    print("================================================")
    print("proposals:", proposals)
    print("================================================")
    print("stats_dict:", stats_dict)
    return 0

if __name__ == "__main__":
    # 设置设备
    print(f"Using GPU: '{'cuda'}'")
    # module = importlib.import_module("model_difflow")
    # model = getattr(module, 'PointConvBidirection')(iters=4)
    # pretrain = "model_difflow_355_0.0114.pth"
    # model.load_state_dict(torch.load(pretrain))  # , strict=False
    # print(f'Loaded model {pretrain}')
    # model = model.to('cuda')
    # gt_flow = torch.tensor(np.loadtxt('flow_np.txt') * 50, dtype=torch.float32).unsqueeze(0).cuda()
    # sample_and_save(model,'test', 'output/', 20000,gt_flow, True)
    # model_path = "/home/liuyuyan/OISR/OISR/checkpoints/sliderlossepoch10/models/final_model.pth"
    # net1 = GAPartNet().cuda()
    # print(f'Loading pretrained model from {model_path}')
    # net1.load_state_dict(torch.load(model_path))

    pl.seed_everything(42)
    inference_single("/home/liuyuyan/GAPartNet-release/gapartnet/ckpt/release.ckpt")