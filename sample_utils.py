import os
import sys
from os.path import join as pjoin
import numpy as np
from numpy.random.mtrand import sample

import torch

CUDA = torch.cuda.is_available()

if CUDA:
    import utils.pointnet_lib.pointnet2_utils as futils


# def farthest_point_sample(xyz, npoint):
#     """
#     Input:
#         xyz: pointcloud data, [B, N, 3]
#         npoint: number of samples
#     Return:
#         centroids: sampled pointcloud index, [B, npoint]
#     """
#     device = xyz.device
#     B, N, C = xyz.shape
#     centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
#     distance = torch.ones(B, N).to(device) * 1e10
#     is_padding = torch.all(xyz == 0, dim=-1)  # Find padding points
#     valid_points_count = (~is_padding).sum(dim=-1)  # Count valid points
#
#     # 为每个batch选择有效点的索引
#     farthest = torch.zeros(B, dtype=torch.long).to(device)
#     for i in range(B):
#         valid_indices = torch.arange(valid_points_count[i], device=xyz.device)
#         farthest[i] = valid_indices[torch.randint(0, valid_points_count[i], (1,))]
#
#     batch_indices = torch.arange(B, dtype=torch.long).to(device)
#     for i in range(npoint):
#         if (farthest >= valid_points_count).any():
#             raise ValueError(f"{i}:Selected point index {farthest} is out of bounds for the valid points!{valid_points_count}")
#         centroids[:, i] = farthest
#         centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
#         dist = torch.sum((xyz - centroid) ** 2, -1)
#         dist[is_padding] = float('inf')
#         mask = dist < distance
#         distance[mask] = dist[mask]
#         distance[is_padding] = 0.0
#         farthest = torch.max(distance, -1)[1]
#     return centroids

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
    if CUDA:
        idx = futils.furthest_point_sample(xyz, npoint).long()
        return idx

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



def FPS(pcs, npoint, device):
    """
    Input:
        pcs: pointcloud data, [N, 3]
        npoint: number of samples
    Return:
        sampled_pcs: [npoint, 3]
        fps_idx: sampled pointcloud index, [npoint, ]
    """
    if pcs.shape[0] < npoint:
        print('Error! shape[0] of point cloud is less than npoint!')
        return None, None

    if pcs.shape[0] == npoint:
        return pcs, np.arange(pcs.shape[0])

    pcs_tensor = torch.from_numpy(np.expand_dims(pcs, 0)).float().to(device)
    fps_idx_tensor = farthest_point_sample(pcs_tensor, npoint)
    fps_idx = fps_idx_tensor.cpu().numpy()[0]
    sampled_pcs = pcs[fps_idx]
    return sampled_pcs, fps_idx
# def FPS(pcs, npoint, device):
#     """
#     Input:
#         pcs: NumPy 数组, 形状为 [B, N, 3]，表示批量点云数据
#         npoint: 需要采样的点数
#         device: 运行设备 (cuda/cpu)
#     Output:
#         sampled_pcs: 采样后的点云, 形状 [B, npoint, 3]
#         fps_idx: 采样点的索引, 形状 [B, npoint]
#     """
#     B, N, C = pcs.shape  # B: batch size, N: 点数, C: 3 维坐标

#     if N < npoint:
#         print("Error! N (点数) 小于 npoint (采样数)!")
#         return None, None

#     # 转换 NumPy 到 PyTorch Tensor
#     pcs_tensor = torch.tensor(pcs, dtype=torch.float32, device=device)

#     # 进行最远点采样
#     fps_idx_tensor = farthest_point_sample(pcs_tensor, npoint).long()  # 形状: [B, npoint]

#     # 通过 gather 进行索引
#     sampled_pcs_tensor = torch.gather(pcs_tensor, 1, fps_idx_tensor.unsqueeze(-1).expand(-1, -1, 3))

#     # 转换回 NumPy
#     sampled_pcs = sampled_pcs_tensor.cpu().numpy()
#     fps_idx = fps_idx_tensor.cpu().numpy()

#     return sampled_pcs, fps_idx




if __name__ == "__main__":
    pc = np.random.random((50000, 3))
    pc_sampled, idx = FPS(pc, 10000)
    print(pc_sampled)
    print(idx)