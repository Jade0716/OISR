from train import GAPartNetDataset, collate_fn
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
from pyntcloud import PyntCloud
from pandas import DataFrame


def PCA(data, correlation=False, sort=True):
    data_mean = np.mean(data, axis=0)
    data_normalize = data - data_mean
    H = np.dot(data_normalize.T, data_normalize)
    eigenvectors, eigenvalues, eigenvectors_t = np.linalg.svd(H)
    if sort:
        sort = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:, sort]
    return eigenvectors, eigenvalues




def find_farthest_point(file_path: str):

    points = np.loadtxt(file_path)
    distances = np.linalg.norm(points, axis=1)
    max_index = np.argmax(distances)
    farthest_point = points[max_index]
    max_distance = distances[max_index]
    print(f"文件 {file_path} 中，距离原点最远的点的坐标: {farthest_point}")
    print(f"距离: {max_distance:.6f}")


def visualize_instance(points: torch.Tensor, instance_label: torch.Tensor, pc_id):
    """
    根据实例标签可视化点云。
    :param points: torch.Tensor, 形状为(N, 3)，表示原始点云。
    :param instance_label: torch.Tensor, 形状为(N,) ，表示实例标签。
    :param pc_id: str, 需要显示在窗口上的点云ID。
    """
    assert points.shape[0] == instance_label.shape[0], "points 和 instance_label 维度不匹配"

    points_np = points.cpu().numpy()
    instance_label_np = instance_label.cpu().numpy()

    # 生成不同实例的颜色
    unique_labels = np.unique(instance_label_np)
    colors = plt.cm.get_cmap("tab10", len(unique_labels))
    color_map = {label: colors(i)[:3] for i, label in enumerate(unique_labels)}

    # 颜色映射到点云
    point_colors = np.array([color_map[label] for label in instance_label_np])

    # 创建 Open3D 点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)
    pcd.colors = o3d.utility.Vector3dVector(point_colors)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Point Cloud {pc_id}")
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()


def visualize_two_point_clouds(file1: str, file2: str):

    # 读取点云数据
    points1 = np.loadtxt(file1)  # 读取第一个点云
    points2 = np.loadtxt(file2)  # 读取第二个点云

    # 创建 Open3D 点云对象
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(points1)
    pcd1.paint_uniform_color([1, 0, 0])  # 红色

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(points2)
    pcd2.paint_uniform_color([0, 0, 1])  # 蓝色

    # 可视化
    o3d.visualization.draw_geometries([pcd1, pcd2])



def visualize_flow_with_instance_selection(points, flows, instance_labels: np.ndarray, selected_labels: list, pc_id):



    # 转换为 numpy 数组并计算移动后的点云位置
    points_np = points[:,:3]
    flows_np = points[:,3:6] - points[:,:3]
    points_flowed = points[:,3:6]

    # 根据选定的实例标签过滤点
    mask = np.isin(instance_labels, selected_labels) & (instance_labels != -100)
    filtered_points_np = points_np[mask]
    filtered_flows_np = flows_np[mask]
    filtered_points_flowed = filtered_points_np + filtered_flows_np

    # 创建原始点云 Open3D 对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)
    pcd.paint_uniform_color([1, 0, 0])  # 红色
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(points[:,3:6])
    pcd1.paint_uniform_color([0, 1, 0])

    # 创建线集来表示点与它们的流动方向
    lines = [[i, i+len(filtered_points_np)] for i in range(len(filtered_points_np))]
    colors = [[0, 0, 1] for _ in range(len(lines))]  # 线条颜色为蓝色
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.concatenate((filtered_points_np, filtered_points_flowed)))
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    # 可视化
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Point Cloud {pc_id}")
    vis.add_geometry(pcd)  # 添加原始点云
    vis.add_geometry(pcd1)  # 添加原始点云
    vis.add_geometry(line_set)  # 添加线条集
    vis.run()
    vis.destroy_window()
    print("Visualization finished")

def visualize_flow_with_instance_colors(points, flows, instance_labels: np.ndarray, pc_id):
    # 原始点和 flow（偏移量）
    points_np = points[:, :3]
    flows_np = points[:, 3:6] - points[:, :3]
    points_flowed = points[:, 3:6]

    # 获取所有有效的实例标签
    valid_mask = instance_labels != -100
    unique_labels = np.unique(instance_labels[valid_mask])
    num_labels = len(unique_labels)

    # 使用 matplotlib colormap 分配颜色
    cmap = plt.get_cmap("tab20", num_labels)
    label2color = {label: cmap(i)[:3] for i, label in enumerate(unique_labels)}

    # 设置点云颜色
    point_colors = np.array([label2color.get(label, [0, 0, 0]) for label in instance_labels])

    # 创建点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)
    pcd.colors = o3d.utility.Vector3dVector(point_colors)

    # 创建 flow 线条
    lines = []
    line_colors = []
    line_points = []

    for i in range(len(points_np)):
        if instance_labels[i] == -100:
            continue
        start = points_np[i]
        end = points_flowed[i]
        color = label2color[instance_labels[i]]  # 同样使用实例颜色

        line_points.append(start)
        line_points.append(end)
        lines.append([2 * i, 2 * i + 1])
        line_colors.append(color)

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.array(line_points))
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(np.array(line_colors))

    # 可视化
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Point Cloud {pc_id}")
    vis.add_geometry(pcd)
    vis.add_geometry(line_set)
    vis.run()
    vis.destroy_window()
    print("Visualization finished")


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
def visualize_flow_with_all_axis(points, flows, instance_labels: np.ndarray, selected_labels: list, pc_id,
                                           axis_data):

    # 确保 points 和 flows 维度匹配，并且与 instance_labels 的长度一致
    assert points.shape == flows.shape and points.shape[0] == instance_labels.shape[0], "维度不匹配"

    # 转换为 numpy 数组并计算移动后的点云位置
    points_np = points
    flows_np = flows
    points_flowed = points_np + flows_np

    # 根据选定的实例标签过滤点
    mask = np.isin(instance_labels, selected_labels) & (instance_labels != -100)
    filtered_points_np = points_np[mask]
    filtered_flows_np = flows_np[mask]
    filtered_points_flowed = filtered_points_np + filtered_flows_np

    # 创建原始点云 Open3D 对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)
    pcd.paint_uniform_color([1, 0, 0])  # 红色
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(points_flowed)
    pcd1.paint_uniform_color([0, 1, 0])

    # 创建线集来表示点与它们的流动方向
    lines = [[i, i + len(filtered_points_np)] for i in range(len(filtered_points_np))]
    colors = [[0, 0, 1] for _ in range(len(lines))]  # 线条颜色为蓝色
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.concatenate((filtered_points_np, filtered_points_flowed)))
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)



    # 绘制旋转轴
    axis_lines = []
    axis_points = []
    axis_colors = []
    if axis_data.ndim == 1:
        axis_data = axis_data.reshape(1, -1)
    for axis in axis_data:
        p_opt = axis[:3]  # 旋转轴上的点
        n_opt = axis[3:]  # 旋转轴方向
        n_opt = n_opt / np.linalg.norm(n_opt)  # 归一化方向

        # 设定旋转轴的长度
        k = 3
        p_end = p_opt + k * n_opt  # 终点
        p_opt = p_opt - k * n_opt
        start_idx = len(axis_points)
        axis_points.append(p_opt)
        axis_points.append(p_end)
        axis_lines.append([start_idx, start_idx + 1])
        axis_colors.append([1, 0, 0])

    # 创建旋转轴的 LineSet
    if axis_points:
        axis_line_set = o3d.geometry.LineSet()
        axis_line_set.points = o3d.utility.Vector3dVector(np.array(axis_points))
        axis_line_set.lines = o3d.utility.Vector2iVector(axis_lines)
        axis_line_set.colors = o3d.utility.Vector3dVector(axis_colors)

    # 可视化
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Point Cloud {pc_id}")
    vis.add_geometry(pcd)  # 添加原始点云
    vis.add_geometry(pcd1)  # 添加变换后的点云
    vis.add_geometry(line_set)  # 添加流动方向线
    if axis_points:
        vis.add_geometry(axis_line_set)  # 添加旋转轴线

    vis.run()
    vis.destroy_window()
    print("Visualization finished")

def visualize_offsets(pt_xyz: np.ndarray, instance_regions: np.ndarray, instance_labels: np.ndarray):

    # 红色点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pt_xyz)
    pcd.paint_uniform_color([1.0, 0.0, 0.0])

    points = []
    lines = []
    colors = []

    line_id = 0
    for i in range(pt_xyz.shape[0]):
        if instance_labels[i] == -100:
            continue

        start = pt_xyz[i]
        end = instance_regions[i, :3]  # 中心点
        points.append(start)
        points.append(end)
        lines.append([line_id, line_id + 1])
        colors.append([0.0, 0.0, 1.0])  # 蓝色
        line_id += 2

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pcd, line_set])

def visualize_point_cloud(points):

    pcd0 = o3d.geometry.PointCloud()
    pcd0.points = o3d.utility.Vector3dVector(points[:,:3])
    pcd0.paint_uniform_color([1, 0, 0])  # 红色
    # pcd1 = o3d.geometry.PointCloud()
    # pcd1.points = o3d.utility.Vector3dVector(points[:,3:6])
    # pcd1.paint_uniform_color([0, 0, 1])  # 红色

    # 使用Open3D可视化点云
    o3d.visualization.draw_geometries([pcd0])
def main():
    # parser = argparse.ArgumentParser(description='Point Cloud Registration')
    # parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
    #                     help='Name of the experiment')
    # parser.add_argument('--eval', action='store_true', default=False,
    #                     help='evaluate the model')
    # args = parser.parse_args()
    #
    #
    # root_dir: str = "/16T/liuyuyan/GAPartNetAllWithFlows"
    # max_points: int = 20000
    # voxel_size: Tuple[float, float, float] = (1 / 100, 1 / 100, 1 / 100)
    # train_batch_size: int = 1
    # val_batch_size: int = 32
    # test_batch_size: int = 32
    # num_workers: int = 16
    # pos_jitter: float = 0.
    # color_jitter: float = 0.1
    # flip_prob: float = 0.
    # rotate_prob: float = 0.
    # train_few_shot: bool = False
    # val_few_shot: bool = False
    # intra_few_shot: bool = False
    # inter_few_shot: bool = False
    # few_shot_num: int = 256
    # train_data_files = GAPartNetDataset(
    #     Path(root_dir) / "train" / "pth",
    #     shuffle=True,
    #     max_points=max_points,
    #     augmentation=True,
    #     voxel_size=voxel_size,
    #     few_shot=train_few_shot,
    #     few_shot_num=few_shot_num,
    #     pos_jitter=pos_jitter,
    #     color_jitter=color_jitter,
    #     flip_prob=flip_prob,
    #     rotate_prob=rotate_prob,
    # )
    # inter_data_files = GAPartNetDataset(
    #     Path(root_dir) / "test_inter" / "pth",
    #     shuffle=True,
    #     max_points=max_points,
    #     augmentation=False,
    #     voxel_size=voxel_size,
    #     few_shot=inter_few_shot,
    #     few_shot_num=few_shot_num,
    #     pos_jitter=pos_jitter,
    #     color_jitter=color_jitter,
    #     flip_prob=flip_prob,
    #     rotate_prob=rotate_prob,
    # )
    # train_dataloader = DataLoader(train_data_files,
    #                               batch_size=train_batch_size,
    #                               shuffle=True,
    #                               num_workers=num_workers,
    #                               collate_fn=collate_fn,
    #                               pin_memory=True,
    #                               drop_last=True,
    #                               )
    # val_dataloader = DataLoader(inter_data_files,
    #                              batch_size=1,
    #                              shuffle=False,
    #                              num_workers=num_workers,
    #                              collate_fn=collate_fn,
    #                              pin_memory=True,
    #                              drop_last=False
    #                              )
    # for pc in val_dataloader:
    #     pc = [Point.to('cuda') for Point in pc]  # List["PointCloud"]
    #     if len(pc)==1 and pc[0].pc_id.startswith("Door"):
    #
    #         points = pc[0].points.cpu().numpy()
    #         pc_id = pc[0].pc_id
    #         print(pc_id)
    #         instance_sem_labels = pc[0].instance_sem_labels
    #         instance_labels = pc[0].instance_labels.cpu().numpy()
    #         sem_labels = pc[0].sem_labels
    #         # 构建 Open3D 点云对象
    #         raw_xyz = points[:,:3]
    #         point_cloud_o3d = o3d.geometry.PointCloud()
    #         point_cloud_o3d.points = o3d.utility.Vector3dVector(raw_xyz)
    #
    #         point_cloud_o3d.estimate_normals(
    #             search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30)  # 你可以调节邻居点数
    #         )
    #
    #         point_cloud_o3d.orient_normals_consistent_tangent_plane(k=30)
    #         # 获取法向量
    #         normals = np.asarray(point_cloud_o3d.normals)  # (N, 3)
    #         xyz = np.asarray(point_cloud_o3d.points)  # (N, 3)
    #
    #         # 拼接 xyz 和 normals 成 (N, 6) 的数组
    #         xyz_normals = np.hstack((xyz, normals))  # shape: (N, 6)
    #         print(xyz_normals.shape)
    #         # 保存为 .npy 文件
    #         np.save(f"pointcloud_with_normals_{pc_id}.npy", xyz_normals)
    #         print(f"Saved to pointcloud_with_normals_{pc_id}.npy")
    #
    #         # 可视化点云和法向量
    #         o3d.visualization.draw_geometries([point_cloud_o3d], point_show_normal=True)
    #
    #
    #
    #         # print(instance_sem_labels)
    #         # visualize_point_cloud(points.cpu().numpy())
    #         # visualize_offsets(points[:,:3].cpu().numpy(),pc[0].instance_regions.cpu().numpy(),instance_labels)
    #         # np.savetxt(f"{pc[0].pc_id}.txt", points, fmt="%.6f")
    #         # np.savetxt(f"{pc[0].pc_id}_ins.txt", instance_labels.cpu().numpy(), fmt="%.6f")
    #         # np.savetxt(f"{pc[0].pc_id}_ins_sem.txt", instance_sem_labels.cpu().numpy(), fmt="%.6f")
    #         # flow = points[:, 3:6] - points[:, :3]
    #         # visualize_flow_with_instance_selection(points, flow, instance_labels, [0,3], 'fuck')
    #         input("continue")
    # data = np.load("/16T/wangzhi/ArtImage/laptop/train/SdfSamples/010988.npz")['sdf_data']
    # point_cloud0,_ = FPS(data[data[:, 3] == 0, :3],10000, 'cuda')
    point_cloud0 = np.loadtxt("/16T/liuyuyan/fuck0.ply")[:,:3]
    point_cloud1 = np.load("/home/liuyuyan/GaPartNet/gapartnet/pointcloud_with_normals_Door_9117_0_2.npy")
    # 创建 Open3D 点云对象
    pcd0 = o3d.geometry.PointCloud()
    pcd1 = o3d.geometry.PointCloud()

    # 设置点的位置
    pcd0.points = o3d.utility.Vector3dVector(point_cloud0[:, :3])
    pcd0.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30)  # 你可以调节邻居点数
    )
    pcd0.orient_normals_consistent_tangent_plane(k=30)  # 基于相邻点一致性调整方向

    pcd1.points = o3d.utility.Vector3dVector(point_cloud1[:, :3])
    pcd1.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30)  # 你可以调节邻居点数
    )
    pcd1.orient_normals_consistent_tangent_plane(k=30)  # 基于相邻点一致性调整方向






    # 为每个点云设置不同的颜色以便区分
    pcd0.paint_uniform_color([1, 0, 0])  # 红色
    pcd1.paint_uniform_color([0, 1, 0])  # 绿色
    o3d.visualization.draw_geometries([pcd0], point_show_normal=True)



    normals0 = np.asarray(pcd0.normals)  # (N, 3)
    xyz0 = np.asarray(pcd0.points)  # (N, 3)
    #
    # 拼接 xyz 和 normals 成 (N, 6) 的数组
    xyz_normals0 = np.hstack((xyz0, normals0))  # shape: (N, 6)
    print(xyz_normals0.shape)
    # 保存为 .npy 文件
    np.save(f"laptop.npy", xyz_normals0)

    normals1= np.asarray(pcd1.normals)  # (N, 3)
    xyz1 = np.asarray(pcd1.points)  # (N, 3)
    #
    # 拼接 xyz 和 normals 成 (N, 6) 的数组
    xyz_normals1 = np.hstack((xyz1, normals1))  # shape: (N, 6)
    print(xyz_normals1.shape)
    # 保存为 .npy 文件
    # np.save(f"moving.npy", xyz_normals1)

    print(f"Saved to pointcloud_with_normals.npy")




    # print("数组数据类型:", points.dtype)
    # print("数组维度:", points.ndim)
    #
    # # 打印前5个元素
    # print("前5个元素:", points[:5])


    # np.savetxt(f"/home/liuyuyan/MeshAnythingV2-main/pc_examples/points.txt", points, fmt="%.6f")
    # instance_labels = np.loadtxt(f"{filename}_ins.txt")
    # axis_df = pd.read_csv(f"{filename}_pos.csv")
    # axis_data = axis_df[["p_x", "p_y", "p_z", "n_x", "n_y", "n_z"]].values  # 转为 numpy array
    # instance_sem_labels = np.loadtxt(f"{filename}_ins_sem.txt").astype(int)
    # for instance_sem_label in instance_sem_labels:
    #     print(PART_ID2NAME[instance_sem_label])
    # flow = points[:,3:6] - points[:,:3]
    # visualize_flow_with_instance_colors(points,flow,instance_labels,instance_sem_labels)
    # visualize_flow_with_instance_selection(points, flow, instance_labels, [0, 1, 2, 3, 4, 5, 6], 'fuck')
    # visualize_flow_with_all_axis(points, flow, instance_labels, [1, 2, 3, 4, 5, 6], 'fuck', axis_data)
if __name__ == '__main__':
    main()