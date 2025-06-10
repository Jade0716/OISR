import torch
import numpy as np
import open3d as o3d
from ..misc.info import OBJECT_NAME2ID  # 假设你的标签字典存储在这里

# 假设已经定义好 PointCloud 类
class PointCloud:
    def __init__(self, pc_id, obj_cat, points, sem_labels, instance_labels, gt_npcs):
        self.pc_id = pc_id
        self.obj_cat = obj_cat
        self.points = points
        self.sem_labels = sem_labels
        self.instance_labels = instance_labels
        self.gt_npcs = gt_npcs

    # Convert points to Open3D PointCloud format
    def to_o3d(self):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)
        return pcd

# 读取 .pth 文件并加载点云数据
def load_data(file_path: str, no_label: bool = False):
    if not no_label:
        pc_data = torch.load(file_path)  # 加载数据
    else:
        raise NotImplementedError

    pc_id = file_path.split("/")[-1].split(".")[0]  # 提取点云 ID
    object_cat = OBJECT_NAME2ID[pc_id.split("_")[0]]  # 获取物体类别

    # 构造 PointCloud 对象
    return PointCloud(
        pc_id=pc_id,
        obj_cat=object_cat,
        points=np.concatenate([pc_data[0], pc_data[1]], axis=-1, dtype=np.float32),  # 合并点云坐标
        sem_labels=pc_data[2].astype(np.int64),  # 语义标签
        instance_labels=pc_data[3].astype(np.int32),  # 实例标签
        gt_npcs=pc_data[4].astype(np.float32)  # NPCs 标签
    )

# 可视化点云
def visualize_point_cloud(file_path: str):
    # 读取数据
    point_cloud = load_data(file_path)

    # 转换为 Open3D PointCloud 对象
    pcd = point_cloud.to_o3d()

    # 可视化点云
    o3d.visualization.draw_geometries([pcd])

# 测试代码，读取一个具体的 .pth 文件并进行可视化
file_path = "/GAPartNet_All/test_inter/pth/Door_8867_00_000.pth"
visualize_point_cloud(file_path)
