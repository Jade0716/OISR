import torch
import numpy as np
import open3d as o3d
from structure.instances import Instances
from misc.info import OBJECT_NAME2ID, PART_ID2NAME
from typing import Optional, Tuple, Union, List
from epic_ops.voxelize import voxelize
from misc.pose_fitting import estimate_pose_from_npcs
from misc.visu import visualize_gapartnet
from network.model import GAPartNet
from train import GAPartNetDataset, collate_fn
from pathlib import Path
from torch.utils.data import ConcatDataset, DataLoader
from calculate import energy_function
import copy
from scipy.spatial import cKDTree
from network.grouping_utils import (apply_nms, cluster_proposals, compute_ap, compute_npcs_loss, filter_invalid_proposals,)
from structure.point_cloud import PointCloud, PointCloudBatch
import json
import csv
import os


def visualize_point_cloud(point0,rgb):

    pcd0 = o3d.geometry.PointCloud()
    pcd0.points = o3d.utility.Vector3dVector(point0)
    pcd0.colors = o3d.utility.Vector3dVector(rgb)





    # 使用Open3D可视化点云
    o3d.visualization.draw_geometries([pcd0])


def visualize_flow(points, flows):
    """
    可视化指定实例标签的点云（红色）及其flow线段（蓝色）。
    :param points: torch.Tensor, 形状为(N, 3)，表示原始点云。
    :param flows: torch.Tensor, 形状为(N, 3)，表示每个点的流动。
    :param instance_labels: np.ndarray, 形状为(N,)，每个点的实例标签。
    :param selected_labels: list, 要显示的实例标签列表。
    :param pc_id: str, 需要显示在窗口上的点云ID。
    """


    # 转换为 numpy 数组并计算移动后的点云位置
    points_np = points
    flows_np = flows
    points_flowed = points_np + flows_np



    # 创建原始点云 Open3D 对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)
    pcd.paint_uniform_color([1, 0, 0])  # 红色
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(points_flowed)
    pcd1.paint_uniform_color([0, 1, 0])

    # 创建线集来表示点与它们的流动方向
    lines = [[i, i+len(flows_np)] for i in range(len(flows_np))]
    colors = [[0, 0, 1] for _ in range(len(lines))]  # 线条颜色为蓝色
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.concatenate((points_np, points_flowed)))
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    # 可视化
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Point Cloud")
    vis.add_geometry(pcd)  # 添加原始点云
    vis.add_geometry(pcd1)  # 添加原始点云
    vis.add_geometry(line_set)  # 添加线条集
    vis.run()
    vis.destroy_window()
    print("Visualization finished")

def analyze_proposals_sorted_by_score(proposals, top_k=None):
    if proposals is None or not hasattr(proposals, 'proposal_offsets'):
        print("No valid proposals.")
        return

    offsets = proposals.proposal_offsets
    num_proposals = offsets.shape[0] - 1

    pt_sem_classes = proposals.pt_sem_classes.long()
    scores = proposals.score_preds
    # ious = proposals.ious
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
        # iou = ious[i].max().item()
        num_points = end - start

        # 获取 GT 类别
        gt_sem_class = -1
        if instance_labels is not None and instance_sem_labels is not None:
            inst_id = instance_labels[start].item()
            if inst_id >= 0 and batch_indices is not None:
                batch_idx = batch_indices[start].item()
                if batch_idx < instance_sem_labels.shape[0] and inst_id < instance_sem_labels.shape[1]:
                    gt_sem_class = instance_sem_labels[batch_idx, inst_id].item()

        results.append((score, i, pred_class, gt_sem_class, num_points))

    # 按 score 降序排序
    results.sort(reverse=True, key=lambda x: x[0])

    if top_k:
        results = results[:top_k]

    print(f"{'Idx':<6} {'Score':<8} {'PredCls':<8} {'GTCls':<6} {'Points':<8} ")
    for score, i, pred_class, gt_class, num_points in results:
        print(f"{i:<6} {score:<8.4f} {pred_class:<8} {gt_class:<6} {num_points:<8}")



def get_top_scored_proposal_instances(proposals, target_classes):
    if proposals is None or not hasattr(proposals, 'proposal_offsets'):
        print("No valid proposals.")
        return None

    # 提取 proposal 相关的属性
    pt_sem_classes = proposals.pt_sem_classes.long()  # 每个 proposal 的语义类别
    scores = proposals.score_preds  # 每个 proposal 的得分
    proposal_offsets = proposals.proposal_offsets  # proposal 的偏移
    num_proposals = proposal_offsets.shape[0] - 1  # proposal 的数量

    best_score = -float('inf')
    best_idx = -1

    # 遍历所有 proposals，选择符合 target_classes 且得分最高的 proposal
    for i in range(num_proposals):
        if pt_sem_classes[i].item() in target_classes:  # 如果 proposal 的语义类别在 target_classes 中
            score = scores[i].item()  # 获取该 proposal 的得分
            if score > best_score:  # 选择得分最高的 proposal
                best_score = score
                best_idx = i

    if best_idx == -1:
        print("No matching proposal found.")
        return None


    # 根据 best_idx 取出对应的 proposal 数据
    start = proposal_offsets[best_idx].item()
    end = proposal_offsets[best_idx + 1].item()

    # 获取对应的点和其他信息
    indices = torch.arange(start, end, device=proposal_offsets.device)

    # 构造新的 proposal_offsets（只有一个 proposal）
    new_offsets = torch.tensor([0, end - start], dtype=torch.int32, device=proposal_offsets.device)

    # 构造新的 Instances
    proposals_ = Instances(
        pt_xyz=proposals.pt_xyz[indices],
        batch_indices=proposals.batch_indices[indices],
        score_preds=scores[best_idx:best_idx+1],
        pt_sem_classes=proposals.pt_sem_classes[best_idx:best_idx+1],
        proposal_offsets=new_offsets,
        sorted_indices=proposals.sorted_indices[indices] if hasattr(proposals, 'sorted_indices') else None,
        valid_mask=proposals.valid_mask
    )

    return proposals_




def visualize_instance_with_axis(points_all, rgb, instance_points, axis_params):


    # 转成 numpy
    if isinstance(points_all, torch.Tensor):
        points_all = points_all.detach().cpu().numpy()
    if isinstance(instance_points, torch.Tensor):
        instance_points = instance_points.detach().cpu().numpy()
    if isinstance(axis_params, torch.Tensor):
        axis_params = axis_params.detach().cpu().numpy()

    # 创建 open3d 点云对象
    pcd_all = o3d.geometry.PointCloud()
    pcd_all.points = o3d.utility.Vector3dVector(points_all)
    
    # 设置点云颜色：rgb是(N,3)数组，每个点都有自己的颜色
    pcd_all.colors = o3d.utility.Vector3dVector(rgb)

    pcd_inst = o3d.geometry.PointCloud()
    pcd_inst.points = o3d.utility.Vector3dVector(instance_points)
    pcd_inst.paint_uniform_color([222/255, 235/255, 247/255])  # 浅蓝色

    # 创建转轴
    axis_line = None
    if len(axis_params) == 6:
        axis_point = axis_params[:3]
        axis_dir = axis_params[3:]
    elif len(axis_params) == 3:
        axis_point = instance_points.mean(axis=0)  # 默认用实例点云中心
        axis_dir = axis_params
    else:
        raise ValueError("axis_params 长度必须是 3 或 6")

    axis_dir = axis_dir / np.linalg.norm(axis_dir)
    line_length = 10  # 可视化长度
    line_pts = np.array([
        axis_point - axis_dir * line_length,
        axis_point + axis_dir * line_length
    ])

    axis_line = o3d.geometry.LineSet()
    axis_line.points = o3d.utility.Vector3dVector(line_pts)
    axis_line.lines = o3d.utility.Vector2iVector([[0, 1]])
    axis_line.colors = o3d.utility.Vector3dVector([[0, 1.0, 0]])  # 绿色线段

    o3d.visualization.draw_geometries([pcd_all, pcd_inst, axis_line])

def visualize_multiple_axes(points_all, rgb, all_axis_params, axis_colors=None, line_radius=0.02):
    # 转成 numpy
    if isinstance(points_all, torch.Tensor):
        points_all = points_all.detach().cpu().numpy()
    if isinstance(rgb, torch.Tensor):
        rgb = rgb.detach().cpu().numpy()

    # 创建背景点云
    pcd_all = o3d.geometry.PointCloud()
    pcd_all.points = o3d.utility.Vector3dVector(points_all)
    pcd_all.colors = o3d.utility.Vector3dVector(rgb)

    # 创建所有旋转轴
    axis_cylinders = []
    if axis_colors is None:
        # 默认颜色：红、绿、蓝、黄、紫、青、橙、粉
        default_colors = [
            [0.8, 0.2, 0.2],  # 红
            [0.2, 0.8, 0.2],  # 绿
            [0.2, 0.2, 0.8],  # 蓝
            [0.8, 0.8, 0.2],  # 黄
            [0.8, 0.2, 0.8],  # 紫
            [0.2, 0.8, 0.8],  # 青
            [0.8, 0.5, 0.2],  # 橙
            [0.8, 0.2, 0.5],  # 粉
        ]
        axis_colors = default_colors

    for i, axis_params in enumerate(all_axis_params):
        if isinstance(axis_params, torch.Tensor):
            axis_params = axis_params.detach().cpu().numpy()
        
        # 创建转轴
        if len(axis_params) == 6:
            axis_point = axis_params[:3]
            axis_dir = axis_params[3:]
        elif len(axis_params) == 3:
            axis_point = points_all.mean(axis=0)  # 默认用点云中心
            axis_dir = axis_params
        else:
            continue

        axis_dir = axis_dir / np.linalg.norm(axis_dir)
        line_length = 5  # 可视化长度
        
        # 创建圆柱体代替线段
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(
            radius=line_radius, 
            height=line_length * 2,
            resolution=10  # 圆柱体的分辨率，数值越大越平滑
        )
        
        # 计算旋转矩阵，使圆柱体沿着轴方向
        # 默认圆柱体沿着z轴，需要旋转到目标方向
        z_axis = np.array([0, 0, 1])
        if np.allclose(axis_dir, z_axis) or np.allclose(axis_dir, -z_axis):
            # 如果轴方向就是z轴，不需要旋转
            rotation_matrix = np.eye(3)
        else:
            # 计算从z轴到目标方向的旋转矩阵
            rotation_axis = np.cross(z_axis, axis_dir)
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            cos_angle = np.dot(z_axis, axis_dir)
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            
            # 使用罗德里格斯公式计算旋转矩阵
            K = np.array([[0, -rotation_axis[2], rotation_axis[1]],
                         [rotation_axis[2], 0, -rotation_axis[0]],
                         [-rotation_axis[1], rotation_axis[0], 0]])
            rotation_matrix = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        
        # 应用旋转和平移
        cylinder.rotate(rotation_matrix, center=[0, 0, 0])
        cylinder.translate(axis_point)
        
        # 设置颜色
        color_idx = i % len(axis_colors)
        cylinder.paint_uniform_color(axis_colors[color_idx])
        
        axis_cylinders.append(cylinder)

    # 可视化所有轴
    geometries = [pcd_all] + axis_cylinders
    o3d.visualization.draw_geometries(geometries)






def write_urdf(filename_prefix, p_opt, n_opt,
               body_obj="cabinet_body_rest.obj",
               door_obj="cabinet_door_part.obj"):

    urdf_filename = f"{filename_prefix}.urdf"

    # 仅创建非空路径
    dir_path = os.path.dirname(urdf_filename)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    axis_norm = n_opt / np.linalg.norm(n_opt)
    axis_str = " ".join(map(str, axis_norm))
    p_str = " ".join(map(str, p_opt))
    p2_str = " ".join(map(str, -p_opt))

    # 生成 URDF 字符串
    urdf_str = f"""<?xml version="1.0"?>
<robot name="{filename_prefix}">

  <link name="base"/>

  <link name="base_link">
    <visual>
      <geometry>
        <mesh filename="{body_obj}" scale="1 1 1"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <mesh filename="{body_obj}" scale="1 1 1"/>
      </geometry>
    </collision>
  </link>
  
  <joint name="fixed_base_to_baselink" type="fixed">
    <parent link="base"/>
    <child link="base_link"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
  </joint>
  
  <link name="door_link">
    <visual>
      <origin xyz="{p2_str}" rpy="0 0 0"/>
      <geometry>
        <mesh filename="{door_obj}" scale="1 1 1"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="{p2_str}" rpy="0 0 0"/>
      <geometry>
        <mesh filename="{door_obj}" scale="1 1 1"/>
      </geometry>
    </collision>
  </link>

  <joint name="door_hinge" type="revolute">
    <parent link="base_link"/>
    <child link="door_link"/>
    <origin xyz="{p_str}" rpy="0 0 0"/>
    <axis xyz="{axis_str}"/>
    <limit lower="0.0" upper="1.6901768476313088"/>
  </joint>

</robot>
"""

    with open(urdf_filename, "w") as f:
        f.write(urdf_str)

    print(f"[✔] URDF 文件已保存到：{urdf_filename}")


def load_meta_and_restore_points(meta_path, points_normalized):
    # 读取meta文件的四行数据
    with open(meta_path, 'r') as f:
        lines = f.readlines()
        max_radius = float(lines[0].strip())
        center_x = float(lines[1].strip())
        center_y = float(lines[2].strip())
        center_z = float(lines[3].strip())

    center = np.array([center_x, center_y, center_z], dtype=np.float32)

    # 还原点云
    points_restored = points_normalized * max_radius + center
    return points_restored, max_radius, center
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
    net.load_state_dict(torch.load("/home/liuyuyan/OISR/checkpoints/flowxyzwithmotion/models/final_model.pth"), strict=True)
    net.to(device)
    root_dir: str = "/mnt/4dba1798-fc0d-4700-a472-04acb2f7b630/liuyuyan/GAPartNetAllWithFlows/"
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
                                  batch_size=1,
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
        if len(pc)!=0 and pc[0].pc_id.startswith("Oven_7201"):
            data_batch = PointCloud.collate(pc)  # PointCloudBatch
            net.eval()
            with torch.no_grad():
                pc_ids, sem_seg, proposals, _ = net(data_batch,10)
            points = pc[0].points.cpu().numpy()
            pc_id = pc[0].pc_id
            points1 = points[:, :3]
            flow = points[:, 3:6]
            points0 = points1-flow
            rgb = points[:, 6:9]
            meta_path = f"/15T/liuyuyan/GAPartNetAllWithFlows/test_inter/meta/{pc_id}.txt"
            points0, max_radius, center = load_meta_and_restore_points(meta_path, points0)
            # with open(f"/15T/liuyuyan/example_rendered_3dgs/metafile/{filename}_1.json", "r") as f:
            #     meta = json.load(f)
            # R_c2w = np.array(meta['world2camera_rotation'], dtype=np.float32).reshape(3, 3)
            # t_c2w = np.array(meta['camera2world_translation'], dtype=np.float32).reshape(3)
            # points0 = (R_c2w @ points0.T).T+ t_c2w

            points1, max_radius, center = load_meta_and_restore_points(meta_path, points1)
            # points1 = (R_c2w @ points1.T).T + t_c2w
            visualize_flow(points0,flow)
            sample_ids = range(len(pc_ids))
            sample_id = 0
            batch_id = sample_id // 1
            batch_sample_id = sample_id % 1
            if proposals is not None:
                proposals.pt_sem_classes = proposals.sem_preds[proposals.proposal_offsets[:-1].long()].long()
                print(f"beyond:{proposals.proposal_offsets.shape[0]-1}")
                analyze_proposals_sorted_by_score(proposals)
                val_min_points_per_class_use = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
                proposals = filter_invalid_proposals(
                    proposals,
                    score_threshold=net.val_score_threshold,
                    val_min_points_per_class=val_min_points_per_class_use,
                )
                proposals = apply_nms(proposals, net.val_nms_iou_threshold)  # 非极大值抑制（NMS），用来过滤掉重叠太多的重复 proposal
                print(f"after:{proposals.proposal_offsets.shape[0]-1}")
                if proposals is not None:
                    proposals.pt_sem_classes = proposals.sem_preds[proposals.proposal_offsets[:-1].long()]
                    # analyze_proposals_sorted_by_score(proposals)
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
                        GAPARTNET_DATA_ROOT="/15T/liuyuyan/GAPartNetAllWithFlows",
                        # save_option=["raw", "pc", "sem_pred", "sem_gt", "ins_pred", "ins_gt", "npcs_pred", "npcs_gt", "bbox_gt", "bbox_gt_pure", "bbox_pred", "bbox_pred_pure"],
                        save_option=["raw", "pc", "sem_pred", "ins_pred", "npcs_pred", "bbox_pred", "bbox_pred_pure"],
                        name=pc_ids[sample_id],
                        split="test_inter",
                        sem_preds=sample_sem_pred.cpu().numpy().squeeze(),  # type: ignore
                        ins_preds=sample_ins_seg_pred.cpu().numpy().squeeze(),
                        npcs_preds=sample_npcs_map.cpu().numpy().squeeze(),
                        bboxes=bboxes,
                    )
                    # 获取所有符合条件的proposals，按得分排序
                    target_classes = [4, 5, 6, 7]
                    all_qualified_proposals = []
                    
                    # 遍历所有proposals，找到符合条件的
                    for i in range(proposals.proposal_offsets.shape[0] - 1):
                        start = proposals.proposal_offsets[i].item()
                        end = proposals.proposal_offsets[i + 1].item()
                        pred_class = proposals.pt_sem_classes[i].item()
                        score = proposals.score_preds[i].item()
                        
                        if pred_class in target_classes:
                            all_qualified_proposals.append((score, i, pred_class))
                    
                    # 按得分降序排序，取前8个
                    all_qualified_proposals.sort(reverse=True, key=lambda x: x[0])
                    top_8_proposals = all_qualified_proposals[:8]
                    
                    print(f"找到 {len(all_qualified_proposals)} 个符合条件的proposals，处理前8个")
                    
                    # 存储所有优化后的旋转轴参数
                    all_optimized_axes = []
                    all_axes_info = []
                    
                    # 循环处理前8个实例
                    for proposal_idx, (score, proposal_id, pred_class) in enumerate(top_8_proposals):
                        print(f"处理第 {proposal_idx + 1} 个实例，得分: {score:.4f}, 类别: {pred_class}")
                        
                        # 获取当前proposal的点云数据
                        start = proposals.proposal_offsets[proposal_id].item()
                        end = proposals.proposal_offsets[proposal_id + 1].item()
                        
                        valid_mask = proposals.valid_mask
                        sorted_indices = proposals.sorted_indices
                        valid_indices = valid_mask.nonzero(as_tuple=False).squeeze(1)
                        proposal_sorted_idx = sorted_indices[start:end]
                        proposal_original_indices = valid_indices[proposal_sorted_idx]
                        
                        sem_label_name = PART_ID2NAME[pred_class]
                        
                        P_inst = torch.tensor((points0), dtype=torch.float32).cuda()[proposal_original_indices]
                        F_inst = torch.tensor((points1-points0), dtype=torch.float32).cuda()[proposal_original_indices]
                        
                        instance_data = [{
                            "inst_idx": proposal_idx,
                            "sem_label": sem_label_name,
                            "points": P_inst,
                            "flow": F_inst
                        }]
                        
                        # 确定参数数量
                        if sem_label_name in {"hinge_lid", "hinge_door"}:
                            num_params = 6
                        elif sem_label_name in {"slider_drawer", "slider_lid"}:
                            num_params = 3
                        else:
                            print(f"跳过不支持的类别: {sem_label_name}")
                            continue
                        
                        # 优化参数
                        x = torch.rand(num_params, dtype=torch.float32, device='cuda', requires_grad=True)
                        optimizer = torch.optim.Adam([x], lr=0.001)
                        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.1)
                        
                        loss = np.inf
                        step = 0
                        while step < 10000:
                            optimizer.zero_grad()
                            loss = energy_function([(sem_label_name, proposal_idx, x)], instance_data)
                            
                            if isinstance(loss, torch.Tensor):
                                loss.backward()
                                optimizer.step()
                                scheduler.step()
                            
                            step += 1
                            if step % 1000 == 0:
                                print(f"  实例 {proposal_idx + 1} Step {step} Loss: {loss.item():.6f}")
                        
                        print(f"  实例 {proposal_idx + 1} 优化完成，最终Loss: {loss.item():.6f}")
                        
                        # 保存优化结果
                        all_optimized_axes.append(x.detach().clone())
                        all_axes_info.append({
                            'proposal_idx': proposal_idx,
                            'score': score,
                            'sem_label': sem_label_name,
                            'num_points': P_inst.shape[0]
                        })
                    
                    # 可视化所有旋转轴
                    if all_optimized_axes:
                        print("可视化所有旋转轴...")
                        visualize_multiple_axes(points0, rgb, all_optimized_axes)
                            
                    # mesh = o3d.io.read_triangle_mesh("/home/liuyuyan/PGSR/output/test/mesh/tsdf_fusion_post.obj")
                    # mesh.compute_vertex_normals()
                    # assert mesh.has_vertex_colors(), "Mesh must have vertex colors"

                    # # 构建 KD 树用于查找顶点最近邻
                    # pcd = o3d.geometry.PointCloud()
                    # pcd.points = mesh.vertices
                    # pcd.colors = mesh.vertex_colors
                    # pcd_tree = o3d.geometry.KDTreeFlann(pcd)

                    # # 查找 mesh 中靠近 P_inst 的顶点索引
                    # selected_vertex_indices = set()
                    # for point in P_inst.cpu().numpy():
                    #     [_, idxs, _] = pcd_tree.search_knn_vector_3d(point, 5)
                    #     selected_vertex_indices.update(idxs)
                    # selected_vertex_indices = set(selected_vertex_indices)

                    # # 分割三角形面片：属于 selected 的顶点 vs 不属于的
                    # triangles = np.asarray(mesh.triangles)
                    # vertices = np.asarray(mesh.vertices)
                    # colors = np.asarray(mesh.vertex_colors)

                    # # 创建两个面片集合
                    # part_triangles = []
                    # rest_triangles = []

                    # for i, tri in enumerate(triangles):
                    #     v0, v1, v2 = tri
                    #     # 如果3个顶点中任意一个属于选中点，划为part；否则划为rest
                    #     if v0 in selected_vertex_indices or v1 in selected_vertex_indices or v2 in selected_vertex_indices:
                    #         part_triangles.append(tri)
                    #     else:
                    #         rest_triangles.append(tri)

                    # # 构造两个子 mesh（保留原始顶点、颜色、拓扑）
                    # def build_mesh(triangles_idx_list):
                    #     submesh = o3d.geometry.TriangleMesh()
                    #     submesh.vertices = mesh.vertices
                    #     submesh.vertex_colors = mesh.vertex_colors
                    #     submesh.triangles = o3d.utility.Vector3iVector(np.array(triangles_idx_list))
                    #     return submesh

                    # part_mesh = build_mesh(part_triangles)
                    # rest_mesh = build_mesh(rest_triangles)


                    # o3d.io.write_triangle_mesh("cabinet_door_part.obj", part_mesh, write_vertex_colors=True, write_triangle_uvs=True)
                    # o3d.io.write_triangle_mesh("cabinet_body_rest.obj", rest_mesh)

                    # part_vertices = np.asarray(part_mesh.vertices)
                    # part_mesh.vertex_colors = o3d.utility.Vector3dVector(
                    #     np.tile(np.array([[1.0, 0.0, 0.0]]), (part_vertices.shape[0], 1))
                    # )

                    # # 给 rest_mesh 染绿色
                    # rest_vertices = np.asarray(rest_mesh.vertices)
                    # rest_mesh.vertex_colors = o3d.utility.Vector3dVector(
                    #     np.tile(np.array([[0.0, 1.0, 0.0]]), (rest_vertices.shape[0], 1))
                    # )

                    # # 添加坐标轴
                    # coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

                    # # 可视化
                    # o3d.visualization.draw_geometries([part_mesh, rest_mesh, coord_frame])



                    
                    # 记录优化后的旋转轴数据
                    all_axes = []
                    header = ["instance_name", "p_x", "p_y", "p_z", "n_x", "n_y", "n_z", "score", "num_points"]
                    
                    for i, (axis_params, info) in enumerate(zip(all_optimized_axes, all_axes_info)):
                        axis_params = axis_params.cpu().detach().numpy()
                        
                        if len(axis_params) > 3:
                            p_opt = axis_params[:3]
                            n_opt = axis_params[3:]
                        else:
                            p_opt = np.array([])
                            n_opt = axis_params[:3]
                        
                        instance_name = f"{info['sem_label']}_{info['proposal_idx']}"
                        print(f"实例 {instance_name} 优化后的旋转轴位置 p:", p_opt)
                        if n_opt.size > 0:
                            print(f"实例 {instance_name} 优化后的旋转轴方向 n:", n_opt)
                        
                        # 统一格式
                        row = [instance_name]
                        row.extend(p_opt.tolist() if p_opt.size > 0 else [None, None, None])
                        row.extend(n_opt.tolist() if n_opt.size > 0 else [None, None, None])
                        row.extend([info['score'], info['num_points']])
                        
                        all_axes.append(row)
                    
                    # 保存优化结果到 CSV
                    csv_filename = f"{pc_id}_pos.csv"
                    with open(csv_filename, mode="w", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(header)
                        writer.writerows(all_axes)
                    
                    print(f"优化结果已保存到 {csv_filename}")
                    
                    # 如果有优化结果，生成URDF文件（使用第一个实例的参数）
                    if all_optimized_axes:
                        first_axis = all_optimized_axes[0].cpu().detach().numpy()
                        if len(first_axis) > 3:
                            p_opt = first_axis[:3]
                            n_opt = first_axis[3:]
                        else:
                            p_opt = np.array([])
                            n_opt = first_axis[:3]
                        
                        if p_opt.size > 0 and n_opt.size > 0:
                             write_urdf(f"{pc_id}_model", p_opt, n_opt)
if __name__ == '__main__':
    main()