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
import copy
from scipy.spatial import cKDTree
from network.grouping_utils import (apply_nms, cluster_proposals, compute_ap, compute_npcs_loss, filter_invalid_proposals,)
from structure.point_cloud import PointCloud, PointCloudBatch
import json
import csv
import os
def compute_distance(P, p, n):
    n = n / torch.norm(n)  # 归一化法向量
    return torch.norm(torch.cross(P - p, n.expand_as(P)), dim=1)

lambda_C = torch.nn.Parameter(torch.tensor(1.0, requires_grad=True, device='cuda'))

def visualize_point_cloud(point0):

    pcd0 = o3d.geometry.PointCloud()
    pcd0.points = o3d.utility.Vector3dVector(point0)
    pcd0.paint_uniform_color([0.6, 0.2, 0.2])


    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1, origin=[0, 0, 0]
    )



    # 使用Open3D可视化点云
    o3d.visualization.draw_geometries([pcd0, coord_frame])
def energy_function(params, instance_data):
    total_loss = 0.0
    total_EV = 0.0
    total_EC = 0.0
    x = params[0][2]  # 只有一个实例
    inst = instance_data[0]
    sem_label = inst["sem_label"]

    if sem_label in {"hinge_lid", "hinge_door"}:
        p = x[:3]
        n = x[3:]
        n = n / torch.norm(n)

        P_inst = inst["points"]
        F_inst = inst["flow"]
        P_number = P_inst.shape[0]
        F_norm = torch.norm(F_inst, dim=1)
        distances = compute_distance(P_inst, p, n)

        D_ratio = distances.unsqueeze(1) / distances.unsqueeze(0)
        F_ratio = F_norm.unsqueeze(1) / F_norm.unsqueeze(0)
        EC = torch.sum(torch.abs(D_ratio - F_ratio))

        top_k = max(1, P_number // 5)
        threshold = torch.topk(F_norm, top_k, largest=True)[0][-1]
        mask = F_norm >= threshold

        P_inst = P_inst[mask]
        F_inst = F_inst[mask]
        F_norm = F_norm[mask]
        P_number_big = P_inst.shape[0]

        if P_number_big == 0:
            print("not cal")
            return torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)

        W = F_norm / torch.sum(F_norm)
        F_dot_n = torch.sum(F_inst * n, dim=1)
        EV = torch.sum(W * torch.abs(F_dot_n / torch.clamp(F_norm, min=1e-3)))

        loss = EV / P_number_big + EC / (P_number * P_number)
        EV = EV / P_number_big
        EC = EC / (P_number * P_number)

    elif sem_label in {"slider_drawer", "slider_lid"} :
        n = x[:3]
        n = n / torch.norm(n)

        P_inst = inst["points"]
        F_inst = inst["flow"]
        P_number = P_inst.shape[0]
        F_norm = torch.norm(F_inst, dim=1)
        top_k = max(1, P_number // 5)
        threshold = torch.topk(F_norm, top_k, largest=True)[0][-1]
        mask = F_norm >= threshold

        P_inst = P_inst[mask]
        F_inst = F_inst[mask]
        F_norm = F_norm[mask]
        P_number_big = P_inst.shape[0]

        # 计算 flow 与 n 的内积
        F_dot_n = torch.sum(F_inst * n, dim=1)

        # 计算 flow 的模长
        F_norm = torch.norm(F_inst, dim=1)
        n_norm = torch.norm(n)

        # 根据公式计算每个项的残差
        residuals = F_dot_n - n_norm * F_norm

        # 计算总和
        loss = torch.sum(torch.abs(residuals))/P_number_big



        if P_number == 0:
            print("not cal")
            return torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)
    else:
        print("not cal")

        # 其他类型
        loss = torch.tensor(0.0)
        EV = torch.tensor(0.0)
        EC = torch.tensor(0.0)
    total_loss += loss
    return total_loss

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

def visualize_instance_with_axis(points_all, instance_points, axis_params):


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
    pcd_all.paint_uniform_color([0, 0, 1.0])  # 蓝色

    pcd_inst = o3d.geometry.PointCloud()
    pcd_inst.points = o3d.utility.Vector3dVector(instance_points)
    pcd_inst.paint_uniform_color([1.0, 0, 0])  # 红色

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

def FindMaxDis(pointcloud):
    max_xyz = pointcloud.max(0)
    min_xyz = pointcloud.min(0)
    center = (max_xyz + min_xyz) / 2
    max_radius = ((((pointcloud - center) ** 2).sum(1)) ** 0.5).max()
    return max_radius, center
def WorldSpaceToBallSpace(pointcloud):

    max_radius, center = FindMaxDis(pointcloud)
    pointcloud_normalized = (pointcloud - center) / max_radius
    return pointcloud_normalized, max_radius, center
def match_points_by_nn(points0, points1):

    tree = cKDTree(points1)
    _, indices = tree.query(points0, k=1)  # 每个points0中点在points1中最近的点的索引
    return points1[indices]



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
    torch.backends.cudnn.deterministic = True
    # torch.manual_seed(1234)
    # torch.cuda.manual_seed_all(1234)
    # np.random.seed(1234)


    #载入分割网络
    model_path = "/home/liuyuyan/GAOR/gaor/checkpoints/100flowstats0.02flowxyzoffsetslider/models/final_model.pth"
    net1 = GAPartNet().cuda()
    print(f'Loading pretrained model from {model_path}')
    net1.load_state_dict(torch.load(model_path))
    filename = "StorageFurniture_46859_0_24"

    points = np.loadtxt(f"{filename}.txt").astype(np.float32)
    points0 = points[:, :3]
    meta_path = f"/16T/liuyuyan/GAPartNetAllWithFlows/train/meta/{filename}.txt"
    points0, max_radius, center = load_meta_and_restore_points(meta_path, points0)
    with open(f"/16T/liuyuyan/example_rendered_3dgs/metafile/{filename}_1.json", "r") as f:
        meta = json.load(f)
    R_c2w = np.array(meta['world2camera_rotation'], dtype=np.float32).reshape(3, 3)
    t_c2w = np.array(meta['camera2world_translation'], dtype=np.float32).reshape(3)
    points0 = (R_c2w @ points0.T).T+ t_c2w
    points1 = points[:, 3:6]
    points1, max_radius, center = load_meta_and_restore_points(meta_path, points1)
    points1 = (R_c2w @ points1.T).T + t_c2w
    visualize_point_cloud(points1)
    pc = PointCloud(pc_id=filename, points=points)
    pc = pc.to_tensor()
    pc = apply_voxelization(pc, voxel_size = (1 / 100, 1 / 100, 1 / 100))
    pc = [pc.to(net1.device)]  # List["PointCloud"]
    data_batch = PointCloud.collate(pc)  # PointCloudBatch
    net1.eval()
    with torch.no_grad():
        pc_ids, sem_seg, proposals, _ = net1(data_batch)
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
            score_threshold=net1.val_score_threshold,
            val_min_points_per_class=val_min_points_per_class_use,
        )
        proposals = apply_nms(proposals, net1.val_nms_iou_threshold)  # 非极大值抑制（NMS），用来过滤掉重叠太多的重复 proposal
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
                GAPARTNET_DATA_ROOT="/16T/liuyuyan/GAPartNetAllWithFlows",
                # save_option=["raw", "pc", "sem_pred", "sem_gt", "ins_pred", "ins_gt", "npcs_pred", "npcs_gt", "bbox_gt", "bbox_gt_pure", "bbox_pred", "bbox_pred_pure"],
                save_option=["raw", "pc", "sem_pred", "ins_pred", "npcs_pred", "bbox_pred", "bbox_pred_pure"],
                name=pc_ids[sample_id],
                split="train",
                sem_preds=sample_sem_pred.cpu().numpy().squeeze(),  # type: ignore
                ins_preds=sample_ins_seg_pred.cpu().numpy().squeeze(),
                npcs_preds=sample_npcs_map.cpu().numpy().squeeze(),
                bboxes=bboxes,
            )
            target_classes = [ 5, 6, 7]
            top_proposal = get_top_scored_proposal_instances(proposals, target_classes)
            if top_proposal is not None:
                print(f"Top proposal: {top_proposal}")


            valid_mask = top_proposal.valid_mask  # [N]
            sorted_indices = top_proposal.sorted_indices  # [M], 是 valid_mask 为 True 的点中的子集
            valid_indices = valid_mask.nonzero(as_tuple=False).squeeze(1)  # [K], K <= N
            start = top_proposal.proposal_offsets[0].item()
            end = top_proposal.proposal_offsets[1].item()
            proposal_sorted_idx = top_proposal.sorted_indices[start:end]  # 聚类后 proposal 的点索引（相对于 valid_indices）
            proposal_original_indices = valid_indices[proposal_sorted_idx]

            inst_sem_label = top_proposal.pt_sem_classes.item()
            sem_label_name = PART_ID2NAME[inst_sem_label]

            P_inst = torch.tensor((points0), dtype=torch.float32).cuda()[proposal_original_indices] # 点坐标
            F_inst =  torch.tensor((points1-points0), dtype=torch.float32).cuda()[proposal_original_indices]  # 光流
            instance_data = [{
                "inst_idx": 0,
                "sem_label": sem_label_name,
                "points": P_inst,
                "flow": F_inst
            }]
            params = []
            if sem_label_name in {"hinge_lid", "hinge_door"}:
                num_params = 6
            elif sem_label_name in {"slider_drawer", "slider_lid"}:
                num_params = 3
            else:
                num_params = 0

            if num_params > 0:
                x = torch.rand(num_params, dtype=torch.float32, device='cuda', requires_grad=True)
                params.append((sem_label_name, 0, x))
            optimizer = torch.optim.Adam([x for _, _, x in params], lr=0.001)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3000, gamma=0.1)

            loss = np.inf
            step = 0
            while step < 6000:
                optimizer.zero_grad()
                loss = energy_function(params, instance_data)


                loss.backward()
                optimizer.step()
                scheduler.step()
                step += 1
                if step % 1000 == 0:
                    print(f"Step {step}  Loss: {loss.item():.6f}")
                    print(f"当前学习率: {scheduler.get_last_lr()[0]}")
                    # print(f"total_EV: {total_EV}, total_EC: {total_EC}")

            print(f"Step {step}  Loss: {loss.item():.6f}")

    _, _, x_opt = params[0]
    visualize_instance_with_axis(points0, P_inst, x_opt)
    mesh = o3d.io.read_triangle_mesh("/home/liuyuyan/PGSR/output/test/mesh/tsdf_fusion_post.obj")
    mesh.compute_vertex_normals()
    assert mesh.has_vertex_colors(), "Mesh must have vertex colors"

    # 构建 KD 树用于查找顶点最近邻
    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices
    pcd.colors = mesh.vertex_colors
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)

    # 查找 mesh 中靠近 P_inst 的顶点索引
    selected_vertex_indices = set()
    for point in P_inst.cpu().numpy():
        [_, idxs, _] = pcd_tree.search_knn_vector_3d(point, 5)
        selected_vertex_indices.update(idxs)
    selected_vertex_indices = set(selected_vertex_indices)

    # 分割三角形面片：属于 selected 的顶点 vs 不属于的
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    colors = np.asarray(mesh.vertex_colors)

    # 创建两个面片集合
    part_triangles = []
    rest_triangles = []

    for i, tri in enumerate(triangles):
        v0, v1, v2 = tri
        # 如果3个顶点中任意一个属于选中点，划为part；否则划为rest
        if v0 in selected_vertex_indices or v1 in selected_vertex_indices or v2 in selected_vertex_indices:
            part_triangles.append(tri)
        else:
            rest_triangles.append(tri)

    # 构造两个子 mesh（保留原始顶点、颜色、拓扑）
    def build_mesh(triangles_idx_list):
        submesh = o3d.geometry.TriangleMesh()
        submesh.vertices = mesh.vertices
        submesh.vertex_colors = mesh.vertex_colors
        submesh.triangles = o3d.utility.Vector3iVector(np.array(triangles_idx_list))
        return submesh

    part_mesh = build_mesh(part_triangles)
    rest_mesh = build_mesh(rest_triangles)


    o3d.io.write_triangle_mesh("cabinet_door_part.obj", part_mesh, write_vertex_colors=True, write_triangle_uvs=True)
    o3d.io.write_triangle_mesh("cabinet_body_rest.obj", rest_mesh)

    part_vertices = np.asarray(part_mesh.vertices)
    part_mesh.vertex_colors = o3d.utility.Vector3dVector(
        np.tile(np.array([[1.0, 0.0, 0.0]]), (part_vertices.shape[0], 1))
    )

    # 给 rest_mesh 染绿色
    rest_vertices = np.asarray(rest_mesh.vertices)
    rest_mesh.vertex_colors = o3d.utility.Vector3dVector(
        np.tile(np.array([[0.0, 1.0, 0.0]]), (rest_vertices.shape[0], 1))
    )

    # 添加坐标轴
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

    # 可视化
    o3d.visualization.draw_geometries([part_mesh, rest_mesh, coord_frame])

    # 记录优化后的旋转轴数据
    all_axes = []
    header = ["instance_name", "p_x", "p_y", "p_z", "n_x", "n_y", "n_z"]  # 固定表头
    for sem_label, inst_idx, x_opt in params:
        x_opt = x_opt.cpu().detach().numpy()
        if len(x_opt) > 3:
            p_opt = x_opt[:3]
            n_opt = x_opt[3:]
        else:
            p_opt = np.array([])  # 如果没有 p_opt，则设置为空数组
            n_opt = x_opt[:3]

        instance_name = f"{sem_label}_{inst_idx}"  # 生成唯一标识
        print(f"实例 {instance_name} 优化后的旋转轴位置 p:", p_opt)
        if n_opt.size > 0:
            print(f"实例 {instance_name} 优化后的旋转轴方向 n:", n_opt)

        # 统一格式：没有的位置填充空值
        row = [instance_name]
        row.extend(p_opt.tolist() if p_opt.size > 0 else [None, None, None])
        row.extend(n_opt.tolist() if n_opt.size > 0 else [None, None, None])

        all_axes.append(row)
    write_urdf("cabinet_model", p_opt, n_opt)
    # 保存优化结果到 CSV
    csv_filename = f"{filename}_pos.csv"
    with open(csv_filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(all_axes)

    print(f"优化结果已保存到 {csv_filename}")
if __name__ == '__main__':
    main()