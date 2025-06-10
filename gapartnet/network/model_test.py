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
def load_data(file_path: str, no_label: bool = False):  # 加载样本数据
    if not no_label:
        pc_data = torch.load(file_path)  # 加载点云数据，包括点、颜色、语义标签、实例标签等
    else:
        # 测试数据没有标签的情况（目前未实现）
        raise NotImplementedError

    pc_id = file_path.split("/")[-1].split(".")[0]
    object_cat = OBJECT_NAME2ID[pc_id.split("_")[0]]

    # 拼接 XYZ + 颜色 + flows
    points = np.concatenate([pc_data[0], pc_data[1], pc_data[2]], axis=-1, dtype=np.float32)
    flows = pc_data[2].astype(np.float32)

    return PointCloud(
        pc_id=pc_id,
        obj_cat=object_cat,
        points=points,
        sem_labels=None,
        instance_labels=None,
        gt_npcs=None,
        flows=flows
    )
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
    net.load_state_dict(torch.load("/home/liuyuyan/GaPartNet/gapartnet/checkpoints/04081625/models/final_model.pth"), strict=False)
    net.to(device)
    filepath = "/home/liuyuyan/GaPartNet/gapartnet/StorageFurniture_44962_0_4"
    pth_save_path = "/home/liuyuyan/GaPartNet/gapartnet/"
    points = np.loadtxt(f"{filepath}.txt")[:, :3]
    rgb = np.loadtxt(f"{filepath}.txt")[:, 3:6]
    flow = np.loadtxt(f"{filepath}.txt")[:,6:9]
    # pcs0_sampled, fps_idx = FPS(points, 20000)  #if need
    pcs_sampled_normalized, max_radius, center = WorldSpaceToBallSpace(points)
    torch.save((pcs_sampled_normalized.astype(np.float32), rgb.astype(
        np.float32), flow.astype(np.float32)), filepath + '.pth')
    file = load_data(filepath + '.pth', no_label=False)  # Pointlcould类
    file = file.to_tensor().to(device)
    voxel_size = (1 / 100, 1 / 100, 1 / 100)
    pc = apply_voxelization(file, voxel_size=voxel_size)
    pc = [pc.to(net.device)]  # List["PointCloud"]
    data_batch = PointCloud.collate(pc)  # PointCloudBatch
    net.eval()
    with torch.no_grad():
        pc_ids, sem_seg, proposals, stats_dict = net(data_batch)
    print(pc_ids, sem_seg, proposals, stats_dict)
    # 计算 loss
    loss = sum([stats_dict[key] for key in stats_dict.keys() if
                'loss' in key and isinstance(stats_dict[key], torch.Tensor)])
    print(loss)

    sample_ids = range(len(pc_ids))

    for sample_id in sample_ids:
        batch_id = sample_id // 1
        batch_sample_id = sample_id % 1
        proposals_ = proposals[batch_id]

        mask = proposals_.valid_mask.reshape(-1, 20000)[batch_sample_id]

        if proposals_ is not None:
            pt_xyz = proposals_.pt_xyz
            batch_indices = proposals_.batch_indices
            proposal_offsets = proposals_.proposal_offsets
            num_points_per_proposal = proposals_.num_points_per_proposal
            num_proposals = num_points_per_proposal.shape[0]
            score_preds = proposals_.score_preds
            mask = proposals_.valid_mask

            indices = torch.arange(mask.shape[0], dtype=torch.int64, device=sem_seg.sem_preds.device)
            proposal_indices = indices[proposals_.valid_mask][proposals_.sorted_indices]

            ins_seg_preds = torch.ones(mask.shape[0]) * 0
            for ins_i in range(len(proposal_offsets) - 1):
                ins_seg_preds[proposal_indices[proposal_offsets[ins_i]:proposal_offsets[ins_i + 1]]] = ins_i + 1

            npcs_maps = torch.ones(proposals_.valid_mask.shape[0], 3, device=proposals_.valid_mask.device) * 0.0
            valid_index = torch.where(proposals_.valid_mask == True)[0][
                proposals_.sorted_indices.long()[torch.where(proposals_.npcs_valid_mask == True)]]
            npcs_maps[valid_index] = proposals_.npcs_preds

            # bounding box
            bboxes = []
            bboxes_batch_index = []
            for proposal_i in range(len(proposal_offsets) - 1):
                npcs_i = npcs_maps[proposal_indices[proposal_offsets[proposal_i]:proposal_offsets[proposal_i + 1]]]
                npcs_i = npcs_i - 0.5
                xyz_i = pt_xyz[proposal_offsets[proposal_i]:proposal_offsets[proposal_i + 1]]
                # import pdb; pdb.set_trace()
                if xyz_i.shape[0] < 10:
                    continue
                bbox_xyz, scale, rotation, translation, out_transform, best_inlier_idx = estimate_pose_from_npcs(
                    xyz_i.cpu().numpy(), npcs_i.cpu().numpy())
                # import pdb; pdb.set_trace()
                if scale[0] == None:
                    continue
                bboxes_batch_index.append(batch_indices[proposal_offsets[proposal_i]])
                bboxes.append(bbox_xyz.tolist())

        # get the sampled data point
        sample_sem_pred = sem_seg.sem_preds.reshape(-1, 20000)[sample_id]
        sample_ins_seg_pred = ins_seg_preds.reshape(-1, 20000)[batch_sample_id]
        sample_npcs_map = npcs_maps.reshape(-1, 20000, 3)[batch_sample_id]
        sample_bboxes = [bboxes[i] for i in range(len(bboxes)) if bboxes_batch_index[i] == batch_sample_id]

        visualize_gapartnet(
            SAVE_ROOT="output/GAPartNetWithFlow_result",
            RAW_IMG_ROOT="data/image_kuafu",
            GAPARTNET_DATA_ROOT="/16T/liuyuyan/GAPartNetAllWithFlows",
            save_option=["raw", "pc", "sem_pred", "sem_gt", "ins_pred", "ins_gt", "npcs_pred", "npcs_gt", "bbox_gt", "bbox_gt_pure", "bbox_pred", "bbox_pred_pure"],
            name=pc_ids[sample_id],
            split="test_inter",
            sem_preds=sample_sem_pred.cpu().numpy(),  # type: ignore
            ins_preds=sample_ins_seg_pred.cpu().numpy(),
            npcs_preds=sample_npcs_map.cpu().numpy(),
            bboxes=sample_bboxes,
        )
if __name__ == '__main__':
    main()
