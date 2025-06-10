'''
Convert the rendered data into the input format for the GAPartNet framework.

Output .pth format:
point_cloud: (N,3), float32, (x,y,z) in camera coordinate
per_point_rgb: (N,3), float32, ranging in [0,1] (R,G,B)
semantic_label: (N, ), int32, ranging in [0,nClass], 0 for others, [1, nClass] for part categories
instance_label: (N, ), int32, ranging in {-100} \cup [0,nInstance-1], -100 for others, [0, nInstance-1] for parts
NPCS: (N,3), float32, ranging in [-1,1] (x,y,z)
idx: (N,2), int32, (y,x) in the image coordinate
'''

import os
from os.path import join as pjoin
from argparse import ArgumentParser
import numpy as np
from collections import defaultdict
import torch
import time
import open3d as o3d
import importlib
from tqdm import tqdm
from utils.read_utils import load_rgb_image, load_depth_map, load_anno_dict, load_meta, load_flow
from utils.sample_utils import FPS

# LOG_PATH = './log_sample.txt'

OBJECT_CATEGORIES = ['Bucket']
AKB48_OBJECT_CATEGORIES = [
    'Box', 'TrashCan', 'Bucket', 'Drawer'
]
time_ = None
MAX_INSTANCE_NUM = 1000
def load_point_cloud_to_tensor(file_path, batch_size=32):
    # 从txt文件中读取点云数据
    flow = torch.tensor(np.loadtxt(file_path)*50, dtype=torch.float32).unsqueeze(0).cuda()  # (1, N, 3)

    return flow

def log_string(file, s):
    file.write(s + '\n')
    print(s)


def get_point_cloud(rgb_image, depth_map, sem_seg_map, ins_seg_map, npcs_map, meta):
    width = meta['width']
    height = meta['height']
    K = np.array(meta['camera_intrinsic']).reshape(3, 3)

    point_cloud = []
    per_point_rgb = []
    per_point_sem_label = []
    per_point_ins_label = []
    per_point_npcs = []
    per_point_idx = []

    for y_ in range(height):
        for x_ in range(width):
            if sem_seg_map[y_, x_] == -2 or ins_seg_map[y_, x_] == -2:
                continue
            z_new = float(depth_map[y_, x_])
            x_new = (x_ - K[0, 2]) * z_new / K[0, 0]
            y_new = (y_ - K[1, 2]) * z_new / K[1, 1]
            point_cloud.append([x_new, y_new, z_new])
            per_point_rgb.append((rgb_image[y_, x_] / 255.0))
            per_point_sem_label.append(sem_seg_map[y_, x_])
            per_point_ins_label.append(ins_seg_map[y_, x_])
            per_point_npcs.append(npcs_map[y_, x_])
            per_point_idx.append([y_, x_])

    return np.array(point_cloud), np.array(per_point_rgb), np.array(per_point_sem_label).astype(np.int8), np.array(
        per_point_ins_label).astype(np.int8), np.array(per_point_npcs), np.array(per_point_idx)


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


# def sample_and_save_with_flow(model, filename0, filename1, data_path, save_path, num_points, gt_flow, visualize=False, device='cuda:0'):
#     global time_
#     filename = '_'.join(filename0.split('_')[:-1])  # 去掉最后一部分（如 _0 或 _1）
#     pth_save_path = pjoin(save_path, 'pth')
#     os.makedirs(pth_save_path, exist_ok=True)
#     meta_save_path = pjoin(save_path, 'meta')
#     os.makedirs(meta_save_path, exist_ok=True)
#     gt_save_path = pjoin(save_path, 'gt')
#     os.makedirs(gt_save_path, exist_ok=True)

#     anno_dict0 = load_anno_dict(data_path, filename0)
#     metafile0 = load_meta(data_path, filename0)
#     rgb_image0 = load_rgb_image(data_path, filename0)
#     depth_map0 = load_depth_map(data_path, filename0)

#     anno_dict1 = load_anno_dict(data_path, filename1)
#     metafile1 = load_meta(data_path, filename1)
#     rgb_image1 = load_rgb_image(data_path, filename1)
#     depth_map1 = load_depth_map(data_path, filename1)



#     # Get point cloud from back-projection
#     pcs0, pcs0_rgb, pcs_sem, pcs_ins, pcs_npcs, pcs_idx = get_point_cloud(rgb_image0,
#                                                                         depth_map0,
#                                                                         anno_dict0['semantic_segmentation'],
#                                                                         anno_dict0['instance_segmentation'],
#                                                                         anno_dict0['npcs_map'],
#                                                                         metafile0)
#     # Get point cloud from back-projection
#     pcs1, pcs1_rgb, _, _, _, _ = get_point_cloud(rgb_image1,
#                                                                         depth_map1,
#                                                                         anno_dict1['semantic_segmentation'],
#                                                                         anno_dict1['instance_segmentation'],
#                                                                         anno_dict1['npcs_map'],
#                                                                         metafile1)

#     assert ((pcs_sem == -1) == (pcs_ins == -1)).all(), 'Semantic and instance labels do not match!'
#     print(f"0:{time.time() - time_}")


#     # FPS sampling
#     pcs0_sampled, fps_idx = FPS(pcs0, num_points, device)
#     pcs1_sampled, fps_idx1 = FPS(pcs1, num_points, device)
#     if pcs0_sampled is None:
#         return -1
#     print(f"1:{time.time() - time_}")


#     pcs_sem_sampled = pcs_sem[fps_idx]
#     pcs_ins_sampled = pcs_ins[fps_idx]
#     pcs_npcs_sampled = pcs_npcs[fps_idx].astype(np.float32)
#     pcs_idx_sampled = pcs_idx[fps_idx].astype(np.int32)
#     #calculate flow
#     gt_flow, _ = load_point_cloud_to_tensor('flow_np.txt', batch_size=1)
#     points0 = torch.tensor(pcs0_sampled, dtype=torch.float32).unsqueeze(0).cuda()  # (1, N, 3)
#     points1 = torch.tensor(pcs1_sampled, dtype=torch.float32).unsqueeze(0).cuda()  # (1, N, 3)
#     color0 = torch.tensor(pcs0_rgb[fps_idx], dtype=torch.float32).unsqueeze(0).cuda()  # (1, N, 3)
#     color1 = torch.tensor(pcs1_rgb[fps_idx1], dtype=torch.float32).unsqueeze(0).cuda()  # (1, N, 3)
#     model = model.eval()
#     with torch.no_grad():
#         flows, fps_pc1_idxs, _, _, _ = model(points0, points1, color0, color1, gt_flow)
#     pcs_flow_sampled = flows[0][0].squeeze(0).detach().cpu().numpy().transpose(1, 0)  # flow_np:(2048,3)
#     # pcs_flow_sampled = flows[0][0].squeeze(0)
#     print(f"2:{time.time() - time_}")
#     # normalize point cloud
#     pcs_sampled_normalized, max_radius, center = WorldSpaceToBallSpace(pcs0_sampled)
#     scale_param = np.array([max_radius, center[0], center[1], center[2]])


#     # convert semantic and instance labels
#     # old label:
#     # semantic label: -1 for others, [0, nClass-1] for part categories
#     # instance label: -1 for others, [0, nInstance-1] for parts
#     # new label:
#     # semantic label: 0 for others, [1, nClass] for part categories
#     # instance label: -100 for others, [0, nInstance-1] for parts
#     pcs_sem_sampled_converted = pcs_sem_sampled + 1
#     pcs_ins_sampled_converted = pcs_ins_sampled.copy()
#     mask = pcs_ins_sampled_converted == -1
#     pcs_ins_sampled_converted[mask] = -100

#     # re-label instance label to be continuous (discontinuous because of FPS sampling)
#     j = 0
#     while (j < pcs_ins_sampled_converted.max()):
#         if (len(np.where(pcs_ins_sampled_converted == j)[0]) == 0):
#             mask = pcs_ins_sampled_converted == pcs_ins_sampled_converted.max()
#             pcs_ins_sampled_converted[mask] = j
#         j += 1
#     # pcs_ins_sampled_converted = pcs_ins_sampled.clone()  # 保证原数据不变
#     # max_instance = pcs_ins_sampled_converted.max().item()  # 获取最大实例标签
#     # for j in range(max_instance + 1):
#     #     mask = pcs_ins_sampled_converted == j
#     #     if mask.sum().item() == 0:  # 如果没有找到当前标签，则跳过
#     #         next_instance = pcs_ins_sampled_converted.max().item() + 1  # 获取下一个标签
#     #         pcs_ins_sampled_converted[mask] = next_instance  # 更新标签
#     # pcs_ins_sampled_converted = pcs_ins_sampled_converted.cpu().numpy()
#     # visualize
#     if visualize:
#         pcd = o3d.geometry.PointCloud()
#         pcd.points = o3d.utility.Vector3dVector(pcs_sampled_normalized)
#         pcd.colors = o3d.utility.Vector3dVector(pcs0_rgb[fps_idx])
#         o3d.visualization.draw_geometries([pcd])

#     torch.save((pcs_sampled_normalized.astype(np.float32), pcs0_rgb[fps_idx].astype(
#         np.float32), pcs_sem_sampled_converted.astype(np.int32), pcs_ins_sampled_converted.astype(
#             np.int32), pcs_npcs_sampled.astype(np.float32), pcs_idx_sampled.astype(np.int32), pcs_flow_sampled.astype(np.float32)), pjoin(pth_save_path, filename + '.pth'))
#     np.savetxt(pjoin(meta_save_path, filename + '.txt'), scale_param, delimiter=',')

#     # save gt for evaluation
#     label_sem_ins = np.ones(pcs_ins_sampled_converted.shape, dtype=np.int32) * (-100)
#     inst_num = int(pcs_ins_sampled_converted.max() + 1)
#     for inst_id in range(inst_num):
#         instance_mask = np.where(pcs_ins_sampled_converted == inst_id)[0]
#         if instance_mask.shape[0] == 0:
#             raise ValueError(f'{filename} has a part missing from point cloud, instance label is not continuous')
#         semantic_label = int(pcs_sem_sampled_converted[instance_mask[0]])
#         if semantic_label == 0:
#             raise ValueError(f'{filename} has a part with semantic label [others]')
#         label_sem_ins[instance_mask] = semantic_label * MAX_INSTANCE_NUM + inst_id

#     np.savetxt(pjoin(gt_save_path, filename + '.txt'), label_sem_ins, fmt='%d')

#     return 0


def sample_and_save_with_flow_batch(model, filenames0, filenames1, data_path, save_path, num_points, gt_flow, visualize=False, device='cuda'):
    # Ensure that the two lists have the same length.
    # global time_
    # print(f"0:{time.time() - time_}")
    assert len(filenames0) == len(filenames1), "Batch sizes do not match!"
    batch_size = len(filenames0)

    # Load batch data
    anno_dicts0 = [load_anno_dict(data_path, filename0) for filename0 in filenames0]
    metafiles0 = [load_meta(data_path, filename0) for filename0 in filenames0]
    rgb_images0 = [load_rgb_image(data_path, filename0) for filename0 in filenames0]
    depth_maps0 = [load_depth_map(data_path, filename0) for filename0 in filenames0]

    anno_dicts1 = [load_anno_dict(data_path, filename1) for filename1 in filenames1]
    metafiles1 = [load_meta(data_path, filename1) for filename1 in filenames1]
    rgb_images1 = [load_rgb_image(data_path, filename1) for filename1 in filenames1]
    depth_maps1 = [load_depth_map(data_path, filename1) for filename1 in filenames1]

    # Generate point clouds for the batch
    pcs_batch0, pcs0_rgbs, pcs_sems, pcs_inss, pcs_npcss, pcs_idxs = zip(*[
        get_point_cloud(rgb_image0, depth_map0,
                        anno_dict0['semantic_segmentation'],
                        anno_dict0['instance_segmentation'],
                        anno_dict0['npcs_map'], metafile0)
        for rgb_image0, depth_map0, anno_dict0, metafile0 in zip(rgb_images0, depth_maps0, anno_dicts0, metafiles0)])
    pcs_batch1, pcs1_rgbs, _, _, _, _ = zip(*[
        get_point_cloud(rgb_image1, depth_map1,
                        anno_dict1['semantic_segmentation'],
                        anno_dict1['instance_segmentation'],
                        anno_dict1['npcs_map'], metafile1)
        for rgb_image1, depth_map1, anno_dict1, metafile1 in zip(rgb_images1, depth_maps1, anno_dicts1, metafiles1)])
    max_points = max(pc.shape[0] for pc in pcs_batch0)
    pcs_batch0_padded = []
    for pc in pcs_batch0:
        points_num, _ = pc.shape
        if points_num < max_points:
            padding = np.zeros((max_points - points_num, 3))
            padded_pc = np.vstack((pc, padding))
        else:
            padded_pc = pc[:max_points]
        pcs_batch0_padded.append(padded_pc)
    max_points = max(pc.shape[0] for pc in pcs_batch1)
    pcs_batch1_padded = []
    for pc in pcs_batch1:
        points_num, _ = pc.shape
        if points_num < max_points:
            padding = np.zeros((max_points - points_num, 3))
            padded_pc = np.vstack((pc, padding))
        else:
            padded_pc = pc[:max_points]
        pcs_batch1_padded.append(padded_pc)
    # Convert point clouds to numpy arrays suitable for FPS function
    pcs_batch0_np = np.stack(pcs_batch0_padded)
    pcs_batch1_np = np.stack(pcs_batch1_padded)
    # print(f"1:{time.time() - time_}")
    # Assuming num_points is a list or array with length B indicating how many points to sample for each point cloud
    sampled_points_batch0, fps_idxs_batch0 = FPS(pcs_batch0_np, num_points, device=device)
    sampled_points_batch1, fps_idxs_batch1 = FPS(pcs_batch1_np, num_points, device=device)
    # print(f"2:{time.time() - time_}")
    # Calculate flow for the batch
    points0_batch = torch.tensor(sampled_points_batch0, dtype=torch.float32).to(device)
    points1_batch = torch.tensor(sampled_points_batch1, dtype=torch.float32).to(device)
    color0_batch = torch.tensor(np.stack([pcs0_rgb[fps_idx] for pcs0_rgb, fps_idx in zip(pcs0_rgbs, fps_idxs_batch0)]), dtype=torch.float32).to(device)
    color1_batch = torch.tensor(np.stack([pcs1_rgb[fps_idx] for pcs1_rgb, fps_idx in zip(pcs1_rgbs, fps_idxs_batch1)]), dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        flows_batch, _, _, _, _ = model(points0_batch, points1_batch, color0_batch, color1_batch, gt_flow)
    # print(f"3:{time.time() - time_}")
    # Process and save results for each file in the batch
    for i, (filename0, filename1) in enumerate(zip(filenames0, filenames1)):
        filename = '_'.join(filename0.split('_')[:-1])  # Remove the last part of the filename
        pth_save_path = pjoin(save_path, 'pth')
        os.makedirs(pth_save_path, exist_ok=True)
        meta_save_path = pjoin(save_path, 'meta')
        os.makedirs(meta_save_path, exist_ok=True)
        gt_save_path = pjoin(save_path, 'gt')
        os.makedirs(gt_save_path, exist_ok=True)

        pcs_sem_sampled = pcs_sems[i][fps_idxs_batch0[i]]
        pcs_ins_sampled = pcs_inss[i][fps_idxs_batch0[i]]
        pcs_npcs_sampled = pcs_npcss[i][fps_idxs_batch0[i]].astype(np.float32)
        pcs_idx_sampled = pcs_idxs[i][fps_idxs_batch0[i]].astype(np.int32)
        pcs_flow_sampled = flows_batch[0][0][i].cpu().numpy().transpose(1, 0)

        pcs_sampled_normalized, max_radius, center = WorldSpaceToBallSpace(sampled_points_batch0[i])
        scale_param = np.array([max_radius, center[0], center[1], center[2]])

        pcs_sem_sampled_converted = pcs_sem_sampled + 1
        pcs_ins_sampled_converted = pcs_ins_sampled.copy()
        mask = pcs_ins_sampled_converted == -1
        pcs_ins_sampled_converted[mask] = -100

        j = 0
        while j < pcs_ins_sampled_converted.max():
            if len(np.where(pcs_ins_sampled_converted == j)[0]) == 0:
                mask = pcs_ins_sampled_converted == pcs_ins_sampled_converted.max()
                pcs_ins_sampled_converted[mask] = j
            j += 1

        if visualize:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pcs_sampled_normalized)
            pcd.colors = o3d.utility.Vector3dVector(pcs0_rgbs[i][fps_idxs_batch0[i]])
            o3d.visualization.draw_geometries([pcd])

        torch.save((pcs_sampled_normalized.astype(np.float32), pcs0_rgbs[i][fps_idxs_batch0[i]].astype(
            np.float32), pcs_sem_sampled_converted.astype(np.int32), pcs_ins_sampled_converted.astype(
            np.int32), pcs_npcs_sampled.astype(np.float32), pcs_idx_sampled.astype(np.int32),
                   pcs_flow_sampled.astype(np.float32)), pjoin(pth_save_path, filename + '.pth'))
        np.savetxt(pjoin(meta_save_path, filename + '.txt'), scale_param, delimiter=',')

        label_sem_ins = np.ones(pcs_ins_sampled_converted.shape, dtype=np.int32) * (-100)
        inst_num = int(pcs_ins_sampled_converted.max() + 1)
        for inst_id in range(inst_num):
            instance_mask = np.where(pcs_ins_sampled_converted == inst_id)[0]
            if instance_mask.shape[0] == 0:
                raise ValueError(f'{filename} has a part missing from point cloud, instance label is not continuous')
            semantic_label = int(pcs_sem_sampled_converted[instance_mask[0]])
            if semantic_label == 0:
                raise ValueError(f'{filename} has a part with semantic label [others]')
            label_sem_ins[instance_mask] = semantic_label * MAX_INSTANCE_NUM + inst_id

        np.savetxt(pjoin(gt_save_path, filename + '.txt'), label_sem_ins, fmt='%d')
        # print(f"4:{time.time() - time_}")
    return 0


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='partnet', help='Specify the dataset to render')
    parser.add_argument('--data_path', type=str, default='/16T/liuyuyan/example_rendered',
                        help='Specify the path to the rendered data')
    parser.add_argument('--save_path', type=str, default='/16T/liuyuyan/GAPartNetWithFlows_data',
                        help='Specify the path to save the sampled data')
    parser.add_argument('--num_points', type=int, default=20000, help='Specify the number of points to sample')
    parser.add_argument('--visualize', type=bool, default=False, help='Whether to visualize the sampled point cloud')
    args = parser.parse_args()
    # 设置设备
    print(f"Using GPU: {'cuda.get_current_device()'}")

    # 加载模型
    module = importlib.import_module("model_difflow")
    model = getattr(module, 'PointConvBidirection')(iters=4)

    # 加载预训练模型
    pretrain = "model_difflow_355_0.0114.pth"
    model.load_state_dict(torch.load(pretrain))  # , strict=False
    print(f'Loaded model {pretrain}')

    # 将模型移到指定 GPU 上
    model = model.to('cuda')
    model.eval()
    gt_flow = load_point_cloud_to_tensor('flow_np.txt', batch_size=32).to('cuda')
    # 禁用 BatchNorm 的统计
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.track_running_stats = False

    DATASET = args.dataset
    DATA_PATH = args.data_path
    SAVE_PATH = args.save_path
    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)
    NUM_POINTS = args.num_points
    VISUALIZE = args.visualize
    if DATASET == 'partnet':
        OBJECT_CATEGORIES = OBJECT_CATEGORIES
    elif DATASET == 'akb48':
        OBJECT_CATEGORIES = AKB48_OBJECT_CATEGORIES
    else:
        raise ValueError(f'Unknown dataset {DATASET}')
    
    filename_list = sorted([x.split('.')[0] for x in os.listdir(pjoin(DATA_PATH, 'rgb'))])
    filename_dict = {x: [] for x in OBJECT_CATEGORIES}

    for fn in filename_list:
        for x in OBJECT_CATEGORIES:
            if fn.startswith(x):
                filename_dict[x].append(fn)
                break
    
    # LOG_FILE = open(LOG_PATH, 'w')

    # def log_writer(s):
    #     log_string(LOG_FILE, s)
    batch_size = 1
    for category in tqdm(filename_dict, desc="Processing Categories", unit="category"):
        fn_list = filename_dict[category]

        # 使用defaultdict来收集属于同一个base_name的所有文件
        paired_files_batched = defaultdict(lambda: ([], []))
        for fn in fn_list:
            base_name_parts = fn.split('_')
            base_name = '_'.join(base_name_parts[:-2])  # 获取基础名称，去掉最后两个下划线后的部分
            suffix = int(base_name_parts[-1])  # 获取最后一个数字作为后缀，用于区分0和1
            paired_files_batched[base_name][suffix].append(fn)

        valid_batches = []
        for base_name, (files_0, files_1) in paired_files_batched.items():
            if len(files_0) == len(files_1) and len(files_0) > 0:  # 确保每组都有对应的文件
                valid_batches.append((files_0, files_1))

        for batch_0, batch_1 in tqdm(valid_batches, desc=f'Processing {category}', unit='batch'):
            # 根据batch_size拆分文件
            num_batches = len(batch_0) // batch_size
            for i in range(num_batches):
                batch_0_sub = batch_0[i * batch_size:(i + 1) * batch_size]
                batch_1_sub = batch_1[i * batch_size:(i + 1) * batch_size]

                # 处理每个子batch
                # time_ = time.time()
                ret = sample_and_save_with_flow_batch(model, batch_0_sub, batch_1_sub, DATA_PATH, SAVE_PATH, NUM_POINTS, gt_flow, VISUALIZE,
                                                    'cuda')
                if ret == -1:
                    print(f'Error processing batch starting with {batch_0_sub[0]} and {batch_1_sub[0]}, num of points less than NUM_POINTS!')
                else:
                    print(f'Finish processing batch starting with {batch_0_sub[0]} and {batch_1_sub[0]}')


        print(f'Finish: {category}')

    
    # LOG_FILE.close()

    print('All finished!')

