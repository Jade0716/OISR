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
OBJECT_CATEGORIES = [['Box'],[ 'TrashCan'], ['Bucket'], ['Drawer']]

# OBJECT_CATEGORIES = ['Refrigerator']


AKB48_OBJECT_CATEGORIES = [
    'Box', 'TrashCan', 'Bucket', 'Drawer'
]
time_ = None
MAX_INSTANCE_NUM = 1000


def load_point_cloud_to_tensor(file_path):
    # 从txt文件中读取点云数据
    flow = torch.tensor(np.loadtxt(file_path) * 50, dtype=torch.float32).unsqueeze(0).cuda()  # (1, N, 3)

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
            point_cloud.append([x_new*10, y_new*10, z_new*10])
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
    max_radius = ((((pointcloud - center) ** 2).sum(1)) ** 0.5).max()
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


def sample_and_save_with_flow(model, filename0, filename1, data_path, save_path, num_points, gt_flow, visualize=False, device='cuda:0'):
    filename = '_'.join(filename0.split('_')[:-1])  # 去掉最后一部分（如 _0 或 _1）
    pth_save_path = pjoin(save_path, 'pth')
    os.makedirs(pth_save_path, exist_ok=True)
    meta_save_path = pjoin(save_path, 'meta')
    os.makedirs(meta_save_path, exist_ok=True)
    gt_save_path = pjoin(save_path, 'gt')
    os.makedirs(gt_save_path, exist_ok=True)

    anno_dict0 = load_anno_dict(data_path, filename0)
    metafile0 = load_meta(data_path, filename0)
    rgb_image0 = load_rgb_image(data_path, filename0)
    depth_map0 = load_depth_map(data_path, filename0)

    anno_dict1 = load_anno_dict(data_path, filename1)
    metafile1 = load_meta(data_path, filename1)
    rgb_image1 = load_rgb_image(data_path, filename1)
    depth_map1 = load_depth_map(data_path, filename1)



    # Get point cloud from back-projection
    pcs0, pcs0_rgb, pcs_sem, pcs_ins, pcs_npcs, pcs_idx = get_point_cloud(rgb_image0,
                                                                        depth_map0,
                                                                        anno_dict0['semantic_segmentation'],
                                                                        anno_dict0['instance_segmentation'],
                                                                        anno_dict0['npcs_map'],
                                                                        metafile0)
    # Get point cloud from back-projection
    pcs1, pcs1_rgb, _, _, _, _ = get_point_cloud(rgb_image1,
                                                                        depth_map1,
                                                                        anno_dict1['semantic_segmentation'],
                                                                        anno_dict1['instance_segmentation'],
                                                                        anno_dict1['npcs_map'],
                                                                        metafile1)

    assert ((pcs_sem == -1) == (pcs_ins == -1)).all(), 'Semantic and instance labels do not match!'

    if pcs0.shape[0] < num_points or pcs1.shape[0] < num_points:
        return -1
    else:
        # FPS sampling
        pcs0_sampled, fps_idx0 = FPS(pcs0, num_points, device)
        pcs1_sampled, fps_idx1 = FPS(pcs1, num_points, device)
        if pcs0_sampled is None or  pcs1_sampled is None:
            return -1

    # pcs_rgb_sampled = pcs0_rgb[fps_idx0]
    pcs_sem_sampled = pcs_sem[fps_idx0]
    pcs_ins_sampled = pcs_ins[fps_idx0]
    pcs_npcs_sampled = pcs_npcs[fps_idx0].astype(np.float32)
    pcs_idx_sampled = pcs_idx[fps_idx0].astype(np.int32)
    #calculate flow and normalize point cloud
    pcs0_sampled_normalized, max_radius, center = WorldSpaceToBallSpace(pcs0_sampled)
    pcs1_sampled_normalized = (pcs1_sampled - center)/max_radius

    scale_param = np.array([max_radius, center[0], center[1], center[2]])

    points0 = torch.tensor(pcs0_sampled_normalized*10, dtype=torch.float32).unsqueeze(0).cuda()  # (1, N, 3)
    points1 = torch.tensor(pcs1_sampled_normalized*10, dtype=torch.float32).unsqueeze(0).cuda()  # (1, N, 3)
    color0 = torch.tensor(pcs0_rgb[fps_idx0], dtype=torch.float32).unsqueeze(0).cuda()  # (1, N, 3)
    color1 = torch.tensor(pcs1_rgb[fps_idx1], dtype=torch.float32).unsqueeze(0).cuda()  # (1, N, 3)
    model = model.eval()
    with torch.no_grad():
        flows, fps_pc1_idxs, _, _, _ = model(points0, points1, color0, color1, gt_flow)
    pcs_flow_sampled = flows[0][3].squeeze(0).detach().cpu().numpy().transpose(1, 0)  # flow_np:(2048,3)
    pcs0_with_flow = pcs0_sampled_normalized + pcs_flow_sampled/10

    # convert semantic and instance labels
    # old label:
    # semantic label: -1 for others, [0, nClass-1] for part categories
    # instance label: -1 for others, [0, nInstance-1] for parts
    # new label:
    # semantic label: 0 for others, [1, nClass] for part categories
    # instance label: -100 for others, [0, nInstance-1] for parts
    pcs_sem_sampled_converted = pcs_sem_sampled + 1
    pcs_ins_sampled_converted = pcs_ins_sampled.copy()
    mask = pcs_ins_sampled_converted == -1
    pcs_ins_sampled_converted[mask] = -100

    # re-label instance label to be continuous (discontinuous because of FPS sampling)
    j = 0
    while (j < pcs_ins_sampled_converted.max()):
        if (len(np.where(pcs_ins_sampled_converted == j)[0]) == 0):
            mask = pcs_ins_sampled_converted == pcs_ins_sampled_converted.max()
            pcs_ins_sampled_converted[mask] = j
        j += 1

    # # visualize
    # if visualize:
    #         pcd = o3d.geometry.PointCloud()
    #         pcd.points = o3d.utility.Vector3dVector(pcs0_sampled)
    #         pcd.colors = o3d.utility.Vector3dVector(np.tile(np.array([[0, 0, 1]]), (pcs0_sampled.shape[0], 1)))  # 蓝色
    #
    #         # 添加 flow 后的点云（红色）
    #         pcd1 = o3d.geometry.PointCloud()
    #         pcd1.points = o3d.utility.Vector3dVector(pcs0_sampled + pcs0_flow_sampled)
    #         pcd1.colors = o3d.utility.Vector3dVector(np.tile(np.array([[1, 0, 0]]), (pcs0_sampled.shape[0], 1)))  # 红色
    #
    #         # 可视化两个点云
    #         o3d.visualization.draw_geometries([pcd, pcd1])

    torch.save((pcs0_sampled_normalized.astype(np.float32),  pcs0_with_flow.astype(np.float32), pcs0_rgb[fps_idx0].astype(
        np.float32), pcs_sem_sampled_converted.astype(np.int32), pcs_ins_sampled_converted.astype(
            np.int32), pcs_npcs_sampled.astype(np.float32), pcs_idx_sampled.astype(np.int32)), pjoin(pth_save_path, filename + '.pth'))
    np.savetxt(pjoin(meta_save_path, filename + '.txt'), scale_param, delimiter=',')

    # save gt for evaluation
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

    return 0




if __name__ == "__main__":

    akb48_id_list_path = '/home/liuyuyan/GaPartNet/dataset/render_tools/meta/akb48_all_id_list.txt'
    with open(akb48_id_list_path, 'r') as f:
        akb48_id_list = [line.strip().replace(' ', '_') for line in f.readlines()]  # 把空格换成下划线，方便匹配
    akb48_id_set = set(akb48_id_list)
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='partnet', help='Specify the dataset to render')
    parser.add_argument('--data_path', type=str, default='/16T/liuyuyan/example_rendered',
                        help='Specify the path to the rendered data')
    parser.add_argument('--save_path', type=str, default='/16T/liuyuyan/GAPartNetWithFlows_data',
                        help='Specify the path to save the sampled data')
    parser.add_argument('--idx', type=int, default=0)
    parser.add_argument('--num_points', type=int, default=20000, help='Specify the number of points to sample')
    parser.add_argument('--visualize', type=bool, default=False, help='Whether to visualize the sampled point cloud')
    args = parser.parse_args()

    # 设置设备
    print(f"Using GPU: '{'cuda'}'")

    DATASET = args.dataset
    DATA_PATH = args.data_path
    SAVE_PATH = args.save_path
    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)
    NUM_POINTS = args.num_points
    VISUALIZE = args.visualize
    OBJECT_CATEGORIES = OBJECT_CATEGORIES[args.idx]
    filename_list = sorted([
        x.split('.')[0]
        for x in os.listdir(pjoin(DATA_PATH, 'rgb'))
        if '_'.join(x.split('.')[0].split('_')[:2]) in akb48_id_set])
    filename_dict = {x: [] for x in OBJECT_CATEGORIES}

    for fn in filename_list:
        for x in OBJECT_CATEGORIES:
            if fn.startswith(x):
                filename_dict[x].append(fn)
                break
    # 载入flow网络
    module = importlib.import_module("model_difflow")
    model = getattr(module, 'PointConvBidirection')(iters=4)
    pretrain = "model_difflow_355_0.0114.pth"
    model.load_state_dict(torch.load(pretrain))  # , strict=False
    print(f'Loaded model {pretrain}')
    model = model.to('cuda')
    gt_flow = torch.tensor(np.loadtxt('flow_np.txt') * 50, dtype=torch.float32).unsqueeze(0).cuda()
    # LOG_FILE = open(LOG_PATH, 'w')

    # def log_writer(s):
    #     log_string(LOG_FILE, s)
    for category in tqdm(filename_dict, desc="Processing Categories", unit="category"):
        fn_list = filename_dict[category]

        paired_files = defaultdict(lambda: ([], []))
        for fn in fn_list:
            base_name_parts = fn.split('_')
            base_name = '_'.join(base_name_parts[:-2])
            suffix = int(base_name_parts[-1])
            paired_files[base_name][suffix].append(fn)

        for base_name, (files_0, files_1) in tqdm(paired_files.items(), desc=f'Processing {category}', unit='pair'):
            if len(files_0) == len(files_1) and len(files_0) > 0:
                for f0, f1 in zip(files_0, files_1):
                    ret = sample_and_save_with_flow(model, f0, f1, DATA_PATH, SAVE_PATH, NUM_POINTS, gt_flow, VISUALIZE,
                                                    'cuda')
                    if ret == -1:
                        print(f'Error processing {f0} and {f1}, num of points less than NUM_POINTS!')
                    else:
                        print(f'Finish processing {f0} and {f1}')

        print(f'Finish: {category}')

    # LOG_FILE.close()

    print('All finished!')

