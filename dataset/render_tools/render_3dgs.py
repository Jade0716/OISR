import os
import sys
from os.path import join as pjoin
import numpy as np
from argparse import ArgumentParser
import scipy.spatial.transform
from scipy.spatial.transform import Rotation as R

from utils.config_utils import PARTNET_DATASET_PATH, AKB48_DATASET_PATH, PARTNET_ID_PATH, AKB48_ID_PATH, \
    PARTNET_CAMERA_POSITION_RANGE, \
    AKB48_CAMERA_POSITION_RANGE, TARGET_GAPARTS, BACKGROUND_RGB, SAVE_PATH
from utils.read_utils import get_id_category, read_joints_from_urdf_file, save_rgb_image, save_depth_map, \
    save_anno_dict, save_meta
from utils.render_utils import get_cam_pos, set_all_scene, render_rgb_image, render_depth_map, \
    render_sem_ins_seg_map, add_background_color_for_image, get_camera_pos_mat, merge_joint_qpos, get_12_cam_positions
from utils.pose_utils import query_part_pose_from_joint_qpos, get_NPCS_map_from_oriented_bbox


def render_one_image(dataset_name, model_id, camera_idx, height, width, use_raytracing=False,
                     replace_texture=False):
    # 1. read the id list to get the category; set path, camera range, and base link name
    if dataset_name == 'partnet':
        category = get_id_category(model_id, PARTNET_ID_PATH)
        if category is None:
            raise ValueError(f'Cannot find the category of model {model_id}')
        data_path = pjoin(PARTNET_DATASET_PATH, str(model_id))
        camera_position_range = PARTNET_CAMERA_POSITION_RANGE
        base_link_name = 'base'

    elif dataset_name == 'akb48':
        category = get_id_category(model_id, AKB48_ID_PATH)
        if category is None:
            raise ValueError(f'Cannot find the category of model {model_id}')
        data_path = pjoin(AKB48_DATASET_PATH, category, str(model_id))
        camera_position_range = AKB48_CAMERA_POSITION_RANGE
        base_link_name = 'root'

    else:
        raise ValueError(f'Unknown dataset {dataset_name}')

    # 2. read the urdf file,  get the kinematic chain, and collect all the joints information
    joints_dict = read_joints_from_urdf_file(data_path, 'mobility_annotation_gapartnet.urdf')

    # 3. generate the joint qpos randomly in the limit range
    joint_qpos = {}
    for joint_name in joints_dict:
        joint_type = joints_dict[joint_name]['type']
        if joint_type == 'prismatic' or joint_type == 'revolute':
            joint_limit = joints_dict[joint_name]['limit']
            joint_qpos[joint_name] = joint_limit[0]
        elif joint_type == 'fixed':
            joint_qpos[joint_name] = 0.0  # ! the qpos of fixed joint must be 0.0
        elif joint_type == 'continuous':
            joint_qpos[joint_name] = 0.0
        else:
            raise ValueError(f'Unknown joint type {joint_type}')

    # 4. generate the camera pose randomly in the specified range
    camera_range = camera_position_range[category][camera_idx]
    camera_pos = get_12_cam_positions(theta_deg=70.0, distance=4.1)

    for camera_idx, camera_pos in enumerate(camera_pos):
        # 5. set scene and camera with current camera_pos
        scene, camera, engine, robot = set_all_scene(data_path=data_path,
                                                     urdf_file='mobility_annotation_gapartnet.urdf',
                                                     cam_pos=camera_pos,
                                                     width=width,
                                                     height=height,
                                                     use_raytracing=False,
                                                     joint_qpos_dict=joint_qpos)


        # 6. compute part poses
        link_pose_dict = query_part_pose_from_joint_qpos(data_path=data_path,
                                                         anno_file='link_annotation_gapartnet.json',
                                                         joint_qpos=joint_qpos,
                                                         joints_dict=joints_dict,
                                                         target_parts=TARGET_GAPARTS,
                                                         base_link_name=base_link_name,
                                                         robot=robot)

        # 7. render
        rgb_image = render_rgb_image(camera=camera)
        depth_map = render_depth_map(camera=camera)
        sem_seg_map, ins_seg_map, valid_linkName_to_instId = render_sem_ins_seg_map(scene=scene,
                                                                                    camera=camera,
                                                                                    link_pose_dict=link_pose_dict,
                                                                                    depth_map=depth_map)
        valid_link_pose_dict = {link_name: link_pose_dict[link_name] for link_name in valid_linkName_to_instId.keys()}

        # 8. camera matrix
        camera_intrinsic, world2camera_rotation, camera2world_translation,  T_cam2world_sapien = get_camera_pos_mat(camera)

        # 9. NPCS map
        valid_linkPose_RTS_dict, valid_NPCS_map = get_NPCS_map_from_oriented_bbox(depth_map,
                                                                                  ins_seg_map,
                                                                                  valid_linkName_to_instId,
                                                                                  valid_link_pose_dict,
                                                                                  camera_intrinsic,
                                                                                  world2camera_rotation,
                                                                                  camera2world_translation)

        # 10. optional texture render
        if replace_texture:
            assert dataset_name == 'partnet', 'Texture replacement is only needed for PartNet dataset'
            texture_joints_dict = read_joints_from_urdf_file(data_path, 'mobility_texture_gapartnet.urdf')
            texture_joint_qpos = merge_joint_qpos(joint_qpos, joints_dict, texture_joints_dict)
            scene, camera, engine, robot = set_all_scene(data_path=data_path,
                                                         urdf_file='mobility_texture_gapartnet.urdf',
                                                         cam_pos=camera_pos,
                                                         width=width,
                                                         height=height,
                                                         use_raytracing=use_raytracing,
                                                         joint_qpos_dict=texture_joint_qpos,
                                                         engine=engine)
            rgb_image = render_rgb_image(camera=camera)

        # 11. add background
        rgb_image = add_background_color_for_image(rgb_image, depth_map, BACKGROUND_RGB)

        # 12. save all
        save_name = f"{category}_{model_id}_{camera_idx}"
        os.makedirs(SAVE_PATH, exist_ok=True)
        T_world2cam_sapien = np.linalg.inv(T_cam2world_sapien)
        R_sapien = T_world2cam_sapien[:3, :3]
        t_sapien = T_world2cam_sapien[:3, 3]

        # 2. 坐标系转换
        R_convert = np.array([
            [1, 0, 0],  # X unchanged
            [0, -1, 0],  # Y flipped
            [0, 0, -1]  # Z flipped
        ])
    
        R_colmap = R_convert @ R_sapien
        t_colmap = R_convert @ t_sapien



        # 4. 转为 COLMAP 四元数格式（[qw, qx, qy, qz]）
        quat = R.from_matrix(R_colmap).as_quat()  # [x, y, z, w]
        qx, qy, qz, qw = quat
        
        qw, qx, qy, qz = qw, qx, qy, qz  # 调整顺序为 [w, x, y, z]

        # 平移向量
        tx, ty, tz = t_colmap

        # 构造路径和行内容
        pose_file = os.path.join(SAVE_PATH, "images.txt")
        pose_line = f"{camera_idx + 1} {qw:.17f} {qx:.17f} {qy:.17f} {qz:.17f} {tx:.17f} {ty:.17f} {tz:.17f} 1 {save_name}.png\n\n"

        # 文件开头写一次说明（仅第一次）
        if not os.path.exists(pose_file):
            with open(pose_file, "w") as f:
                f.write("# Image list with two lines of data per image:\n")
                f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n\n")

        # 追加写入当前位姿信息
        with open(pose_file, "a") as f:
            f.write(pose_line)
        save_rgb_image(rgb_image, SAVE_PATH, save_name)
        save_depth_map(depth_map, SAVE_PATH, save_name)

        bbox_pose_dict = {
            link_name: {
                'bbox': valid_link_pose_dict[link_name]['bbox'],
                'category_id': valid_link_pose_dict[link_name]['category_id'],
                'instance_id': valid_linkName_to_instId[link_name],
                'pose_RTS_param': valid_linkPose_RTS_dict[link_name],
            } for link_name in valid_link_pose_dict
        }

        anno_dict = {
            'semantic_segmentation': sem_seg_map,
            'instance_segmentation': ins_seg_map,
            'npcs_map': valid_NPCS_map,
            'bbox_pose_dict': bbox_pose_dict,
        }
        save_anno_dict(anno_dict, SAVE_PATH, save_name)

        metafile = {
            'model_id': model_id,
            'category': category,
            'camera_idx': camera_idx,
            'width': width,
            'height': height,
            'joint_qpos': joint_qpos,
            'camera_pos': camera_pos.reshape(-1).tolist(),
            'camera_intrinsic': camera_intrinsic.reshape(-1).tolist(),
            'world2camera_rotation': world2camera_rotation.reshape(-1).tolist(),
            'camera2world_translation': camera2world_translation.reshape(-1).tolist(),
            'target_gaparts': TARGET_GAPARTS,
            'use_raytracing': use_raytracing,
            'replace_texture': replace_texture,
        }
        save_meta(metafile, SAVE_PATH, save_name)


        print(f"Rendered {save_name} successfully!")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='partnet', help='Specify the dataset to render')
    parser.add_argument('--model_id', type=int, default=46859, help='Specify the model id to render')
    parser.add_argument('--camera_idx', type=int, default=0, help='Specify the camera range index to render')
    parser.add_argument('--height', type=int, default=800, help='Specify the height of the rendered image')
    parser.add_argument('--width', type=int, default=800, help='Specify the width of the rendered image')
    parser.add_argument('--ray_tracing', type=bool, default=False,
                        help='Specify whether to use ray tracing in rendering')
    parser.add_argument('--replace_texture', type=bool, default=False,
                        help='Specify whether to replace the texture of the rendered image using the original model')

    args = parser.parse_args()

    assert args.dataset in ['partnet', 'akb48'], f'Unknown dataset {args.dataset}'
    if args.dataset == 'akb48':
        assert not args.replace_texture, 'Texture replacement is not needed for AKB48 dataset'

    render_one_image(args.dataset, args.model_id, args.camera_idx, args.height, args.width,
                     args.ray_tracing, args.replace_texture)

    print("Done!")
