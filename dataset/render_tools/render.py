import os
import sys
from os.path import join as pjoin
import math
import numpy as np
from argparse import ArgumentParser

from utils.config_utils import PARTNET_DATASET_PATH, AKB48_DATASET_PATH, PARTNET_ID_PATH, AKB48_ID_PATH, PARTNET_CAMERA_POSITION_RANGE, \
    AKB48_CAMERA_POSITION_RANGE, TARGET_GAPARTS, BACKGROUND_RGB, SAVE_PATH
from utils.read_utils import get_id_category, read_joints_from_urdf_file, save_rgb_image, save_depth_map, save_anno_dict, save_meta
from utils.render_utils import get_cam_pos, set_all_scene, render_rgb_image, render_depth_map, \
    render_sem_ins_seg_map, add_background_color_for_image, get_camera_pos_mat, merge_joint_qpos
from utils.pose_utils import query_part_pose_from_joint_qpos, get_NPCS_map_from_oriented_bbox



def get_interleaved_indices(n):
    first_half = list(range((n + 1) // 2))
    second_half = list(range((n + 1) // 2, n))
    result = []
    for a, b in zip(first_half, second_half):
        result.extend([a, b])
    if len(first_half) > len(second_half):
        result.append(first_half[-1])
    return result

def render_two_images(dataset_name, model_id, camera_idx, render_idx, height, width, use_raytracing=False, replace_texture=False):

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

    # 4. generate the camera pose randomly in the specified range
    camera_range = camera_position_range[category][camera_idx]
    camera_pos = get_cam_pos(
        theta_min=camera_range['theta_min'], theta_max=camera_range['theta_max'],
        phi_min=camera_range['phi_min'], phi_max=camera_range['phi_max'],
        dis_min=camera_range['distance_min'], dis_max=camera_range['distance_max']
    )
    flow_idx = 0
    while flow_idx < 2:
        joint_qpos = {}
        all_joint_qpos = {}
        prismatic_joints = [name for name, info in joints_dict.items() if info['type'] == 'prismatic']
        revolute_joints = [name for name, info in joints_dict.items() if info['type'] == 'revolute']

        prismatic_num = len(prismatic_joints)
        revolute_num = len(revolute_joints)

        # 预计算打散的 ratio 顺序
        prismatic_ratios = [(i + 1) / (prismatic_num + 1) * 0.8 for i in range(prismatic_num)]
        revolute_ratios = [(i + 1) / (revolute_num + 1) * 0.8 for i in range(revolute_num)]

        prismatic_order = get_interleaved_indices(prismatic_num)
        revolute_order = get_interleaved_indices(revolute_num)

        shuffled_prismatic_ratios = [prismatic_ratios[i] for i in prismatic_order]
        shuffled_revolute_ratios = [revolute_ratios[i] for i in revolute_order]

        for joint_name in joints_dict:
            joint_type = joints_dict[joint_name]['type']
            joint_limit = joints_dict[joint_name].get('limit', [0.0, 0.0])

            if joint_type in ['prismatic', 'revolute']:
                if flow_idx == 0:
                    # 所有关节都设为 limit[0]
                    joint_qpos[joint_name] = joint_limit[0]
                else:
                    if joint_type == 'prismatic':
                        joint_list = prismatic_joints
                        shuffled_ratios = shuffled_prismatic_ratios
                    else:
                        joint_list = revolute_joints
                        shuffled_ratios = shuffled_revolute_ratios

                    idx = joint_list.index(joint_name)
                    if len(joint_list) == 1:
                        ratio = 0.2
                    else:
                        ratio = shuffled_ratios[idx]

                    joint_qpos[joint_name] = joint_limit[0] + (joint_limit[1] - joint_limit[0]) * ratio

            elif joint_type == 'fixed':
                joint_qpos[joint_name] = 0.0

            elif joint_type == 'continuous':
                joint_qpos[joint_name] = -2000.0 if flow_idx == 0 else 2000.0

            else:
                raise ValueError(f'Unknown joint type {joint_type}')
            if flow_idx == 1:
                all_joint_qpos[joint_name] = {
            'joint_type':joint_type,
            'qpos': joint_qpos[joint_name],
            'limit': joint_limit
        }



        # 5. pass the joint qpos and the augmentation parameters to set up render environment and robot
        scene, camera, engine, robot = set_all_scene(data_path=data_path,
                                            urdf_file='mobility_annotation_gapartnet.urdf',
                                            cam_pos=camera_pos,
                                            width=width,
                                            height=height,
                                            use_raytracing=False,
                                            joint_qpos_dict=joint_qpos)

        # 6. use qpos to calculate the gapart poses
        link_pose_dict = query_part_pose_from_joint_qpos(data_path=data_path, anno_file='link_annotation_gapartnet.json', joint_qpos=joint_qpos, joints_dict=joints_dict, target_parts=TARGET_GAPARTS, base_link_name=base_link_name, robot=robot)

        # 7. render the rgb, depth, mask, valid(visible) gapart
        rgb_image = render_rgb_image(camera=camera)
        depth_map = render_depth_map(camera=camera)
        sem_seg_map, ins_seg_map, valid_linkName_to_instId = render_sem_ins_seg_map(scene=scene, camera=camera, link_pose_dict=link_pose_dict, depth_map=depth_map)
        valid_link_pose_dict = {link_name: link_pose_dict[link_name] for link_name in valid_linkName_to_instId.keys()}

        # 8. acquire camera intrinsic and extrinsic matrix
        camera_intrinsic, R_c2w, t_c2w , T_cam2world= get_camera_pos_mat(camera)

        # 9. calculate NPCS map
        valid_linkPose_RTS_dict, valid_NPCS_map = get_NPCS_map_from_oriented_bbox(depth_map, ins_seg_map, valid_linkName_to_instId, valid_link_pose_dict, camera_intrinsic, R_c2w, t_c2w)

        # 10. (optional, only for [partnet] dataset) use texture to render rgb to replace the previous rgb (texture issue during cutting the mesh)
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

        # 11. add background color
        rgb_image = add_background_color_for_image(rgb_image, depth_map, BACKGROUND_RGB)

        # 12. save the rendered results
        save_name = f"{category}_{model_id}_{camera_idx}_{render_idx}_{flow_idx}"
        if not os.path.exists(SAVE_PATH):
            os.mkdir(SAVE_PATH)

        save_rgb_image(rgb_image, SAVE_PATH, save_name)

        save_depth_map(depth_map, SAVE_PATH, save_name)

        bbox_pose_dict = {}
        for link_name in valid_link_pose_dict:
            bbox_pose_dict[link_name] = {
                'bbox': valid_link_pose_dict[link_name]['bbox'],
                'category_id': valid_link_pose_dict[link_name]['category_id'],
                'instance_id': valid_linkName_to_instId[link_name],
                'pose_RTS_param': valid_linkPose_RTS_dict[link_name],
            }
        anno_dict = {
            'semantic_segmentation': sem_seg_map,
            'instance_segmentation': ins_seg_map,
            'npcs_map': valid_NPCS_map,
            'bbox_pose_dict': bbox_pose_dict,
        }
        save_anno_dict(anno_dict, SAVE_PATH, save_name)
        if flow_idx==1:
            metafile = {
                'model_id': model_id,
                'category': category,
                'camera_idx': camera_idx,
                'render_idx': render_idx,
                'flow_idx': flow_idx,
                'width': width,
                'height': height,
                'joint_qpos': joint_qpos,
                'camera_pos': camera_pos.reshape(-1).tolist(),
                'camera_intrinsic': camera_intrinsic.reshape(-1).tolist(),
                'R_c2w': R_c2w.reshape(-1).tolist(),
                't_c2w': t_c2w.reshape(-1).tolist(),
                'target_gaparts': TARGET_GAPARTS,
                'use_raytracing': use_raytracing,
                'replace_texture': replace_texture,
                'full_joint_qpos' : all_joint_qpos,
            }
            save_meta(metafile, SAVE_PATH, save_name)

        print(f"Rendered {save_name} successfully!")
        flow_idx = flow_idx + 1


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='partnet', help='Specify the dataset to render')
    parser.add_argument('--model_id', type=int, default=10040, help='Specify the model id to render')
    parser.add_argument('--camera_idx', type=int, default=0, help='Specify the camera range index to render')
    parser.add_argument('--render_idx', type=int, default=0, help='Specify the render index to render')
    parser.add_argument('--height', type=int, default=800, help='Specify the height of the rendered image')
    parser.add_argument('--width', type=int, default=800, help='Specify the width of the rendered image')
    parser.add_argument('--ray_tracing', type=bool, default=False, help='Specify whether to use ray tracing in rendering')
    parser.add_argument('--replace_texture', type=bool, default=False, help='Specify whether to replace the texture of the rendered image using the original model')
    args = parser.parse_args()
    assert args.dataset in ['partnet', 'akb48'], f'Unknown dataset {args.dataset}'
    if args.dataset == 'akb48':
        assert not args.replace_texture, 'Texture replacement is not needed for AKB48 dataset'

    render_two_images(args.dataset, args.model_id, args.camera_idx, args.render_idx, args.height, args.width, args.ray_tracing, args.replace_texture)

    # print("Done!")

