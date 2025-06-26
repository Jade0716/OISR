import os
import numpy as np
from os.path import join as pjoin
from utils.config_utils import SAVE_PATH, VISU_SAVE_PATH
from utils.read_utils import load_rgb_image, load_depth_map, load_anno_dict, load_meta
from utils.visu_utils import visu_point_cloud
from tqdm import tqdm  # 导入tqdm

if __name__ == "__main__":
    # 遍历save_path中的所有文件
    rgb_directory = pjoin(SAVE_PATH, 'rgb')  # 假设'rgb'文件夹在VISU_SAVE_PATH目录下
    files = [filename for filename in os.listdir(rgb_directory) if filename.endswith(".png")]  # 获取所有png文件

    # 使用tqdm包装files列表，添加进度条
    for filename in tqdm(files, desc="Processing files", unit="file"):
        # 解析文件名，得到模型ID、类别、渲染索引、相机位置索引
        parts = filename.split('_')
        if len(parts) >= 4:
            CATEGORY = parts[0]
            MODEL_ID = int(parts[1])
            CAMERA_POSITION_INDEX = int(parts[2])
            RENDER_INDEX = int(parts[3].split('.')[0])  # 去掉扩展名

            save_path = pjoin(VISU_SAVE_PATH, f'{CATEGORY}_{MODEL_ID}_{CAMERA_POSITION_INDEX}_{RENDER_INDEX}')
            txt_save_path = pjoin(save_path, f'{CATEGORY}_{MODEL_ID}_{CAMERA_POSITION_INDEX}_{RENDER_INDEX}.txt')

            # 检查结果文件是否已经存在
            if os.path.exists(txt_save_path):
                continue

            if not os.path.exists(save_path):
                os.makedirs(save_path)

            # 加载相关数据
            rgb_image = load_rgb_image(SAVE_PATH, f'{CATEGORY}_{MODEL_ID}_{CAMERA_POSITION_INDEX}_{RENDER_INDEX}')
            depth_map = load_depth_map(SAVE_PATH, f'{CATEGORY}_{MODEL_ID}_{CAMERA_POSITION_INDEX}_{RENDER_INDEX}')
            metafile = load_meta(SAVE_PATH, f'{CATEGORY}_{MODEL_ID}_{CAMERA_POSITION_INDEX}_{RENDER_INDEX}')

            # 生成点云并保存为txt文件
            point_cloud_with_rgb = visu_point_cloud(rgb_image, depth_map, metafile)
            np.savetxt(txt_save_path, point_cloud_with_rgb, fmt='%f', delimiter=',')

    print('Done!')