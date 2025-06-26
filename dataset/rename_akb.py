import os
import re


def load_category_mapping(file_path):
    """ 读取类别和ID的映射，返回集合以便快速查找 """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    category_set = set()
    for line in lines:
        line = line.strip()
        if line:
            category_set.add(line.replace(' ', '_'))  # 统一格式，例如 Box 128 -> Box_128
    return category_set


def rename_files(base_dirs, category_set):
    """ 遍历指定目录并重命名符合规则的文件 """
    for base_dir in base_dirs:
        if not os.path.exists(base_dir):
            print(f"Warning: {base_dir} does not exist, skipping...")
            continue

        for file_name in os.listdir(base_dir):
            match = re.match(r"([A-Za-z]+)_(\d+)(_\d+_\d+\.(pth|txt))", file_name)
            if match:
                category, id_num, suffix, ext = match.groups()
                original_name = f"{category}_{id_num}{suffix}"
                new_name = f"AKB{category}_{id_num}{suffix}"

                if f"{category}_{id_num}" in category_set:
                    old_path = os.path.join(base_dir, original_name)
                    new_path = os.path.join(base_dir, new_name)
                    os.rename(old_path, new_path)
                    print(f"Renamed: {old_path} -> {new_path}")


directories = [
'/16T/liuyuyan/GAPartNetAllWithFlows/train/meta/', '/16T/liuyuyan/GAPartNetAllWithFlows/train/pth/', '/16T/liuyuyan/GAPartNetAllWithFlows/train_gt/',
'/16T/liuyuyan/GAPartNetAllWithFlows/val/meta/', '/16T/liuyuyan/GAPartNetAllWithFlows/val/pth/', '/16T/liuyuyan/GAPartNetAllWithFlows/val_gt/',
'/16T/liuyuyan/GAPartNetAllWithFlows/test_intra/meta/', '/16T/liuyuyan/GAPartNetAllWithFlows/test_intra/pth/', '/16T/liuyuyan/GAPartNetAllWithFlows/test_intra_gt/',
]

category_set = load_category_mapping('render_tools/meta/akb48_all_id_list.txt')
rename_files(directories, category_set)
