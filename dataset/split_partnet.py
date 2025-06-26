import json
import os
from glob import glob
from tqdm import tqdm

# 读取 JSON 文件
json_paths = [
    "render_tools/meta/partnet_all_split.json",
    "render_tools/meta/akb48_all_split.json"
]

merged_data = {"seen_category": {}, "unseen_category": {}}

for json_path in json_paths:
    with open(json_path, "r") as f:
        data = json.load(f)

    # 合并 seen_category
    for category, instances in data["seen_category"].items():
        if category not in merged_data["seen_category"]:
            merged_data["seen_category"][category] = {"seen_instance": [], "unseen_instance": []}
        merged_data["seen_category"][category]["seen_instance"].extend(instances["seen_instance"])
        merged_data["seen_category"][category]["unseen_instance"].extend(instances["unseen_instance"])

    # 合并 unseen_category
    for category, instances in data["unseen_category"].items():
        if category not in merged_data["unseen_category"]:
            merged_data["unseen_category"][category] = {"seen_instance": [], "unseen_instance": []}
        merged_data["unseen_category"][category]["seen_instance"].extend(instances["seen_instance"])
        merged_data["unseen_category"][category]["unseen_instance"].extend(instances["unseen_instance"])
# 定义数据集根目录和目标目录
root = "/16T/liuyuyan/GAPartNetWithFlows_data/"
tar_root = "/16T/liuyuyan/GAPartNetAllWithFlows/"

# 定义数据集类别
splits = {
    "train": {"pth": [], "meta": [], "gt": []},
    "val": {"pth": [], "meta": [], "gt": []},
    "test_intra": {"pth": [], "meta": [], "gt": []},
    "test_inter": {"pth": [], "meta": [], "gt": []},
}


# 定义需要处理的文件类型
file_types = ["pth", "meta", "gt"]

# 处理 seen_category（已见类别）
for category, instances in tqdm(merged_data["seen_category"].items(), desc="Processing seen_category"):
    for instance in instances["seen_instance"]:
        files = {
            "pth": sorted(glob(os.path.join(root, "pth", f"{category}_{instance}_0_*.pth"))),
            "meta": sorted(glob(os.path.join(root, "meta", f"{category}_{instance}_0_*.txt"))),
            "gt": sorted(glob(os.path.join(root, "gt", f"{category}_{instance}_0_*.txt")))
        }
        for file_type in file_types:
            splits["train"][file_type].extend(files[file_type][:28])  # 前 28 份进入训练集
            splits["val"][file_type].extend(files[file_type][28:])  # 后 4 份进入验证集

    for instance in instances["unseen_instance"]:
        for file_type in file_types:
            if file_type == 'pth':
                splits["test_intra"][file_type].extend(glob(os.path.join(root, "pth", f"{category}_{instance}_0_*.pth")))  # 类内测试集的pth文件
            else:
                splits["test_intra"][file_type].extend(glob(os.path.join(root, file_type, f"{category}_{instance}_0_*.txt")))  # 类内测试集的meta和gt文件
# 处理 unseen_category（未见类别）
for category, instances in tqdm(merged_data["unseen_category"].items(), desc="Processing unseen_category"):
    for instance in instances["seen_instance"] + instances["unseen_instance"]:
        for file_type in file_types:
            if file_type == 'pth':
                splits["test_inter"][file_type].extend(glob(os.path.join(root, "pth", f"{category}_{instance}_0_*.pth")))  # 类内测试集的pth文件
            else:
                splits["test_inter"][file_type].extend(glob(os.path.join(root, file_type, f"{category}_{instance}_0_*.txt")))

# 移动文件到目标目录
def move_files(file_dict, split_name):
    target_dir = os.path.join(tar_root, split_name)
    os.makedirs(os.path.join(target_dir, "meta"), exist_ok=True)
    os.makedirs(os.path.join(target_dir, "pth"), exist_ok=True)
    os.makedirs(os.path.join(tar_root, f"{split_name}_gt"), exist_ok=True)

    # 遍历每个文件类型
    for file_type, file_list in file_dict.items():
        for file in tqdm(file_list, desc=f"Moving {file_type} files for {split_name}"):
            file_name = os.path.basename(file)

            if file_type == "gt":
                target_file = os.path.join(tar_root, f"{split_name}_gt", file_name)
            else:
                target_file = os.path.join(target_dir, file_type, file_name)


            os.system(f"mv {file} {target_file}")

# 执行文件移动
for split_name, file_dict in splits.items():
    print(f"Processing {split_name} with {sum(len(files) for files in file_dict.values())} files...")
    move_files(file_dict, split_name)  # 为每个split名称执行移动

print("Dataset split completed!")
# def move_files(file_list, split_name):
#     target_dir = os.path.join(tar_root, split_name)
#     os.makedirs(os.path.join(target_dir, "meta"), exist_ok=True)
#     os.makedirs(os.path.join(target_dir, "pth"), exist_ok=True)
#     os.makedirs(os.path.join(tar_root, f"{split_name}_gt"), exist_ok=True)
#     for file in tqdm(file_list, desc=f"Moving files for {split_name}"):
#         file_type = file.split(".")[-1]
#         file_name = os.path.basename(file)
#
#         if file_type == "gt":
#             target_file = os.path.join(tar_root, f"{split_name}_gt", file_name)
#         else:
#             target_file = os.path.join(target_dir, file_type, file_name)
#
#         if not os.path.exists(target_file):
#             os.system(f"mv {file} {target_file}")
#
# # 执行文件移动
# for split_name, file_list in splits.items():
#     print(f"Processing {split_name} with {sum(len(files) for files in file_list.values())} files...")
#     move_files(file_list, split_name)   #split["train"], train
#
# print("Dataset split completed!")
