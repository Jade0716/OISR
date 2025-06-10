import os
import torch
import glob

# 每组路径，按照 pth/meta/gt 顺序排列
directory_groups = [
    {
        'pth': '/16T/liuyuyan/GAPartNetAllWithFlows/train/pth/',
        'meta': '/16T/liuyuyan/GAPartNetAllWithFlows/train/meta/',
        'gt': '/16T/liuyuyan/GAPartNetAllWithFlows/train_gt/',
    },
    {
        'pth': '/16T/liuyuyan/GAPartNetAllWithFlows/val/pth/',
        'meta': '/16T/liuyuyan/GAPartNetAllWithFlows/val/meta/',
        'gt': '/16T/liuyuyan/GAPartNetAllWithFlows/val_gt/',
    },
    {
        'pth': '/16T/liuyuyan/GAPartNetAllWithFlows/test_intra/pth/',
        'meta': '/16T/liuyuyan/GAPartNetAllWithFlows/test_intra/meta/',
        'gt': '/16T/liuyuyan/GAPartNetAllWithFlows/test_intra_gt/',
    },
]

for group in directory_groups:
    pth_dir = group['pth']
    meta_dir = group['meta']
    gt_dir = group['gt']

    for file in os.listdir(pth_dir):
        if not file.endswith('.pth'):
            continue

        file_path = os.path.join(pth_dir, file)
        basename = os.path.splitext(file)[0]

        try:
            pc_data = torch.load(file_path)
            if pc_data[2].shape != (20000, 3):
                print(f"[DELETE] {file} - shape: {pc_data[2].shape}")

                # 删除 .pth 文件
                os.remove(file_path)

                # 删除对应 meta 文件（所有以basename开头的文件）
                for meta_file in glob.glob(os.path.join(meta_dir, f"{basename}.*")):
                    os.remove(meta_file)
                    print(f"  └─ Deleted meta: {meta_file}")

                # 删除对应 gt 文件（所有以basename开头的文件）
                for gt_file in glob.glob(os.path.join(gt_dir, f"{basename}.*")):
                    os.remove(gt_file)
                    print(f"  └─ Deleted gt: {gt_file}")

        except Exception as e:
            print(f"[ERROR] Failed to process {file_path}: {e}")
