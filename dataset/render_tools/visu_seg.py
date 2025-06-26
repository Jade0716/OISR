import numpy as np
import matplotlib.pyplot as plt

def visualize_npz_file(npz_file_path):
    # 加载 .npz 文件
    npz_file = np.load(npz_file_path)

    # 查看文件中包含的所有键
    print("Available keys in the npz file:", npz_file.files)

    # 获取实例分割和语义分割数据
    instance_segmentation = npz_file['instance_segmentation']
    semantic_segmentation = npz_file['semantic_segmentation']

    # 可视化实例分割结果
    plt.figure(figsize=(10, 5))

    # 实例分割可视化
    plt.subplot(1, 2, 1)
    plt.imshow(instance_segmentation, cmap='jet')  # 使用 jet 色图来区分不同的实例
    plt.title('Instance Segmentation')
    # plt.colorbar()  # 显示色条

    # 语义分割可视化
    plt.subplot(1, 2, 2)
    plt.imshow(semantic_segmentation, cmap='tab20')  # 使用 tab20 色图来区分不同的语义类别
    plt.title('Semantic Segmentation')
    # 显示图像
    plt.show()

# 替换为你的 .npz 文件路径
npz_file_path = 'example_rendered/segmentation/Laptop_10040_0_0.npz'

# 可视化数据
visualize_npz_file(npz_file_path)
