import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import sys

sys.path.append("..")


# 可视化函数（与你的脚本相同，保留）
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


# 加载 color 图片
color_image = cv2.imread('color_1.png')
color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
H, W, _ = color_image.shape

# 计算图片中心点
input_point = np.array([[W // 2, H // 2]])  # 中心点
input_label = np.array([1])  # 前景点

# 加载 SAM 模型
from segment_anything import sam_model_registry, SamPredictor

sam_checkpoint = "/home/liuyuyan/MyAnygrasp/example_data/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)
predictor.set_image(color_image)

# 生成 masks
masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)

# # 可视化（可选）
# for i, (mask, score) in enumerate(zip(masks, scores)):
#     plt.figure(figsize=(10, 10))
#     plt.imshow(color_image)
#     show_mask(mask, plt.gca())
#     show_points(input_point, input_label, plt.gca())
#     plt.title(f"Mask {i + 1}, Score: {score:.3f}", fontsize=18)
#     plt.axis('off')
#     plt.show()

# 取第三个 mask（索引为 2）
selected_mask = masks[2].astype(np.uint8) * 255  # 转成 0/255 图像

color_mask_array = np.where(selected_mask[..., None] > 0, color_image, 0)

# 保存 color_mask.png
color_mask_image = Image.fromarray(color_mask_array)
color_mask_image.save("color_1_mask.png")
print("已保存 color_mask.png")

# 加载 depth.png 并应用 mask
depth_image = Image.open("depth_1.png").convert("L")
depth_array = np.array(depth_image)

# 保持尺寸一致检查
if depth_array.shape != selected_mask.shape:
    raise ValueError("depth.png 和 color.png 分辨率不一致！")

# 应用 mask
depth_mask_array = np.where(selected_mask > 0, depth_array, 0)

# 保存 depth_mask.png
depth_mask_image = Image.fromarray(depth_mask_array)
depth_mask_image.save("depth_1_mask.png")
print("已保存 depth_mask.png")
