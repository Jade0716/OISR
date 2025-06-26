from datasets import Dataset
from PIL import Image
import io
import matplotlib.pyplot as plt

# 加载 .arrow 文件
dataset_path = "/16T/liuyuyan/xarm6/xarm6_pick_bottle_in_box/xarm6_pick_bottle_in_box-train-00053-of-00060.arrow"
ds = Dataset.from_file(dataset_path)

# 查看总共有多少条数据
print(f"总样本数: {len(ds)}")

# 选一条样本（比如第 0 条）
sample = ds[0]

# 提取图像字段（注意是二进制形式）
image_1_bytes = sample["image_1"]
image_2_bytes = sample["image_2"]

# 使用 PIL 加载为图片
image_1 = Image.open(io.BytesIO(image_1_bytes))
image_2 = Image.open(io.BytesIO(image_2_bytes))

# 可视化
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(image_1)
plt.title("image_1")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(image_2)
plt.title("image_2")
plt.axis("off")

plt.tight_layout()
plt.show()
