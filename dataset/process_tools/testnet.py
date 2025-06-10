import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleConvNet(nn.Module):
    def __init__(self):
        super(SimpleConvNet, self).__init__()

        # 使用ModuleList存储每一层卷积
        self.mlp_convs = nn.ModuleList()

        # 添加卷积层
        self.mlp_convs.append(nn.Conv2d(3, 8, 1))  # 从3个通道到8个通道，卷积核大小1
        self.mlp_convs.append(nn.Conv2d(8, 8, 1))  # 从8个通道到8个通道，卷积核大小1
        self.mlp_convs.append(nn.Conv2d(8, 8, 1))  # 从8个通道到8个通道，卷积核大小1

    def forward(self, x):
        # 依次应用每一层卷积和ReLU
        for conv in self.mlp_convs:
            x = F.relu(conv(x))
        return x


# 随机生成一个形状为 (2, 3, 32, 20000) 的tensor
input_tensor = torch.randn(2, 3, 32, 20000).to('cuda:1')

# 初始化网络
model = SimpleConvNet().to('cuda:1')
print(next(model.parameters()).device)

# 将输入数据传递给网络
output_tensor = model(input_tensor)

# 打印输出形状
print("Output shape:", output_tensor.shape)
