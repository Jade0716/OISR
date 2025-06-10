from typing import List
import spconv.pytorch as spconv
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

def feature_transform_reguliarzer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss

class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class PointNetEncoder(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, channel=3):
        super(PointNetEncoder, self).__init__()
        self.stn = STN3d(channel)
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        B, D, N = x.size()
        trans = self.stn(x)
        x = x.transpose(2, 1)
        if D > 3:
            feature = x[:, :, 3:]
            x = x[:, :, :3]
        x = torch.bmm(x, trans)
        if D > 3:
            x = torch.cat([x, feature], dim=2)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


class PointNetSegBackbone(nn.Module):
    def __init__(self, pc_dim, fea_dim):
        super(PointNetSegBackbone, self).__init__()
        self.fea_dim = fea_dim
        self.feat = PointNetEncoder(global_feat=False, feature_transform=True, channel=3+pc_dim)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 256, 1)
        self.conv4 = torch.nn.Conv1d(256, self.fea_dim, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        fea = x.transpose(2,1).contiguous()
        return fea
        # x = F.log_softmax(x.view(-1,self.k), dim=-1)
        # x = x.view(batchsize, n_pts, self.k)
        # return x, trans_feat

class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat, weight):
        loss = F.nll_loss(pred, target, weight = weight)
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)
        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss

class ResBlock(spconv.SparseModule):
    def __init__(
        self, in_channels: int, out_channels: int, norm_fn: nn.Module, indice_key=None
    ):
        super().__init__()

        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            # assert False
            self.shortcut = spconv.SparseSequential(
                spconv.SubMConv3d(in_channels, out_channels, kernel_size=1, \
                bias=False),
                norm_fn(out_channels),
            )

        self.conv1 = spconv.SparseSequential(
            spconv.SubMConv3d(
                in_channels, out_channels, kernel_size=3,
                padding=1, bias=False, indice_key=indice_key,
            ),
            norm_fn(out_channels),
        )

        self.conv2 = spconv.SparseSequential(
            spconv.SubMConv3d(
                out_channels, out_channels, kernel_size=3,
                padding=1, bias=False, indice_key=indice_key,
            ),
            norm_fn(out_channels),
        )

    def forward(self, x: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        shortcut = self.shortcut(x)

        x = self.conv1(x)
        x = x.replace_feature(F.relu(x.features))

        x = self.conv2(x)
        x = x.replace_feature(F.relu(x.features + shortcut.features))

        return x

class UBlock(nn.Module):
    def __init__(
        self,
        channels: List[int],
        block_fn: nn.Module,
        block_repeat: int,
        norm_fn: nn.Module,
        indice_key_id: int = 1,
    ):
        super().__init__()

        self.channels = channels

        encoder_blocks = [
            block_fn(
                channels[0], channels[0], norm_fn, indice_key=f"subm{indice_key_id}"
            )
            for _ in range(block_repeat)
        ]
        self.encoder_blocks = spconv.SparseSequential(*encoder_blocks)

        if len(channels) > 1:
            self.downsample = spconv.SparseSequential(
                spconv.SparseConv3d(
                    channels[0], channels[1], kernel_size=2, stride=2,
                    bias=False, indice_key=f"spconv{indice_key_id}",
                ),
                norm_fn(channels[1]),
                nn.ReLU(),
            )

            self.ublock = UBlock(
                channels[1:], block_fn, block_repeat, norm_fn, indice_key_id + 1
            )

            self.upsample = spconv.SparseSequential(
                spconv.SparseInverseConv3d(
                    channels[1], channels[0], kernel_size=2,
                    bias=False, indice_key=f"spconv{indice_key_id}",
                ),
                norm_fn(channels[0]),
                nn.ReLU(),
            )

            decoder_blocks = [
                block_fn(
                    channels[0] * 2, channels[0], norm_fn,
                    indice_key=f"subm{indice_key_id}",
                ),
            ]
            for _ in range(block_repeat -1):
                decoder_blocks.append(
                    block_fn(
                        channels[0], channels[0], norm_fn,
                        indice_key=f"subm{indice_key_id}",
                    )
                )
            self.decoder_blocks = spconv.SparseSequential(*decoder_blocks)

    def forward(self, x: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        x = self.encoder_blocks(x)
        shortcut = x

        if len(self.channels) > 1:
            x = self.downsample(x)
            x = self.ublock(x)
            x = self.upsample(x)

            x = x.replace_feature(torch.cat([x.features, shortcut.features],\
                 dim=-1))
            x = self.decoder_blocks(x)

        return x

class SparseUNet(nn.Module):
    def __init__(self, stem: nn.Module, ublock: UBlock):
        super().__init__()

        self.stem = stem
        self.ublock = ublock

    def forward(self, x):
        if self.stem is not None:
            x = self.stem(x)
        x = self.ublock(x)
        return x

    @classmethod
    def build(
        cls,
        in_channels: int,
        channels: List[int],
        block_repeat: int,
        norm_fn: nn.Module,
        without_stem: bool = False,
    ):
        if not without_stem:
            stem = spconv.SparseSequential(
                spconv.SubMConv3d(
                    in_channels, channels[0], kernel_size=3,
                    padding=1, bias=False, indice_key="subm1",
                ),
                norm_fn(channels[0]),  #16
                nn.ReLU(),
            )
        else:
            stem = spconv.SparseSequential(
                norm_fn(channels[0]),
                nn.ReLU(),
            )

        block = UBlock(channels, ResBlock, block_repeat, norm_fn, \
            indice_key_id=1)

        return SparseUNet(stem, block)


class GroupedSparseUNet(nn.Module):
    def __init__(self, branch1, branch2, branch3, fusion, ublock: UBlock):
        super().__init__()
        self.branch1 = branch1  # 原始点云分支
        self.branch2 = branch2  # 变换后的点云分支
        self.branch3 = branch3  # RGB分支
        self.fusion = fusion  # 特征融合模块
        self.ublock = ublock

    def forward(self, x):
        # 提取完整特征并分割
        all_features = x.features  # [N, 9]
        x1_features = all_features[:, 0:3]  # 原始点云 (通道0-2)
        x2_features = all_features[:, 3:6]  # 变换后的点云 (通道3-5)
        x3_features = all_features[:, 6:9]  # RGB (通道6-8)

        # 创建新的稀疏张量 (复用原始坐标和空间结构)
        x1 = spconv.SparseConvTensor(
            features=x1_features,
            indices=x.indices,
            spatial_shape=x.spatial_shape,
            batch_size=x.batch_size
        )
        x2 = spconv.SparseConvTensor(
            features=x2_features,
            indices=x.indices,
            spatial_shape=x.spatial_shape,
            batch_size=x.batch_size
        )
        x3 = spconv.SparseConvTensor(
            features=x3_features,
            indices=x.indices,
            spatial_shape=x.spatial_shape,
            batch_size=x.batch_size
        )

        # 独立分支处理
        x1 = self.branch1(x1)
        x2 = self.branch2(x2)
        x3 = self.branch3(x3)

        # # 计算差异特征
        # diff_features = x2.features - x1.features

        # 特征融合
        fused_features = torch.cat([
            x1.features,
            x2.features,
            x3.features
        ], dim=1)  # [N, 8+8+8=24]

        # 创建融合后的稀疏张量
        x_fused = spconv.SparseConvTensor(
            features=fused_features,
            indices=x1.indices,  # 使用处理后的坐标 (可能已被分支修改)
            spatial_shape=x1.spatial_shape,
            batch_size=x1.batch_size
        )

        # 通过融合模块
        x_fused = self.fusion(x_fused)

        # 输入到UBlock
        return self.ublock(x_fused)

    @classmethod
    def build(
            cls,
            in_channels: int,
            channels: List[int],
            block_repeat: int,
            norm_fn: nn.Module,
            without_stem: bool = False,
    ):
        # 分支1：原始点云
        branch1 = spconv.SparseSequential(
            spconv.SubMConv3d(3, 8, kernel_size=3, padding=1),
            norm_fn(8),
            nn.ReLU(),
        )

        # 分支2：变换后的点云
        branch2 = spconv.SparseSequential(
            spconv.SubMConv3d(3, 8, kernel_size=3, padding=1),
            norm_fn(8),
            nn.ReLU(),
        )

        # 分支3：RGB
        branch3 = spconv.SparseSequential(
            spconv.SubMConv3d(3, 8, kernel_size=3, padding=1),
            norm_fn(8),
            nn.ReLU(),
        )

        # 融合模块（输入通道=8+8+8=24，输出通道=24）
        fusion = spconv.SparseSequential(
            spconv.SubMConv3d(24, 24, kernel_size=3, padding=1),
            norm_fn(24),
            nn.ReLU(),
        )


        block = UBlock(channels, ResBlock, block_repeat, norm_fn, \
            indice_key_id=1)

        return GroupedSparseUNet(branch1, branch2, branch3, fusion, block)
class UBlock_NoSkip(nn.Module):
    def __init__(
        self,
        channels: List[int],
        block_fn: nn.Module,
        block_repeat: int,
        norm_fn: nn.Module,
        indice_key_id: int = 1,
    ):
        super().__init__()

        self.channels = channels

        encoder_blocks = [
            block_fn(
                channels[0], channels[0], norm_fn, indice_key=f"subm{indice_key_id}"
            )
            for _ in range(block_repeat)
        ]
        self.encoder_blocks = spconv.SparseSequential(*encoder_blocks)

        if len(channels) > 1:
            self.downsample = spconv.SparseSequential(
                spconv.SparseConv3d(
                    channels[0], channels[1], kernel_size=2, stride=2,
                    bias=False, indice_key=f"spconv{indice_key_id}",
                ),
                norm_fn(channels[1]),
                nn.ReLU(),
            )

            self.ublock = UBlock(
                channels[1:], block_fn, block_repeat, norm_fn, indice_key_id + 1
            )

            self.upsample = spconv.SparseSequential(
                spconv.SparseInverseConv3d(
                    channels[1], channels[0], kernel_size=2,
                    bias=False, indice_key=f"spconv{indice_key_id}",
                ),
                norm_fn(channels[0]),
                nn.ReLU(),
            )

            decoder_blocks = [
                block_fn(
                    channels[0], channels[0], norm_fn,
                    indice_key=f"subm{indice_key_id}",
                ),
            ]
            for _ in range(block_repeat -1):
                decoder_blocks.append(
                    block_fn(
                        channels[0], channels[0], norm_fn,
                        indice_key=f"subm{indice_key_id}",
                    )
                )
            self.decoder_blocks = spconv.SparseSequential(*decoder_blocks)

    def forward(self, x: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        x = self.encoder_blocks(x)
        # shortcut = x

        if len(self.channels) > 1:
            x = self.downsample(x)
            x = self.ublock(x)
            x = self.upsample(x)

            # x = x.replace_feature(torch.cat([x.features, shortcut.features],\
            #      dim=-1))
            x = self.decoder_blocks(x)

        return x

class SparseUNet_NoSkip(nn.Module):
    def __init__(self, stem: nn.Module, ublock: UBlock_NoSkip):
        super().__init__()

        self.stem = stem
        self.ublock = ublock

    def forward(self, x):
        if self.stem is not None:
            x = self.stem(x)
        x = self.ublock(x)
        return x

    @classmethod
    def build(
        cls,
        in_channels: int,
        channels: List[int],
        block_repeat: int,
        norm_fn: nn.Module,
        without_stem: bool = False,
    ):
        if not without_stem:
            stem = spconv.SparseSequential(
                spconv.SubMConv3d(
                    in_channels, channels[0], kernel_size=3,
                    padding=1, bias=False, indice_key="subm1",
                ),
                norm_fn(channels[0]),
                nn.ReLU(),
            )
        else:
            stem = spconv.SparseSequential(
                norm_fn(channels[0]),
                nn.ReLU(),
            )

        block = UBlock(channels, ResBlock, block_repeat, norm_fn, \
            indice_key_id=1)

        return SparseUNet(stem, block)


class PointNetBackbone(nn.Module):
    def __init__(
        self,
        pc_dim: int,
        feature_dim: int,
    ):
        super().__init__()
        self.pc_dim = pc_dim
        self.feature_dim = feature_dim
        self.backbone = PointNetSegBackbone(self.pc_dim,self.feature_dim)
    
    def forward(self, input_pc):
        others = {}
        return self.backbone(input_pc), others

