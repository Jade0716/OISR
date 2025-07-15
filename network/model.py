from __future__ import print_function

import functools
from typing import Dict
from typing import Optional, Tuple, List

import numpy as np
import open3d as o3d
import spconv.pytorch as spconv
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from epic_ops.iou import batch_instance_seg_iou
from epic_ops.reduce import segmented_maxpool

from misc.info import get_symmetry_matrix
from network.grouping_utils import (cluster_proposals, compute_npcs_loss, get_gt_scores, segmented_voxelize)
from network.losses import focal_loss, dice_loss, pixel_accuracy
from structure.instances import Instances
from structure.point_cloud import PointCloudBatch
from structure.segmentation import Segmentation
from .backbone import SparseUNet


class GAPartNet(nn.Module):
    def __init__(
            self,
            in_channels: int = 9,
            num_part_classes: int = 10,
            backbone_type: str = "SparseUNet",
            # backbone_cfg: Dict = {"channels": [24, 48, 64, 80, 96, 112, 128],"block_repeat": 2},
            backbone_cfg: Dict = {"channels": [16, 32, 48, 64, 80, 96, 112], "block_repeat": 2},
            # semantic segmentation
            ignore_sem_label: int = -100,
            use_sem_focal_loss: bool = True,
            use_sem_dice_loss: bool = True,
            # instance segmentation
            instance_seg_cfg: Dict = {      "ball_query_radius": 0.01,
      "max_num_points_per_query": 50,
      "min_num_points_per_proposal": 5 ,
      "max_num_points_per_query_shift": 300,
      "score_fullscale": 28,
      "score_scale": 50},
            # npcs segmentation
            symmetry_indices: List = [0, 1, 3, 3, 2, 0, 3, 2, 4, 1],
            # training
            training_schedule: List = [5 ,10],
            # validation
            val_score_threshold: float = 0.09,
            # val_min_num_points_per_proposal: int = 3,
            val_nms_iou_threshold: float = 0.3,
            val_ap_iou_threshold: float = 0.5,
            angle_thresh_deg: float = 180.0,
            flow_diff_thresh: float = 1,
            # testing
            visualize_cfg: Dict = {"visualize_offsets": False},
            use_adaptive_clustering: bool = False,
    ):
        super(GAPartNet, self).__init__()
        self.validation_step_outputs = []
        self.visualize_cfg = visualize_cfg
        self.use_adaptive_clustering = use_adaptive_clustering
        self.in_channels = in_channels
        self.num_part_classes = num_part_classes
        self.backbone_type = backbone_type
        self.backbone_cfg = backbone_cfg
        self.ignore_sem_label = ignore_sem_label
        self.use_sem_focal_loss = use_sem_focal_loss
        self.use_sem_dice_loss = use_sem_dice_loss
        self.val_nms_iou_threshold = val_nms_iou_threshold
        #self.val_ap_iou_threshold = val_ap_iou_threshold
        self.val_score_threshold = val_score_threshold
        # self.val_min_num_points_per_proposal = val_min_num_points_per_proposal
        self.symmetry_indices = torch.as_tensor(symmetry_indices, dtype=torch.int64)
        self.start_scorenet, self.start_npcs = training_schedule
        self.ball_query_radius = instance_seg_cfg["ball_query_radius"]
        self.max_num_points_per_query = instance_seg_cfg["max_num_points_per_query"]
        self.min_num_points_per_proposal = instance_seg_cfg["min_num_points_per_proposal"]
        self.max_num_points_per_query_shift = instance_seg_cfg["max_num_points_per_query_shift"]
        self.angle_thresh_deg = angle_thresh_deg
        self.flow_diff_thresh = flow_diff_thresh
        self.score_fullscale = instance_seg_cfg["score_fullscale"]
        self.score_scale = instance_seg_cfg["score_scale"]

        ## network
        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)
        # backbone
        channels = self.backbone_cfg["channels"]
        block_repeat = self.backbone_cfg["block_repeat"]
        fea_dim = channels[0]
        # 静态分支：xyz+rgb
        self.backbone0 = SparseUNet.build(6, channels, block_repeat, norm_fn)
        # 动态分支：flow - 增强感受野
        # 使用更深的网络配置来扩大感受野
        flow_channels = channels
        flow_block_repeat = block_repeat  # 增加重复次数
        # 增加更多下采样层来扩大感受野
        extended_flow_channels = flow_channels 
        self.backbone1 = SparseUNet.build(3, extended_flow_channels, flow_block_repeat, norm_fn)

        # semantic segmentation head
        # self.sem_seg_head = nn.Linear(fea_dim, self.num_part_classes)
        self.sem_seg_head = nn.Sequential(
            nn.Linear(fea_dim, fea_dim // 2),
            norm_fn(fea_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(fea_dim // 2, self.num_part_classes),
        )

        # 特征融合 MLP
        self.feature_fusion = nn.Sequential(
            nn.Linear(fea_dim * 2, fea_dim),
            norm_fn(fea_dim),
            nn.ReLU(),
        )

        # offset prediction
        self.offset_head = nn.Sequential(
            nn.Linear(fea_dim, fea_dim // 2),
            norm_fn(fea_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(fea_dim // 2 , 3),
        )
        self.offset_flow_head = nn.Sequential(
            nn.Linear(fea_dim, fea_dim // 2),
            norm_fn(fea_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(fea_dim // 2 , 3),
        )

        self.score_unet = SparseUNet.build(
            fea_dim, channels[:2], block_repeat, norm_fn, without_stem=True
        )
        self.score_head = nn.Linear(fea_dim, self.num_part_classes - 1)

        self.npcs_unet = SparseUNet.build(
            fea_dim, channels[:2], block_repeat, norm_fn, without_stem=True
        )
        self.npcs_head = nn.Linear(fea_dim, 3 * (self.num_part_classes - 1))

        (
            symmetry_matrix_1, symmetry_matrix_2, symmetry_matrix_3
        ) = get_symmetry_matrix()
        self.symmetry_matrix_1 = symmetry_matrix_1
        self.symmetry_matrix_2 = symmetry_matrix_2
        self.symmetry_matrix_3 = symmetry_matrix_3
        self.device = torch.device('cuda')
        #----------------------------------



    def forward(self, data_batch, current_epoch):   #PointCloudBatch
        points = data_batch.points
        pc_ids = data_batch.pc_ids
        batch_indices = data_batch.batch_indices
        sem_labels = data_batch.sem_labels
        instance_regions = data_batch.instance_regions
        instance_regions_flow = data_batch.instance_regions_flow
        instance_labels = data_batch.instance_labels
        instance_sem_labels = data_batch.instance_sem_labels
        num_points_per_instance = data_batch.num_points_per_instance
        gt_npcs = data_batch.gt_npcs
        pt_xyz = points[:, :3]-points[:,3:6]
        flow = points[:,3:6]
        flow_xyz = points[:, :3]
        

        # Forward pass through backbone
        pc_feature = self.forward_backbone(pc_batch=data_batch)   #用u-net得到点云体素化特征

        # Semantic segmentation
        sem_logits = self.forward_sem_seg(pc_feature)        #全连接层分割预测
        sem_preds = torch.argmax(sem_logits.detach(), dim=-1)  #每个点的得分
        if sem_labels is not None:
            loss_sem_seg = self.loss_sem_seg(sem_logits, sem_labels)
            all_accu = (sem_preds == sem_labels).sum().float() / (sem_labels.shape[0])

        else:
            loss_sem_seg = 0.0
            all_accu = 0.0

        if sem_labels is not None:
            instance_mask = sem_labels > 0
            pixel_accu = pixel_accuracy(sem_preds[instance_mask], sem_labels[instance_mask])
        else:
            pixel_accu = 0.0

        sem_seg = Segmentation(
            batch_size=len(points),
            sem_preds=sem_preds,
            sem_labels=sem_labels,
            all_accu=all_accu,
            pixel_accu=pixel_accu,
        )

        # Offset prediction
        offsets_preds = self.forward_offset(pc_feature)          #偏移预测
        offsets_preds_flow = self.forward_offset_flow(pc_feature)
        

        
        if instance_regions is not None:
            offsets_gt = instance_regions[:, :3] - pt_xyz
            loss_offset_dist, loss_offset_dir = self.loss_offset(
                offsets_preds, offsets_gt, sem_labels, instance_labels,   
            )
            offsets_flow_gt = instance_regions_flow[:, :3] - flow_xyz
            loss_offset_flow_dist, loss_offset_flow_dir = self.loss_offset(
                offsets_preds_flow, offsets_flow_gt, sem_labels, instance_labels,
            )
        else:
            loss_offset_dist, loss_offset_dir = 0., 0.
            loss_offset_flow_dist, loss_offset_flow_dir = 0., 0.

        # 可视化偏移预测
        if self.visualize_cfg.get('visualize_offsets', False):
            # 转换为numpy进行可视化，但不改变原始tensor
            pt_xyz_np = pt_xyz.detach().cpu().numpy()
            flow_xyz_np = flow_xyz.detach().cpu().numpy()
            offsets_preds_np = offsets_preds.detach().cpu().numpy()
            offsets_preds_flow_np = offsets_preds_flow.detach().cpu().numpy()
            instance_labels_np = instance_labels.detach().cpu().numpy() if instance_labels is not None else np.full(pt_xyz_np.shape[0], -100)
            print("可视化静态偏移预测...")
            self.visualize_offsets_with_lines(pt_xyz_np, offsets_preds_np, instance_labels_np, title="静态偏移预测")
            
            print("可视化动态偏移预测...")
            self.visualize_offsets_with_lines(flow_xyz_np, offsets_flow_gt.detach().cpu().numpy(), instance_labels_np, title="动态偏移预测")
        #运动一致性损失 - 帮助区分slider和hinge
        # if  instance_labels is not None and current_epoch >= 5:
        #     loss_motion_consistency = 10*self.motion_consistency_loss(flow, sem_logits, instance_labels, sem_labels)
        #     # print(f"loss_motion_consistency: {loss_motion_consistency}")
        # else:
        #     loss_motion_consistency = torch.tensor(0.0, device=flow.device)

        # if self.current_epoch >= self.start_clustering:
        # Proposal clustering and revoxelization
        voxel_tensor, pc_voxel_id, proposals = self.proposal_clustering_and_revoxelize(   #利用预测语义和偏移量，再体素化得到proposals
            pt_xyz=pt_xyz,
            flow_xyz=flow_xyz,
            batch_indices=batch_indices,
            pt_features=pc_feature,
            sem_preds=sem_preds,
            offset_preds=offsets_preds,
            offset_preds_flow = offsets_preds_flow,
            instance_labels=instance_labels,
        )

        if sem_labels is not None and proposals is not None:
            proposals.sem_labels = sem_labels[proposals.valid_mask][
                proposals.sorted_indices
            ]
        if proposals is not None:
            proposals.instance_sem_labels = instance_sem_labels

        # Clustering and scoring
        # if self.current_epoch >= self.start_scorenet
        if voxel_tensor is not None and proposals is not None and current_epoch >= self.start_scorenet :#and voxel_tensor.batch_size > 1:
            score_logits = self.forward_proposal_score(
                voxel_tensor, pc_voxel_id, proposals
            )
            proposal_offsets_begin = proposals.proposal_offsets[:-1].long()
            if proposals.sem_labels is not None:
                proposal_sem_labels = proposals.sem_labels[proposal_offsets_begin].long()
            else:
                proposal_sem_labels = proposals.sem_preds[proposal_offsets_begin].long()

            score_logits = score_logits.gather(    #得到对sem_preds预测的实例类别的得分
                1, proposal_sem_labels[:, None] - 1
            ).squeeze(1)
            proposals.score_preds = score_logits.detach().sigmoid()
            if num_points_per_instance is not None:
                #loss_slider_consistency = self.slider_flow_consistency_loss(flow,proposals,score_logits,
                 #                                                           proposal_sem_labels, [4, 7, 8, 9])
                loss_slider_consistency = torch.tensor(0.0, device=flow.device)
                loss_prop_score = self.loss_proposal_score(
                    score_logits, proposals, num_points_per_instance,
                )
            else:
                # import pdb
                # pdb.set_trace()
                loss_prop_score = 0.0
                loss_slider_consistency = 0.0
        else:
            loss_prop_score = 0.0
            loss_slider_consistency = 0.0



        # if self.current_epoch >= self.start_npcs
        # NPCS prediction
        if voxel_tensor is not None and current_epoch >= self.start_npcs:# and voxel_tensor.batch_size > 1:
            npcs_logits = self.forward_proposal_npcs(
                voxel_tensor, pc_voxel_id
            )
            if gt_npcs is not None:
                gt_npcs = gt_npcs[proposals.valid_mask][proposals.sorted_indices]
                loss_prop_npcs = self.loss_proposal_npcs(npcs_logits, gt_npcs, proposals)
            else:
                proposals.npcs_valid_mask = torch.ones(proposals.sorted_indices.shape[0], dtype=torch.bool, device=proposals.sorted_indices.device)
                npcs_logits = npcs_logits[proposals.npcs_valid_mask]
                sem_preds = proposals.sem_preds[proposals.npcs_valid_mask].long()
                proposal_indices = proposals.proposal_indices[proposals.npcs_valid_mask]

                npcs_logits = rearrange(npcs_logits, "n (k c) -> n k c", c=3)
                npcs_logits = npcs_logits.gather(
                    1, index=repeat(sem_preds - 1, "n -> n one c", one=1, c=3)
                ).squeeze(1)

                proposals.npcs_preds = npcs_logits.detach()
                loss_prop_npcs = 0.0
        else:
            loss_prop_npcs = 0.0
            npcs_preds = None
        
        # self.visualize_offsets(flow_xyz.cpu().numpy(), offsets_preds.cpu().numpy(), instance_labels.cpu().numpy())
        # self.visualize_offsets(pt_xyz.cpu().numpy(), offsets_preds.cpu().numpy(), instance_labels.cpu().numpy())
        dict = {
            #'loss_slider_consistency': loss_slider_consistency,
            'loss_sem_seg': loss_sem_seg,
            'loss_offset_dist': loss_offset_dist,
            'loss_offset_dir': loss_offset_dir,
            'loss_offset_flow_dist': loss_offset_flow_dist,
            'loss_offset_flow_dir': loss_offset_flow_dir,
            'loss_prop_score': loss_prop_score,
            'loss_prop_npcs': loss_prop_npcs,
            #'loss_motion_consistency': loss_motion_consistency,
        }


        return pc_ids, sem_seg, proposals, dict

    def visualize_offsets(self, pt_xyz: np.ndarray, offsets: np.ndarray, instance_labels: np.ndarray):

        moved_points = np.zeros_like(pt_xyz)
        colors = np.zeros((pt_xyz.shape[0], 3))

        for i in range(pt_xyz.shape[0]):
            if instance_labels[i] == -100:
                moved_points[i] = pt_xyz[i]
                colors[i] = [1.0, 0.0, 0.0]  # 红色
            else:
                moved_points[i] = pt_xyz[i] + offsets[i]
                colors[i] = [0.0, 0.0, 1.0]  # 蓝色

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(moved_points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        o3d.visualization.draw_geometries([pcd])

    def visualize_offsets_with_lines(self, pt_xyz: np.ndarray, offsets: np.ndarray, instance_labels: np.ndarray, title: str = "偏移可视化"):
        """
        可视化偏移预测，显示原始点云和偏移后的点云，用线段连接对应点
        """
        # 计算偏移后的点云
        moved_points = pt_xyz + offsets
        
        # 创建原始点云（红色）
        original_pcd = o3d.geometry.PointCloud()
        original_pcd.points = o3d.utility.Vector3dVector(pt_xyz)
        original_colors = np.full((pt_xyz.shape[0], 3), [1.0, 0.0, 0.0])  # 红色
        original_pcd.colors = o3d.utility.Vector3dVector(original_colors)
        
        # 创建偏移后的点云（蓝色）
        moved_pcd = o3d.geometry.PointCloud()
        moved_pcd.points = o3d.utility.Vector3dVector(moved_points)
        moved_colors = np.full((moved_points.shape[0], 3), [0.0, 0.0, 1.0])  # 蓝色
        moved_pcd.colors = o3d.utility.Vector3dVector(moved_colors)
        
        # 创建连接线
        lines = []
        line_colors = []
        
        # 只对有实例标签的点创建连接线
        valid_mask = instance_labels >= 0
        valid_indices = np.where(valid_mask)[0]
        
        for i in valid_indices:
            # 创建从原点到偏移点的线段
            lines.append([i, i + len(pt_xyz)])  # 连接原始点和偏移点
            line_colors.append([0.5, 0.5, 0.5])  # 灰色线条
        
        # 创建线段几何体
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(np.vstack([pt_xyz, moved_points]))
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(line_colors)
        
        # 显示几何体
        geometries = [original_pcd, moved_pcd, line_set]
        
        # 设置窗口标题
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=title, width=1200, height=800)
        
        for geom in geometries:
            vis.add_geometry(geom)
        
        
        # 运行可视化
        vis.run()
        vis.destroy_window()

    def forward_backbone(
            self,
            pc_batch: PointCloudBatch,
    ):
        if self.backbone_type == "SparseUNet":
            static_voxel_tensor = pc_batch.static_voxel_tensor
            dynamic_voxel_tensor = pc_batch.dynamic_voxel_tensor
            pc_voxel_id = pc_batch.pc_voxel_id
        #     voxel_features = self.backbone(voxel_tensor)
        #     pc_feature = voxel_features.features[pc_voxel_id]
        # elif self.backbone_type == "PointNet":
        #     pc_feature = self.backbone(pc_batch.points.reshape(-1, 6, 20000))[0]
        #     pc_feature = pc_feature.reshape(-1, pc_feature.shape[-1])
            # 替换特征
            # 通过两个backbone
            static_voxel_features = self.backbone0(static_voxel_tensor)
            dynamic_voxel_features = self.backbone1(dynamic_voxel_tensor)
            # 提取点特征
            static_pc_features = static_voxel_features.features[pc_voxel_id]
            dynamic_pc_features = dynamic_voxel_features.features[pc_voxel_id]
            # 特征拼接融合
            pc_feature = self.feature_fusion(torch.cat([static_pc_features, dynamic_pc_features], dim=-1))
        else:
            raise NotImplementedError("Only SparseUNet dual-branch supported in this version.")
        return pc_feature

    def forward_sem_seg(
            self,
            pc_feature: torch.Tensor,
    ) -> torch.Tensor:
        sem_logits = self.sem_seg_head(pc_feature)

        return sem_logits


    def forward_offset(
            self,
            pc_feature: torch.Tensor,
    ) -> torch.Tensor:
        offset = self.offset_head(pc_feature)

        return offset

    def forward_offset_flow(
            self,
            pc_feature: torch.Tensor,
    ) -> torch.Tensor:
        offset = self.offset_flow_head(pc_feature)

        return offset


    def forward_proposal_score(
            self,
            voxel_tensor: spconv.SparseConvTensor,
            pc_voxel_id: torch.Tensor,
            proposals: Instances,
    ):
        proposal_offsets = proposals.proposal_offsets
        proposal_offsets_begin = proposal_offsets[:-1]   
        proposal_offsets_end = proposal_offsets[1:]   
        score_features = self.score_unet(voxel_tensor)
        score_features = score_features.features[pc_voxel_id]
        pooled_score_features, _ = segmented_maxpool(
            score_features, proposal_offsets_begin, proposal_offsets_end
        )
        score_logits = self.score_head(pooled_score_features)

        return score_logits


    def forward_proposal_npcs(
            self,
            voxel_tensor: spconv.SparseConvTensor,
            pc_voxel_id: torch.Tensor,
    ) -> torch.Tensor:
        npcs_features = self.npcs_unet(voxel_tensor)
        npcs_logits = self.npcs_head(npcs_features.features)
        npcs_logits = npcs_logits[pc_voxel_id]

        return npcs_logits

    def loss_sem_seg(
            self,
            sem_logits: torch.Tensor,
            sem_labels: torch.Tensor,
    ) -> torch.Tensor:
        #创建一个类别权重，给handle类别加权
        class_weights=torch.ones(self.num_part_classes,device=sem_logits.device)
        class_weights[1]=2.0
        class_weights[2]=3.0
        class_weights[9]=3.0
        if self.use_sem_focal_loss:
                       loss = focal_loss(
                           sem_logits, sem_labels,
                           alpha=None,
                           gamma=2.0,
                           ignore_index=self.ignore_sem_label,
                           reduction="mean",
                       )
            # 计算小类别的自适应focal loss
            # small_classes = [1, 2, 8, 9]  # handle + knob classes
            # is_small_class = torch.isin(sem_labels, torch.tensor(small_classes, device=sem_labels.device))

            # if is_small_class.any():
            #     # 小类别使用更高的gamma值(3.0)以更关注难分类样本
            #     small_mask = is_small_class
            #     other_mask = ~small_mask

            #     loss = torch.zeros_like(sem_logits[:, 0])
            #     if small_mask.any():
            #         loss[small_mask] = focal_loss(
            #             sem_logits[small_mask], sem_labels[small_mask],
            #             alpha=None, gamma=3.0, ignore_index=self.ignore_sem_label, reduction="none"
            #         )
            #     if other_mask.any():
            #         loss[other_mask] = focal_loss(
            #             sem_logits[other_mask], sem_labels[other_mask],
            #             alpha=None, gamma=2.0, ignore_index=self.ignore_sem_label, reduction="none"
            #         )
            #     loss = loss.mean()
            # else:
            #     loss = focal_loss(
            #         sem_logits, sem_labels,
            #         alpha=None, gamma=2.0, ignore_index=self.ignore_sem_label, reduction="mean"
            #     )
        else:
            loss = F.cross_entropy(
                sem_logits, sem_labels,
                weight=None,
                ignore_index=self.ignore_sem_label,
                reduction="mean",
            )

        if self.use_sem_dice_loss:
            loss += dice_loss(
                sem_logits[:, :, None, None], sem_labels[:, None, None],
            )
        return loss


    def loss_offset(
            self,
            offsets: torch.Tensor,
            gt_offsets: torch.Tensor,
            sem_labels: torch.Tensor,
            instance_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        valid_instance_mask = (sem_labels > 0) & (instance_labels >= 0)

        pt_diff = offsets - gt_offsets
        pt_dist = torch.sum(pt_diff.abs(), dim=-1)
        # pt_dist = torch.norm(pt_diff, p=2, dim=-1)
        loss_offset_dist = pt_dist[valid_instance_mask].mean()
        # print(gt_offsets.abs()[valid_instance_mask].mean())

        gt_offsets_norm = torch.norm(gt_offsets, p=2, dim=-1)
        gt_offsets = gt_offsets / (gt_offsets_norm[:, None] + 1e-8)

        offsets_norm = torch.norm(offsets, p=2, dim=-1)
        offsets = offsets / (offsets_norm[:, None] + 1e-8)

        dir_diff = -(gt_offsets * offsets).sum(-1)
        loss_offset_dir = dir_diff[valid_instance_mask].mean()

        return loss_offset_dist, loss_offset_dir


    # def loss_proposal_score(
    #         self,
    #         score_logits: torch.Tensor,
    #         proposals: Instances,
    #         num_points_per_instance: torch.Tensor,
    # ) -> torch.Tensor:
    #     ious = batch_instance_seg_iou(
    #         proposals.proposal_offsets,   
    #         proposals.instance_labels,   
    #         proposals.batch_indices,   
    #         num_points_per_instance,
    #     )
    #     proposals.ious = ious
    #     proposals.num_points_per_instance = num_points_per_instance

    #     ious_max, gt_assignment = ious.max(-1)  # 每个 proposal 对应的 GT 实例索引
    #     gt_scores = get_gt_scores(ious_max, 0.75, 0.25)  # 原始基于IoU的soft标签

    #     # 需要的输入
    #     flow = getattr(proposals, 'flow', None)
    #     sem_preds = proposals.sem_preds
    #     instance_labels = proposals.instance_labels
    #     sorted_indices = proposals.sorted_indices
    #     proposal_offsets = proposals.proposal_offsets

    #     if flow is None:
    #         raise ValueError("proposals 需要包含 flow 字段（每个点的 flow）")

    #     proposal_flow_scores = []
    #     gt_flow_scores = []

    #     for k in range(proposal_offsets.shape[0] - 1):
    #         start = proposal_offsets[k].item()
    #         end = proposal_offsets[k + 1].item()
    #         indices = sorted_indices[start:end]  # 属于该 proposal 的点索引

    #         prop_flow = flow[indices]
    #         prop_lengths = prop_flow.norm(dim=1)
    #         length_mean = prop_lengths.mean()
    #         norm_lengths = prop_lengths / (length_mean + 1e-8)
    #         length_std = norm_lengths.std()
    #         cv = length_std  # coefficient of variation

    #         pred_cls = sem_preds[indices[0]].item()
    #         if pred_cls in [4, 7]:  # hinge
    #             prop_score = 1.0 - torch.sigmoid(cv * 5)
    #         elif pred_cls in [5, 6]:  # slider
    #             prop_score = torch.sigmoid(cv * 5)
    #         else:
    #             prop_score = torch.tensor(1.0, device=flow.device)  # 其他类不影响分数
    #         # print(f"prop_cls: {pred_cls} prop_score: {prop_score}")
    #         proposal_flow_scores.append(prop_score)

    #         # === 计算GT flow stats ===
    #         gt_cls = instance_labels[indices[0]].item()
    #         gt_instance_mask = (instance_labels == gt_assignment[k].item())
    #         if gt_instance_mask.sum() < 3:
    #             gt_flow_scores.append(torch.tensor(1.0, device=flow.device))  # 忽略小GT
    #             continue
    #         gt_flow = flow[gt_instance_mask]
    #         gt_lengths = gt_flow.norm(dim=1)
    #         gt_mean = gt_lengths.mean()
    #         gt_norm = gt_lengths / (gt_mean + 1e-8)
    #         gt_std = gt_norm.std()
    #         gt_cv = gt_std
    #         gt_sem_cls = sem_preds[gt_instance_mask][0].item()

    #         if gt_sem_cls in [4, 7]:  # hinge
    #             gt_score = 1.0 - torch.sigmoid(gt_cv * 5)
    #         elif gt_sem_cls in [5, 6]:  # slider
    #             gt_score = torch.sigmoid(gt_cv * 5)
    #         else:
    #             gt_score = torch.tensor(1.0, device=flow.device)
    #         # print(f"gt_sem_cls:{gt_sem_cls} gt_score: {gt_score}")
    #         gt_flow_scores.append(gt_score)

    #     # 组合统计量差值惩罚
    #     proposal_flow_scores = torch.stack(proposal_flow_scores)
    #     gt_flow_scores = torch.stack(gt_flow_scores)
    #     flow_diff = (proposal_flow_scores - gt_flow_scores).abs()
    #     flow_penalty = flow_diff  # 值越大 -> 惩罚越强
    #     final_scores = gt_scores * (1 - flow_penalty)

    #     return F.binary_cross_entropy_with_logits(score_logits, final_scores)
    def loss_proposal_score(
        self,
        score_logits: torch.Tensor,
        proposals: Instances,
        num_points_per_instance: torch.Tensor,
    ) -> torch.Tensor:
        ious = batch_instance_seg_iou(
            proposals.proposal_offsets,
            proposals.instance_labels,
            proposals.batch_indices,
            num_points_per_instance,
        )
        proposals.ious = ious
        proposals.num_points_per_instance = num_points_per_instance

        ious_max, gt_assignment = ious.max(-1)
        gt_scores = get_gt_scores(ious_max, 0.75, 0.25)

        flow = getattr(proposals, 'flow', None)
        sem_preds = proposals.sem_preds
        instance_labels = proposals.instance_labels
        sorted_indices = proposals.sorted_indices
        proposal_offsets = proposals.proposal_offsets

        if flow is None:
            raise ValueError("proposals needs to contain flow field (flow for each point)")

        num_proposals = proposal_offsets.shape[0] - 1
        if num_proposals == 0:
            return torch.tensor(0.0, device=score_logits.device)
        all_flow_lengths = flow.norm(dim=1)
        point_proposal_indices = torch.empty_like(sorted_indices, dtype=torch.long)
        for i in range(num_proposals):
            start = proposal_offsets[i].item()
            end = proposal_offsets[i + 1].item()
            point_proposal_indices[sorted_indices[start:end]] = i
        sum_lengths_per_proposal = torch.zeros(num_proposals, device=flow.device)
        sum_lengths_per_proposal.scatter_add_(0, point_proposal_indices[sorted_indices], all_flow_lengths[sorted_indices])
        points_count_per_proposal = torch.zeros(num_proposals, device=flow.device)
        ones = torch.ones_like(sorted_indices, dtype=flow.dtype)
        points_count_per_proposal.scatter_add_(0, point_proposal_indices[sorted_indices], ones)
        points_count_per_proposal[points_count_per_proposal == 0] = 1 # Set to 1 to avoid NaN in division, will be masked later
        mean_lengths_per_proposal = sum_lengths_per_proposal / points_count_per_proposal
        normalized_lengths_per_point = all_flow_lengths / (mean_lengths_per_proposal[point_proposal_indices] + 1e-8)

        # Sum of squared normalized lengths for variance
        sum_sq_norm_lengths_per_proposal = torch.zeros(num_proposals, device=flow.device)
        sum_sq_norm_lengths_per_proposal.scatter_add_(0, point_proposal_indices[sorted_indices], normalized_lengths_per_point[sorted_indices]**2)

        # Variance and Standard Deviation
        variance_per_proposal = (sum_sq_norm_lengths_per_proposal / points_count_per_proposal) - (sum_lengths_per_proposal / points_count_per_proposal)**2
        std_per_proposal = torch.sqrt(torch.relu(variance_per_proposal)) # Use relu to prevent sqrt of negative due to small numerical errors

        # Coefficient of Variation (CV) for proposals
        cv_per_proposal = std_per_proposal

        # Predict classes for each proposal (assuming homogeneous class within a proposal)
        # We can get the class for the first point of each proposal
        first_point_indices = proposal_offsets[:-1]
        pred_cls_for_proposals = sem_preds[sorted_indices[first_point_indices]]

        # Calculate proposal scores based on predicted class and CV
        proposal_flow_scores = torch.ones(num_proposals, device=flow.device)
        hinge_mask = (pred_cls_for_proposals == 4) | (pred_cls_for_proposals == 7)
        slider_mask = (pred_cls_for_proposals == 5) | (pred_cls_for_proposals == 6)

        proposal_flow_scores[hinge_mask] = 1.0 - torch.sigmoid(cv_per_proposal[hinge_mask] * 5)
        proposal_flow_scores[slider_mask] = torch.sigmoid(cv_per_proposal[slider_mask] * 5)

        # --- Parallelized GT Flow Scores ---
        # Get the GT instance mask for each proposal's assigned GT
        # This creates a boolean mask for all points indicating if they belong to the assigned GT instance
        gt_instance_ids = gt_assignment[torch.arange(num_proposals, device=flow.device)]
        gt_instance_masks_all_points = (instance_labels.unsqueeze(0) == gt_instance_ids.unsqueeze(1)) # shape: (num_proposals, num_total_points)

        # Filter out small GT instances (less than 3 points) from consideration for GT flow scores
        gt_instance_point_counts = gt_instance_masks_all_points.sum(dim=1).float()
        valid_gt_mask = gt_instance_point_counts >= 3

        # Prepare for masked calculations
        # We only want to compute for valid GTs
        if valid_gt_mask.sum() == 0: # If no valid GTs, set all GT flow scores to 1.0
            gt_flow_scores = torch.ones(num_proposals, device=flow.device)
        else:
            # Replicate flow and instance labels for each proposal's assigned GT
            flow_expanded = flow.unsqueeze(0).expand(num_proposals, -1, -1) # (num_proposals, num_total_points, 3)
            gt_flow_for_valid = flow_expanded[valid_gt_mask] # Flows for points belonging to valid assigned GTs

            # Calculate lengths for points within valid GT instances
            gt_flow_lengths_for_valid = gt_flow_for_valid.norm(dim=2) # (num_valid_gt_instances, num_points_in_each_gt)

            # Get the actual point counts for each valid GT
            num_points_in_valid_gt = gt_instance_point_counts[valid_gt_mask].long()

            # Calculate mean lengths for valid GTs.
            # We can use `torch.segment_reduce` or a manual sum/count approach with indexing
            # For simplicity and broad compatibility, let's use a sum and count with masks
            sum_gt_lengths = (gt_flow_for_valid.norm(dim=2) * gt_instance_masks_all_points[valid_gt_mask]).sum(dim=1)
            mean_gt_lengths = sum_gt_lengths / (gt_instance_point_counts[valid_gt_mask] + 1e-8)

            # Normalize lengths for CV calculation for valid GTs
            normalized_gt_lengths = (gt_flow_for_valid.norm(dim=2) * gt_instance_masks_all_points[valid_gt_mask]) / \
                                    (mean_gt_lengths.unsqueeze(1) * gt_instance_masks_all_points[valid_gt_mask] + 1e-8)

            # Sum of squared normalized lengths for variance for valid GTs
            sum_sq_norm_gt_lengths = (normalized_gt_lengths**2 * gt_instance_masks_all_points[valid_gt_mask]).sum(dim=1)

            # Variance and Standard Deviation for valid GTs
            variance_gt = (sum_sq_norm_gt_lengths / (gt_instance_point_counts[valid_gt_mask] + 1e-8)) - mean_gt_lengths**2
            std_gt = torch.sqrt(torch.relu(variance_gt))

            # Coefficient of Variation (CV) for valid GTs
            cv_gt = std_gt

            # Get semantic classes for valid GTs (first point of the GT instance)
            gt_sem_cls_for_valid = sem_preds[instance_labels == gt_assignment[valid_gt_mask].to(instance_labels.device)].unique()
            # This needs to be done carefully. We need the sem_pred for the *assigned* GT instance.
            # A more robust way: use `gt_assignment` to map back to original instance_labels indices.
            gt_sem_cls_per_proposal_assigned_gt = sem_preds[instance_labels == gt_assignment.unsqueeze(1)].unique(dim=1)[0] # Assuming each GT has a consistent sem_pred

            gt_flow_scores = torch.ones(num_proposals, device=flow.device)
            # Apply to valid GTs only
            gt_sem_cls_valid_proposals = gt_sem_cls_per_proposal_assigned_gt[valid_gt_mask]
            
            gt_hinge_mask = (gt_sem_cls_valid_proposals == 4) | (gt_sem_cls_valid_proposals == 7)
            gt_slider_mask = (gt_sem_cls_valid_proposals == 5) | (gt_sem_cls_valid_proposals == 6)

            gt_flow_scores[valid_gt_mask][gt_hinge_mask] = 1.0 - torch.sigmoid(cv_gt[gt_hinge_mask] * 5)
            gt_flow_scores[valid_gt_mask][gt_slider_mask] = torch.sigmoid(cv_gt[gt_slider_mask] * 5)

        # --- Combine Scores ---
        flow_diff = (proposal_flow_scores - gt_flow_scores).abs()
        flow_penalty = flow_diff
        final_scores = gt_scores * (1 - flow_penalty)

        return F.binary_cross_entropy_with_logits(score_logits, final_scores)


    def loss_proposal_score(
        self,
        score_logits: torch.Tensor,
        proposals: 'Instances',
        num_points_per_instance: torch.Tensor,
    ) -> torch.Tensor:
        # === IoU & GT soft score ===
        ious = batch_instance_seg_iou(
            proposals.proposal_offsets,
            proposals.instance_labels,
            proposals.batch_indices,
            num_points_per_instance,
        )
        proposals.ious = ious
        proposals.num_points_per_instance = num_points_per_instance

        ious_max, gt_assignment = ious.max(-1)
        gt_scores = get_gt_scores(ious_max, 0.75, 0.25)

        flow = getattr(proposals, 'flow', None)
        if flow is None:
            raise ValueError("proposals needs to contain flow field")

        sem_preds = proposals.sem_preds
        instance_labels = proposals.instance_labels
        sorted_indices = proposals.sorted_indices
        proposal_offsets = proposals.proposal_offsets

        num_props = proposal_offsets.shape[0] - 1
        if num_props == 0:
            return torch.tensor(0.0, device=score_logits.device)

        # === Proposal flow stats (并行) ===
        all_flow_lengths = flow.norm(dim=1)
        point_to_proposal = torch.zeros_like(sorted_indices, dtype=torch.long)
        point_to_proposal[sorted_indices] = torch.bucketize(
            torch.arange(len(sorted_indices), device=flow.device),
            proposal_offsets[1:]
        )

        sum_lengths = torch.zeros(num_props, device=flow.device).scatter_add_(
            0, point_to_proposal[sorted_indices], all_flow_lengths[sorted_indices]
        )
        count_lengths = torch.zeros(num_props, device=flow.device).scatter_add_(
            0, point_to_proposal[sorted_indices], torch.ones_like(sorted_indices, dtype=flow.dtype)
        )
        count_lengths[count_lengths == 0] = 1
        mean_lengths = sum_lengths / count_lengths

        normalized_lengths = all_flow_lengths / (mean_lengths[point_to_proposal] + 1e-8)
        norm_lengths_sq = normalized_lengths ** 2

        sum_sq = torch.zeros(num_props, device=flow.device).scatter_add_(
            0, point_to_proposal[sorted_indices], norm_lengths_sq[sorted_indices]
        )
        var = (sum_sq / count_lengths) - (mean_lengths ** 2)
        std = torch.sqrt(torch.relu(var))
        cv_proposals = std

        first_point_indices = proposal_offsets[:-1]
        proposal_sem_cls = sem_preds[sorted_indices[first_point_indices]]

        hinge_mask = (proposal_sem_cls == 4) | (proposal_sem_cls == 7)
        slider_mask = (proposal_sem_cls == 5) | (proposal_sem_cls == 6)

        prop_flow_scores = torch.ones(num_props, device=flow.device)
        prop_flow_scores[hinge_mask] = 1.0 - torch.sigmoid(cv_proposals[hinge_mask] * 5)
        prop_flow_scores[slider_mask] = torch.sigmoid(cv_proposals[slider_mask] * 5)

        # === GT flow stats (并行) ===
        point_idx = torch.arange(flow.shape[0], device=flow.device)
        gt_mask_matrix = instance_labels.unsqueeze(0) == gt_assignment.unsqueeze(1)  # (P, N)
        gt_point_counts = gt_mask_matrix.sum(dim=1)
        valid_mask = gt_point_counts >= 3

        # Expand flow
        flow_expanded = flow.unsqueeze(0).expand(num_props, -1, -1)
        flow_lengths = flow_expanded.norm(dim=2)  # (P, N)

        masked_lengths = flow_lengths * gt_mask_matrix
        mean_gt = masked_lengths.sum(dim=1) / (gt_point_counts + 1e-8)
        norm_gt = masked_lengths / (mean_gt.unsqueeze(1) + 1e-8)
        norm_sq_gt = norm_gt ** 2
        var_gt = (norm_sq_gt * gt_mask_matrix).sum(dim=1) / (gt_point_counts + 1e-8) - mean_gt ** 2
        std_gt = torch.sqrt(torch.relu(var_gt))
        cv_gt = std_gt

        # 获取每个 proposal 对应 GT 实例的第一个点
        gt_assignment_exp = gt_assignment.unsqueeze(1)
        match_matrix = (instance_labels.unsqueeze(0) == gt_assignment_exp)  # (P, N)
        first_gt_indices = match_matrix.float().cumsum(dim=1).eq(1).float().argmax(dim=1)
        gt_sem_cls = sem_preds[first_gt_indices]

        gt_flow_scores = torch.ones(num_props, device=flow.device)
        valid_cv = cv_gt[valid_mask]
        valid_cls = gt_sem_cls[valid_mask]

        hinge_mask = (valid_cls == 4) | (valid_cls == 7)
        slider_mask = (valid_cls == 5) | (valid_cls == 6)

        gt_flow_scores_valid = torch.ones_like(valid_cv)
        gt_flow_scores_valid[hinge_mask] = 1.0 - torch.sigmoid(valid_cv[hinge_mask] * 5)
        gt_flow_scores_valid[slider_mask] = torch.sigmoid(valid_cv[slider_mask] * 5)

        gt_flow_scores[valid_mask] = gt_flow_scores_valid

        # === Final score ===
        flow_penalty = (prop_flow_scores - gt_flow_scores).abs()
        final_scores = gt_scores * (1 - flow_penalty)

        return F.binary_cross_entropy_with_logits(score_logits, final_scores)
    #compute loss for flow
    def slider_flow_consistency_loss(self, flow, proposals, score_logits,
                                      proposal_sem_labels, hinge_id) -> torch.Tensor:
        device = flow.device
        losses = []
        scores = []

        offsets = proposals.proposal_offsets
        num_proposals = offsets.shape[0] - 1
        sorted_indices = proposals.sorted_indices  # [M]
        valid_mask = proposals.valid_mask
        valid_indices = valid_mask.nonzero(as_tuple=False).squeeze(1)

        for k in range(num_proposals):
            if proposal_sem_labels[k].item() in hinge_id:
                continue

            score = torch.sigmoid(score_logits[k])
            scores.append(score)

            start = offsets[k].item()
            end = offsets[k + 1].item()

            proposal_sorted_idx = sorted_indices[start:end]
            proposal_original_indices = valid_indices[proposal_sorted_idx]

            if len(proposal_original_indices) == 0:
                loss = torch.tensor(0.0, device=device)
            else:
                proposal_flow = flow[valid_indices[proposal_sorted_idx]]

                # 直接按 proposal 内 flow 一致性计算
                flow_mean = proposal_flow.mean(dim=0, keepdim=True)
                flow_consistency = torch.norm(proposal_flow - flow_mean, dim=1)
                loss = flow_consistency.mean()

            losses.append(loss)

        if len(losses) == 0:
            print("losses's len is 0")
            return torch.tensor(0.0, device=device)

        scores = torch.stack(scores)
        losses = torch.stack(losses)

        # Softmax 归一化得分作为权重
        weights = torch.softmax(scores, dim=0)
        final_loss = (weights * losses).sum()

        return final_loss

    def motion_consistency_loss(self, flow, sem_logits, instance_labels, sem_labels):
        device = flow.device
        loss = torch.tensor(0.0, device=device)
        
        # 定义运动部件类别
        slider_classes = [5, 6]  # slider类别
        hinge_classes = [4, 7]   # hinge类别
        
        # 计算flow长度
        flow_lengths = torch.norm(flow, p=2, dim=1)  # [N]
        
        # 获取最可能的预测结果（argmax）
        sem_preds = torch.argmax(sem_logits, dim=1)  # [N]
        
        # 只在指定的类别上计算损失
        target_classes = [4, 5, 6, 7]
        target_mask = torch.isin(sem_labels, torch.tensor(target_classes, device=device))
        
        # 用于加权平均的计数器
        total_instances = 0
        total_loss = 0.0
        
        # 处理slider类别
        for class_id in slider_classes:
            # 使用最可能的预测结果，并且只在目标类别上计算
            class_mask = (instance_labels >= 0) & (sem_preds == class_id) & target_mask
            
            if class_mask.sum() == 0:
                continue
                
            class_flow = flow[class_mask]
            class_flow_lengths = flow_lengths[class_mask]
            class_instances = instance_labels[class_mask]
            
            # 对每个实例计算长度一致性
            unique_instances = torch.unique(class_instances)
            
            for inst_id in unique_instances:
                inst_mask = class_instances == inst_id
                if inst_mask.sum() < 3:
                    continue
                    
                inst_flow_lengths = class_flow_lengths[inst_mask]
                
                # slider应该具有相对一致的长度分布
                length_mean = inst_flow_lengths.mean()
                normalized_flow_lengths = inst_flow_lengths / (length_mean + 1e-8)
                length_std = normalized_flow_lengths.std()
                coefficient_of_variation = length_std
                # print(f"coefficient_of_variation: {coefficient_of_variation}")
                # 鼓励小的变异系数（长度一致）- 限制在0-1范围
                slider_loss = torch.sigmoid(coefficient_of_variation * 5)  # 缩放并限制在[0,1]
                # print(f"slider_loss: {slider_loss}")
                total_loss += slider_loss
                total_instances += 1
        
        # 处理hinge类别
        for class_id in hinge_classes:
            # 使用最可能的预测结果，并且只在目标类别上计算
            class_mask = (instance_labels >= 0) & (sem_preds == class_id) & target_mask
            
            if class_mask.sum() == 0:
                continue
                
            class_flow_lengths = flow_lengths[class_mask]
            class_instances = instance_labels[class_mask]
            
            # 对每个实例计算长度分布模式
            unique_instances = torch.unique(class_instances)
            
            for inst_id in unique_instances:
                inst_mask = class_instances == inst_id
                if inst_mask.sum() < 5:
                    continue
                    
                inst_flow_lengths = class_flow_lengths[inst_mask]
                
                # hinge的flow长度应该变化较大（旋转运动）
                length_mean = inst_flow_lengths.mean()
                normalized_flow_lengths = inst_flow_lengths / (length_mean + 1e-8)
                length_std = normalized_flow_lengths.std()
                coefficient_of_variation = length_std
                # print(f"coefficient_of_variation: {coefficient_of_variation}")
                # 鼓励大的变异系数（长度变化大）- 限制在0-1范围
                # 当变异系数大时，损失应该小；当变异系数小时，损失应该大
                hinge_loss = 1.0 - torch.sigmoid(coefficient_of_variation * 5)  # 限制在[0,1]
                # print(f"hinge_loss: {hinge_loss}")
                total_loss += hinge_loss
                total_instances += 1
        
        # 计算加权平均
        if total_instances > 0:
            loss = total_loss / total_instances
        else:
            loss = torch.tensor(0.0, device=device)
        
        return loss
    def proposal_clustering_and_revoxelize(
            self,
            pt_xyz: torch.Tensor,
            flow_xyz: torch.Tensor,
            batch_indices: torch.Tensor,
            pt_features: torch.Tensor,
            sem_preds: torch.Tensor,
            offset_preds: torch.Tensor,
            offset_preds_flow : torch.Tensor,
            instance_labels: Optional[torch.Tensor],
    ):
        device = self.device

        if instance_labels is not None:
            valid_mask = (sem_preds > 0) & (instance_labels >= 0)
        else:
            valid_mask = sem_preds > 0

        pt_xyz = pt_xyz[valid_mask]
        flow_xyz = flow_xyz[valid_mask]
        batch_indices = batch_indices[valid_mask]
        pt_features = pt_features[valid_mask]
        sem_preds = sem_preds[valid_mask].int()
        offset_preds = offset_preds[valid_mask]
        offset_preds_flow = offset_preds_flow[valid_mask]
        flow = flow_xyz - pt_xyz

        if instance_labels is not None:
            instance_labels = instance_labels[valid_mask]

        # get batch offsets (csr) from batch indices
        _, batch_indices_compact, num_points_per_batch = torch.unique_consecutive(
            batch_indices, return_inverse=True, return_counts=True
        )
        batch_indices_compact = batch_indices_compact.int()
        batch_offsets = torch.zeros(
            (num_points_per_batch.shape[0] + 1,), dtype=torch.int32, device=device
        )
        batch_offsets[1:] = num_points_per_batch.cumsum(0)

        # cluster proposals: dual set
        sorted_cc_labels, sorted_indices = cluster_proposals(
            pt_xyz + offset_preds, flow, batch_indices_compact, batch_offsets, sem_preds,
            self.ball_query_radius, self.max_num_points_per_query,
            self.angle_thresh_deg, self.flow_diff_thresh,
            use_adaptive=self.use_adaptive_clustering,
        )

        sorted_cc_labels_shift, sorted_indices_shift = cluster_proposals(
            flow_xyz + offset_preds_flow, flow, batch_indices_compact, batch_offsets, sem_preds,
            self.ball_query_radius, self.max_num_points_per_query,
            self.angle_thresh_deg, self.flow_diff_thresh,
            use_adaptive=self.use_adaptive_clustering,
        )

        # combine clusters
        sorted_cc_labels = torch.cat([
            sorted_cc_labels,
            sorted_cc_labels_shift + sorted_cc_labels.shape[0],
        ], dim=0)
        sorted_indices = torch.cat([sorted_indices, sorted_indices_shift], dim=0)
        # compact the proposal ids
        _, proposal_indices, num_points_per_proposal = torch.unique_consecutive(
            sorted_cc_labels, return_inverse=True, return_counts=True
        )

        # # remove small proposals
        # valid_proposal_mask = (
        #         num_points_per_proposal >= self.min_num_points_per_proposal
        # )

        # 获取每个proposal的类别 (使用第一个点的语义类别代表该proposal)
        num_proposals = num_points_per_proposal.shape[0]
        proposal_offsets_temp = torch.zeros(num_proposals + 1, dtype=torch.int32, device=device)
        proposal_offsets_temp[1:] = num_points_per_proposal.cumsum(0)
        proposal_classes = sem_preds[sorted_indices[proposal_offsets_temp[:-1].long()]]
        '''        
        # 针对小类别进行额外的精细聚类
        small_classes = [1, 2, 8, 9]  # handle + knob classes  
        small_mask = torch.isin(sem_preds, torch.tensor(small_classes, device=sem_preds.device))

        if small_mask.any():
            # 小类别使用更小的聚类半径进行精细聚类
            fine_radius = self.ball_query_radius * 0.7  # 缩小到70%
            sorted_cc_labels_fine, sorted_indices_fine = cluster_proposals(
                (flow_xyz + offset_preds)[small_mask], 
                batch_indices_compact[small_mask], 
                batch_offsets, 
                sem_preds[small_mask],
                fine_radius, 
                self.max_num_points_per_query,
            )
            # 合并精细聚类结果
            if sorted_cc_labels_fine.shape[0] > 0:
                sorted_cc_labels_fine += sorted_cc_labels.shape[0] + sorted_cc_labels_shift.shape[0]
                sorted_indices_fine = torch.nonzero(small_mask, as_tuple=True)[0][sorted_indices_fine]
                sorted_cc_labels = torch.cat([sorted_cc_labels, sorted_cc_labels_shift, sorted_cc_labels_fine], dim=0)
                sorted_indices = torch.cat([sorted_indices, sorted_indices_shift, sorted_indices_fine], dim=0)
            else:
                sorted_cc_labels = torch.cat([sorted_cc_labels, sorted_cc_labels_shift], dim=0)
                sorted_indices = torch.cat([sorted_indices, sorted_indices_shift], dim=0)
        else:
            # 没有小类别时的原始合并逻辑
            sorted_cc_labels = torch.cat([sorted_cc_labels, sorted_cc_labels_shift], dim=0)
            sorted_indices = torch.cat([sorted_indices, sorted_indices_shift], dim=0)
        '''



        # remove small proposals (handle和knob类别使用更低的阈值)
        small_class_mask = (proposal_classes == 1) | (proposal_classes == 2) | (proposal_classes == 8) | (proposal_classes == 9)
        min_points_threshold = torch.where(small_class_mask, 2, self.min_num_points_per_proposal)
        valid_proposal_mask = num_points_per_proposal >= min_points_threshold

        # proposal to point
        valid_point_mask = valid_proposal_mask[proposal_indices]

        sorted_indices = sorted_indices[valid_point_mask]
        if sorted_indices.shape[0] == 0:
            return None, None, None

        batch_indices = batch_indices[sorted_indices]
        pt_xyz = pt_xyz[sorted_indices]
        pt_features = pt_features[sorted_indices]
        sem_preds = sem_preds[sorted_indices]
        if instance_labels is not None:
            instance_labels = instance_labels[sorted_indices]

        # re-compact the proposal ids
        proposal_indices = proposal_indices[valid_point_mask]
        _, proposal_indices, num_points_per_proposal = torch.unique_consecutive(
            proposal_indices, return_inverse=True, return_counts=True
        )
        num_proposals = num_points_per_proposal.shape[0]

        # get proposal batch offsets
        proposal_offsets = torch.zeros(
            num_proposals + 1, dtype=torch.int32, device=device
        )
        proposal_offsets[1:] = num_points_per_proposal.cumsum(0)

        # voxelization
        voxel_features, voxel_coords, pc_voxel_id = segmented_voxelize(
            pt_xyz, pt_features,
            proposal_offsets, proposal_indices,
            num_points_per_proposal,
            self.score_fullscale, self.score_scale,
        )
        voxel_tensor = spconv.SparseConvTensor(
            voxel_features, voxel_coords.int(),
            spatial_shape=[self.score_fullscale] * 3,
            batch_size=num_proposals,
        )
        if not (pc_voxel_id >= 0).all():
            import pdb
            pdb.set_trace()
        flow = flow_xyz[sorted_indices] - pt_xyz

        proposals = Instances(
            valid_mask=valid_mask,
            sorted_indices=sorted_indices,
            pt_xyz=pt_xyz,
            batch_indices=batch_indices,
            proposal_offsets=proposal_offsets,
            proposal_indices=proposal_indices,
            num_points_per_proposal=num_points_per_proposal,
            sem_preds=sem_preds,
            instance_labels=instance_labels,
            flow=flow,
        )

        return voxel_tensor, pc_voxel_id, proposals
