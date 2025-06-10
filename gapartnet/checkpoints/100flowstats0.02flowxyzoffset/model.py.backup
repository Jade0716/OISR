from __future__ import print_function
from typing import Optional, Dict, Tuple, List
import functools
import torch
import torch.nn as nn
import numpy as np
import spconv.pytorch as spconv
import torch.nn.functional as F
import spconv.pytorch as spconv
from einops import rearrange, repeat
from epic_ops.iou import batch_instance_seg_iou
from epic_ops.reduce import segmented_maxpool
from fontTools.misc.psOperators import ps_integer
from epic_ops.reduce import segmented_maxpool

from structure.point_cloud import PointCloudBatch, PointCloud
from structure.instances import Instances

from misc.info import OBJECT_NAME2ID, PART_ID2NAME, PART_NAME2ID, get_symmetry_matrix
from .backbone import SparseUNet, GroupedSparseUNet
from network.losses import focal_loss, dice_loss, pixel_accuracy, mean_iou
from network.grouping_utils import (apply_nms, cluster_proposals, compute_ap,
                               compute_npcs_loss, filter_invalid_proposals,
                               get_gt_scores, segmented_voxelize)
from structure.segmentation import Segmentation
from structure.instances import Instances
import copy
import json
from functools import partial
from pathlib import Path
from typing import Optional, Tuple, Union, List

import torchdata.datapipes as dp
from epic_ops.voxelize import voxelize
from torch.utils.data import Dataset
import random
from glob import glob
from tqdm import tqdm
from structure.point_cloud import PointCloud, PointCloudBatch
from dataset import data_utils

from misc.info import OBJECT_NAME2ID, PART_ID2NAME, PART_NAME2ID, get_symmetry_matrix

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
import open3d as o3d


class GAPartNet(nn.Module):
    def __init__(
            self,
            in_channels: int = 12,
            num_part_classes: int = 10,
            backbone_type: str = "SparseUNet",
            backbone_cfg: Dict = {"channels": [24, 48, 64, 80, 96, 112, 128],"block_repeat": 2},
            # backbone_cfg: Dict = {"channels": [16, 32, 48, 64, 80, 96, 112], "block_repeat": 2},
            # semantic segmentation
            ignore_sem_label: int = -100,
            use_sem_focal_loss: bool = True,
            use_sem_dice_loss: bool = True,
            # instance segmentation
            instance_seg_cfg: Dict = {      "ball_query_radius": 0.02,
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
            # testing
            visualize_cfg: Dict = {},
    ):
        super(GAPartNet, self).__init__()
        self.validation_step_outputs = []
        self.visualize_cfg = visualize_cfg
        self.in_channels = in_channels
        self.num_part_classes = num_part_classes
        self.backbone_type = backbone_type
        self.backbone_cfg = backbone_cfg
        self.ignore_sem_label = ignore_sem_label
        self.use_sem_focal_loss = use_sem_focal_loss
        self.use_sem_dice_loss = use_sem_dice_loss
        self.val_nms_iou_threshold = val_nms_iou_threshold
        self.val_ap_iou_threshold = val_ap_iou_threshold
        self.val_score_threshold = val_score_threshold
        # self.val_min_num_points_per_proposal = val_min_num_points_per_proposal
        self.symmetry_indices = torch.as_tensor(symmetry_indices, dtype=torch.int64)
        self.start_scorenet, self.start_npcs = training_schedule
        self.ball_query_radius = instance_seg_cfg["ball_query_radius"]
        self.max_num_points_per_query = instance_seg_cfg["max_num_points_per_query"]
        self.min_num_points_per_proposal = instance_seg_cfg["min_num_points_per_proposal"]
        self.max_num_points_per_query_shift = instance_seg_cfg["max_num_points_per_query_shift"]
        self.score_fullscale = instance_seg_cfg["score_fullscale"]
        self.score_scale = instance_seg_cfg["score_scale"]

        ## network
        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)
        # backbone
        channels = self.backbone_cfg["channels"]
        block_repeat = self.backbone_cfg["block_repeat"]
        fea_dim = channels[0]
        self.backbone = SparseUNet.build(in_channels, channels, block_repeat, norm_fn)
        # semantic segmentation head
        # self.sem_seg_head = nn.Linear(fea_dim, self.num_part_classes)
        self.sem_seg_head = nn.Sequential(
            nn.Linear(fea_dim, fea_dim // 2),
            norm_fn(fea_dim // 2),
            nn.ReLU(),
            nn.Linear(fea_dim // 2, self.num_part_classes)
        )

        # offset prediction
        self.offset_head = nn.Sequential(
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



    def forward(self, data_batch):   #PointCloudBatch
        points = data_batch.points
        pc_ids = data_batch.pc_ids
        batch_indices = data_batch.batch_indices
        sem_labels = data_batch.sem_labels
        instance_regions = data_batch.instance_regions
        instance_labels = data_batch.instance_labels
        instance_sem_labels = data_batch.instance_sem_labels
        num_points_per_instance = data_batch.num_points_per_instance
        gt_npcs = data_batch.gt_npcs
        pt_xyz = points[:, :3]
        flow_xyz = points[:,3:6]

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
        if instance_regions is not None:
            offsets_gt = instance_regions[:, :3] - pt_xyz
            loss_offset_dist, loss_offset_dir = self.loss_offset(
                offsets_preds, offsets_gt, sem_labels, instance_labels,   
            )
        else:
            loss_offset_dist, loss_offset_dir = 0., 0.
        # if self.current_epoch >= self.start_clustering:
        # Proposal clustering and revoxelization
        voxel_tensor, pc_voxel_id, proposals = self.proposal_clustering_and_revoxelize(   #利用预测语义和偏移量，再体素化得到proposals
            pt_xyz=pt_xyz,
            flow_xyz=flow_xyz,
            batch_indices=batch_indices,
            pt_features=pc_feature,
            sem_preds=sem_preds,
            offset_preds=offsets_preds,
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
        if voxel_tensor is not None and proposals is not None :#and voxel_tensor.batch_size > 1:
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
                loss_prop_score = self.loss_proposal_score(
                    score_logits, proposals, num_points_per_instance,   
                )
            else:
                # import pdb
                # pdb.set_trace()
                loss_prop_score = 0.0
        else:
            loss_prop_score = 0.0

        # if self.current_epoch >= self.start_npcs
        # NPCS prediction
        if voxel_tensor is not None:# and voxel_tensor.batch_size > 1:
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
        # print(f"loss_sem_seg: {loss_sem_seg}, loss_offset_dist: {loss_offset_dist}, loss_offset_dir: {loss_offset_dir}, loss_prop_score: {loss_prop_score}")
        dict = {
            'loss_sem_seg': loss_sem_seg,
            'loss_offset_dist': loss_offset_dist,
            'loss_offset_dir': loss_offset_dir,
            'loss_prop_score': loss_prop_score,
            'loss_prop_npcs': loss_prop_npcs,
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

    def forward_backbone(
            self,
            pc_batch: PointCloudBatch,
    ):
        if self.backbone_type == "SparseUNet":
            voxel_tensor = pc_batch.voxel_tensor
            pc_voxel_id = pc_batch.pc_voxel_id
            voxel_features = self.backbone(voxel_tensor)
            pc_feature = voxel_features.features[pc_voxel_id]
        elif self.backbone_type == "PointNet":
            pc_feature = self.backbone(pc_batch.points.reshape(-1, 6, 20000))[0]
            pc_feature = pc_feature.reshape(-1, pc_feature.shape[-1])

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
        if self.use_sem_focal_loss:
            loss = focal_loss(
                sem_logits, sem_labels,
                alpha=None,
                gamma=2.0,
                ignore_index=self.ignore_sem_label,
                reduction="mean",
            )
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
        loss_offset_dist = pt_dist[valid_instance_mask].mean()

        gt_offsets_norm = torch.norm(gt_offsets, p=2, dim=-1)
        gt_offsets = gt_offsets / (gt_offsets_norm[:, None] + 1e-8)

        offsets_norm = torch.norm(offsets, p=2, dim=-1)
        offsets = offsets / (offsets_norm[:, None] + 1e-8)

        dir_diff = -(gt_offsets * offsets).sum(-1)
        loss_offset_dir = dir_diff[valid_instance_mask].mean()

        return loss_offset_dist, loss_offset_dir


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

        ious_max = ious.max(-1)[0]
        gt_scores = get_gt_scores(ious_max, 0.75, 0.25)

        return F.binary_cross_entropy_with_logits(score_logits, gt_scores)


    def loss_proposal_npcs(
            self,
            npcs_logits: torch.Tensor,
            gt_npcs: torch.Tensor,
            proposals: Instances,
    ) -> torch.Tensor:
        sem_preds, sem_labels = proposals.sem_preds, proposals.sem_labels
        proposal_indices = proposals.proposal_indices
        valid_mask = (sem_preds == sem_labels) & (gt_npcs != 0).any(dim=-1)




        npcs_logits = npcs_logits[valid_mask]
        gt_npcs = gt_npcs[valid_mask]
        sem_preds = sem_preds[valid_mask].long()
        sem_labels = sem_labels[valid_mask]
        proposal_indices = proposal_indices[valid_mask]

        npcs_logits = rearrange(npcs_logits, "n (k c) -> n k c", c=3)
        npcs_logits = npcs_logits.gather(
            1, index=repeat(sem_preds - 1, "n -> n one c", one=1, c=3)
        ).squeeze(1)

        proposals.npcs_preds = npcs_logits.detach()
        proposals.gt_npcs = gt_npcs
        proposals.npcs_valid_mask = valid_mask

        loss_npcs = 0
        # import pdb; pdb.set_trace()
        self.symmetry_indices = self.symmetry_indices.to(sem_preds.device)
        self.symmetry_matrix_1 = self.symmetry_matrix_1.to(sem_preds.device)
        self.symmetry_matrix_2 = self.symmetry_matrix_2.to(sem_preds.device)
        self.symmetry_matrix_3 = self.symmetry_matrix_3.to(sem_preds.device)
        # import pdb; pdb.set_trace()
        symmetry_indices = self.symmetry_indices[sem_preds]
        # group #1
        group_1_mask = symmetry_indices < 3
        symmetry_indices_1 = symmetry_indices[group_1_mask]
        if symmetry_indices_1.shape[0] > 0:
            loss_npcs += compute_npcs_loss(
                npcs_logits[group_1_mask], gt_npcs[group_1_mask],
                proposal_indices[group_1_mask],
                self.symmetry_matrix_1[symmetry_indices_1]
            )

        # group #2
        group_2_mask = symmetry_indices == 3
        symmetry_indices_2 = symmetry_indices[group_2_mask]
        if symmetry_indices_2.shape[0] > 0:
            loss_npcs += compute_npcs_loss(
                npcs_logits[group_2_mask], gt_npcs[group_2_mask],
                proposal_indices[group_2_mask],
                self.symmetry_matrix_2[symmetry_indices_2 - 3]
            )

        # group #3
        group_3_mask = symmetry_indices == 4
        symmetry_indices_3 = symmetry_indices[group_3_mask]
        if symmetry_indices_3.shape[0] > 0:
            loss_npcs += compute_npcs_loss(
                npcs_logits[group_3_mask], gt_npcs[group_3_mask],
                proposal_indices[group_3_mask],
                self.symmetry_matrix_3[symmetry_indices_3 - 4]
            )

        return loss_npcs

    def proposal_clustering_and_revoxelize(
            self,
            pt_xyz: torch.Tensor,
            flow_xyz: torch.Tensor,
            batch_indices: torch.Tensor,
            pt_features: torch.Tensor,
            sem_preds: torch.Tensor,
            offset_preds: torch.Tensor,
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
            flow_xyz + offset_preds, batch_indices_compact, batch_offsets, sem_preds,
            self.ball_query_radius, self.max_num_points_per_query,
        )

        sorted_cc_labels_shift, sorted_indices_shift = cluster_proposals(
            pt_xyz + offset_preds, batch_indices_compact, batch_offsets, sem_preds,
            self.ball_query_radius, self.max_num_points_per_query_shift,
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

        # remove small proposals
        valid_proposal_mask = (
                num_points_per_proposal >= self.min_num_points_per_proposal
        )
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
        )

        return voxel_tensor, pc_voxel_id, proposals
