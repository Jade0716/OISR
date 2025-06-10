from __future__ import print_function

import spconv.pytorch as spconv
from einops import rearrange, repeat
from epic_ops.iou import batch_instance_seg_iou

from losses import focal_loss, dice_loss, pixel_accuracy, mean_iou
from grouping_utils import (apply_nms, cluster_proposals, compute_ap,
                               compute_npcs_loss, filter_invalid_proposals,
                               get_gt_scores, segmented_voxelize)
from ..structure.segmentation import Segmentation
from ..structure.instances import Instances
import copy
import json
from functools import partial
from pathlib import Path
from typing import Optional, Tuple, Union, List
import argparse
import torchdata.datapipes as dp
from epic_ops.voxelize import voxelize
from torch.utils.data import Dataset
import random
from glob import glob

from ..structure.point_cloud import PointCloud
from ..dataset import data_utils

from ..misc.info import OBJECT_NAME2ID, PART_ID2NAME, PART_NAME2ID, get_symmetry_matrix


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from model import GAPartNet


class IOStream:
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


def _init_(args):
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)
    if not os.path.exists('checkpoints/' + args.exp_name + '/' + 'models'):
        os.makedirs('checkpoints/' + args.exp_name + '/' + 'models')
    os.system('cp main.py checkpoints' + '/' + args.exp_name + '/' + 'main.py.backup')
    os.system('cp model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp data.py checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')


def weights_init(m):
    classname=m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.kaiming_normal_(m.weight.data)
    if classname.find('Conv1d') != -1:
        nn.init.kaiming_normal_(m.weight.data)
        
class GAPartNetDataset(Dataset):
    def __init__(
            self,
            root_dir: Union[str, Path, List] = "",
            shuffle: bool = False,
            max_points: int = 20000,
            augmentation: bool = False,
            voxel_size: Tuple[float, float, float] = (1 / 100, 1 / 100, 1 / 100),
            few_shot=False,
            few_shot_num=512,
            pos_jitter: float = 0.,
            color_jitter: float = 0.,
            flip_prob: float = 0.,
            rotate_prob: float = 0.,
            nopart_path: str = "data/nopart.txt",
            no_label=False,
    ):
        if type(root_dir) == list:
            file_paths = []
            for rt in root_dir:
                file_paths += glob(str(rt) + "/*.pth")
        else:
            file_paths = glob(str(root_dir) + "/*.pth")
        self.nopart_files = open(nopart_path, "r").readlines()[0].split(" ")
        self.nopart_names = [p.split("/")[-1].split(".")[0] for p in self.nopart_files]
        file_paths = [path for path in file_paths
                      if path.split("/")[-1].split(".")[0] not in self.nopart_names]
        if shuffle:
            random.shuffle(file_paths)
        if few_shot:
            file_paths = file_paths[:few_shot_num]
        self.pc_paths = file_paths
        self.no_label = no_label
        self.augmentation = augmentation
        self.pos_jitter = pos_jitter
        self.color_jitter = color_jitter
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob
        self.voxel_size = voxel_size
        self.max_points = max_points

    def __len__(self):
        return len(self.pc_paths)

    def __getitem__(self, idx):
        path = self.pc_paths[idx]
        file = load_data(path, no_label=self.no_label)
        if not bool((file.instance_labels != -100).any()):
            import ipdb;
            ipdb.set_trace()
        file = downsample(file, max_points=self.max_points)
        file = compact_instance_labels(file)
        if self.augmentation:
            file = apply_augmentations(file,
                                       pos_jitter=self.pos_jitter,
                                       color_jitter=self.color_jitter,
                                       flip_prob=self.flip_prob,
                                       rotate_prob=self.rotate_prob, )
        file = generate_inst_info(file)
        file = file.to_tensor()
        file = apply_voxelization(file, voxel_size=self.voxel_size)
        return file


def apply_augmentations(
        pc: PointCloud,
        *,
        pos_jitter: float = 0.,
        color_jitter: float = 0.,
        flip_prob: float = 0.,
        rotate_prob: float = 0.,
) -> PointCloud:
    pc = copy.copy(pc)

    m = np.eye(3)
    if pos_jitter > 0:
        m += np.random.randn(3, 3) * pos_jitter

    if flip_prob > 0:
        if np.random.rand() < flip_prob:
            m[0, 0] = -m[0, 0]

    if rotate_prob > 0:
        if np.random.rand() < flip_prob:
            theta = np.random.rand() * np.pi * 2
            m = m @ np.asarray([
                [np.cos(theta), np.sin(theta), 0],
                [-np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ])

    pc.points = pc.points.copy()
    pc.points[:, :3] = pc.points[:, :3] @ m

    if color_jitter > 0:
        pc.points[:, 3:] += np.random.randn(
            1, pc.points.shape[1] - 3
        ) * color_jitter

    return pc


def downsample(pc: PointCloud, *, max_points: int = 20000) -> PointCloud:
    pc = copy.copy(pc)

    num_points = pc.points.shape[0]

    if num_points > max_points:
        assert False, (num_points, max_points)

    return pc


def compact_instance_labels(pc: PointCloud) -> PointCloud:
    pc = copy.copy(pc)

    valid_mask = pc.instance_labels >= 0
    instance_labels = pc.instance_labels[valid_mask]
    _, instance_labels = np.unique(instance_labels, return_inverse=True)
    pc.instance_labels[valid_mask] = instance_labels

    return pc


def generate_inst_info(pc: PointCloud) -> PointCloud:
    pc = copy.copy(pc)

    num_points = pc.points.shape[0]

    num_instances = int(pc.instance_labels.max()) + 1
    instance_regions = np.zeros((num_points, 9), dtype=np.float32)
    num_points_per_instance = []
    instance_sem_labels = []

    assert num_instances > 0

    for i in range(num_instances):
        indices = np.where(pc.instance_labels == i)[0]

        xyz_i = pc.points[indices, :3]
        min_i = xyz_i.min(0)
        max_i = xyz_i.max(0)
        mean_i = xyz_i.mean(0)
        instance_regions[indices, 0:3] = mean_i
        instance_regions[indices, 3:6] = min_i
        instance_regions[indices, 6:9] = max_i

        num_points_per_instance.append(indices.shape[0])
        instance_sem_labels.append(int(pc.sem_labels[indices[0]]))

    pc.num_instances = num_instances
    pc.instance_regions = instance_regions
    pc.num_points_per_instance = np.asarray(num_points_per_instance, dtype=np.int32)
    pc.instance_sem_labels = np.asarray(instance_sem_labels, dtype=np.int32)

    return pc


def apply_voxelization(
        pc: PointCloud, *, voxel_size: Tuple[float, float, float]
) -> PointCloud:
    pc = copy.copy(pc)

    num_points = pc.points.shape[0]
    pt_xyz = pc.points[:, :3]
    points_range_min = pt_xyz.min(0)[0] - 1e-4
    points_range_max = pt_xyz.max(0)[0] + 1e-4
    voxel_features, voxel_coords, _, pc_voxel_id = voxelize(
        pt_xyz, pc.points,
        batch_offsets=torch.as_tensor([0, num_points], dtype=torch.int64, device=pt_xyz.device),
        voxel_size=torch.as_tensor(voxel_size, device=pt_xyz.device),
        points_range_min=torch.as_tensor(points_range_min, device=pt_xyz.device),
        points_range_max=torch.as_tensor(points_range_max, device=pt_xyz.device),
        reduction="mean",
    )
    assert (pc_voxel_id >= 0).all()

    voxel_coords_range = (voxel_coords.max(0)[0] + 1).clamp(min=128, max=None)

    pc.voxel_features = voxel_features
    pc.voxel_coords = voxel_coords
    pc.voxel_coords_range = voxel_coords_range.tolist()
    pc.pc_voxel_id = pc_voxel_id

    return pc


def load_data(file_path: str, no_label: bool = False):
    if not no_label:
        pc_data = torch.load(file_path)
    else:
        # testing data type, e.g. real world point cloud without GT semantic label.
        raise NotImplementedError

    pc_id = file_path.split("/")[-1].split(".")[0]
    object_cat = OBJECT_NAME2ID[pc_id.split("_")[0]]

    return PointCloud(
        pc_id=pc_id,
        obj_cat=object_cat,
        points=np.concatenate(
            [pc_data[0], pc_data[1]],
            axis=-1, dtype=np.float32,
        ),
        sem_labels=pc_data[2].astype(np.int64),
        instance_labels=pc_data[3].astype(np.int32),
        gt_npcs=pc_data[4].astype(np.float32),
    )


def from_folder(
        root_dir: Union[str, Path] = "",
        split: str = "train_new",
        shuffle: bool = False,
        max_points: int = 20000,
        augmentation: bool = False,
        voxel_size: Tuple[float, float, float] = (1 / 100, 1 / 100, 1 / 100),
        pos_jitter: float = 0.,
        color_jitter: float = 0.1,
        flip_prob: float = 0.,
        rotate_prob: float = 0.,
):
    root_dir = Path(root_dir)

    with open(root_dir / f"{split}.json") as f:
        file_names = json.load(f)

    pipe = dp.iter.IterableWrapper(file_names)

    # pipe = pipe.filter(filter_fn=lambda x: x == "pth_new/StorageFurniture_41004_00_013.pth")

    pipe = pipe.distributed_sharding_filter()
    if shuffle:
        pipe = pipe.shuffle()

    # Load data
    pipe = pipe.map(partial(load_data, root_dir=root_dir))
    # Remove empty samples
    pipe = pipe.filter(filter_fn=lambda x: bool((x.instance_labels != -100).any()))

    # Downsample
    # TODO: Crop
    pipe = pipe.map(partial(downsample, max_points=max_points))
    pipe = pipe.map(compact_instance_labels)

    # Augmentations
    if augmentation:
        pipe = pipe.map(partial(
            apply_augmentations,
            pos_jitter=pos_jitter,
            color_jitter=color_jitter,
            flip_prob=flip_prob,
            rotate_prob=rotate_prob,
        ))

    # Generate instance info
    pipe = pipe.map(generate_inst_info)

    # To tensor
    pipe = pipe.map(lambda pc: pc.to_tensor())

    # Voxelization
    pipe = pipe.map(partial(apply_voxelization, voxel_size=voxel_size))

    return pipe




def loss_sem_seg(
        model,
        sem_logits: torch.Tensor,
        sem_labels: torch.Tensor,
) -> torch.Tensor:
    if model.use_sem_focal_loss:
        loss = focal_loss(
            sem_logits, sem_labels,
            alpha=None,
            gamma=2.0,
            ignore_index=model.ignore_sem_label,
            reduction="mean",
        )
    else:
        loss = F.cross_entropy(
            sem_logits, sem_labels,
            weight=None,
            ignore_index=model.ignore_sem_label,
            reduction="mean",
        )

    if model.use_sem_dice_loss:
        loss += dice_loss(
            sem_logits[:, :, None, None], sem_labels[:, None, None],
        )
    return loss

def loss_offset(
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
    score_logits: torch.Tensor,
    proposals: Instances,
    num_points_per_instance: torch.Tensor,
) -> torch.Tensor:
    ious = batch_instance_seg_iou(
        proposals.proposal_offsets, # type: ignore
        proposals.instance_labels, # type: ignore
        proposals.batch_indices, # type: ignore
        num_points_per_instance,
    )
    proposals.ious = ious
    proposals.num_points_per_instance = num_points_per_instance

    ious_max = ious.max(-1)[0]
    gt_scores = get_gt_scores(ious_max, 0.75, 0.25)

    return F.binary_cross_entropy_with_logits(score_logits, gt_scores)


def loss_proposal_npcs(
    model,
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
    model.symmetry_indices = model.symmetry_indices.to(sem_preds.device)
    model.symmetry_matrix_1 = model.symmetry_matrix_1.to(sem_preds.device)
    model.symmetry_matrix_2 = model.symmetry_matrix_2.to(sem_preds.device)
    model.symmetry_matrix_3 = model.symmetry_matrix_3.to(sem_preds.device)
    # import pdb; pdb.set_trace()
    symmetry_indices = model.symmetry_indices[sem_preds]
    # group #1
    group_1_mask = symmetry_indices < 3
    symmetry_indices_1 = symmetry_indices[group_1_mask]
    if symmetry_indices_1.shape[0] > 0:
        loss_npcs += compute_npcs_loss(
            npcs_logits[group_1_mask], gt_npcs[group_1_mask],
            proposal_indices[group_1_mask],
            model.symmetry_matrix_1[symmetry_indices_1]
        )

    # group #2
    group_2_mask = symmetry_indices == 3
    symmetry_indices_2 = symmetry_indices[group_2_mask]
    if symmetry_indices_2.shape[0] > 0:
        loss_npcs += compute_npcs_loss(
            npcs_logits[group_2_mask], gt_npcs[group_2_mask],
            proposal_indices[group_2_mask],
            model.symmetry_matrix_2[symmetry_indices_2 - 3]
        )

    # group #3
    group_3_mask = symmetry_indices == 4
    symmetry_indices_3 = symmetry_indices[group_3_mask]
    if symmetry_indices_3.shape[0] > 0:
        loss_npcs += compute_npcs_loss(
            npcs_logits[group_3_mask], gt_npcs[group_3_mask],
            proposal_indices[group_3_mask],
            model.symmetry_matrix_3[symmetry_indices_3 - 4]
        )

    return loss_npcs
    
def proposal_clustering_and_revoxelize(
        model,
        pt_xyz: torch.Tensor,
        batch_indices: torch.Tensor,
        pt_features: torch.Tensor,
        sem_preds: torch.Tensor,
        offset_preds: torch.Tensor,
        instance_labels: Optional[torch.Tensor],
):
    device = model.device #need modify

    if instance_labels is not None:
        valid_mask = (sem_preds > 0) & (instance_labels >= 0)
    else:
        valid_mask = sem_preds > 0

    pt_xyz = pt_xyz[valid_mask]
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
        pt_xyz, batch_indices_compact, batch_offsets, sem_preds,
        model.ball_query_radius, model.max_num_points_per_query,
    )

    sorted_cc_labels_shift, sorted_indices_shift = cluster_proposals(
        pt_xyz + offset_preds, batch_indices_compact, batch_offsets, sem_preds,
        model.ball_query_radius, model.max_num_points_per_query_shift,
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
            num_points_per_proposal >= model.min_num_points_per_proposal
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
        model.score_fullscale, model.score_scale,
    )
    voxel_tensor = spconv.SparseConvTensor(
        voxel_features, voxel_coords.int(),
        spatial_shape=[model.score_fullscale] * 3,
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

def train_one_epoch(
        net,
        point_clouds: List[PointCloud],
        opt,
        device="cuda",
):
    net.train()
    batch_size = len(point_clouds)

    # 将网络模型转移到指定设备
    net.to(device)

    # data batch parsing
    data_batch = PointCloud.collate(point_clouds)
    points = data_batch.points.to(device)
    sem_labels = data_batch.sem_labels.to(device) if data_batch.sem_labels is not None else None
    pc_ids = data_batch.pc_ids
    instance_regions = data_batch.instance_regions.to(device) if data_batch.instance_regions is not None else None
    instance_labels = data_batch.instance_labels.to(device) if data_batch.instance_labels is not None else None
    batch_indices = data_batch.batch_indices.to(device)
    instance_sem_labels = data_batch.instance_sem_labels.to(device) if data_batch.instance_sem_labels is not None else None
    num_points_per_instance = data_batch.num_points_per_instance
    gt_npcs = data_batch.gt_npcs.to(device) if data_batch.gt_npcs is not None else None

    pt_xyz = points[:, :3]

    # Forward pass through backbone
    pc_feature = net.forward_backbone(pc_batch=data_batch)

    # Semantic segmentation
    sem_logits = net.forward_sem_seg(pc_feature)
    sem_preds = torch.argmax(sem_logits.detach(), dim=-1)

    # Compute loss for semantic segmentation
    loss_sem_se = loss_sem_seg(net, sem_logits, sem_labels)

    # Calculate accuracy
    all_accu = (sem_preds == sem_labels).sum().float() / sem_labels.shape[0]
    if sem_labels is not None:
        instance_mask = sem_labels > 0
        pixel_accu = pixel_accuracy(sem_preds[instance_mask], sem_labels[instance_mask])
    else:
        pixel_accu = 0.0

    sem_seg = Segmentation(
        batch_size=batch_size,
        sem_preds=sem_preds,
        sem_labels=sem_labels,
        all_accu=all_accu,
        pixel_accu=pixel_accu,
    )

    # Offset prediction
    offsets_preds = net.forward_offset(pc_feature)
    if instance_regions is not None:
        offsets_gt = instance_regions[:, :3] - pt_xyz
        loss_offset_dist, loss_offset_dir = loss_offset(
            offsets_preds, offsets_gt, sem_labels, instance_labels,  # type: ignore
        )
    else:
        loss_offset_dist, loss_offset_dir = 0., 0.

    # Proposal clustering and revoxelization
    voxel_tensor, pc_voxel_id, proposals = net.proposal_clustering_and_revoxelize(
        pt_xyz=pt_xyz,
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
    score_logits = net.forward_proposal_score(
        voxel_tensor, pc_voxel_id, proposals
    )  # type: ignore
    proposal_offsets_begin = proposals.proposal_offsets[:-1].long()  # type: ignore

    if proposals.sem_labels is not None:  # type: ignore
        proposal_sem_labels = proposals.sem_labels[proposal_offsets_begin].long()  # type: ignore
    else:
        proposal_sem_labels = proposals.sem_preds[proposal_offsets_begin].long()  # type: ignore
    score_logits = score_logits.gather(
        1, proposal_sem_labels[:, None] - 1
    ).squeeze(1)
    proposals.score_preds = score_logits.detach().sigmoid()  # type: ignore
    if num_points_per_instance is not None:  # type: ignore
        loss_prop_score = loss_proposal_score(
            net, score_logits, proposals, num_points_per_instance,  # type: ignore
        )
    else:
        loss_prop_score = 0.0

    # NPCS prediction
    npcs_logits = net.forward_proposal_npcs(
        voxel_tensor, pc_voxel_id
    )
    if gt_npcs is not None:
        gt_npcs = gt_npcs[proposals.valid_mask][proposals.sorted_indices]
        loss_prop_npcs = loss_proposal_npcs(npcs_logits, gt_npcs, proposals)
    else:
        loss_prop_npcs = 0.0

    # Total loss
    loss = loss_sem_se + loss_offset_dist + loss_offset_dir + loss_prop_score + loss_prop_npcs

    # Backpropagation and optimization
    opt.zero_grad()
    loss.backward()
    opt.step()

    return pc_ids, sem_seg, proposals, loss.item()


def test_one_epoch(
        net,
        point_clouds: List[PointCloud],
        device="cuda",
):
    net.eval()
    batch_size = len(point_clouds)

    # 将网络模型转移到指定设备
    net.to(device)

    # data batch parsing
    data_batch = PointCloud.collate(point_clouds)
    points = data_batch.points.to(device)
    sem_labels = data_batch.sem_labels.to(device) if data_batch.sem_labels is not None else None
    pc_ids = data_batch.pc_ids
    instance_regions = data_batch.instance_regions.to(device) if data_batch.instance_regions is not None else None
    instance_labels = data_batch.instance_labels.to(device) if data_batch.instance_labels is not None else None
    batch_indices = data_batch.batch_indices.to(device)
    instance_sem_labels = data_batch.instance_sem_labels.to(
        device) if data_batch.instance_sem_labels is not None else None
    num_points_per_instance = data_batch.num_points_per_instance
    gt_npcs = data_batch.gt_npcs.to(device) if data_batch.gt_npcs is not None else None

    pt_xyz = points[:, :3]
    with torch.no_grad():
        # Forward pass through backbone
        pc_feature = net.forward_backbone(pc_batch=data_batch)

        # Semantic segmentation
        sem_logits = net.forward_sem_seg(pc_feature)
        sem_preds = torch.argmax(sem_logits.detach(), dim=-1)

        # Compute loss for semantic segmentation
        loss_sem_se = loss_sem_seg(net, sem_logits, sem_labels)

        # Calculate accuracy
        all_accu = (sem_preds == sem_labels).sum().float() / sem_labels.shape[0]
        if sem_labels is not None:
            instance_mask = sem_labels > 0
            pixel_accu = pixel_accuracy(sem_preds[instance_mask], sem_labels[instance_mask])
        else:
            pixel_accu = 0.0

        sem_seg = Segmentation(
            batch_size=batch_size,
            sem_preds=sem_preds,
            sem_labels=sem_labels,
            all_accu=all_accu,
            pixel_accu=pixel_accu,
        )

        # Offset prediction
        offsets_preds = net.forward_offset(pc_feature)
        if instance_regions is not None:
            offsets_gt = instance_regions[:, :3] - pt_xyz
            loss_offset_dist, loss_offset_dir = loss_offset(
                offsets_preds, offsets_gt, sem_labels, instance_labels,  # type: ignore
            )
        else:
            loss_offset_dist, loss_offset_dir = 0., 0.

        # Proposal clustering and revoxelization
        voxel_tensor, pc_voxel_id, proposals = net.proposal_clustering_and_revoxelize(
            pt_xyz=pt_xyz,
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
        score_logits = net.forward_proposal_score(
            voxel_tensor, pc_voxel_id, proposals
        )  # type: ignore
        proposal_offsets_begin = proposals.proposal_offsets[:-1].long()  # type: ignore

        if proposals.sem_labels is not None:  # type: ignore
            proposal_sem_labels = proposals.sem_labels[proposal_offsets_begin].long()  # type: ignore
        else:
            proposal_sem_labels = proposals.sem_preds[proposal_offsets_begin].long()  # type: ignore
        score_logits = score_logits.gather(
            1, proposal_sem_labels[:, None] - 1
        ).squeeze(1)
        proposals.score_preds = score_logits.detach().sigmoid()  # type: ignore
        if num_points_per_instance is not None:  # type: ignore
            loss_prop_score = loss_proposal_score(
                net, score_logits, proposals, num_points_per_instance,  # type: ignore
            )
        else:
            loss_prop_score = 0.0

        # NPCS prediction
        npcs_logits = net.forward_proposal_npcs(
            voxel_tensor, pc_voxel_id
        )
        if gt_npcs is not None:
            gt_npcs = gt_npcs[proposals.valid_mask][proposals.sorted_indices]
            loss_prop_npcs = loss_proposal_npcs(npcs_logits, gt_npcs, proposals)
        else:
            loss_prop_npcs = 0.0

        # Total loss
        loss = loss_sem_se + loss_offset_dist + loss_offset_dir + loss_prop_score + loss_prop_npcs

    return pc_ids, sem_seg, proposals, loss.item()


def train(args, epochs, net, point_clouds0: List[PointCloud], point_clouds1: List[PointCloud], textio):
    learning_rate : float = 1e-3
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),lr=learning_rate,)

    best_test_loss = np.inf
    for epoch in range(epochs):
        textio.cprint('==epoch: %d, learning rate: %f==' % (epoch, opt.param_groups[0]['lr']))
        _, _, proposals, train_loss = train_one_epoch(net, point_clouds0, opt)
        textio.cprint('mean train loss: %f' % train_loss)

        _, _, _, test_loss = test_one_epoch(net, point_clouds1, opt)
        textio.cprint('mean test loss: %f' % test_loss)
        if best_test_loss >= test_loss:
            best_test_loss = test_loss
            textio.cprint('best test loss till now: %f' % test_loss)
            if torch.cuda.device_count() > 1:
                torch.save(net.module.state_dict(), 'checkpoints/%s/models/model.best.t7' % args.exp_name)
            else:
                torch.save(net.state_dict(), 'checkpoints/%s/models/model.best.t7' % args.exp_name)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='evaluate the model')
    args = parser.parse_args()
    # CUDA settings
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    np.random.seed(1234)
    _init_(args)
    textio = IOStream('checkpoints/' + args.exp_name + '/run.log')
    textio.cprint(str(args))

    
    root_dir: str = "data/GAPartNet_All"
    max_points: int = 20000
    voxel_size: Tuple[float, float, float] = (1 / 100, 1 / 100, 1 / 100)
    train_batch_size: int = 32
    val_batch_size: int = 32
    test_batch_size: int = 32
    num_workers: int = 16
    pos_jitter: float = 0.
    color_jitter: float = 0.1
    flip_prob: float = 0.
    rotate_prob: float = 0.
    train_few_shot: bool = False
    val_few_shot: bool = False
    intra_few_shot: bool = False
    inter_few_shot: bool = False
    few_shot_num: int = 256
    train_with_all: bool = False
    train_data_files = GAPartNetDataset(
                    Path(root_dir) / "train" / "pth",
                    shuffle=True,
                    max_points=max_points,
                    augmentation=True,
                    voxel_size=voxel_size,
                    few_shot = train_few_shot,
                    few_shot_num=few_shot_num,
                    pos_jitter = pos_jitter,
                    color_jitter = color_jitter,
                    flip_prob = flip_prob,
                    rotate_prob = rotate_prob,
                )
    test_data_files = GAPartNetDataset(
                Path(root_dir) / "test_inter" / "pth",
                shuffle=True,
                max_points= max_points,
                augmentation=False,
                voxel_size= voxel_size,
                few_shot = inter_few_shot,
                few_shot_num= few_shot_num,
                pos_jitter = pos_jitter,
                color_jitter = color_jitter,
                flip_prob = flip_prob,
                rotate_prob = rotate_prob,
            )
    train_loader = DataLoader(
            train_data_files,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=data_utils.trivial_batch_collator,
            pin_memory=True,
            drop_last=True,
        )
    test_loader = DataLoader(
            test_data_files,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=data_utils.trivial_batch_collator,
            pin_memory=True,
            drop_last=False,
        )
    net = GAPartNet().cuda()
    net.apply(weights_init)
    net = nn.DataParallel(net)
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    if args.eval:
        train(args, 2, net, train_loader, test_loader, textio)
    else:
        train(args, 2, net, train_loader, test_loader, textio)

    print('FINISH')
    print('FINISH')


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '1, 2'
    main()