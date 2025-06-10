0326:
AKB48数据集坐标大小有问题，载入时需*10
不如ga的：50_round_fixed_handle，50_slider_button，50_slider_drawer，50_slider_lid，
差不多：50_line_fixed_handle，50_hinge_door，50_hinge_knob，50_revolute_handle
好：50_hinge_lid
AKB:BOX,BUCKET,DRAWER,TRASHCAN
hinge_lid,revolute_handle,slider_drawer
0328:仅重新渲染AKB：1，3 未渲染：6，7
jax版本报错
0330:一个实例有2000个点，逐点计算EC会相加4000000个结果
优化公式，消除点的影响
0410:目前可得到的proposals：
Instances(valid_mask=tensor([False, False,  True,  ..., False,  True, False], device='cuda:0'), sorted_indices=tensor([   0,    3,    4,  ..., 6558, 6584, 7907], device='cuda:0'), pt_xyz=tensor([[ 0.3856,  0.6636,  0.0605],
        [ 0.3860,  0.1455, -0.3694],
        [-0.1159,  0.5239, -0.2004],
        ...,
        [ 0.5095, -0.2053, -0.2033],
        [ 0.5189, -0.1858, -0.1517],
        [ 0.5091, -0.1969, -0.1775]], device='cuda:0'), batch_indices=tensor([0, 0, 0,  ..., 0, 0, 0], device='cuda:0', dtype=torch.int32), proposal_offsets=tensor([    0,  2544,  4189,  6564,  8171,  9553,  9705,  9726,  9739,  9754,
         9775, 12313, 13953, 19441, 19448, 19461, 19467, 19480, 19489, 19496,
        19503, 19510], device='cuda:0', dtype=torch.int32), proposal_indices=tensor([ 0,  0,  0,  ..., 20, 20, 20], device='cuda:0'), num_points_per_proposal=tensor([2544, 1645, 2375, 1607, 1382,  152,   21,   13,   15,   21, 2538, 1640,
        5488,    7,   13,    6,   13,    9,    7,    7,    7], device='cuda:0'), sem_preds=tensor([4, 4, 4,  ..., 5, 5, 5], device='cuda:0', dtype=torch.int32), pt_sem_classes=None, score_preds=tensor([9.9998e-01, 9.9996e-01, 4.0879e-01, 1.7837e-01, 2.2999e-01, 9.9568e-05,
        8.2560e-01, 9.7316e-01, 8.4107e-01, 9.6515e-01, 9.9998e-01, 9.9997e-01,
        9.9857e-01, 1.8572e-01, 7.1403e-01, 3.1159e-01, 2.3493e-04, 3.9474e-01,
        3.7806e-03, 1.0899e-03, 2.2942e-04], device='cuda:0'), npcs_preds=None, sem_labels=None, instance_labels=None, instance_sem_labels=None, num_points_per_instance=None, gt_npcs=None, npcs_valid_mask=None, ious=None, cls_preds=None, cls_labels=None, name=None)
0413:
1706仅修改了通道数
0414:
2035:修改了通道，分割头，特征3：points+flow，未对投票做修改
0416:仅重新渲染table
withoutcolor:去除颜色信息，通道改6，改load_data,apply_voxelizationtr,所有最小点个数为3
train。generate_inst_info里加入motion_type_labels
0417:
看一下model_id45238
渲染table,st,ref，dr-A
改Sd.Ld urdf
Camera_102417_0_4.
0418:convert_rendered_into_input四个重新渲染的
0423:重新渲染table,st,ref，dr-A，渲染完后需全部重新convert
0424:重新渲染完成:KitchenPot,CoffeeMachine, Trashcan, Toilet,Table, Refrigerator