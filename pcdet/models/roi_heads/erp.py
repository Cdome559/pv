import torch
import torch.nn as nn
import torch.nn.functional as F

from ...ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_stack_modules
from ...utils import common_utils
from .roi_head_template import RoIHeadTemplate


class ERPHead(RoIHeadTemplate):
    def __init__(self, input_channels, model_cfg, num_class=1, **kwargs):
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.model_cfg = model_cfg

        self.roi_grid_pool_layer, num_c_out = pointnet2_stack_modules.build_local_aggregation_module(
            input_channels=input_channels, config=self.model_cfg.ROI_GRID_POOL
        )

        self.grid_size = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        self.num_c_out = num_c_out

        # 自适应扩张参数
        adaptive_cfg = getattr(self.model_cfg.ROI_GRID_POOL, 'ADAPTIVE_EXPAND', None)
        if adaptive_cfg is not None:
            self.use_adaptive_expand = adaptive_cfg.ENABLED
            self.base_scale = adaptive_cfg.BASE_SCALE
            self.min_scale = adaptive_cfg.MIN_SCALE
            self.max_scale = adaptive_cfg.MAX_SCALE
        else:
            self.use_adaptive_expand = False
            self.base_scale = 1.2
            self.min_scale = 1.1
            self.max_scale = 1.3

        # -------------------------
        # grid 分支
        # -------------------------
        pre_channel = self.grid_size * self.grid_size * self.grid_size * num_c_out
        shared_fc_list = []
        for k in range(len(self.model_cfg.SHARED_FC)):
            shared_fc_list.extend([
                nn.Conv1d(pre_channel, self.model_cfg.SHARED_FC[k], kernel_size=1, bias=False),
                nn.BatchNorm1d(self.model_cfg.SHARED_FC[k]),
                nn.ReLU()
            ])
            pre_channel = self.model_cfg.SHARED_FC[k]

            if k != len(self.model_cfg.SHARED_FC) - 1 and self.model_cfg.DP_RATIO > 0:
                shared_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))

        self.shared_fc_layer = nn.Sequential(*shared_fc_list)

        # -------------------------
        # corner 分支
        # 8 个 corner，每个 corner 一个 num_c_out 维特征
        # -------------------------
        corner_pre_channel = 8 * num_c_out
        corner_shared_fc_list = []
        for k in range(len(self.model_cfg.CORNER_SHARED_FC)):
            corner_shared_fc_list.extend([
                nn.Conv1d(corner_pre_channel, self.model_cfg.CORNER_SHARED_FC[k], kernel_size=1, bias=False),
                nn.BatchNorm1d(self.model_cfg.CORNER_SHARED_FC[k]),
                nn.ReLU()
            ])
            corner_pre_channel = self.model_cfg.CORNER_SHARED_FC[k]

            if k != len(self.model_cfg.CORNER_SHARED_FC) - 1 and self.model_cfg.CORNER_DP_RATIO > 0:
                corner_shared_fc_list.append(nn.Dropout(self.model_cfg.CORNER_DP_RATIO))

        self.corner_shared_fc_layer = nn.Sequential(*corner_shared_fc_list)

        combined_input_channel = corner_pre_channel + pre_channel

        self.cls_layers = self.make_fc_layers(
            input_channels=combined_input_channel,
            output_channels=self.num_class,
            fc_list=self.model_cfg.CLS_FC
        )
        self.reg_layers = self.make_fc_layers(
            input_channels=combined_input_channel,
            output_channels=self.box_coder.code_size * self.num_class,
            fc_list=self.model_cfg.REG_FC
        )
        self.init_weights(weight_init='xavier')

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        nn.init.normal_(self.reg_layers[-1].weight, mean=0, std=0.001)

    def _compute_adaptive_scales(self, rois, point_coords, point_cls_scores):
        """
        基于 point_cls_scores 计算每个 RoI 的前景点比例 a_i，
        再得到自适应扩张尺度:
            lambda_i = min_scale + (max_scale - min_scale) * a_i

        Args:
            rois: (B, N, 7 + C) 或至少前 7 维有效
            point_coords: (num_points, 4) [bs_idx, x, y, z]
            point_cls_scores: (num_points,)

        Returns:
            adaptive_scales: (B*N,)
        """
        batch_size, num_rois = rois.shape[:2]
        rois_reshaped = rois[..., :7].contiguous().view(-1, 7)

        # 用 base_scale=1.2 的参考扩张区域统计前景点比例
        ref_rois = rois_reshaped.clone()
        ref_rois[:, 3:6] = ref_rois[:, 3:6] * self.base_scale

        xyz = point_coords[:, 1:4]
        batch_idx = point_coords[:, 0].long()
        point_scores = point_cls_scores.view(-1)

        adaptive_scales = rois_reshaped.new_full((rois_reshaped.shape[0],), self.base_scale)

        for roi_idx in range(rois_reshaped.shape[0]):
            cur_roi = ref_rois[roi_idx]
            cur_batch = roi_idx // num_rois

            cur_points_mask = (batch_idx == cur_batch)
            cur_points = xyz[cur_points_mask]
            cur_scores = point_scores[cur_points_mask]

            if cur_points.shape[0] == 0:
                adaptive_scales[roi_idx] = self.base_scale
                continue

            # 变换到当前 roi 局部坐标系
            local_points = cur_points - cur_roi[0:3].unsqueeze(0)

            cosa = torch.cos(-cur_roi[6])
            sina = torch.sin(-cur_roi[6])
            rot_mat = cur_points.new_tensor([
                [cosa, -sina, 0.0],
                [sina,  cosa, 0.0],
                [0.0,   0.0,  1.0]
            ])
            local_points = torch.matmul(local_points, rot_mat.T)

            dx, dy, dz = cur_roi[3], cur_roi[4], cur_roi[5]
            inside_mask = (
                (local_points[:, 0].abs() <= dx / 2) &
                (local_points[:, 1].abs() <= dy / 2) &
                (local_points[:, 2].abs() <= dz / 2)
            )

            if inside_mask.sum() == 0:
                a_i = cur_scores.new_tensor(0.5)
            else:
                # a_i = 区域内前景置信度均值
                a_i = cur_scores[inside_mask].mean()

            adaptive_scales[roi_idx] = self.min_scale + (self.max_scale - self.min_scale) * a_i

        return adaptive_scales

    def roi_grid_pool(self, batch_dict):
        """
        Returns:
            pooled_features: (B*N, grid_size^3, C)
            pooled_features_corners: (B*N, 8, C)
            adaptive_scales: (B*N,)
        """
        batch_size = batch_dict['batch_size']
        rois = batch_dict['rois']
        point_coords = batch_dict['point_coords']
        point_features = batch_dict['point_features']

        # 前景置信度加权
        point_cls_scores = batch_dict['point_cls_scores'].view(-1)
        point_features = point_features * point_cls_scores.view(-1, 1)

        # 计算自适应扩张尺度
        if self.use_adaptive_expand:
            adaptive_scales = self._compute_adaptive_scales(
                rois=rois,
                point_coords=point_coords,
                point_cls_scores=point_cls_scores
            )
        else:
            adaptive_scales = rois.new_full((batch_size * rois.shape[1],), self.base_scale)

        # 原始 RoI 的规则网格点
        global_roi_grid_points, _ = self.get_global_grid_points_of_roi(
            rois, grid_size=self.grid_size
        )  # (B*N, grid_size^3, 3)
        global_roi_grid_points = global_roi_grid_points.view(batch_size, -1, 3)

        # 自适应扩张后的 corner points
        expanded_global_corners = self.get_expanded_corners_of_roi(
            rois, scales=adaptive_scales
        )  # (B*N, 8, 3)
        expanded_global_corners = expanded_global_corners.view(batch_size, -1, 3)

        xyz = point_coords[:, 1:4]
        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        batch_idx = point_coords[:, 0]

        for k in range(batch_size):
            xyz_batch_cnt[k] = (batch_idx == k).sum()

        # -------------------------
        # 1) grid pooling
        # -------------------------
        new_xyz = global_roi_grid_points.view(-1, 3)
        new_xyz_batch_cnt = xyz.new_zeros(batch_size).int().fill_(global_roi_grid_points.shape[1])

        _, pooled_features = self.roi_grid_pool_layer(
            xyz=xyz.contiguous(),
            xyz_batch_cnt=xyz_batch_cnt,
            new_xyz=new_xyz,
            new_xyz_batch_cnt=new_xyz_batch_cnt,
            features=point_features.contiguous(),
        )

        # -------------------------
        # 2) corner pooling
        # -------------------------
        new_xyz = expanded_global_corners.view(-1, 3)
        new_xyz_batch_cnt = xyz.new_zeros(batch_size).int().fill_(expanded_global_corners.shape[1])

        _, pooled_features_corners = self.roi_grid_pool_layer(
            xyz=xyz.contiguous(),
            xyz_batch_cnt=xyz_batch_cnt,
            new_xyz=new_xyz,
            new_xyz_batch_cnt=new_xyz_batch_cnt,
            features=point_features.contiguous(),
        )

        pooled_features = pooled_features.view(
            -1, self.grid_size ** 3, pooled_features.shape[-1]
        )  # (B*N, grid_size^3, C)

        pooled_features_corners = pooled_features_corners.view(
            -1, 8, pooled_features_corners.shape[-1]
        )  # (B*N, 8, C)

        return pooled_features, pooled_features_corners, adaptive_scales

    def get_global_grid_points_of_roi(self, rois, grid_size):
        rois = rois.view(-1, rois.shape[-1])
        batch_size_rcnn = rois.shape[0]

        local_roi_grid_points = self.get_dense_grid_points(rois, batch_size_rcnn, grid_size)
        global_roi_grid_points = common_utils.rotate_points_along_z(
            local_roi_grid_points.clone(), rois[:, 6]
        ).squeeze(dim=1)
        global_center = rois[:, 0:3].clone()
        global_roi_grid_points += global_center.unsqueeze(dim=1)
        return global_roi_grid_points, local_roi_grid_points

    @staticmethod
    def get_dense_grid_points(rois, batch_size_rcnn, grid_size):
        faked_features = rois.new_ones((grid_size, grid_size, grid_size))
        dense_idx = faked_features.nonzero()
        dense_idx = dense_idx.repeat(batch_size_rcnn, 1, 1).float()

        local_roi_size = rois.view(batch_size_rcnn, -1)[:, 3:6]
        roi_grid_points = (dense_idx + 0.5) / grid_size * local_roi_size.unsqueeze(dim=1) \
                          - (local_roi_size.unsqueeze(dim=1) / 2)
        return roi_grid_points

    def get_expanded_corners_of_roi(self, rois, scales):
        """
        Args:
            rois: (B, N, 7 + C)
            scales: (B*N,)
        Returns:
            expanded_global_corners: (B*N, 8, 3)
        """
        rois = rois.view(-1, rois.shape[-1])
        enlarged_rois = rois.clone()
        enlarged_rois[:, 3:6] = enlarged_rois[:, 3:6] * scales.view(-1, 1)

        corners_local = self.get_corner_grid_points(enlarged_rois)
        expanded_global_corners = common_utils.rotate_points_along_z(
            corners_local.clone(), enlarged_rois[:, 6]
        ).squeeze(dim=1)

        global_center = enlarged_rois[:, 0:3].clone()
        expanded_global_corners += global_center.unsqueeze(dim=1)

        return expanded_global_corners

    def get_corner_grid_points(self, rois):
        faked_features = rois.new_ones((2, 2, 2))
        dense_idx = faked_features.nonzero()  # (8, 3)

        batch_size_rcnn = rois.shape[0]
        dense_idx = dense_idx.repeat(batch_size_rcnn, 1, 1).float()  # (B, 8, 3)

        local_roi_size = rois.view(batch_size_rcnn, -1)[:, 3:6]
        corner_grid_points = (dense_idx + 0.5) / 2 * local_roi_size.unsqueeze(dim=1) \
                             - (local_roi_size.unsqueeze(dim=1) / 2)
        return corner_grid_points

    def forward(self, batch_dict):
        targets_dict = self.proposal_layer(
            batch_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
        )

        if self.training:
            targets_dict = batch_dict.get('roi_targets_dict', None)
            if targets_dict is None:
                targets_dict = self.assign_targets(batch_dict)
                batch_dict['rois'] = targets_dict['rois']
                batch_dict['roi_labels'] = targets_dict['roi_labels']

        # RoI aware pooling
        pooled_features, corner_pooled_features, adaptive_scales = self.roi_grid_pool(batch_dict)

        grid_size = self.grid_size
        batch_size_rcnn = pooled_features.shape[0]

        pooled_features = pooled_features.permute(0, 2, 1).contiguous().view(
            batch_size_rcnn, -1, 1
        )  # (B*N, grid_size^3*C, 1)

        corner_pooled_features = corner_pooled_features.permute(0, 2, 1).contiguous().view(
            batch_size_rcnn, -1, 1
        )  # (B*N, 8*C, 1)

        shared_features_corners = self.corner_shared_fc_layer(corner_pooled_features)
        shared_features_grids = self.shared_fc_layer(pooled_features)

        combined_features = torch.cat([shared_features_corners, shared_features_grids], dim=1)

        rcnn_cls = self.cls_layers(combined_features).transpose(1, 2).contiguous().squeeze(dim=1)
        rcnn_reg = self.reg_layers(combined_features).transpose(1, 2).contiguous().squeeze(dim=1)

        # 方便可视化或调试
        batch_dict['adaptive_expand_scales'] = adaptive_scales.view(batch_dict['batch_size'], -1)

        if not self.training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'],
                rois=batch_dict['rois'],
                cls_preds=rcnn_cls,
                box_preds=rcnn_reg
            )
            batch_dict['batch_cls_preds'] = batch_cls_preds
            batch_dict['batch_box_preds'] = batch_box_preds
            batch_dict['cls_preds_normalized'] = False
        else:
            targets_dict['rcnn_cls'] = rcnn_cls
            targets_dict['rcnn_reg'] = rcnn_reg
            self.forward_ret_dict = targets_dict

        return batch_dict
