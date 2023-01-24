# Copyright (c) OpenMMLab. All rights reserved.
import copy
import inspect
import math
import warnings

import cv2
import mmcv
import numpy as np
from numpy import random

from mmdet.core import BitmapMasks, PolygonMasks, find_inside_bboxes
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from mmdet.utils import log_img_scale
from ..builder import PIPELINES

try:
    from imagecorruptions import corrupt
except ImportError:
    corrupt = None


@PIPELINES.register_module()
class VerticalHalfCutMix:

    def __init__(self,
                 img_scale=(640, 640),
                 max_iters=15,
                 prob=0.5):
        # log_img_scale(img_scale, skip_square=True)
        assert isinstance(img_scale, tuple)
        assert 0 <= prob <= 1
        self.dynamic_scale = img_scale
        self.max_iters = max_iters
        self.prob = prob


    def __call__(self, results):
        """Call function to make a mixup of image.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Result dict with mixup transformed.
        """
        if random.random() < self.prob:
            results = self._vertical_cutmix_transform(results)
        return results

    def get_indexes(self, dataset):
        """Call function to collect indexes.

        Args:
            dataset (:obj:`MultiImageMixDataset`): The dataset.

        Returns:
            list: indexes.
        """

        for i in range(self.max_iters):
            index = random.randint(0, len(dataset))
            gt_bboxes_i = dataset.get_ann_info(index)['bboxes']
            if len(gt_bboxes_i) != 0:
                break

        return index

    def _vertical_cutmix_transform(self, results):
        """CutMix transform function.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Updated result dict.
        """
        assert 'mix_results' in results
        assert len(
            results['mix_results']) == 1, 'MixUp only support 2 images now !'

        if results['mix_results'][0]['gt_bboxes'].shape[0] == 0:
            # empty bbox
            return results
        

        img_2_results = results['mix_results'][0]
        img_2 = img_2_results['img']

        # resize img_2
        scale_ratio = min(self.dynamic_scale[0] / img_2.shape[0],
                          self.dynamic_scale[1] / img_2.shape[1])
        img_2 = mmcv.imresize(img_2, (self.dynamic_scale[1], self.dynamic_scale[0]))

        img_1 = results['img']

        img_1_bboxes = results['gt_bboxes']
        img_1_labels = results['gt_labels']
        img_2_bboxes = img_2_results['gt_bboxes']
        img_2_labels = img_2_results['gt_labels']

        # adjust img_2 bbox
        img_2_bboxes[:, 0::2] = img_2_bboxes[:, 0::2] * scale_ratio
        img_2_bboxes[:, 1::2] = img_2_bboxes[:, 1::2] * scale_ratio

        origin_h, origin_w = img_1.shape[:2]
        target_h, target_w = img_2.shape[:2]
        half_target_w = int(target_w / 2)

        # create empty img
        cutmix_img = np.zeros((origin_h, origin_w, 3)).astype(np.uint8)

        # choose cutmix position 
        cutmix_val = 0
        if np.random.randint(0, 2) == 1:
            cutmix_val = 1

        # cutmix
        if cutmix_val == 1:
            cutmix_img[:, 0:half_target_w] = img_1[:, 0:half_target_w]
            cutmix_img[:, half_target_w: ] = img_2[:, half_target_w: ]

            idxes_1, idxes_2 = (img_1_bboxes[:, 0] < half_target_w), (img_2_bboxes[:, 0] >= half_target_w)
            l_1, l_2 = img_1_labels[idxes_1], img_2_labels[idxes_2]
            b_1, b_2 = img_1_bboxes[idxes_1], img_2_bboxes[idxes_2]

        else:
            cutmix_img[:, 0:half_target_w] = img_2[:, 0:half_target_w]
            cutmix_img[:, half_target_w: ] = img_1[:, half_target_w: ]

            idxes_1, idxes_2 = (img_1_bboxes[:, 0] >= half_target_w), (img_2_bboxes[:, 0] < half_target_w)
            l_1, l_2 = img_1_labels[idxes_1], img_2_labels[idxes_2]
            b_1, b_2 = img_1_bboxes[idxes_1], img_2_bboxes[idxes_2]

        cutmix_labels = np.concatenate((l_1, l_2), axis=0)
        cutmix_bboxes = np.concatenate((b_1, b_2), axis=0)

        results['img'] = cutmix_img.astype(np.uint8)
        results['img_shape'] = cutmix_img.shape
        results['gt_bboxes'] = cutmix_bboxes
        results['gt_labels'] = cutmix_labels

        return results
        