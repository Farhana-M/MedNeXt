#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import numpy as np
from batchgenerators.transforms.abstract_transforms import AbstractTransform


class RemoveKeyTransform(AbstractTransform):
    def __init__(self, key_to_remove):
        self.key_to_remove = key_to_remove

    def __call__(self, **data_dict):
        _ = data_dict.pop(self.key_to_remove, None)
        return data_dict


class MaskTransform(AbstractTransform):
    def __init__(self, dct_for_where_it_was_used, mask_idx_in_seg=1, set_outside_to=0, data_key="data", seg_key="seg"):
        """
        data[mask < 0] = 0
        Sets everything outside the mask to 0. CAREFUL! outside is defined as < 0, not =0 (in the Mask)!!!

        :param dct_for_where_it_was_used:
        :param mask_idx_in_seg:
        :param set_outside_to:
        :param data_key:
        :param seg_key:
        """
        self.dct_for_where_it_was_used = dct_for_where_it_was_used
        self.seg_key = seg_key
        self.data_key = data_key
        self.set_outside_to = set_outside_to
        self.mask_idx_in_seg = mask_idx_in_seg

    def __call__(self, **data_dict):
        seg = data_dict.get(self.seg_key)
        if seg is None or seg.shape[1] < self.mask_idx_in_seg:
            raise Warning("mask not found, seg may be missing or seg[:, mask_idx_in_seg] may not exist")
        data = data_dict.get(self.data_key)
        for b in range(data.shape[0]):
            mask = seg[b, self.mask_idx_in_seg]
            for c in range(data.shape[1]):
                if self.dct_for_where_it_was_used[c]:
                    data[b, c][mask < 0] = self.set_outside_to
        data_dict[self.data_key] = data
        return data_dict


def convert_3d_to_2d_generator(data_dict):
    shp = data_dict['data'].shape
    data_dict['data'] = data_dict['data'].reshape((shp[0], shp[1] * shp[2], shp[3], shp[4]))
    data_dict['orig_shape_data'] = shp
    shp = data_dict['seg'].shape
    data_dict['seg'] = data_dict['seg'].reshape((shp[0], shp[1] * shp[2], shp[3], shp[4]))
    data_dict['orig_shape_seg'] = shp
    return data_dict


def convert_2d_to_3d_generator(data_dict):
    shp = data_dict['orig_shape_data']
    current_shape = data_dict['data'].shape
    data_dict['data'] = data_dict['data'].reshape((shp[0], shp[1], shp[2], current_shape[-2], current_shape[-1]))
    shp = data_dict['orig_shape_seg']
    current_shape_seg = data_dict['seg'].shape
    data_dict['seg'] = data_dict['seg'].reshape((shp[0], shp[1], shp[2], current_shape_seg[-2], current_shape_seg[-1]))
    return data_dict


class Convert3DTo2DTransform(AbstractTransform):
    def __init__(self):
        pass

    def __call__(self, **data_dict):
        return convert_3d_to_2d_generator(data_dict)


class Convert2DTo3DTransform(AbstractTransform):
    def __init__(self):
        pass

    def __call__(self, **data_dict):
        return convert_2d_to_3d_generator(data_dict)


class ConvertSegmentationToRegionsTransform(AbstractTransform):
    def __init__(self, regions: dict, seg_key: str = "seg", output_key: str = "seg", seg_channel: int = 0):
        """
        regions are tuple of tuples where each inner tuple holds the class indices that are merged into one region, example:
        regions= ((1, 2), (2, )) will result in 2 regions: one covering the region of labels 1&2 and the other just 2
        :param regions:
        :param seg_key:
        :param output_key:
        """
        self.seg_channel = seg_channel
        self.output_key = output_key
        self.seg_key = seg_key
        self.regions = regions

    def __call__(self, **data_dict):
        seg = data_dict.get(self.seg_key)
        num_regions = len(self.regions)
        if seg is not None:
            seg_shp = seg.shape
            output_shape = list(seg_shp)
            output_shape[1] = num_regions
            region_output = np.zeros(output_shape, dtype=seg.dtype)
            for b in range(seg_shp[0]):
                for r, k in enumerate(self.regions.keys()):
                    for l in self.regions[k]:
                        region_output[b, r][seg[b, self.seg_channel] == l] = 1
            data_dict[self.output_key] = region_output
        return data_dict

class RandomBiasFieldTransform(AbstractTransform):
    """
    Simulate MRI bias field artifact as described by Leemput et al. (1999).
    Adds a smooth multiplicative field to simulate intensity non-uniformity.

    Args:
        order (int): Order of the polynomial basis.
        coeff_range (tuple): Range of random coefficients (min, max).
        p_per_sample (float): Probability of applying the transform to a sample.
    """
    def __init__(self, order=3, coeff_range=(0.3, 0.8), p_per_sample=0.25):
        self.order = self._validate_order(order)
        self.coeff_range = coeff_range
        self.p_per_sample = p_per_sample

    def __call__(self, **data_dict):
        if np.random.rand() >= self.p_per_sample:
            return data_dict

        image = data_dict['data']
        shape = image.shape[1:]  # exclude channel dim (assume shape is (C, X, Y, Z))

        bias_field = self.generate_bias_field(shape, self.order, self.coeff_range)
        bias_field = np.exp(bias_field).astype(np.float32)

        # Apply bias to all channels
        data_dict['data'] *= bias_field[np.newaxis, ...]
        return data_dict

    def generate_bias_field(self, shape, order, coeff_range):
        coords = [np.linspace(-1, 1, s, dtype=np.float32) for s in shape]
        mesh = np.meshgrid(*coords, indexing='ij')
        bias_field = np.zeros(shape, dtype=np.float32)

        for xo in range(order + 1):
            for yo in range(order + 1 - xo):
                for zo in range(order + 1 - (xo + yo)):
                    coeff = np.random.uniform(*coeff_range)
                    term = coeff * (mesh[0] ** xo) * (mesh[1] ** yo) * (mesh[2] ** zo)
                    bias_field += term

        return bias_field

    def _validate_order(self, order):
        if not isinstance(order, int) or order < 0:
            raise ValueError("Order must be a non-negative integer.")
        return order
