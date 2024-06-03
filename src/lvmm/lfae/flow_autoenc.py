from typing import Dict

import torch.nn as nn

from .bg_motion_predictor import BGMotionPredictor
from .generator import Generator
from .region_predictor import RegionPredictor


class FlowAE(nn.Module):
    def __init__(self, 
                 num_regions, num_channels, revert_axis_swap, 
                 estimate_affine, 
                 generator_params: Dict = {},
                 region_predictor_params: Dict = {},
                 bg_predictor_params: Dict = {}):
        super(FlowAE, self).__init__()

        self.generator = Generator(
            num_regions=num_regions,
            num_channels=num_channels,
            revert_axis_swap=revert_axis_swap,
            **generator_params)
        self.region_predictor = RegionPredictor(
            num_regions=num_regions,
            num_channels=num_channels,
            estimate_affine=estimate_affine,
            **region_predictor_params).cuda()
        self.bg_predictor = BGMotionPredictor(
            num_channels=num_channels,
            **bg_predictor_params)

        self.ref_img = None
        self.dri_img = None
        self.generated = None

    def forward(self):
        source_region_params = self.region_predictor(self.ref_img)
        self.driving_region_params = self.region_predictor(self.dri_img)

        bg_params = self.bg_predictor(self.ref_img, self.dri_img)
        self.generated = self.generator(
            self.ref_img,
            source_region_params=source_region_params,
            driving_region_params=self.driving_region_params,
            bg_params=bg_params)
        self.generated.update({
            'source_region_params':
            source_region_params,
            'driving_region_params':
            self.driving_region_params
        })

    def set_train_input(self, ref_img, dri_img):
        self.ref_img = ref_img.cuda()
        self.dri_img = dri_img.cuda()
