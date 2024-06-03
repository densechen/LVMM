import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import grad
from torchvision import models

from .util import AntiAliasInterpolation2d, make_coordinate_grid


class Vgg19(nn.Module):
    """
    Vgg19 network for perceptual loss.
    """

    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        if os.path.exists("models/vgg19-dcbb9e9d.pth"):
            vgg = models.vgg19()
            vgg.load_state_dict(torch.load("models/vgg19-dcbb9e9d.pth", map_location="cpu"))
        else:
            vgg = models.vgg19(pretained=True)
        
        vgg_pretrained_features = vgg.features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        self.mean = nn.Parameter(data=torch.Tensor(
            np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))),
                                       requires_grad=False)
        self.std = nn.Parameter(data=torch.Tensor(
            np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))),
                                      requires_grad=False)

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = (x - self.mean) / self.std
        h_relu1 = self.slice1(x)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class ImagePyramide(nn.Module):
    """
    Create image pyramide for computing pyramide perceptual loss.
    """

    def __init__(self, scales, num_channels):
        super(ImagePyramide, self).__init__()
        downs = {}
        for scale in scales:
            downs[str(scale).replace('.', '-')] = AntiAliasInterpolation2d(
                num_channels, scale)
        self.downs = nn.ModuleDict(downs)

    def forward(self, x):
        out_dict = {}
        for scale, down_module in self.downs.items():
            out_dict['prediction_' +
                     str(scale).replace('-', '.')] = down_module(x)
        return out_dict


class Transform(object):
    """
    Random tps transformation for equivariance constraints.
    """

    def __init__(self, bs, **kwargs):
        noise = torch.normal(mean=0,
                             std=kwargs['sigma_affine'] *
                             torch.ones([bs, 2, 3]))
        self.theta = noise + torch.eye(2, 3).view(1, 2, 3)
        self.bs = bs

        if ('sigma_tps' in kwargs) and ('points_tps' in kwargs):
            self.tps = True
            self.control_points = make_coordinate_grid(
                (kwargs['points_tps'], kwargs['points_tps']),
                type=self.theta.type())
            self.control_points = self.control_points.unsqueeze(0)
            self.control_params = torch.normal(
                mean=0,
                std=kwargs['sigma_tps'] *
                torch.ones([bs, 1, kwargs['points_tps']**2]))
        else:
            self.tps = False

    def transform_frame(self, frame):
        grid = make_coordinate_grid(frame.shape[2:],
                                    type=frame.type()).unsqueeze(0)
        grid = grid.view(1, frame.shape[2] * frame.shape[3], 2)
        grid = self.warp_coordinates(grid).view(self.bs, frame.shape[2],
                                                frame.shape[3], 2)
        return F.grid_sample(frame, grid, padding_mode="reflection", align_corners=True)

    def warp_coordinates(self, coordinates):
        theta = self.theta.type(coordinates.type())
        theta = theta.unsqueeze(1)
        transformed = torch.matmul(
            theta[:, :, :, :2], coordinates.unsqueeze(-1)) + theta[:, :, :, 2:]
        transformed = transformed.squeeze(-1)

        if self.tps:
            control_points = self.control_points.type(coordinates.type())
            control_params = self.control_params.type(coordinates.type())
            distances = coordinates.view(coordinates.shape[0], -1, 1,
                                         2) - control_points.view(1, 1, -1, 2)
            distances = torch.abs(distances).sum(-1)

            # TODO this part may have bugs
            result = distances**2
            result = result * torch.log(distances + 1e-6)
            result = result * control_params
            result = result.sum(dim=2).view(self.bs, coordinates.shape[1], 1)
            transformed = transformed + result

        return transformed

    def jacobian(self, coordinates):
        new_coordinates = self.warp_coordinates(coordinates)
        grad_x = grad(new_coordinates[..., 0].sum(),
                      coordinates,
                      create_graph=True)
        grad_y = grad(new_coordinates[..., 1].sum(),
                      coordinates,
                      create_graph=True)
        jacobian = torch.cat(
            [grad_x[0].unsqueeze(-2), grad_y[0].unsqueeze(-2)], dim=-2)
        return jacobian


class ReconstructionModel(nn.Module):
    """
    Merge all updates into single model for better multi-gpu usage
    """

    def __init__(self, flow_ae, train_params):
        super(ReconstructionModel, self).__init__()
        self.flow_ae = flow_ae
        self.train_params = train_params
        self.scales = train_params['scales']
        self.pyramid = ImagePyramide(self.scales, self.flow_ae.generator.num_channels)
        self.loss_weights = train_params['loss_weights']

        if sum(self.loss_weights['perceptual']) != 0:
            self.vgg = Vgg19()
            self.vgg.requires_grad_(False)
            self.vgg.eval()

    def forward(self, x):
        source_region_params = self.flow_ae.region_predictor(x['source'])
        driving_region_params = self.flow_ae.region_predictor(x['driving'])

        bg_params = self.flow_ae.bg_predictor(x['source'], x['driving'])
        generated = self.flow_ae.generator(x['source'],
                                   source_region_params=source_region_params,
                                   driving_region_params=driving_region_params,
                                   bg_params=bg_params)
        generated.update({
            'source_region_params': source_region_params,
            'driving_region_params': driving_region_params
        })

        loss_values = {}

        pyramide_real = self.pyramid(x['driving'])
        pyramide_generated = self.pyramid(generated['prediction'])

        if sum(self.loss_weights['perceptual']) != 0:
            value_total = 0
            for scale in self.scales:
                x_vgg = self.vgg(pyramide_generated['prediction_' +
                                                    str(scale)])
                y_vgg = self.vgg(pyramide_real['prediction_' + str(scale)])

                for i, weight in enumerate(self.loss_weights['perceptual']):
                    value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
                    value_total += weight * value
            loss_values['perceptual'] = value_total

        if (self.loss_weights['equivariance_shift'] +
                self.loss_weights['equivariance_affine']) != 0:
            transform = Transform(x['driving'].shape[0],
                                  **self.train_params['transform_params'])
            transformed_frame = transform.transform_frame(x['driving'])
            transformed_region_params = self.flow_ae.region_predictor(
                transformed_frame)

            generated['transformed_frame'] = transformed_frame
            generated['transformed_region_params'] = transformed_region_params

            if self.loss_weights['equivariance_shift'] != 0:
                value = torch.abs(
                    driving_region_params['shift'] -
                    transform.warp_coordinates(
                        transformed_region_params['shift'])).mean()
                loss_values['equivariance_shift'] = self.loss_weights[
                    'equivariance_shift'] * value

            if self.loss_weights['equivariance_affine'] != 0:
                affine_transformed = torch.matmul(
                    transform.jacobian(transformed_region_params['shift']),
                    transformed_region_params['affine'])

                normed_driving = torch.inverse(driving_region_params['affine'])
                normed_transformed = affine_transformed
                value = torch.matmul(normed_driving, normed_transformed)
                eye = torch.eye(2).view(1, 1, 2, 2).type(value.type())

                if self.flow_ae.generator.pixelwise_flow_predictor.revert_axis_swap:
                    value = value * torch.sign(value[:, :, 0:1, 0:1])

                value = torch.abs(eye - value).mean()
                loss_values['equivariance_affine'] = self.loss_weights[
                    'equivariance_affine'] * value

        return loss_values, generated
