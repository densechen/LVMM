import math

import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from skimage.draw import disk as circle


def draw_colored_heatmap(heatmap, colormap, bg_color):
    parts = []
    weights = []
    bg_color = np.array(bg_color).reshape((1, 1, 1, 3))
    num_regions = heatmap.shape[-1]
    for i in range(num_regions):
        color = np.array(colormap(i / num_regions))[:3]
        color = color.reshape((1, 1, 1, 3))
        part = heatmap[:, :, :, i:(i + 1)]
        part = part / np.max(part, axis=(1, 2), keepdims=True)
        weights.append(part)

        color_part = part * color
        parts.append(color_part)

    weight = sum(weights)
    bg_weight = 1 - np.minimum(1, weight)
    weight = np.maximum(1, weight)
    result = sum(parts) / weight + bg_weight * bg_color
    return result


class Visualizer(object):

    def __init__(self,
                 kp_size=5,
                 draw_border=False,
                 colormap='gist_rainbow',
                 region_bg_color=(0, 0, 0)):
        self.kp_size = kp_size
        self.draw_border = draw_border
        self.colormap = plt.get_cmap(colormap)
        self.region_bg_color = np.array(region_bg_color)

    def draw_image_with_kp(self, image, kp_array):
        image = np.copy(image)
        spatial_size = np.array(image.shape[:2][::-1])[np.newaxis]
        kp_array = spatial_size * (kp_array + 1) / 2
        num_regions = kp_array.shape[0]
        for kp_ind, kp in enumerate(kp_array):
            rr, cc = circle((kp[1], kp[0]),
                            self.kp_size,
                            shape=image.shape[:2])
            image[rr, cc] = np.array(self.colormap(kp_ind / num_regions))[:3]
        return image

    def create_image_column_with_kp(self, images, kp):
        image_array = np.array(
            [self.draw_image_with_kp(v, k) for v, k in zip(images, kp)])
        return self.create_image_column(image_array)

    def create_image_column(self, images):
        if self.draw_border:
            images = np.copy(images)
            images[:, :, [0, -1]] = (1, 1, 1)
            images[:, :, [0, -1]] = (1, 1, 1)
        return np.concatenate(list(images), axis=0)

    def create_image_grid(self, *args):
        out = []
        for arg in args:
            if type(arg) == tuple:
                out.append(self.create_image_column_with_kp(arg[0], arg[1]))
            else:
                out.append(self.create_image_column(arg))
        return np.concatenate(out, axis=1)

    @staticmethod
    def sample(x, index):
        return x[index].unsqueeze(dim=0).clone().detach()

    def visualize(self, driving, source, out, index=0):
        images = []

        # Source image with region centers
        source = self.sample(source, index)
        source = source.data.cpu()
        source_region_params = self.sample(
            out['source_region_params']['shift'], index)
        source_region_params = source_region_params.data.cpu().numpy()
        source = np.transpose(source, [0, 2, 3, 1])
        images.append((source, source_region_params))

        if 'heatmap' in out['source_region_params']:
            source_heatmap = self.sample(
                out['source_region_params']['heatmap'], index)
            source_heatmap = F.interpolate(source_heatmap,
                                           size=source.shape[1:3])
            source_heatmap = np.transpose(source_heatmap.data.cpu().numpy(),
                                          [0, 2, 3, 1])
            images.append(
                draw_colored_heatmap(source_heatmap, self.colormap,
                                     self.region_bg_color))

        # Deformed image
        if 'deformed' in out:
            deformed = self.sample(out['deformed'], index)
            deformed = deformed.data.cpu().numpy()
            deformed = np.transpose(deformed, [0, 2, 3, 1])
            images.append(deformed)

        # Equivariance visualization
        if 'transformed_frame' in out:
            transformed = self.sample(out['transformed_frame'], index)
            transformed = transformed.data.cpu().numpy()
            transformed = np.transpose(transformed, [0, 2, 3, 1])
            transformed_kp = self.sample(
                out['transformed_region_params']['shift'], index)
            transformed_kp = transformed_kp.data.cpu().numpy()
            images.append((transformed, transformed_kp))

        # Driving image with region centers
        driving_region_params = self.sample(
            out['driving_region_params']['shift'], index)
        driving_region_params = driving_region_params.data.cpu().numpy()
        driving = self.sample(driving, index)
        driving = driving.data.cpu().numpy()
        driving = np.transpose(driving, [0, 2, 3, 1])
        images.append((driving, driving_region_params))

        # Heatmaps visualizations
        if 'heatmap' in out['driving_region_params']:
            driving_heatmap = self.sample(
                out['driving_region_params']['heatmap'], index)
            driving_heatmap = F.interpolate(driving_heatmap,
                                            size=source.shape[1:3])
            driving_heatmap = np.transpose(driving_heatmap.data.cpu().numpy(),
                                           [0, 2, 3, 1])
            images.append(
                draw_colored_heatmap(driving_heatmap, self.colormap,
                                     self.region_bg_color))

        # Result
        prediction = self.sample(out['prediction'], index)
        prediction = prediction.data.cpu().numpy()
        prediction = np.transpose(prediction, [0, 2, 3, 1])
        images.append(prediction)

        # Occlusion map
        if 'occlusion_map' in out:
            occlusion_map = self.sample(out['occlusion_map'], index)
            occlusion_map = occlusion_map.data.cpu().repeat(1, 3, 1, 1)
            occlusion_map = F.interpolate(occlusion_map,
                                          size=source.shape[1:3]).numpy()
            occlusion_map = np.transpose(occlusion_map, [0, 2, 3, 1])
            images.append(occlusion_map)

        image = self.create_image_grid(*images)
        # reshape (1, 8) to (2, 4)
        H, W, _ = image.shape
        base = H
        num_image = W // H
        row, col = 2, math.ceil(num_image / 2)
        new_image = np.zeros((row * base, col * base, 3), dtype=np.float32)
        cnt = 0
        for ii in range(row):
            for jj in range(col):
                try:
                    new_image[ii * base:(ii + 1) * base, jj * base:(jj + 1) *
                              base, :] = image[:,
                                               cnt * base:(cnt + 1) * base, :]
                except:
                    pass
                cnt += 1
        new_image = (255 * new_image).astype(np.uint8)
        return new_image
