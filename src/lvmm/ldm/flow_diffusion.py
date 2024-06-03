import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ggid.utils import instantiate_from_config

from .gaussian_diffusion import GaussianDiffusion


class FlowDiffusion(nn.Module):
    def __init__(self,
                 flow_ae_params,
                 unet_params,
                 embedder_params,
                 fourier_params,
                 timesteps=1000,
                 sampling_timesteps=250,
                 null_cond_prob=0.1,
                 ddim_sampling_eta=1.,
                 latent_size=64,):
        super().__init__()

        self.flow_ae = instantiate_from_config(flow_ae_params)
        self.unet = instantiate_from_config(unet_params)
        self.embedder = instantiate_from_config(embedder_params)
        self.fourier = instantiate_from_config(fourier_params)

        self.diffusion = GaussianDiffusion(
            self.unet,
            sampling_timesteps=sampling_timesteps,
            timesteps=timesteps,  # number of steps
            loss_type='l2',  # L1 or L2
            null_cond_prob=null_cond_prob,
            ddim_sampling_eta=ddim_sampling_eta,
        )

        # frozen parameters
        self.flow_ae.requires_grad_(False)
        self.flow_ae.eval()
        self.embedder.requires_grad_(False)
        self.embedder.eval()
        self.fourier.requires_grad_(False)
        self.fourier.eval()

        self.num_frames = self.fourier.num_frames
        self.k_frequency = self.fourier.k_frequency
        self.channels = self.unet.channels

        self.latent_size = latent_size

    def forward(self):
        # compute pseudo ground-truth flow
        real_grid_list = []
        real_conf_list = []
        real_out_img_list = []
        real_warped_img_list = []
        with torch.no_grad():
            source_region_params = self.flow_ae.region_predictor(self.ref_img)
            for idx in range(self.num_frames):
                driving_region_params = self.flow_ae.region_predictor(
                    self.real_vid[:, :, idx, :, :])
                bg_params = self.flow_ae.bg_predictor(
                    self.ref_img,
                    self.real_vid[:, :, idx, :, :])
                generated = self.flow_ae.generator(
                    self.ref_img,
                    source_region_params=source_region_params,
                    driving_region_params=driving_region_params,
                    bg_params=bg_params)
                generated.update(
                    {'source_region_params': source_region_params,
                     'driving_region_params': driving_region_params}
                )
                real_grid_list.append(
                    generated["optical_flow"].permute(0, 3, 1, 2))
                # normalized occlusion map
                real_conf_list.append(generated["occlusion_map"])
                real_out_img_list.append(generated["prediction"])
                real_warped_img_list.append(generated["deformed"])
        self.real_vid_grid = torch.stack(real_grid_list, dim=2)
        self.real_vid_conf = torch.stack(real_conf_list, dim=2)
        self.real_out_vid = torch.stack(real_out_img_list, dim=2)
        self.real_warped_vid = torch.stack(real_warped_img_list, dim=2)
        # reference images are the same for different time steps, just pick the final one
        self.ref_img_fea = generated["bottle_neck_feat"].clone().detach()

        # shift occlusion to [0, 1]
        raw_x = torch.cat((self.real_vid_grid, self.real_vid_conf*2-1), dim=1)

        if raw_x.shape[-1] != self.latent_size:
            raw_x = self.interpolate(raw_x, self.latent_size)

        x = self.fourier.deconstruct(raw_x)

        return self.diffusion(x, self.cond)

    def interpolate(self, x, latent_size):
        """
        Args:
            x: batch size, channels, frames, height, width
        """
        num_frames = x.shape[2]
        x = rearrange(x, 'b c f h w -> (b f) c h w')
        y = F.interpolate(x, size=(latent_size, latent_size), mode='bilinear')
        return rearrange(y, "(b f) c h w -> b c f h w", f=num_frames)

    def sample_one_video(self, latent_size=None, cond_scale=1.0):
        """
        Args:
            shape: batch size, channels, num frames, height, width
        """
        batch_size = len(self.sample_img)
        shape = [batch_size, self.channels, self.k_frequency,
                 self.latent_size, self.latent_size]
        noise = torch.randn(shape, device=next(self.parameters()).device)

        # if cond_scale = 1.0, not using unconditional model
        pred = self.diffusion.sample(
            noise, cond=self.cond, cond_scale=cond_scale)

        # transform from fourier frequency domain to real domain
        pred = self.fourier.reconstruct(pred)

        if latent_size != self.latent_size:
            pred = self.interpolate(pred, latent_size)

        self.sample_vid_grid = pred[:, :2, :, :, :]
        self.sample_vid_conf = (pred[:, 2, :, :, :].unsqueeze(dim=1) + 1) * 0.5
        nf = self.sample_vid_grid.size(2)
        with torch.no_grad():
            sample_out_img_list = []
            sample_warped_img_list = []
            for idx in range(nf):
                sample_grid = self.sample_vid_grid[:, :, idx, :, :].permute(
                    0, 2, 3, 1)
                sample_conf = self.sample_vid_conf[:, :, idx, :, :]
                # predict fake out image and fake warped image
                generated = self.flow_ae.generator.forward_with_flow(
                    source_image=self.sample_img,
                    optical_flow=sample_grid,
                    occlusion_map=sample_conf)
                sample_out_img_list.append(generated["prediction"])
                sample_warped_img_list.append(generated["deformed"])
        self.sample_out_vid = torch.stack(sample_out_img_list, dim=2)
        self.sample_warped_vid = torch.stack(sample_warped_img_list, dim=2)

    def set_train_input(self, ref_img, real_vid, ref_text):
        """
        Args:
            ref_img: batch size, 3, width, height
        """
        self.ref_img = ref_img.cuda()
        self.real_vid = real_vid.cuda()
        with torch.no_grad():
            self.cond = self.embedder.encode(ref_img, ref_text)

    def set_sample_input(self, sample_img, sample_text):
        self.sample_img = sample_img.cuda()
        with torch.no_grad():
            self.cond = self.embedder.encode(sample_img, sample_text)

    def get_grid(self, b, nf, H, W, normalize=True):
        if normalize:
            h_range = torch.linspace(-1, 1, H)
            w_range = torch.linspace(-1, 1, W)
        else:
            h_range = torch.arange(0, H)
            w_range = torch.arange(0, W)
        grid = torch.stack(torch.meshgrid(
            [h_range, w_range]), -1).repeat(b, 1, 1, 1).flip(3).float()  # flip h,w to x,y
        return grid.permute(0, 3, 1, 2).unsqueeze(dim=2).repeat(1, 1, nf, 1, 1)


if __name__ == "__main__":
    from omegaconf import OmegaConf
    config = OmegaConf.load("configs/train_lfdm.yaml")

    model = FlowDiffusion(
        flow_ae_params=config.flow_ae_params,
        unet_params=config.unet3d_params,
        embedder_params=config.embedder_params,
        fourier_params=config.fourier_params,
        **config.flow_diffusion_params)
    bs = 5
    channels = 3
    img_size = 256
    num_frames = 40
    ref_text = ["play basketball"] * bs
    ref_img = torch.rand(
        (bs, channels, img_size, img_size), dtype=torch.float32)
    real_vid = torch.rand(
        (bs, channels, num_frames, img_size, img_size), dtype=torch.float32)

    model.cuda()

    # training

    model.diffusion.train()
    optimizer_diff = torch.optim.Adam(model.diffusion.parameters(), lr=1e-4)
    model.set_train_input(
        ref_img=ref_img, real_vid=real_vid, ref_text=ref_text)
    optimizer_diff.zero_grad()
    loss = model()
    loss.backward()
    optimizer_diff.step()
    print(loss.item())

    # model.eval()
    # model.set_sample_input(sample_img=ref_img, sample_text=ref_text)
    # model.sample_one_video(cond_scale=1.0)
