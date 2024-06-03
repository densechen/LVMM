import torch
import torch.nn as nn
from einops import rearrange, repeat

from ggid.utils import instantiate_from_config

from .gaussian_diffusion import GaussianDiffusion


def expand_temporal(x):
    return rearrange(x, "b c h w -> b c 1 h w")


def squeeze_temporal(x):
    return rearrange(x, "b c 1 h w -> b c h w")


class FrameDiffusion(nn.Module):
    def __init__(self,
                 flow_ae_params,
                 unet_params,
                 fourier_params,
                 timesteps=1000,
                 sampling_timesteps=50,
                 null_cond_prob=0.1,
                 ddim_sampling_eta=1.,
                 mini_batch=4,
                 ):
        super().__init__()

        self.flow_ae = instantiate_from_config(flow_ae_params)
        self.unet = instantiate_from_config(unet_params)
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
        self.fourier.requires_grad_(False)
        self.fourier.eval()

        self.num_frames = self.fourier.num_frames
        self.k_frequency = self.fourier.k_frequency
        self.channels = self.unet.channels
        self.mini_batch = mini_batch

    def sample(self, warp_ref_feat_k, dri_feat_k_m_1, index, cond_scale=1.0):
        """
        Args:
            shape: batch size, channels, width, height
        """
        batch_size = len(self.ref_img)
        latent_size = self.ref_img.shape[-1] // 4
        # The shape of noise is halved.
        shape = [batch_size, self.channels // 2, 1, latent_size, latent_size]
        device = next(self.parameters()).device
        noise = torch.randn(shape, device=device)
        # expand_temporal
        # if cond_scale = 1.0, not using unconditional model
        pred = self.diffusion.sample(
            noise,
            cond=dict(
                cat=expand_temporal(warp_ref_feat_k), 
                cross=rearrange(dri_feat_k_m_1, "b c h w -> b (h w) c"),
                seq=torch.tensor(index,device=device).repeat(batch_size),
            ),
            cond_scale=cond_scale)

        return squeeze_temporal(pred)

    @torch.no_grad()
    def prepare_training_data(self):
        nf = self.sample_vid_grid.size(2)
        with torch.no_grad():
            sample_vid_grid = rearrange(
                self.sample_vid_grid, "b c f h w -> (b f) h w c")
            sample_vid_conf = rearrange(
                self.sample_vid_conf, "b c f h w -> (b f) c h w")
            ref_img_fea = repeat(
                self.ref_img_fea, "b c h w -> (b f) c h w", f=nf)

            warp_ref_feat = self.flow_ae.generator.apply_optical(
                input_previous=None,
                input_skip=ref_img_fea,
                # occlusion_map is not used actually.
                motion_params={"optical_flow": sample_vid_grid,
                               "occlusion_map":  sample_vid_conf}
            )
            # dri_feat with the shape of (b f) c h w
            dri_feat_list = []
            for idx in range(nf):
                dri_feat = self.flow_ae.generator.compute_fea(
                    self.real_vid[:, :, idx])
                dri_feat_list.append(dri_feat)
            dri_feat = rearrange(torch.stack(
                dri_feat_list, dim=2), "b c f h w -> (b f) c h w")
        self.dri_feat = dri_feat
        self.warp_ref_feat = warp_ref_feat

    def forward(self):
        dri_feat, warp_ref_feat = self.dri_feat, self.warp_ref_feat
        # Make sure all index >= 1
        index = torch.randperm(
            len(dri_feat) - 1, device=dri_feat.device)[:self.mini_batch] + 1
        dri_feat_k = expand_temporal(
            torch.index_select(dri_feat, dim=0, index=index))
        warp_ref_feat_k = expand_temporal(
            torch.index_select(warp_ref_feat, dim=0, index=index))
        dri_feat_k_m_1 = torch.index_select(dri_feat, dim=0, index=index-1)

        dri_feat_k_m_1 = rearrange(dri_feat_k_m_1, "b c h w -> b (h w) c")

        # warp_ref_feat will concat
        # dri_feat[k] is ground truth, i.e., target
        # dri_feat[k-1] is cross condition
        # warp_ref_feat[k] is cat condition
        return self.diffusion(x=dri_feat_k, cond=dict(cat=warp_ref_feat_k, cross=dri_feat_k_m_1, seq=index))

    def clear_cache(self):

        del self.real_vid, self.ref_img
        del self.real_vid_grid, self.real_vid_conf
        del self.real_out_vid, self.real_warped_vid, self.ref_img_fea
        del self.sample_vid_grid, self.sample_vid_conf

        torch.cuda.empty_cache()
        
    def clear_feat(self):
        del self.dri_feat, self.warp_ref_feat
        torch.cuda.empty_cache()

    @torch.no_grad()
    def preprocess(self, debug=False):
        # Prepare training data
        # compute pseudo ground-truth flow
        real_grid_list = []
        real_conf_list = []
        real_out_img_list = []
        real_warped_img_list = []
        with torch.no_grad():
            source_region_params = self.flow_ae.region_predictor(self.ref_img)
            nf = self.real_vid.shape[2]
            for idx in range(nf):
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

        if debug:
            print("Fourier Estimator Loss", self.fourier.loss(raw_x).item())

        x = self.fourier.deconstruct(raw_x)
        inv_x = self.fourier.reconstruct(x)

        self.sample_vid_grid = inv_x[:, :2, :, :, :]
        self.sample_vid_conf = (
            inv_x[:, 2, :, :, :].unsqueeze(dim=1) + 1) * 0.5

    def set_input(self, ref_img, real_vid, debug=False):
        """
        Args:
            ref_img: batch size, 3, width, height
            real_vid: batch size, 3, num frames, width, height
        """
        self.ref_img = ref_img.cuda()
        self.real_vid = real_vid.cuda()

        self.preprocess(debug=debug)

    def sample_one_video(self, cond_scale=1.):
        nf = self.sample_vid_grid.size(2)
        with torch.no_grad():
            sample_out_img_list = []
            sample_warped_img_list = []
            # set first dri_feat_k_m_1 as ref img's feature
            dri_feat_k_m_1 = self.flow_ae.generator.compute_fea(self.ref_img)
            for idx in range(nf):
                sample_grid = self.sample_vid_grid[:, :, idx, :, :].permute(
                    0, 2, 3, 1)
                sample_conf = self.sample_vid_conf[:, :, idx, :, :]

                # predict fake out image and fake warped image
                generated = self.flow_ae.generator.forward_with_flow(
                    source_image=self.ref_img,
                    optical_flow=sample_grid,
                    occlusion_map=sample_conf,
                    cinns_hook=lambda warp_ref_feat_k: self.sample(warp_ref_feat_k, dri_feat_k_m_1=dri_feat_k_m_1, index=idx, cond_scale=cond_scale))
                sample_out_img_list.append(generated["prediction"])
                sample_warped_img_list.append(generated["deformed"])
                dri_feat_k_m_1 = generated["deformed_feat"]

        self.clear_cache()

        self.sample_out_vid = torch.stack(sample_out_img_list, dim=2)
        self.sample_warped_vid = torch.stack(sample_warped_img_list, dim=2)


if __name__ == "__main__":
    from omegaconf import OmegaConf
    config = OmegaConf.load("configs/train_fdm.yaml")

    model = FrameDiffusion(
        flow_ae_params=config.flow_ae_params,
        unet_params=config.unet3d_params,
        fourier_params=config.fourier_params,
        **config.frame_diffusion_params)
    bs = 5
    channels = 3
    img_size = 256
    num_frames = 150
    ref_text = ["play basketball"] * bs
    ref_img = torch.rand(
        (bs, channels, img_size, img_size), dtype=torch.float32)
    real_vid = torch.rand(
        (bs, channels, num_frames, img_size, img_size), dtype=torch.float32)

    model.cuda()

    # training

    model.diffusion.train()
    optimizer_flow = torch.optim.Adam(model.diffusion.parameters(), lr=1e-5)
    model.set_input(
        ref_img=ref_img, real_vid=real_vid)
    model.prepare_training_data()
    model.clear_cache()

    for _ in range(100):
        optimizer_flow.zero_grad()
        loss = model()
        loss.backward()
        from torch.nn.utils.clip_grad import clip_grad_norm_
        clip_grad_norm_(model.diffusion.parameters(), 1.0)
        optimizer_flow.step()
        print(loss.item())
        del loss
        torch.cuda.empty_cache()

    # model.eval()
    # model.set_input(ref_img=ref_img, real_vid=real_vid)
    # model.sample_one_video()
