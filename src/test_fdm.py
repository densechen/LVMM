import os
from argparse import ArgumentParser

import torch
import torch.distributed as dist
from einops import rearrange
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import trange

from ggid.ldm import FrameDiffusion
from ggid.utils import (instantiate_from_config, video_tensor_to_gif,
                        zero_rank_print)


def main(config, batch_size=2, debug_fourier=False):
    # Initialize distributed training
    dist.init_process_group(backend="nccl")
    global_rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    num_processes = dist.get_world_size()
    is_main_process = global_rank == 0

    torch.manual_seed(config.global_seed)

    # Logging folder
    output_dir = config.output_dir

    # Handle the output folder creation
    os.makedirs(output_dir, exist_ok=True)

    model = FrameDiffusion(
        flow_ae_params=config.flow_ae_params,
        unet_params=config.unet3d_params,
        fourier_params=config.fourier_params,
        **config.frame_diffusion_params,
    )

    model.eval()

    if not os.path.exists(config.unet3d_checkpoints):
        raise FileNotFoundError(f"UNet3D weighs missed.")
    else:
        ckpt = torch.load(config.unet3d_checkpoints, map_location="cuda")
        start_step = ckpt["global_step"]
        model.unet.load_state_dict(ckpt["state_dict"])
        zero_rank_print(
            f"Load UNet3D from: {config.unet3d_checkpoints} at step {start_step}")

    if not os.path.exists(config.flow_ae_checkpoints):
        raise RuntimeError(f"FlowAE weights missed.")
    else:
        ckpt = torch.load(config.flow_ae_checkpoints, map_location="cuda")
        model.flow_ae.load_state_dict(ckpt["state_dict"])
        zero_rank_print(f"Load FlowAE from: {config.flow_ae_checkpoints}")

    dataset = instantiate_from_config(config.dataset_params)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=1,
                            pin_memory=True,
                            drop_last=True)

    model.to(local_rank)

    process_bar = trange(config.samples)
    for i_iter, x in enumerate(dataloader):
        process_bar.update()
        if process_bar.n > config.samples:
            break

        # b c f h w
        x["frames"] = rearrange(x["frames"], "b f c h w -> b c f h w")
        ref_img = x["frames"][:, :, 0]
        real_vid = x["frames"]

        model.set_input(ref_img=ref_img, real_vid=real_vid, debug=debug_fourier)
        
        if debug_fourier:
            continue

        with torch.no_grad():
            model.sample_one_video()
        
        sample_out_vid = model.sample_out_vid
        sample_warped_vid = model.sample_warped_vid

        # b c f h w
        video = torch.cat(
            [real_vid.cpu(), sample_out_vid.cpu(), sample_warped_vid.cpu()], dim=-1)
        # video = video * 255

        for idx, tensor in enumerate(video):
            video_name = "%04d_%s.gif" % (
                i_iter, x["txt"][idx].replace("/", '')[:20])
            video_tensor_to_gif(
                tensor, path=os.path.join(output_dir, video_name))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", help="path to config",
                        default="configs/test_fdm.yaml")
    parser.add_argument("--batch_size",
                        default=4,
                        type=int,
                        help="batch size.")
    parser.add_argument("--debug_fourier", action="store_true", default=False, help="print Fourier Estimator Loss ONLY.")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)

    if not os.environ.get("WORLD_SIZE", None):
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "9010"

    main(config, batch_size=args.batch_size, debug_fourier=args.debug_fourier)
