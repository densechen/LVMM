import os
import timeit
from argparse import ArgumentParser

import torch
import torch.distributed as dist
from einops import rearrange
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ggid.ldm import FlowDiffusion
from ggid.utils import instantiate_from_config, zero_rank_print


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(config, batch_size=2):
    # Initialize distributed training
    dist.init_process_group(backend="nccl")
    global_rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    num_processes = dist.get_world_size()
    is_main_process = global_rank == 0

    seed = config.global_seed + global_rank
    torch.manual_seed(seed)

    # Logging folder
    output_dir = config.output_dir

    # Handle the output folder creation
    if is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "tb"), exist_ok=True)
        OmegaConf.save(dict(config), os.path.join(output_dir, 'config.yaml'))
        writer = SummaryWriter(os.path.join(output_dir, "tb"))

    model = FlowDiffusion(
        flow_ae_params=config.flow_ae_params,
        unet_params=config.unet3d_params,
        embedder_params=config.embedder_params,
        fourier_params=config.fourier_params,
        **config.flow_diffusion_params,
    )

    train_params = config.train_params
    model.diffusion.train()
    optimizer = torch.optim.AdamW(model.diffusion.parameters(), lr=train_params.lr)

    if os.path.exists(config.unet3d_checkpoints):
        ckpt = torch.load(config.unet3d_checkpoints, map_location="cuda")
        start_step = ckpt["global_step"]
        model.unet.load_state_dict(ckpt["state_dict"])
        zero_rank_print(
            f"Restore from: {config.unet3d_checkpoints} at step {start_step}")
        for group in optimizer.param_groups:
            group["initial_lr"] = train_params.lr
    else:
        start_step = 0
        zero_rank_print(f"Train from scratch.")
        
        
    if not os.path.exists(config.flow_ae_checkpoints):
        raise RuntimeError(f"FlowAE weights missed.")
    else:
        ckpt = torch.load(config.flow_ae_checkpoints, map_location="cuda")
        model.flow_ae.load_state_dict(ckpt["state_dict"])
        zero_rank_print(f"Load FlowAE from: {config.flow_ae_checkpoints}")

    scheduler = CosineAnnealingLR(optimizer,
                                  eta_min=1e-6,
                                  last_epoch=start_step - 1,
                                  T_max=train_params.steps)
    if is_main_process:

        def state(name, model):
            zero_rank_print(f"### {name}")
            params = [p for p in model.parameters() if p.requires_grad]
            zero_rank_print(
                f"trainable params number: {len(params)}")
            zero_rank_print(
                f"trainable params scale: {sum(p.numel() for p in params) / 1e6:.3f} M"
            )

        state("Total", model)
        state("FlowAE", model.flow_ae)
        state("UNet3D", model.unet)
        state("Embedder", model.embedder)
        state("Fourier", model.fourier)

    dataset = instantiate_from_config(config.dataset_params)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=train_params['dataloader_workers'],
                            pin_memory=True,
                            drop_last=True)

    model.to(local_rank)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # rewritten by nhm
    batch_time = AverageMeter()
    data_time = AverageMeter()

    total_losses = AverageMeter()

    cnt = 0
    actual_step = start_step
    final_step = train_params.steps

    while actual_step < final_step:
        iter_end = timeit.default_timer()

        for i_iter, x in enumerate(dataloader):
            actual_step = int(start_step + cnt)
            data_time.update(timeit.default_timer() - iter_end)
            optimizer.zero_grad()

            # b c f h w
            x["frames"] = rearrange(x["frames"], "b f c h w -> b c f h w")
            ref_img = x["frames"][:, :, 0]
            real_vid = x["frames"]
            ref_text = x['txt']

            model.module.set_train_input(ref_img=ref_img, real_vid=real_vid, ref_text=ref_text)

            loss = model.module()
            loss.backward()
            
            clip_grad_norm_(model.module.unet.parameters(), max_norm=1.)

            optimizer.step()

            batch_time.update(timeit.default_timer() - iter_end)
            iter_end = timeit.default_timer()

            bs = len(ref_img)
            total_losses.update(loss, bs)

            if is_main_process:
                writer.add_scalar("loss", total_losses.val, global_step=actual_step)

            if actual_step % train_params[
                    "print_freq"] == 0 and is_main_process:

                print(
                    'iter: [{0}]{1}/{2}\t'
                    'loss {loss.val:.4f} ({loss.avg:.4f})'
                    .format(cnt,
                            actual_step,
                            final_step,
                            loss=total_losses))

            if actual_step % train_params[
                    "save_ckpt_freq"] == 0 and cnt != 0 and is_main_process:
                zero_rank_print('taking snapshot...')
                torch.save(
                    {
                        "global_step": actual_step,
                        "state_dict": model.module.unet.state_dict(),
                    },
                    os.path.join(
                        output_dir, "checkpoints",
                        'UNet3D_' + format(actual_step, "09d") + '.pth'))

            if actual_step % train_params[
                    "update_ckpt_freq"] == 0 and is_main_process:
                zero_rank_print('updating snapshot...')
                torch.save(
                    {
                        'global_step': actual_step,
                        "state_dict": model.module.unet.state_dict(),
                    }, os.path.join(output_dir, "checkpoints", 'UNet3D.pth'))

            if actual_step >= final_step:
                break

            cnt += 1

            scheduler.step()
    if is_main_process:
        zero_rank_print('save the final model...')
        torch.save(
            {
                "global_step": actual_step,
                "state_dict": model.module.unet.state_dict(),
            }, os.path.join(output_dir, "checkpoints", 'UNet3D.pth'))
        writer.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config", help="path to config")
    parser.add_argument("--batch_size",
                        default=4,
                        type=int,
                        help="batch size.")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)

    if not os.environ.get("WORLD_SIZE", None):
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "9010"

    train(config, batch_size=args.batch_size)
