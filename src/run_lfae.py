import os
import timeit
from argparse import ArgumentParser

import imageio
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from lvmm.lfae import ReconstructionModel
from lvmm.utils import instantiate_from_config, zero_rank_print
from lvmm.utils.visualizer import Visualizer


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
        os.makedirs(os.path.join(output_dir, "imgshots"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "tb"), exist_ok=True)
        OmegaConf.save(dict(config), os.path.join(output_dir, 'config.yaml'))
        writer = SummaryWriter(os.path.join(output_dir, "tb"))

    flow_ae = instantiate_from_config(config.flow_ae_params)
    flow_ae.requires_grad_(True)

    train_params = config.train_params
    optimizer = torch.optim.AdamW(flow_ae.parameters(), lr=train_params.lr)

    if os.path.exists(config.flow_ae_checkpoints):
        ckpt = torch.load(config.flow_ae_checkpoints, map_location="cuda")
        start_step = ckpt["global_step"]
        m, u = flow_ae.load_state_dict(ckpt["state_dict"], strict=False)
        if len(m):
            print("Missing keys", m)
        if len(u):
            print("Unexpected keys", u)
        zero_rank_print(
            f"Restore from: {config.flow_ae_checkpoints} at step {start_step}")
        for group in optimizer.param_groups:
            group["initial_lr"] = train_params.lr
    else:
        start_step = 0
        zero_rank_print(f"Train from scratch.")

    scheduler = CosineAnnealingLR(optimizer,
                                  eta_min=1e-6,
                                  last_epoch=start_step - 1,
                                  T_max=train_params.steps)
    model = ReconstructionModel(flow_ae, config.train_params)

    if is_main_process:

        def state(name, model):
            zero_rank_print(f"### {name}")
            params = [p for p in model.parameters() if p.requires_grad]
            zero_rank_print(
                f"trainable params number: {len(params)}")
            zero_rank_print(
                f"trainable params scale: {sum(p.numel() for p in params) / 1e6:.3f} M"
            )

        state("Total", flow_ae)
        state("generator", flow_ae.generator)
        state("region_predictor", flow_ae.region_predictor)
        state("bg_predictor", flow_ae.bg_predictor)

    dataset = instantiate_from_config(config.dataset_params)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=train_params['dataloader_workers'],
                            pin_memory=True,
                            drop_last=True)

    visualizer = Visualizer(**config['visualizer_params'])

    model.to(local_rank)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    # rewritten by nhm
    batch_time = AverageMeter()
    data_time = AverageMeter()

    total_losses = AverageMeter()
    losses_perc = AverageMeter()
    losses_equiv_shift = AverageMeter()
    losses_equiv_affine = AverageMeter()

    cnt = 0
    actual_step = start_step
    final_step = train_params.steps

    while actual_step < final_step:
        iter_end = timeit.default_timer()

        for i_iter, x in enumerate(dataloader):
            actual_step = int(start_step + cnt)
            data_time.update(timeit.default_timer() - iter_end)
            optimizer.zero_grad()

            x["source"], x["driving"] = x["frames"].unbind(1)
            losses, generated = model(x)
            loss_values = [val.mean() for val in losses.values()]
            loss = sum(loss_values)
            loss.backward()
            
            clip_grad_norm_(model.parameters(), max_norm=1.)

            optimizer.step()

            batch_time.update(timeit.default_timer() - iter_end)
            iter_end = timeit.default_timer()

            bs = x['source'].size(0)
            total_losses.update(loss, bs)
            losses_perc.update(loss_values[0], bs)
            losses_equiv_shift.update(loss_values[1], bs)
            losses_equiv_affine.update(loss_values[2], bs)


            if is_main_process:
                writer.add_scalars(
                    "train",
                    tag_scalar_dict=dict(loss=total_losses.val,
                                         loss_perc=losses_perc.val,
                                         loss_shift=losses_equiv_shift.val,
                                         loss_affine=losses_equiv_affine.val),
                    global_step=actual_step,
                )

            if actual_step % train_params[
                    "print_freq"] == 0 and is_main_process:

                print(
                    'iter: [{0}]{1}/{2}\t'
                    'loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'loss_perc {loss_perc.val:.4f} ({loss_perc.avg:.4f})\n'
                    'loss_shift {loss_shift.val:.4f} ({loss_shift.avg:.4f})\t'
                    'loss_affine {loss_affine.val:.4f} ({loss_affine.avg:.4f})'
                    .format(cnt,
                            actual_step,
                            final_step,
                            loss=total_losses,
                            loss_perc=losses_perc,
                            loss_shift=losses_equiv_shift,
                            loss_affine=losses_equiv_affine))

            if actual_step % train_params[
                    'save_img_freq'] == 0 and is_main_process:
                save_image = visualizer.visualize(x['driving'],
                                                  x['source'],
                                                  generated,
                                                  index=0)
                save_file = os.path.join(output_dir, "imgshots",
                                         f"{actual_step:09d}.png")
                imageio.imsave(save_file, save_image)

            if actual_step % train_params[
                    "save_ckpt_freq"] == 0 and cnt != 0 and is_main_process:
                zero_rank_print('taking snapshot...')
                torch.save(
                    {
                        "global_step": actual_step,
                        "state_dict": flow_ae.state_dict(),
                    },
                    os.path.join(
                        output_dir, "checkpoints",
                        'RegionMM_' + format(actual_step, "09d") + '.pth'))

            if actual_step % train_params[
                    "update_ckpt_freq"] == 0 and is_main_process:
                zero_rank_print('updating snapshot...')
                torch.save(
                    {
                        'global_step': actual_step,
                        "state_dict": flow_ae.state_dict(),
                    }, os.path.join(output_dir, "checkpoints", 'RegionMM.pth'))

            if actual_step >= final_step:
                break

            cnt += 1

            scheduler.step()
    if is_main_process:
        zero_rank_print('save the final model...')
        torch.save(
            {
                "global_step": actual_step,
                "state_dict": flow_ae.state_dict(),
            }, os.path.join(output_dir, "checkpoints", 'RegionMM.pth'))
        writer.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config", help="path to config")
    parser.add_argument("--batch_size",
                        default=1,
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
