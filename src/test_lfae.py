import argparse
import os

import imageio
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import trange

from lvmm.utils import instantiate_from_config, zero_rank_print
from lvmm.utils.misc import grid2fig


def sample_img(rec_img_batch, index):
    rec_img = rec_img_batch[index].permute(1, 2, 0).data.cpu().numpy().copy()
    rec_img[rec_img < 0] = 0
    rec_img[rec_img > 1] = 1
    rec_img *= 255
    return np.array(rec_img, np.uint8)


def main(config, batch_size):
    # Initialize distributed training
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    
    torch.manual_seed(config.global_seed)
    
    flow_ae = instantiate_from_config(config.flow_ae_params).cuda()

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
    else:
        raise FileNotFoundError
    flow_ae.eval()

    dataset = instantiate_from_config(config.dataset_params)
    testloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=1,
                            pin_memory=True,
                            drop_last=True)
    if os.path.exists(config.output_dir):
        import shutil
        shutil.rmtree(config.output_dir, ignore_errors=True)
    os.makedirs(config.output_dir, exist_ok=True)
    process_bar = trange(config.samples)
    for idx, batch in enumerate(testloader):
        process_bar.update()
        if idx >= config.samples:
            break

        real_vids = rearrange(batch['frames'], "b f c h w -> b c f h w")
        # use first frame of each video as reference frame
        ref_imgs = real_vids[:, :, 0, :, :].clone().detach()
        bs = real_vids.size(0)

        nf = real_vids.size(2)
        out_img_list = []
        warped_img_list = []
        warped_grid_list = []
        # conf_map_list = []
        for frame_idx in trange(nf, leave=False, desc="Sampling"):
            dri_imgs = real_vids[:, :, frame_idx, :, :]
            with torch.no_grad():
                flow_ae.set_train_input(ref_img=ref_imgs, dri_img=dri_imgs)
                flow_ae.forward()
            out_img_list.append(
                flow_ae.generated['prediction'].clone().detach())
            warped_img_list.append(
                flow_ae.generated['deformed'].clone().detach())
            warped_grid_list.append(
                flow_ae.generated['optical_flow'].clone().detach())
            # conf_map_list.append(
            #     flow_ae.generated['occlusion_map'].clone().detach())

        out_img_list_tensor = torch.stack(out_img_list, dim=0)
        warped_img_list_tensor = torch.stack(warped_img_list, dim=0)
        warped_grid_list_tensor = torch.stack(warped_grid_list, dim=0)
        # conf_map_list_tensor = torch.stack(conf_map_list, dim=0)

        for batch_idx in trange(bs, leave=False, desc="Saving"):
            msk_size = ref_imgs.shape[-1]
            new_im_list = []
            for frame_idx in range(nf):
                save_tar_img = sample_img(real_vids[:, :, frame_idx],
                                            batch_idx)
                save_out_img = sample_img(
                    out_img_list_tensor[frame_idx], batch_idx) 
                save_warped_img = sample_img(
                    warped_img_list_tensor[frame_idx], batch_idx)
                save_warped_grid = grid2fig(warped_grid_list_tensor[
                    frame_idx, batch_idx].data.cpu().numpy(),
                                            grid_size=32,
                                            img_size=msk_size)
                # save_conf_map = conf_map_list_tensor[
                #     frame_idx, batch_idx].unsqueeze(dim=0)
                # save_conf_map = save_conf_map.data.cpu()
                # save_conf_map = F.interpolate(
                #     save_conf_map, size=real_vids.shape[3:5]).numpy()
                # save_conf_map = np.transpose(save_conf_map,
                #                                 [0, 2, 3, 1])
                # save_conf_map = np.array(save_conf_map[0, :, :, 0] *
                #                             255,
                #                             dtype=np.uint8)
                new_im = Image.new('RGB', (msk_size * 5, msk_size))
                new_im.paste(Image.fromarray(save_tar_img, 'RGB'), # input video
                                (0, 0))
                new_im.paste(Image.fromarray(save_out_img, 'RGB'), # prediction frame by frame
                                (msk_size, 0))
                new_im.paste(Image.fromarray(save_warped_img, 'RGB'), # deformed frame by frame
                                (msk_size * 2, 0))
                new_im.paste(Image.fromarray(save_warped_grid), # optical flow
                                (msk_size * 3, 0))
                # new_im.paste(Image.fromarray(save_conf_map, "L"), # occlusion mask
                #                 (msk_size * 4, 0)) 
                new_im_list.append(new_im)
            video_name = "%04d_%s.gif" % (idx, batch["txt"][batch_idx].replace("/", '')[:20])
            imageio.mimsave(os.path.join(config.output_dir, video_name),
                            new_im_list)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='configs/test_lfae.yaml', help="path to config")
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
        os.environ["MASTER_PORT"] = "9011"

    main(config, batch_size=args.batch_size)
