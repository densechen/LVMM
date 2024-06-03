import importlib

import torch
import torch.distributed as dist
from PIL import Image
from torchvision import transforms as T


def zero_rank_print(s):
    if (not dist.is_initialized()) or (dist.is_initialized() and dist.get_rank() == 0): print("### " + s)

# tensor of shape (channels, frames, height, width) -> gif

CHANNELS_TO_MODE = {
    1: 'L',
    3: 'RGB',
    4: 'RGBA'
}


def seek_all_images(img, channels=3):
    assert channels in CHANNELS_TO_MODE, f'channels {channels} invalid'
    mode = CHANNELS_TO_MODE[channels]

    i = 0
    while True:
        try:
            img.seek(i)
            yield img.convert(mode)
        except EOFError:
            break
        i += 1



def video_tensor_to_gif(tensor, path, duration=120, loop=0, optimize=True):
    images = map(T.ToPILImage(), tensor.unbind(dim=1))
    first_img, *rest_imgs = images
    first_img.save(path, save_all=True, append_images=rest_imgs,
                   duration=duration, loop=loop, optimize=optimize)
    return images


# gif -> (channels, frame, height, width) tensor

def gif_to_tensor(path, channels=3, transform=T.ToTensor()):
    img = Image.open(path)
    tensors = tuple(map(transform, seek_all_images(img, channels=channels)))
    return torch.stack(tensors, dim=1)



def bf_to_b_f(tensor, b=None, f=None) -> torch.Tensor:
    if b is not None:
        return tensor.reshape(b, -1, *tensor.shape[1:])
    else:
        return tensor.reshape(-1, f, *tensor.shape[1:])

def b_f_to_bf(tensor):
    return tensor.reshape(-1, *tensor.shape[2:])

def b_to_b_f(tensor, f):
    return tensor.unsqueeze(1).repeat(1, f, *[1, ]*(tensor.dim()-1))

def b_to_bf(tensor, f):
    return b_f_to_bf(b_to_b_f(tensor, f))

def f_to_b_f(tensor, b):
    return tensor.unsqueeze(0).repeat(b, *[1, ] * tensor.dim())

def f_to_bf(tensor, b):
    return b_f_to_bf(f_to_b_f(tensor, b))

def bf_to_b(tensor, f):
    """Only keep the first element in f dimension.
    """
    return bf_to_b_f(tensor, f=f)[:, 0]

def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)
