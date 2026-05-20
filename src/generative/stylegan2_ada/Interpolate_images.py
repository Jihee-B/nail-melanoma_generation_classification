# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate a sequence of images by interpolating between two latent codes.

Adapted from NVIDIA StyleGAN2-ADA's ``gen_video.py``: instead of rendering
an interpolation *video*, this saves the intermediate frames as individual
images (``step0000.jpg`` ... ``stepNNNN.jpg``). Used to produce the
sequential NM->NUM morphing figure (paper Methods S2).

Because it imports ``dnnlib`` and ``legacy``, this script must be run from
inside a StyleGAN2-ADA-PyTorch checkout. See the module README for setup.

Example
-------
    python interpolate_images.py \\
        --network /path/to/network-snapshot.pkl \\
        --seed1 18 --seed2 13 --num-steps 10 \\
        --outdir ./morph/results_0018_0013
"""
import os

import click
import dnnlib
import legacy
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seed1', type=int, help='First (start) seed', required=True)
@click.option('--seed2', type=int, help='Second (end) seed', required=True)
@click.option('--num-steps', type=int, help='Number of interpolation steps',
              default=10, show_default=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi',
              default=1, show_default=True)
@click.option('--outdir', help='Where to save the output images', type=str,
              required=True, metavar='DIR')
def generate_images(network_pkl, seed1, seed2, num_steps, truncation_psi, outdir):
    """Linearly interpolate in W space between two seeds and save each step."""
    print('Loading networks from "%s"...' % network_pkl)
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)

    os.makedirs(outdir, exist_ok=True)

    z1 = torch.from_numpy(np.random.RandomState(seed1).randn(1, G.z_dim)).to(device)
    z2 = torch.from_numpy(np.random.RandomState(seed2).randn(1, G.z_dim)).to(device)
    ws1 = G.mapping(z1, None, truncation_psi=truncation_psi)
    ws2 = G.mapping(z2, None, truncation_psi=truncation_psi)

    for step in tqdm(range(num_steps)):
        t = step / (num_steps - 1)
        ws = ws1 + (ws2 - ws1) * t
        img = G.synthesis(ws)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/step{step:04d}.jpg')


if __name__ == "__main__":
    generate_images()  # pylint: disable=no-value-for-parameter