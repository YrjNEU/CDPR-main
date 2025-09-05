# Copyright 2023 Bingxin Ke, ETH Zurich. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------
# If you find this code useful, we kindly ask you to cite our paper in your work.
# Please find bibtex at: https://github.com/prs-eth/Marigold#-citation
# More information about the method can be found at https://marigoldmonodepth.github.io
# --------------------------------------------------------------------------

# python run.py     --checkpoint checkpoint/marigold-pol     --denoise_steps 50     --ensemble_size 10     --input_dir input/test     --output_dir output/in-the-wild_example
import json
import argparse
import logging
import os
from glob import glob

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from CDPR import MarigoldPipeline
from CDPR.ConfidencePredictor import SimpleUNet, CNNBackbone
EXTENSION_LIST = [".jpg", ".jpeg", ".png"]


def get_filenames(input_dir: str, rgb_folder: str = 'rgb', pol_folder: str = 'pol', ext_list=None):
    """
    获取 RGB 图像和对应的 AoLP 和 DoLP 图像路径。

    Args:
        input_dir (str): 根目录路径，包含 rgb 和 pol 文件夹。
        rgb_folder (str): rgb 文件夹名称。
        pol_folder (str): pol 文件夹名称。
        ext_list (list): 允许的文件扩展名列表，例如 ['.png', '.jpg']。

    Returns:
        rgb_filename_list (list): 所有 RGB 图像文件的路径列表。
        pol_filename_list (list): 每个元素是一个二元组，包含 AoLP 和 DoLP 图像的路径。
    """

    # 设置默认的扩展名列表，如果没有提供
    if ext_list is None:
        ext_list = ['.png', '.jpg']

    # 获取 rgb 文件夹路径
    input_rgb_dir = os.path.join(input_dir, rgb_folder)

    # 获取所有的 RGB 文件
    rgb_filename_list = glob(os.path.join(input_rgb_dir, "*"))
    rgb_filename_list = [
        f for f in rgb_filename_list if os.path.splitext(f)[1].lower() in ext_list
    ]
    rgb_filename_list = sorted(rgb_filename_list)
    n_images = len(rgb_filename_list)

    # 如果没有找到图像文件，输出日志并退出
    if n_images > 0:
        logging.info(f"Found {n_images} images in RGB folder.")
    else:
        logging.error(f"No image found in '{input_rgb_dir}'")
        exit(1)

    # 创建 pol_filename_list，每个元素包含 AoLP 和 DoLP 图像路径
    pol_filename_list = []
    for rgb_path in rgb_filename_list:
        # 获取文件名，不包含扩展名
        filename = os.path.splitext(os.path.basename(rgb_path))[0]

        # 对应的 AoLP 和 DoLP 图像路径
        aolp_path = os.path.join(input_dir, pol_folder, f"{filename}_aolp.png")
        dolp_path = os.path.join(input_dir, pol_folder, f"{filename}_dolp.png")

        # 确保 AoLP 和 DoLP 图像存在
        if os.path.exists(aolp_path) and os.path.exists(dolp_path):
            pol_filename_list.append((aolp_path, dolp_path))
        else:
            logging.warning(f"Missing AoLP or DoLP for {rgb_path}, skipping.")

    # 检查 pol_filename_list 是否为空
    if len(pol_filename_list) == 0:
        logging.error(f"No valid AoLP or DoLP files found.")
        exit(1)

    return rgb_filename_list, pol_filename_list


def process_pol_images(aolp_path, dolp_path):
    # Load AoLP and DoLP images using PIL
    aolp_image = Image.open(aolp_path)
    dolp_image = Image.open(dolp_path)

    # Convert images to numpy arrays
    aolp_array = np.array(aolp_image).astype(np.float32)  # Convert AoLP image to numpy array
    dolp_array = np.array(dolp_image).astype(np.float32)  # Convert DoLP image to numpy array

    # Convert to PyTorch tensors
    aolp_tensor = torch.from_numpy(aolp_array).unsqueeze(0)  # Convert AoLP to tensor [1, H, W]
    dolp_tensor = torch.from_numpy(dolp_array).unsqueeze(0)  # Convert DoLP to tensor [1, H, W]

    # Return the processed tensors
    return torch.cat([aolp_tensor, dolp_tensor], dim=0).unsqueeze(0) # [1, 2, H, W]

if "__main__" == __name__:
    logging.basicConfig(level=logging.INFO)

    # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser(
        description="Run single-image depth estimation using Marigold."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        # default="prs-eth/marigold-lcm-v1-0",
        default="checkpoint/depth/marigold-polCNN",
        help="Checkpoint path or hub name.",
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to the input rgb image folder.",
    )

    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory."
    )

    # inference setting
    parser.add_argument(
        "--denoise_steps",
        type=int,
        default=None,
        help="Diffusion denoising steps, more steps results in higher accuracy but slower inference speed. "
             "For the original (DDIM) version, it's recommended to use 10-50 steps, while for LCM 1-4 steps.",
    )
    parser.add_argument(
        "--ensemble_size",
        type=int,
        default=5,
        help="Number of predictions to be ensembled, more inference gives better results but runs slower.",
    )
    parser.add_argument(
        "--half_precision",
        "--fp32",
        action="store_true",
        help="Run with half-precision (16-bit float), might lead to suboptimal result.",
    )

    # resolution setting
    parser.add_argument(
        "--processing_res",
        type=int,
        default=None,
        help="Maximum resolution of processing. 0 for using input image resolution. Default: 768.",
    )
    parser.add_argument(
        "--output_processing_res",
        action="store_true",
        help="When input is resized, out put depth at resized operating resolution. Default: False.",
    )
    parser.add_argument(
        "--resample_method",
        choices=["bilinear", "bicubic", "nearest"],
        default="bilinear",
        help="Resampling method used to resize images and depth predictions. This can be one of `bilinear`, `bicubic` or `nearest`. Default: `bilinear`",
    )

    # depth map colormap
    parser.add_argument(
        "--color_map",
        type=str,
        default="Spectral",
        help="Colormap used to render depth predictions.",
    )

    # other settings
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Reproducibility seed. Set to `None` for unseeded inference.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=0,
        help="Inference batch size. Default: 0 (will be set automatically).",
    )
    parser.add_argument(
        "--apple_silicon",
        action="store_true",
        help="Flag of running on Apple Silicon.",
    )

    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    input_dir = args.input_dir
    output_dir = args.output_dir

    denoise_steps = args.denoise_steps
    ensemble_size = args.ensemble_size
    if ensemble_size > 15:
        logging.warning("Running with large ensemble size will be slow.")
    half_precision = args.half_precision

    processing_res = args.processing_res
    match_input_res = not args.output_processing_res
    if 0 == processing_res and match_input_res is False:
        logging.warning(
            "Processing at native resolution without resizing output might NOT lead to exactly the same resolution, due to the padding and pooling properties of conv layers."
        )
    resample_method = args.resample_method

    color_map = args.color_map
    seed = args.seed
    batch_size = args.batch_size
    apple_silicon = args.apple_silicon
    if apple_silicon and 0 == batch_size:
        batch_size = 1  # set default batchsize

    # -------------------- Preparation --------------------
    # Output directories
    output_dir_color = os.path.join(output_dir, "depth_colored")
    output_dir_tif = os.path.join(output_dir, "depth_bw")
    output_dir_npy = os.path.join(output_dir, "depth_npy")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir_color, exist_ok=True)
    os.makedirs(output_dir_tif, exist_ok=True)
    os.makedirs(output_dir_npy, exist_ok=True)
    logging.info(f"output dir = {output_dir}")

    # -------------------- Device --------------------
    if apple_silicon:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps:0")
        else:
            device = torch.device("cpu")
            logging.warning("MPS is not available. Running on CPU will be slow.")
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
            logging.warning("CUDA is not available. Running on CPU will be slow.")
    logging.info(f"device = {device}")

    # -------------------- Data --------------------
    rgb_filename_list, pol_filename_list = get_filenames(input_dir)
    # rgb_filename_list = glob(os.path.join(input_rgb_dir, "*"))
    # rgb_filename_list = [
    #     f for f in rgb_filename_list if os.path.splitext(f)[1].lower() in EXTENSION_LIST
    # ]
    # rgb_filename_list = sorted(rgb_filename_list)
    # n_images = len(rgb_filename_list)
    # if n_images > 0:
    #     logging.info(f"Found {n_images} images")
    # else:
    #     logging.error(f"No image found in '{input_rgb_dir}'")
    #     exit(1)

    # -------------------- Model --------------------
    if half_precision:
        dtype = torch.float16
        variant = "fp16"
        logging.info(
            f"Running with half precision ({dtype}), might lead to suboptimal result."
        )
    else:
        dtype = torch.float32
        variant = None

    pipe: MarigoldPipeline = MarigoldPipeline.from_pretrained(
        checkpoint_path, variant=variant, torch_dtype=dtype
    )
    # 初始化 confidence_predictor
    pipe.confidence_predictor = CNNBackbone(in_channels=8, out_channels=1)
    pipe.confidence_predictor.load_state_dict(torch.load('checkpoint/depth/marigold-polCNN/confidence_predictor.pth'))
    pipe.confidence_predictor.eval()
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except ImportError:
        pass  # run without xformers

    pipe = pipe.to(device)
    logging.info(
        f"scale_invariant: {pipe.scale_invariant}, shift_invariant: {pipe.shift_invariant}"
    )

    # Print out config
    logging.info(
        f"Inference settings: checkpoint = `{checkpoint_path}`, "
        f"with denoise_steps = {denoise_steps or pipe.default_denoising_steps}, "
        f"ensemble_size = {ensemble_size}, "
        f"processing resolution = {processing_res or pipe.default_processing_resolution}, "
        f"seed = {seed}; "
        f"color_map = {color_map}."
    )

    # -------------------- Inference and saving --------------------
    with torch.no_grad():
        os.makedirs(output_dir, exist_ok=True)

        for rgb_path, pol_paths in tqdm(zip(rgb_filename_list, pol_filename_list), desc="Estimating depth", leave=True):
            # Read RGB image
            input_rgb = Image.open(rgb_path)

            # Read AoLP and DoLP images
            aolp_path, dolp_path = pol_paths
            input_pol = process_pol_images(aolp_path, dolp_path)

            # Random number generator
            if seed is None:
                generator = None
            else:
                generator = torch.Generator(device=device)
                generator.manual_seed(seed)

            # Predict depth
            pipe_out = pipe(
                input_rgb,
                input_pol,
                denoising_steps=denoise_steps,
                ensemble_size=ensemble_size,
                processing_res=processing_res,
                match_input_res=match_input_res,
                batch_size=batch_size,
                color_map=color_map,
                show_progress_bar=True,
                resample_method=resample_method,
                generator=generator,
            )

            depth_pred: np.ndarray = pipe_out.depth_np
            depth_colored: Image.Image = pipe_out.depth_colored

            # Save as npy
            rgb_name_base = os.path.splitext(os.path.basename(rgb_path))[0]
            pred_name_base = rgb_name_base + "_pred"
            npy_save_path = os.path.join(output_dir_npy, f"{pred_name_base}.npy")
            if os.path.exists(npy_save_path):
                logging.warning(f"Existing file: '{npy_save_path}' will be overwritten")
            np.save(npy_save_path, depth_pred)

            # Save as 16-bit uint png
            depth_to_save = (depth_pred * 65535.0).astype(np.uint16)
            png_save_path = os.path.join(output_dir_tif, f"{pred_name_base}.png")
            if os.path.exists(png_save_path):
                logging.warning(f"Existing file: '{png_save_path}' will be overwritten")
            Image.fromarray(depth_to_save).save(png_save_path, mode="I;16")

            # Colorize
            colored_save_path = os.path.join(
                output_dir_color, f"{pred_name_base}_colored.png"
            )
            if os.path.exists(colored_save_path):
                logging.warning(
                    f"Existing file: '{colored_save_path}' will be overwritten"
                )
            depth_colored.save(colored_save_path)
