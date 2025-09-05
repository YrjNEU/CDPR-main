import sys
import os

import argparse
import logging
import numpy as np
import os
import torch
from PIL import Image
from glob import glob
from tqdm.auto import tqdm

from CDPR import MarigoldNormalsPipeline, MarigoldNormalsOutput

EXTENSION_LIST = [".jpg", ".jpeg", ".png"]
# python run_normal.py     --checkpoint checkpoint/normal/marigold-cnn     --denoise_steps 4     --ensemble_size 10     --input_dir input/test     --output_dir output/normaloutput
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
        description="Marigold : Surface Normals Estimation : Multi-image Inference"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="prs-eth/marigold-normals-v1-1",
        help="Checkpoint path or hub name.",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to the input image folder.",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory."
    )
    parser.add_argument(
        "--denoise_steps",
        type=int,
        default=None,
        help="Diffusion denoising steps, more steps results in higher accuracy but slower inference speed. If set to "
        "`None`, default value will be read from checkpoint.",
    )
    parser.add_argument(
        "--processing_res",
        type=int,
        default=None,
        help="Resolution to which the input is resized before performing estimation. `0` uses the original input "
        "resolution; `None` resolves the best default from the model checkpoint. Default: `None`",
    )
    parser.add_argument(
        "--ensemble_size",
        type=int,
        default=1,
        help="Number of predictions to be ensembled. Default: `1`.",
    )
    parser.add_argument(
        "--half_precision",
        "--fp16",
        action="store_true",
        help="Run with half-precision (16-bit float), might lead to suboptimal result.",
    )
    parser.add_argument(
        "--output_processing_res",
        action="store_true",
        help="Setting this flag will output the result at the effective value of `processing_res`, otherwise the "
        "output will be resized to the input resolution.",
    )
    parser.add_argument(
        "--resample_method",
        choices=["bilinear", "bicubic", "nearest"],
        default="bilinear",
        help="Resampling method used to resize images and predictions. This can be one of `bilinear`, `bicubic` or "
        "`nearest`. Default: `bilinear`",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Reproducibility seed. Set to `None` for randomized inference. Default: `None`",
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
        help="Use Apple Silicon for faster inference (subject to availability).",
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
            "Processing at native resolution without resizing output might NOT lead to exactly the same resolution, "
            "due to the padding and pooling properties of conv layers."
        )
    resample_method = args.resample_method

    seed = args.seed
    batch_size = args.batch_size
    apple_silicon = args.apple_silicon
    if apple_silicon and 0 == batch_size:
        batch_size = 1  # set default batchsize

    # -------------------- Preparation --------------------
    # Output directories
    output_dir_vis = os.path.join(output_dir, "normals_vis")
    output_dir_npy = os.path.join(output_dir, "normals_npy")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir_vis, exist_ok=True)
    os.makedirs(output_dir_npy, exist_ok=True)
    logging.info(f"output dir = {output_dir}")

    # -------------------- Device --------------------
    if apple_silicon:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps")
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
    rgb_filename_list, pol_filename_list = get_filenames(input_dir)

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

    pipe: MarigoldNormalsPipeline = MarigoldNormalsPipeline.from_pretrained(
        checkpoint_path, variant=variant, torch_dtype=dtype
    )

    try:
        pipe.enable_xformers_memory_efficient_attention()
    except ImportError:
        pass  # run without xformers

    pipe = pipe.to(device)
    logging.info("Loaded normals pipeline")

    # Print out config
    logging.info(
        f"Inference settings: checkpoint = `{checkpoint_path}`, "
        f"with denoise_steps = {denoise_steps or pipe.default_denoising_steps}, "
        f"ensemble_size = {ensemble_size}, "
        f"processing resolution = {processing_res or pipe.default_processing_resolution}, "
        f"seed = {seed}; "
    )

    # -------------------- Inference and saving --------------------
    with torch.no_grad():
        os.makedirs(output_dir, exist_ok=True)

        for rgb_path, pol_paths in tqdm(
            zip(rgb_filename_list, pol_filename_list), desc="Surface Normals Inference", leave=True
        ):
            # Read input image
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

            # Perform inference
            pipe_out: MarigoldNormalsOutput = pipe(
                input_rgb,
                input_pol,
                denoising_steps=denoise_steps,
                ensemble_size=ensemble_size,
                processing_res=processing_res,
                match_input_res=match_input_res,
                batch_size=batch_size,
                show_progress_bar=True,
                resample_method=resample_method,
                generator=generator,
            )

            normals_pred: np.ndarray = pipe_out.normals_np  # [3,H,W]
            normals_img: Image.Image = pipe_out.normals_img

            # Save as npy
            rgb_name_base = os.path.splitext(os.path.basename(rgb_path))[0]
            pred_name_base = rgb_name_base + "_normals"
            npy_save_path = os.path.join(output_dir_npy, f"{pred_name_base}.npy")
            if os.path.exists(npy_save_path):
                logging.warning(f"Existing file: '{npy_save_path}' will be overwritten")
            np.save(npy_save_path, normals_pred)

            # Save as png
            vis_save_path = os.path.join(output_dir_vis, f"{pred_name_base}.png")
            if os.path.exists(vis_save_path):
                logging.warning(f"Existing file: '{vis_save_path}' will be overwritten")
            normals_img.save(vis_save_path)