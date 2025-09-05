# Copyright 2023 Bingxin Ke, ETH Zurich. All rights reserved.
# Last modified: 2024-05-24
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


import logging
import random
from typing import Dict, Optional, Union, Any

import numpy as np
import torch
from PIL.Image import Image
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DiffusionPipeline,
    LCMScheduler,
    UNet2DConditionModel,
)
from diffusers.utils import BaseOutput
from PIL import Image
# from numpy import ndarray, dtype, _SCT
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import pil_to_tensor, resize
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from .util.batchsize import find_batch_size
from .util.ensemble import ensemble_depth
from .util.image_util import (
    chw2hwc,
    colorize_depth_maps,
    get_tv_resample_method,
    resize_max_res,
)
from marigold.ConfidencePredictor import SimpleUNet, CNNBackbone, ResNet18Backbone, MLPBackbone


class MarigoldDepthOutput(BaseOutput):
    """
    Output class for Marigold monocular depth prediction pipeline.

    Args:
        depth_np (`np.ndarray`):
            Predicted depth map, with depth values in the range of [0, 1].
        depth_colored (`PIL.Image.Image`):
            Colorized depth map, with the shape of [3, H, W] and values in [0, 1].
        uncertainty (`None` or `np.ndarray`):
            Uncalibrated uncertainty(MAD, median absolute deviation) coming from ensembling.
    """

    depth_np: np.ndarray
    depth_colored: Union[None, Image.Image]
    uncertainty: Union[None, np.ndarray]

class MarigoldNormalOutput(BaseOutput):

    normal_np: np.ndarray
    normal_colored: Image.Image
    uncertainty: Union[None, np.ndarray]


class MarigoldPipeline(DiffusionPipeline):
    """
    Pipeline for monocular depth estimation using Marigold: https://marigoldmonodepth.github.io.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        unet (`UNet2DConditionModel`):
            Conditional U-Net to denoise the depth latent, conditioned on image latent.
        vae (`AutoencoderKL`):
            Variational Auto-Encoder (VAE) Model to encode and decode images and depth maps
            to and from latent representations.
        scheduler (`DDIMScheduler`):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents.
        text_encoder (`CLIPTextModel`):
            Text-encoder, for empty text embedding.
        tokenizer (`CLIPTokenizer`):
            CLIP tokenizer.
        scale_invariant (`bool`, *optional*):
            A model property specifying whether the predicted depth maps are scale-invariant. This value must be set in
            the model config. When used together with the `shift_invariant=True` flag, the model is also called
            "affine-invariant". NB: overriding this value is not supported.
        shift_invariant (`bool`, *optional*):
            A model property specifying whether the predicted depth maps are shift-invariant. This value must be set in
            the model config. When used together with the `scale_invariant=True` flag, the model is also called
            "affine-invariant". NB: overriding this value is not supported.
        default_denoising_steps (`int`, *optional*):
            The minimum number of denoising diffusion steps that are required to produce a prediction of reasonable
            quality with the given model. This value must be set in the model config. When the pipeline is called
            without explicitly setting `num_inference_steps`, the default value is used. This is required to ensure
            reasonable results with various model flavors compatible with the pipeline, such as those relying on very
            short denoising schedules (`LCMScheduler`) and those with full diffusion schedules (`DDIMScheduler`).
        default_processing_resolution (`int`, *optional*):
            The recommended value of the `processing_resolution` parameter of the pipeline. This value must be set in
            the model config. When the pipeline is called without explicitly setting `processing_resolution`, the
            default value is used. This is required to ensure reasonable results with various model flavors trained
            with varying optimal processing resolution values.
    """

    """
        该管道用于单目深度估计，基于扩散模型的生成能力进行深度预测。

        Args:
            unet (`UNet2DConditionModel`):
                条件 U-Net 模型，用于深度图去噪。
            vae (`AutoencoderKL`):
                变分自编码器 (VAE)，用于对输入图像进行编码和解码。
            scheduler (`Union[DDIMScheduler, LCMScheduler]`):
                采样调度器，决定去噪的方式。
            scale_invariant (`bool`, optional, default=True):
                预测的深度图是否对尺度变化不敏感。
            shift_invariant (`bool`, optional, default=True):
                预测的深度图是否对平移变化不敏感。
        """

    rgb_latent_scale_factor = 0.18215
    depth_latent_scale_factor = 0.18215
    pol_latent_scale_factor = 0.18215

    def __init__(
        self,
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        scheduler: Union[DDIMScheduler, LCMScheduler],
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        scale_invariant: Optional[bool] = True,
        shift_invariant: Optional[bool] = True,
        default_denoising_steps: Optional[int] = None,
        default_processing_resolution: Optional[int] = None,
        confidence_predictor: SimpleUNet = None,
    ):

        super().__init__()

        backbone = 'MLP'
        if confidence_predictor is None:
            if backbone == 'Unet':
            # 默认初始化
                confidence_predictor = SimpleUNet(in_channels=8, out_channels=1)

            elif backbone == 'CNN':
                confidence_predictor = CNNBackbone(in_channels=8, out_channels=1)

            elif backbone == 'ResNet18':
                confidence_predictor = ResNet18Backbone(in_channels=8, out_channels=1)

            elif backbone == 'MLP':
                confidence_predictor = MLPBackbone(in_channels=8, out_channels=1)

        self.confidence_predictor = confidence_predictor.to(self.device)

        # if Latent_Fuser is None:
        #     # 默认初始化
        #     Latent_Fuser = FusionNet(in_channels=9, base_channels=32, out_channels=4, use_residual=True)
        # self.Latent_Fuser = Latent_Fuser.to(self.device)

        print('device', self.device)
        # register_modules 是 DiffusionPipeline 提供的方法，可以一次性将若干子模块（如 unet, vae, scheduler）注册到当前的 pipeline 中。
        # 这样做不仅能让这些模块自动成为 pipeline 的一部分，也方便在后续保存、加载模型时，能统一管理这些组件。
        self.register_modules(
            unet=unet,
            vae=vae,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            confidence_predictor=self.confidence_predictor,
        )
        # register_to_config是DiffusionPipeline提供的另一个实用方法，用于将关键的超参数（或配置项）记录进 pipeline 的 config 中。
        # 这样在保存/加载 config 时，可以保留这些值，
        self.register_to_config(
            scale_invariant=scale_invariant,
            shift_invariant=shift_invariant,
            default_denoising_steps=default_denoising_steps,
            default_processing_resolution=default_processing_resolution,
        )

        self.scale_invariant = scale_invariant
        self.shift_invariant = shift_invariant
        self.default_denoising_steps = default_denoising_steps
        self.default_processing_resolution = default_processing_resolution
        self.empty_text_embed = None

    @torch.no_grad()
    def __call__(
        self,
        # input_image输入可以是PIL的Image格式或张量格式
        input_image: Union[Image.Image, torch.Tensor],
        input_image_pol: torch.Tensor,
        # 去噪步数设为yaml中的值：50
        denoising_steps: Optional[int] = None,
        # ensemble_size：要进行多少次独立推理并做集成 (ensemble)，默认是 5
        ensemble_size: int = 5,
        # 保持原图分辨率
        processing_res: Optional[int] = None,
        # 将网络输出的深度图resize回原图分辨率
        match_input_res: bool = True,
        # 重采样方法
        resample_method: str = "bilinear",
        # 根据显存自动调整batchsize
        batch_size: int = 0,
        # 随机数生成器 (PyTorch Generator)，可用于控制每次推理时的随机初始化，使结果可复现
        generator: Union[torch.Generator, None] = None,
        color_map: str = "Spectral",
        show_progress_bar: bool = True,
        ensemble_kwargs: Dict = None,
        # 优化目标
        mode: str = None,
    ) -> MarigoldDepthOutput:
        """
        Function invoked when calling the pipeline.
        Args:
            input_image (`Image`):
                Input RGB (or gray-scale) image.
            denoising_steps (`int`, *optional*, defaults to `None`):
                Number of denoising diffusion steps during inference. The default value `None` results in automatic
                selection. The number of steps should be at least 10 with the full Marigold models, and between 1 and 4
                for Marigold-LCM models.
            ensemble_size (`int`, *optional*, defaults to `10`):
                Number of predictions to be ensembled.
            processing_res (`int`, *optional*, defaults to `None`):
                Effective processing resolution. When set to `0`, processes at the original image resolution. This
                produces crisper predictions, but may also lead to the overall loss of global context. The default
                value `None` resolves to the optimal value from the model config.
            match_input_res (`bool`, *optional*, defaults to `True`):
                Resize depth prediction to match input resolution.
                Only valid if `processing_res` > 0.
            resample_method: (`str`, *optional*, defaults to `bilinear`):
                Resampling method used to resize images and depth predictions. This can be one of `bilinear`, `bicubic` or `nearest`, defaults to: `bilinear`.
            batch_size (`int`, *optional*, defaults to `0`):
                Inference batch size, no bigger than `num_ensemble`.
                If set to 0, the script will automatically decide the proper batch size.
            generator (`torch.Generator`, *optional*, defaults to `None`)
                Random generator for initial noise generation.
            show_progress_bar (`bool`, *optional*, defaults to `True`):
                Display a progress bar of diffusion denoising.
            color_map (`str`, *optional*, defaults to `"Spectral"`, pass `None` to skip colorized depth map generation):
                Colormap used to colorize the depth map.
            scale_invariant (`str`, *optional*, defaults to `True`):
                Flag of scale-invariant prediction, if True, scale will be adjusted from the raw prediction.
            shift_invariant (`str`, *optional*, defaults to `True`):
                Flag of shift-invariant prediction, if True, shift will be adjusted from the raw prediction, if False, near plane will be fixed at 0m.
            ensemble_kwargs (`dict`, *optional*, defaults to `None`):
                Arguments for detailed ensembling settings.
        Returns:
            `MarigoldDepthOutput`: Output class for Marigold monocular depth prediction pipeline, including:
            - **depth_np** (`np.ndarray`) Predicted depth map, with depth values in the range of [0, 1]
            - **depth_colored** (`PIL.Image.Image`) Colorized depth map, with the shape of [3, H, W] and values in [0, 1], None if `color_map` is `None`
            - **uncertainty** (`None` or `np.ndarray`) Uncalibrated uncertainty(MAD, median absolute deviation)
                    coming from ensembling. None if `ensemble_size = 1`
        """
        # Model-specific optimal default values leading to fast and reasonable results.
        # 从yaml文件中读取参数
        if denoising_steps is None:
            denoising_steps = self.default_denoising_steps
        if processing_res is None:
            processing_res = self.default_processing_resolution

        assert processing_res >= 0
        assert ensemble_size >= 1

        # Check if denoising step is reasonable
        self._check_inference_step(denoising_steps)
        # 图像缩放的采样方式
        resample_method: InterpolationMode = get_tv_resample_method(resample_method)

        # ----------------- Image Preprocess -----------------
        # Convert to torch tensor
        # RGB图像预处理，[H,W,3]->[1,3,H,W]
        # 只有在run的时候，才是Image格式，训练时均为tensor格式
        if isinstance(input_image, Image.Image):
            input_image = input_image.convert("RGB")
            # convert to torch tensor [H, W, rgb] -> [rgb, H, W]
            rgb = pil_to_tensor(input_image)
            rgb = rgb.unsqueeze(0)  # [1, rgb, H, W]
        elif isinstance(input_image, torch.Tensor):
            rgb = input_image
        else:
            raise TypeError(f"Unknown input type: {type(input_image) = }")
        input_size = rgb.shape
        assert (
            4 == rgb.dim() and 3 == input_size[-3]
        ), f"Wrong input shape {input_size}, expected [1, rgb, H, W]"

        # 判定偏振图维度
        if isinstance(input_image_pol, torch.Tensor):
            pol = input_image_pol
        else:
            raise TypeError(f"Unknown input type: {type(input_image_pol) = }")
        input_size_pol = pol.shape
        assert (
            4 == pol.dim() and 2 == input_size_pol[-3]
        ), f"Wrong input shape {input_size_pol}, expected [1, adolp, H, W]"
        # Resize image
        # 如果 processing_res > 0，就将输入等比例缩放至最大边为 processing_res
        # resize_max_res函数会自动将宽高比调节为一致
        if processing_res > 0:
            rgb = resize_max_res(
                rgb,
                max_edge_resolution=processing_res,
                resample_method=resample_method,
            )
            pol = resize_max_res(
                pol,
                max_edge_resolution=processing_res,
                resample_method=resample_method,
            )

        # Normalize rgb values
        # 将RGB值归一化到[-1,1]之间
        rgb_norm: torch.Tensor = rgb / 255.0 * 2.0 - 1.0  #  [0, 255] -> [-1, 1]
        rgb_norm = rgb_norm.to(self.dtype)
        assert rgb_norm.min() >= -1.0 and rgb_norm.max() <= 1.0

        # 将偏振图像归一化到[-1,1]之间
        dolp_norm: torch.Tensor = pol[:, 1, :, :] / 255.0 * 2.0 - 1.0 # dolp:[-1, 1]
        aolp_rad: torch.Tensor = pol[:, 0, :, :] / 255.0 * torch.pi # aolp_rad:[0, pi]
        aolp_cos = torch.cos(2 * aolp_rad) # aolp_cos:[-1, 1]
        aolp_sin = torch.sin(2 * aolp_rad) # aolp_sin:[-1, 1]
        pol_norm = torch.cat([dolp_norm, aolp_cos, aolp_sin], dim=0).unsqueeze(0) # pol_norm:[1, 3, H, W]
        pol_norm = pol_norm.to(self.dtype)
        # print(rgb_norm.shape)
        # print(pol_norm.shape)
        # print(input_image_pol[:, 0, :, :].min(), input_image_pol[:, 0, :, :].max())
        # print(input_image_pol[:, 1, :, :].min(), input_image_pol[:, 1, :, :].max())
        # print(pol_norm[:, 0, :, :].min(), pol_norm[:, 0, :, :].max())
        assert pol_norm.min() >= -1.0 and pol_norm.max() <= 1.0
        # ----------------- Predicting depth -----------------
        # Batch repeated input image
        # 如果 ensemble_size > 1，需要把同一张图复制多份来进行多次推理
        duplicated_rgb = rgb_norm.expand(ensemble_size, -1, -1, -1)
        duplicated_pol = pol_norm.expand(ensemble_size, -1, -1, -1)  # 扩展偏振图像
        # single_rgb_dataset = TensorDataset(duplicated_rgb)
        single_rgb_pol_dataset = TensorDataset(duplicated_rgb, duplicated_pol)
        # 如果未设定batchsize，则自动寻找
        if batch_size > 0:
            _bs = batch_size
        else:
            _bs = find_batch_size(
                ensemble_size=ensemble_size,
                input_res=max(rgb_norm.shape[1:]),
                dtype=self.dtype,
            )

        single_rgb_pol_loader = DataLoader(
            # single_rgb_dataset, batch_size=_bs, shuffle=False
            single_rgb_pol_dataset, batch_size=_bs, shuffle=False
        )

        # Predict depth maps (batched)
        # 分批地把图像送入 single_infer() 进行单次推理
        # 收集所有批次的结果 depth_pred_raw，拼接到 depth_preds
        geo_pred_ls = []
        if show_progress_bar:
            iterable = tqdm(
                single_rgb_pol_loader, desc=" " * 2 + "Inference batches", leave=False
            )
        else:
            iterable = single_rgb_pol_loader
        for batch in iterable:
            batched_rgb, batched_pol = batch
            geo_pred_raw = self.single_infer(
                rgb_in=batched_rgb,
                pol_in=batched_pol,
                num_inference_steps=denoising_steps,
                show_pbar=show_progress_bar,
                generator=generator,
            )
            geo_pred_ls.append(geo_pred_raw.detach())
        geo_preds = torch.concat(geo_pred_ls, dim=0)
        torch.cuda.empty_cache()  # clear vram cache for ensembling

        # ----------------- Test-time ensembling -----------------
        # ensemble_size > 1，就会调用 ensemble_depth() 来将多张预测进行对齐和融合
        if ensemble_size > 1:
            geo_pred, pred_uncert = ensemble_depth(
                geo_preds,
                scale_invariant=self.scale_invariant,
                shift_invariant=self.shift_invariant,
                max_res=50,
                **(ensemble_kwargs or {}),
            )
        else:
            geo_pred = geo_preds
            pred_uncert = None

        # Resize back to original resolution
        # 若缩小过图像则恢复原图分辨率
        if match_input_res:
            geo_pred = resize(
                geo_pred,
                input_size[-2:],
                interpolation=resample_method,
                antialias=True,
            )

        # Convert to numpy
        # 输出移到 CPU 并转换成 numpy 数组，如果融合过程里有不确定度 pred_uncert，也做相应处理
        geo_pred = geo_pred.squeeze()
        geo_pred = geo_pred.cpu().numpy()
        if pred_uncert is not None:
            pred_uncert = pred_uncert.squeeze().cpu().numpy()

        # Clip output range
        # 深度值裁剪到[0,1]
        depth_pred = geo_pred.clip(0, 1)

        # Colorize
        # 深度图可视化为彩色
        if color_map is not None:
            depth_colored = colorize_depth_maps(
                depth_pred, 0, 1, cmap=color_map
            ).squeeze()  # [3, H, W], value in (0, 1)
            depth_colored = (depth_colored * 255).astype(np.uint8)
            depth_colored_hwc = chw2hwc(depth_colored)
            depth_colored_img = Image.fromarray(depth_colored_hwc)
        else:
            depth_colored_img = None

        return MarigoldDepthOutput(
            depth_np=depth_pred,
            depth_colored=depth_colored_img,
            uncertainty=pred_uncert,
        )
        # elif self.mode == 'normal':
        # normal_pred = geo_pred.clip(-1, 1)
        # normal_colored = ((normal_pred + 1) / 2 * 255).astype(np.uint8)
        # normal_colored_img = Image.fromarray(normal_colored)
        # return MarigoldNormalOutput(
        #     normal_np=normal_pred,
        #     normal_colored=normal_colored_img,
        #     uncertainty=pred_uncert,
        # )

        # else:
        #     raise ValueError(f"Unknown mode: {self.mode}")

    def _check_inference_step(self, n_step: int) -> None:
        """
        Check if denoising step is reasonable
        Args:
            n_step (`int`): denoising steps
        """
        assert n_step >= 1

        if isinstance(self.scheduler, DDIMScheduler):
            if n_step < 10:
                logging.warning(
                    f"Too few denoising steps: {n_step}. Recommended to use the LCM checkpoint for few-step inference."
                )
        elif isinstance(self.scheduler, LCMScheduler):
            if not 1 <= n_step <= 4:
                logging.warning(
                    f"Non-optimal setting of denoising steps: {n_step}. Recommended setting is 1-4 steps."
                )
        else:
            raise RuntimeError(f"Unsupported scheduler type: {type(self.scheduler)}")

    def encode_empty_text(self):
        """
        Encode text embedding for empty prompt
        """
        prompt = ""
        text_inputs = self.tokenizer(
            prompt,
            padding="do_not_pad",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.text_encoder.device)
        self.empty_text_embed = self.text_encoder(text_input_ids)[0].to(self.dtype)

    @torch.no_grad()
    def single_infer(
        self,
        rgb_in: torch.Tensor,
        pol_in: torch.Tensor,
        num_inference_steps: int,
        generator: Union[torch.Generator, None],
        show_pbar: bool,
    ) -> torch.Tensor:
        """
        Perform an individual depth prediction without ensembling.

        Args:
            rgb_in (`torch.Tensor`):
                Input RGB image.
            num_inference_steps (`int`):
                Number of diffusion denoisign steps (DDIM) during inference.
            show_pbar (`bool`):
                Display a progress bar of diffusion denoising.
            generator (`torch.Generator`)
                Random generator for initial noise generation.
        Returns:
            `torch.Tensor`: Predicted depth map.
        """
        device = self.device
        rgb_in = rgb_in.to(device)
        pol_in = pol_in.to(device)

        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps  # [T]

        # Encode image
        # RGB图像和偏振图像的潜在编码
        rgb_latent = self.encode_rgb(rgb_in)
        pol_latent = self.encode_pol(pol_in)
        latent_cat = torch.cat([rgb_latent, pol_latent], dim=1).to(device)  # [B, 8, h, w]
        self.confidence_predictor = self.confidence_predictor.to(device)
        alpha = self.confidence_predictor(latent_cat) # [B, 1, h, w]
        # print('z_rgb', rgb_latent)
        # print('z_pol', pol_latent)
        # print(alpha)
        # B, _, H, W = latent_cat.shape
        # alpha = torch.rand(B, 1, H, W, device=latent_cat.device, dtype=latent_cat.dtype)
        # alpha = 0
        z_fused = rgb_latent + alpha * (pol_latent - rgb_latent) # [B, 4, h, w]

        # Initial depth map (noise)
        # 深度的随机噪声
        depth_latent = torch.randn(
            z_fused.shape,
            device=device,
            dtype=self.dtype,
            generator=generator,
        )  # [B, 4, h, w]

        # Batched empty text embedding
        # 将文本编码为空
        if self.empty_text_embed is None:
            self.encode_empty_text()
        batch_empty_text_embed = self.empty_text_embed.repeat(
            (z_fused.shape[0], 1, 1)
        ).to(device)  # [B, 2, 1024]

        # Denoising loop
        # 去噪
        if show_pbar:
            iterable = tqdm(
                enumerate(timesteps),
                total=len(timesteps),
                leave=False,
                desc=" " * 4 + "Diffusion denoising",
            )
        else:
            iterable = enumerate(timesteps)
        # 逐步去噪
        for i, t in iterable:
            unet_input = torch.cat(
                [z_fused, depth_latent], dim=1
            )  # this order is important

            # predict the noise residual
            noise_pred = self.unet(
                unet_input, t, encoder_hidden_states=batch_empty_text_embed
            ).sample  # [B, 4, h, w]

            # compute the previous noisy sample x_t -> x_t-1
            depth_latent = self.scheduler.step(
                noise_pred, t, depth_latent, generator=generator
            ).prev_sample

        # if self.mode == 'depth':
        depth = self.decode_depth(depth_latent)
        # clip prediction
        depth = torch.clip(depth, -1.0, 1.0)
        # shift to [0, 1]
        depth = (depth + 1.0) / 2.0
        return depth
        # elif self.mode == 'normal':
        #     normal = self.decode_normal(geo_latent)
        #     normal /= (torch.norm(normal, p=2, dim=1, keepdim=True) + 1e-5)
        #     normal *= -1.
        #     return normal
        # else:
        #     raise ValueError(f"Unknown mode: {self.mode}")


    # 将RGB图编码至潜在空间中
    def encode_rgb(self, rgb_in: torch.Tensor) -> torch.Tensor:
        """
        Encode RGB image into latent.

        Args:
            rgb_in (`torch.Tensor`):
                Input RGB image to be encoded.

        Returns:
            `torch.Tensor`: Image latent.
        """
        # encode
        h = self.vae.encoder(rgb_in)
        moments = self.vae.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        # scale latent
        rgb_latent = mean * self.rgb_latent_scale_factor
        return rgb_latent

    def encode_pol(self, pol_in: torch.Tensor) -> torch.Tensor:
        """
        Encode RGB image into latent.

        Args:
            rgb_in (`torch.Tensor`):
                Input RGB image to be encoded.

        Returns:
            `torch.Tensor`: Image latent.
        """
        # encode
        h = self.vae.encoder(pol_in)
        moments = self.vae.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        # scale latent
        pol_latent = mean * self.pol_latent_scale_factor
        return pol_latent

    def decode_depth(self, depth_latent: torch.Tensor) -> torch.Tensor:
        """
        Decode depth latent into depth map.

        Args:
            depth_latent (`torch.Tensor`):
                Depth latent to be decoded.

        Returns:
            `torch.Tensor`: Decoded depth map.
        """
        # scale latent
        depth_latent = depth_latent / self.depth_latent_scale_factor
        # decode
        z = self.vae.post_quant_conv(depth_latent)
        stacked = self.vae.decoder(z)
        # mean of output channels
        depth_mean = stacked.mean(dim=1, keepdim=True)
        return depth_mean

    # def decode_normal(self, normal_latent: torch.Tensor) -> torch.Tensor:
    #     """
    #     Decode normal latent into normal map.
    #     Args:
    #         normal_latent (`torch.Tensor`):
    #             Depth latent to be decoded.
    #     Returns:
    #         `torch.Tensor`: Decoded normal map.
    #     """
    #
    #     # scale latent
    #     normal_latent = normal_latent / self.latent_scale_factor
    #     # decode
    #     z = self.vae.post_quant_conv(normal_latent)
    #     normal = self.vae.decoder(z)
    #     return normal
