# Copyright 2023-2025 Marigold Team, ETH Zürich. All rights reserved.
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
# More information about Marigold:
#   https://marigoldmonodepth.github.io
#   https://marigoldcomputervision.github.io
# Efficient inference pipelines are now part of diffusers:
#   https://huggingface.co/docs/diffusers/using-diffusers/marigold_usage
#   https://huggingface.co/docs/diffusers/api/pipelines/marigold
# Examples of trained models and live demos:
#   https://huggingface.co/prs-eth
# Related projects:
#   https://rollingdepth.github.io/
#   https://marigolddepthcompletion.github.io/
# Citation (BibTeX):
#   https://github.com/prs-eth/Marigold#-citation
# If you find Marigold useful, we kindly ask you to cite our papers.
# --------------------------------------------------------------------------

import logging
import numpy as np
import torch
from PIL import Image
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DiffusionPipeline,
    LCMScheduler,
    UNet2DConditionModel,
)
from diffusers.utils import BaseOutput
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import pil_to_tensor, resize
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from typing import Dict, Optional, Union

from .util.batchsize import find_batch_size
from .util.ensemble import ensemble_normals
from .util.image_util import (
    chw2hwc,
    get_tv_resample_method,
    resize_max_res,
)
from marigold.ConfidencePredictor import SimpleUNet, CNNBackbone, ResNet18Backbone, MLPBackbone

class MarigoldNormalsOutput(BaseOutput):
    """
    Output class for Marigold Surface Normals Estimation pipeline.

    Args:
        normals_np (`np.ndarray`):
            Predicted normals map of shape [3, H, W] with values in the range of [-1, 1] (unit length vectors).
        normals_img (`PIL.Image.Image`):
            Normals image, with the shape of [H, W, 3] and values in [0, 255].
        uncertainty (`None` or `np.ndarray`):
            Uncalibrated uncertainty(MAD, median absolute deviation) coming from ensembling.
    """

    normals_np: np.ndarray
    normals_img: Image.Image
    uncertainty: Union[None, np.ndarray]


class MarigoldNormalsPipeline(DiffusionPipeline):
    """
    Pipeline for Marigold Surface Normals Estimation: https://marigoldcomputervision.github.io.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        unet (`UNet2DConditionModel`):
            Conditional U-Net to denoise the prediction latent, conditioned on image latent.
        vae (`AutoencoderKL`):
            Variational Auto-Encoder (VAE) Model to encode and decode images and predictions
            to and from latent representations.
        scheduler (`DDIMScheduler`):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents.
        text_encoder (`CLIPTextModel`):
            Text-encoder, for empty text embedding.
        tokenizer (`CLIPTokenizer`):
            CLIP tokenizer.
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

    latent_scale_factor = 0.18215

    def __init__(
        self,
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        scheduler: Union[DDIMScheduler, LCMScheduler],
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        default_denoising_steps: Optional[int] = None,
        default_processing_resolution: Optional[int] = None,
        confidence_predictor: SimpleUNet = None,
    ):
        super().__init__()

        backbone = 'CNN'
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

        self.register_modules(
            unet=unet,
            vae=vae,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
        )
        self.register_to_config(
            default_denoising_steps=default_denoising_steps,
            default_processing_resolution=default_processing_resolution,
        )

        self.default_denoising_steps = default_denoising_steps
        self.default_processing_resolution = default_processing_resolution

        self.empty_text_embed = None

    @torch.no_grad()
    def __call__(
        self,
        input_image: Union[Image.Image, torch.Tensor],
        input_image_pol: torch.Tensor,
        denoising_steps: Optional[int] = None,
        ensemble_size: int = 1,
        processing_res: Optional[int] = None,
        match_input_res: bool = True,
        resample_method: str = "bilinear",
        batch_size: int = 0,
        generator: Union[torch.Generator, None] = None,
        show_progress_bar: bool = True,
        ensemble_kwargs: Dict = None,
    ) -> MarigoldNormalsOutput:
        """
        Function invoked when calling the pipeline.

        Args:
            input_image (`Image`):
                Input RGB (or gray-scale) image.
            denoising_steps (`int`, *optional*, defaults to `None`):
                Number of denoising diffusion steps during inference. The default value `None` results in automatic
                selection.
            ensemble_size (`int`, *optional*, defaults to `1`):
                Number of predictions to be ensembled.
            processing_res (`int`, *optional*, defaults to `None`):
                Effective processing resolution. When set to `0`, processes at the original image resolution. This
                produces crisper predictions, but may also lead to the overall loss of global context. The default
                value `None` resolves to the optimal value from the model config.
            match_input_res (`bool`, *optional*, defaults to `True`):
                Resize the prediction to match the input resolution.
                Only valid if `processing_res` > 0.
            resample_method: (`str`, *optional*, defaults to `bilinear`):
                Resampling method used to resize images and predictions. This can be one of `bilinear`, `bicubic` or
                `nearest`, defaults to: `bilinear`.
            batch_size (`int`, *optional*, defaults to `0`):
                Inference batch size, no bigger than `num_ensemble`.
                If set to 0, the script will automatically decide the proper batch size.
            generator (`torch.Generator`, *optional*, defaults to `None`)
                Random generator for initial noise generation.
            show_progress_bar (`bool`, *optional*, defaults to `True`):
                Display a progress bar of diffusion denoising.
            ensemble_kwargs (`dict`, *optional*, defaults to `None`):
                Arguments for detailed ensembling settings.
        Returns:
            `MarigoldNormalsOutput`: Output class for Marigold monocular surface normals estimation pipeline, including:
            - **normals_np** (`np.ndarray`) Predicted normals map of shape [3, H, W] with values in the range of [-1, 1]
                    (unit length vectors)
            - **normals_img** (`PIL.Image.Image`) Normals image, with the shape of [H, W, 3] and values in [0, 255]
            - **uncertainty** (`None` or `np.ndarray`) Uncalibrated uncertainty(MAD, median absolute deviation)
                    coming from ensembling. None if `ensemble_size = 1`
        """
        # Model-specific optimal default values leading to fast and reasonable results.
        if denoising_steps is None:
            denoising_steps = self.default_denoising_steps
        if processing_res is None:
            processing_res = self.default_processing_resolution

        assert processing_res >= 0
        assert ensemble_size >= 1

        # Check if denoising step is reasonable
        self._check_inference_step(denoising_steps)

        resample_method: InterpolationMode = get_tv_resample_method(resample_method)

        # ----------------- Image Preprocess -----------------
        # Convert to torch tensor
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
        rgb_norm: torch.Tensor = rgb / 255.0 * 2.0 - 1.0  #  [0, 255] -> [-1, 1]
        rgb_norm = rgb_norm.to(self.dtype)
        assert rgb_norm.min() >= -1.0 and rgb_norm.max() <= 1.0

        # 将偏振图像归一化到[-1,1]之间
        dolp_norm: torch.Tensor = pol[:, 1, :, :] / 255.0 * 2.0 - 1.0  # dolp:[-1, 1]
        aolp_rad: torch.Tensor = pol[:, 0, :, :] / 255.0 * torch.pi  # aolp_rad:[0, pi]
        aolp_cos = torch.cos(2 * aolp_rad)  # aolp_cos:[-1, 1]
        aolp_sin = torch.sin(2 * aolp_rad)  # aolp_sin:[-1, 1]
        pol_norm = torch.cat([dolp_norm, aolp_cos, aolp_sin], dim=0).unsqueeze(0)  # pol_norm:[1, 3, H, W]
        pol_norm = pol_norm.to(self.dtype)
        # print(rgb_norm.shape)
        # print(pol_norm.shape)
        # print(input_image_pol[:, 0, :, :].min(), input_image_pol[:, 0, :, :].max())
        # print(input_image_pol[:, 1, :, :].min(), input_image_pol[:, 1, :, :].max())
        # print(pol_norm[:, 0, :, :].min(), pol_norm[:, 0, :, :].max())
        assert pol_norm.min() >= -1.0 and pol_norm.max() <= 1.0

        # ----------------- Predicting normals -----------------
        # Batch repeated input image
        duplicated_rgb = rgb_norm.expand(ensemble_size, -1, -1, -1)
        duplicated_pol = pol_norm.expand(ensemble_size, -1, -1, -1)  # 扩展偏振图像
        # single_rgb_dataset = TensorDataset(duplicated_rgb)
        single_rgb_pol_dataset = TensorDataset(duplicated_rgb, duplicated_pol)
        if batch_size > 0:
            _bs = batch_size
        else:
            _bs = find_batch_size(
                ensemble_size=ensemble_size,
                input_res=max(rgb_norm.shape[1:]),
                dtype=self.dtype,
            )

        single_rgb_pol_loader = DataLoader(
            single_rgb_pol_dataset, batch_size=_bs, shuffle=False
        )

        # Predict normals maps (batched)
        target_pred_ls = []
        if show_progress_bar:
            iterable = tqdm(
                single_rgb_pol_loader, desc=" " * 2 + "Inference batches", leave=False
            )
        else:
            iterable = single_rgb_pol_loader
        for batch in iterable:
            # (batched_img,) = batch
            batched_rgb, batched_pol = batch
            target_pred_raw = self.single_infer(
                rgb_in=batched_rgb,
                pol_in=batched_pol,
                num_inference_steps=denoising_steps,
                show_pbar=show_progress_bar,
                generator=generator,
            )
            target_pred_ls.append(target_pred_raw.detach())
        target_preds = torch.concat(target_pred_ls, dim=0)
        torch.cuda.empty_cache()  # clear vram cache for ensembling

        # ----------------- Test-time ensembling -----------------
        if ensemble_size > 1:
            final_pred, pred_uncert = ensemble_normals(
                target_preds,
                **(ensemble_kwargs or {}),
            )
        else:
            final_pred = target_preds
            pred_uncert = None

        # Resize back to original resolution
        if match_input_res:
            final_pred = resize(
                final_pred,
                input_size[-2:],
                interpolation=resample_method,
                antialias=True,
            )

        # Convert to numpy
        final_pred = final_pred.squeeze()
        final_pred = final_pred.cpu().numpy()
        if pred_uncert is not None:
            pred_uncert = pred_uncert.squeeze().cpu().numpy()

        # Clip output range
        final_pred = final_pred.clip(-1, 1)

        # Colorize
        normals_img = ((final_pred + 1) * 127.5).astype(np.uint8)
        normals_img = chw2hwc(normals_img)
        normals_img = Image.fromarray(normals_img)

        return MarigoldNormalsOutput(
            normals_np=final_pred,
            normals_img=normals_img,
            uncertainty=pred_uncert,
        )

    def _check_inference_step(self, n_step: int) -> None:
        """
        Check if denoising step is reasonable
        Args:
            n_step (`int`): denoising steps
        """
        assert n_step >= 1

        if isinstance(self.scheduler, DDIMScheduler):
            if "trailing" != self.scheduler.config.timestep_spacing:
                logging.warning(
                    f"The loaded `DDIMScheduler` is configured with `timestep_spacing="
                    f'"{self.scheduler.config.timestep_spacing}"`; the recommended setting is `"trailing"`. '
                    f"This change is backward-compatible and yields better results. "
                    f"Consider using `prs-eth/marigold-normals-v1-1` for the best experience."
                )
            else:
                if n_step > 10:
                    logging.warning(
                        f"Setting too many denoising steps ({n_step}) may degrade the prediction; consider relying on "
                        f"the default values."
                    )
            if not self.scheduler.config.rescale_betas_zero_snr:
                logging.warning(
                    f"The loaded `DDIMScheduler` is configured with `rescale_betas_zero_snr="
                    f"{self.scheduler.config.rescale_betas_zero_snr}`; the recommended setting is True. "
                    f"Consider using `prs-eth/marigold-normals-v1-1` for the best experience."
                )
        elif isinstance(self.scheduler, LCMScheduler):
            raise RuntimeError(
                "This pipeline implementation does not support the LCMScheduler. Please refer to the project "
                "README.md for instructions about using LCM."
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
        Perform a single prediction without ensembling.

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
            `torch.Tensor`: Predicted targets.
        """
        device = self.device
        rgb_in = rgb_in.to(device)
        pol_in = pol_in.to(device)

        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps  # [T]

        # Encode image
        rgb_latent = self.encode_rgb(rgb_in)  # [B, 4, h, w]
        pol_latent = self.encode_pol(pol_in)  # [B, 4, h, w]
        latent_cat = torch.cat([rgb_latent, pol_latent], dim=1).to(device)  # [B, 8, h, w]
        self.confidence_predictor = self.confidence_predictor.to(device)
        alpha = self.confidence_predictor(latent_cat)  # [B, 1, h, w]
        z_fused = rgb_latent + alpha * (pol_latent - rgb_latent)  # [B, 4, h, w]

        # Noisy latent for outputs
        target_latent = torch.randn(
            z_fused.shape,
            device=device,
            dtype=self.dtype,
            generator=generator,
        )  # [B, 4, h, w]

        # Batched empty text embedding
        if self.empty_text_embed is None:
            self.encode_empty_text()
        batch_empty_text_embed = self.empty_text_embed.repeat(
            (z_fused.shape[0], 1, 1)
        ).to(device)  # [B, 2, 1024]

        # Denoising loop
        if show_pbar:
            iterable = tqdm(
                enumerate(timesteps),
                total=len(timesteps),
                leave=False,
                desc=" " * 4 + "Diffusion denoising",
            )
        else:
            iterable = enumerate(timesteps)

        for i, t in iterable:
            unet_input = torch.cat(
                [rgb_latent, target_latent], dim=1
            )  # this order is important

            # predict the noise residual
            noise_pred = self.unet(
                unet_input, t, encoder_hidden_states=batch_empty_text_embed
            ).sample  # [B, 4, h, w]

            # compute the previous noisy sample x_t -> x_t-1
            target_latent = self.scheduler.step(
                noise_pred, t, target_latent, generator=generator
            ).prev_sample

        normals = self.decode_normals(target_latent)  # [B,3,H,W]

        # clip prediction
        normals = torch.clip(normals, -1.0, 1.0)
        norm = torch.norm(normals, dim=1, keepdim=True)
        normals /= norm.clamp(min=1e-6)  # [B,3,H,W]

        return normals

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
        rgb_latent = mean * self.latent_scale_factor
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
        pol_latent = mean * self.latent_scale_factor
        return pol_latent

    def decode_normals(self, normals_latent: torch.Tensor) -> torch.Tensor:
        """
        Decode normals latent into normals map.

        Args:
            normals_latent (`torch.Tensor`):
                Normals latent to be decoded.

        Returns:
            `torch.Tensor`: Decoded normals map.
        """
        # scale latent
        normals_latent = normals_latent / self.latent_scale_factor
        # decode
        z = self.vae.post_quant_conv(normals_latent)
        stacked = self.vae.decoder(z)
        return stacked
