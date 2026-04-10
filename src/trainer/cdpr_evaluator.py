import logging
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List
import numpy as np
from PIL import Image
from CDPR.CDPR_depth_pipeline import CDPRPipeline, CDPRDepthOutput
from src.util import metric
from src.util.metric import MetricTracker
from src.util.logging_util import tb_logger, eval_dict_to_text
from src.util.alignment import align_depth_least_square
from src.util.seeding import generate_seed_sequence
import random


class CDPREvaluator:
    def __init__(
            self,
            cfg,
            model: CDPRPipeline,
            # val_dataloader: DataLoader,
            device,
            checkpoint_path: str,
            out_dir_eval: str,
            seed: int,
            val_dataloader: List[DataLoader] = None,
    ):
        """
        Initialize the Marigold Evaluator.

        Args:
            cfg (OmegaConf): The configuration object containing evaluation parameters.
            model (CDPRPipeline): The model to be evaluated.
            val_dataloader (DataLoader): DataLoader for the validation dataset.
            device (torch.device): Device (CPU/GPU) for evaluation.
            checkpoint_path (str): Path to the checkpoint directory for loading model weights.
            out_dir_eval (str): Output directory for saving evaluation results.
        """
        self.cfg = cfg
        self.model = model
        self.device = device
        # self.val_loader = val_dataloader
        self.val_loader: List[DataLoader] = val_dataloader
        self.out_dir_eval = out_dir_eval
        # Load the model checkpoint (weights)
        self.load_checkpoint(checkpoint_path)

        # Evaluation metrics setup
        self.metric_funcs = [getattr(metric, _met) for _met in cfg.eval.eval_metrics]
        self.val_metrics = MetricTracker(*[m.__name__ for m in self.metric_funcs])

        # seed = cfg.validation.init_seed
        self.set_random_seed(seed)

        confidence_path = os.path.join(checkpoint_path, "confidence_predictor.pth")
        if os.path.exists(confidence_path):
            logging.info(f"Loading confidence predictor from {confidence_path}")
            state_dict = torch.load(confidence_path, map_location=device)
            self.model.confidence_predictor.load_state_dict(state_dict)
            self.model.confidence_predictor.to(device)
            self.model.confidence_predictor.eval()
            self.model.confidence_predictor.requires_grad_(False)

        else:
            logging.warning(f"Confidence predictor file not found at: {confidence_path}")

        assert not self.model.confidence_predictor.training, "confidence_predictor is still in training mode!"

    def set_random_seed(self, seed):
        """
        Set the random seed for reproducibility.
        """
        random.seed(seed)  # Python's random module
        np.random.seed(seed)  # Numpy's random module
        torch.manual_seed(seed)  # PyTorch CPU seed
        torch.cuda.manual_seed_all(seed)  # PyTorch GPU seed
        torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
        torch.backends.cudnn.benchmark = False  # Disable the cudnn auto-tuner

    def load_checkpoint(self, checkpoint_path):
        """
        Load the pre-trained weights from the checkpoint directory.
        """
        logging.info(f"Loading checkpoint from: {checkpoint_path}")

        self.model = self.model.from_pretrained(checkpoint_path)
        self.model.unet.eval()
        self.model.vae.eval()
        self.model.unet.requires_grad_(False)

    @torch.no_grad()
    def evaluate(self):
        per_sample_filename = os.path.join(self.out_dir_eval, "per_sample_metrics.csv")
        # write title
        with open(per_sample_filename, "w+") as f:
            f.write("filename,")
            f.write(",".join([m.__name__ for m in self.metric_funcs]))
            f.write("\n")
        for i, val_loader in enumerate(self.val_loader):
            val_dataset_name = val_loader.dataset.disp_name
            # val_metric_dic = self.validate_single_dataset(
            #     data_loader=val_loader, metric_tracker=self.val_metrics
            # )

            global_seed_list = generate_seed_sequence(
                initial_seed=self.cfg.validation.init_seed,
                length=len(val_loader)
            )

            val_metric_dic = self.validate_single_dataset(
                data_loader=val_loader,
                metric_tracker=self.val_metrics,
                seed_list=global_seed_list
            )

            # save to file
            eval_text = eval_dict_to_text(
                val_metrics=val_metric_dic,
                dataset_name=val_dataset_name,
                sample_list_path=val_loader.dataset.filename_ls_path,
            )
            _save_to = os.path.join(
                self.out_dir_eval,
                f"eval.txt",
            )
            with open(_save_to, "w+") as f:
                f.write(eval_text)


    @torch.no_grad()
    def validate_single_dataset(
            self,
            data_loader: DataLoader,
            metric_tracker: MetricTracker,
            seed_list,
            save_to_dir: str = None,
    ):
        self.model.to(self.device)
        metric_tracker.reset()
        all_alpha_means = []

        # Generate seed sequence for consistent evaluation
        # val_init_seed = self.cfg.validation.init_seed
        # val_seed_ls = generate_seed_sequence(val_init_seed, len(data_loader))
        val_seed_ls = list(seed_list)

        if self.out_dir_eval is not None:
            csv_path = os.path.join(self.out_dir_eval, "per_sample_metrics.csv")
            with open(csv_path, "w") as f:
                f.write("filename," + ",".join([m.__name__ for m in self.metric_funcs]) + "\n")
        for i, batch in enumerate(
                tqdm(data_loader, desc=f"evaluating on {data_loader.dataset.disp_name}"),
                start=1,
        ):
            assert 1 == data_loader.batch_size
            # Read input image
            rgb_int = batch["rgb_int"]  # [B, 3, H, W]
            pol_raw = batch["pol_raw"]  # [B, 3, H, W]
            # GT depth
            depth_raw_ts = batch["depth_raw_linear"].squeeze()
            depth_raw = depth_raw_ts.numpy()
            depth_raw_ts = depth_raw_ts.to(self.device)
            valid_mask_ts = batch["valid_mask_raw"].squeeze()
            valid_mask = valid_mask_ts.numpy()
            valid_mask_ts = valid_mask_ts.to(self.device)

            # Random number generator
            seed = val_seed_ls.pop()
            if seed is None:
                generator = None
            else:
                generator = torch.Generator(device=self.device)
                generator.manual_seed(seed)

            # Predict depth
            pipe_out: CDPRDepthOutput = self.model(
                rgb_int,
                pol_raw,
                denoising_steps=self.cfg.validation.denoising_steps,
                ensemble_size=self.cfg.validation.ensemble_size,
                processing_res=self.cfg.validation.processing_res,
                match_input_res=self.cfg.validation.match_input_res,
                generator=generator,
                batch_size=1,  # use batch size 1 to increase reproducibility
                color_map=None,
                show_progress_bar=False,
                resample_method=self.cfg.validation.resample_method,
            )

            # ---- record alpha ----
            if hasattr(pipe_out, "alpha") and pipe_out.alpha is not None:
                alpha_mean = pipe_out.alpha.mean()
                all_alpha_means.append(alpha_mean)

            depth_pred: np.ndarray = pipe_out.depth_np

            if "least_square" == self.cfg.eval.alignment:
                depth_pred, scale, shift = align_depth_least_square(
                    gt_arr=depth_raw,
                    pred_arr=depth_pred,
                    valid_mask_arr=valid_mask,
                    return_scale_shift=True,
                    max_resolution=self.cfg.eval.align_max_res,
                )
            else:
                raise RuntimeError(f"Unknown alignment type: {self.cfg.eval.alignment}")

            # Clip to dataset min max
            depth_pred = np.clip(
                depth_pred,
                a_min=data_loader.dataset.min_depth,
                a_max=data_loader.dataset.max_depth,
            )

            # clip to d > 0 for evaluation
            depth_pred = np.clip(depth_pred, a_min=1e-6, a_max=None)

            # Evaluate
            sample_metric = []
            depth_pred_ts = torch.from_numpy(depth_pred).to(self.device)

            for met_func in self.metric_funcs:
                _metric_name = met_func.__name__
                _metric = met_func(depth_pred_ts, depth_raw_ts, valid_mask_ts).item()
                sample_metric.append(_metric.__str__())
                metric_tracker.update(_metric_name, _metric)

            if self.out_dir_eval is not None:
                filename = batch["rgb_relative_path"][0]  # 相对路径名
                with open(csv_path, "a") as f:
                    f.write(filename + ",")
                    f.write(",".join(sample_metric) + "\n")

            # Save as 16-bit uint png
            if save_to_dir is not None:
                img_name = batch["rgb_relative_path"][0].replace("/", "_")
                png_save_path = os.path.join(save_to_dir, f"{img_name}.png")
                depth_to_save = (pipe_out.depth_np * 65535.0).astype(np.uint16)
                Image.fromarray(depth_to_save).save(png_save_path, mode="I;16")

            # ---- save mean alpha for this dataset ----
            if self.out_dir_eval is not None and len(all_alpha_means) > 0:
                alpha_mean_all_samples = float(np.mean(all_alpha_means))
                save_path = os.path.join(self.out_dir_eval, "alpha_mean.txt")
                with open(save_path, "w") as f:
                    f.write(str(alpha_mean_all_samples))
                print(f"[Alpha] mean alpha saved to {save_path}: {alpha_mean_all_samples}")

        return metric_tracker.result()
