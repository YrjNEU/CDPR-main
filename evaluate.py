import argparse
import logging
import os
from datetime import datetime, timedelta
from typing import List

import torch
from omegaconf import OmegaConf
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from CDPR import CDPRDepthPipeline, CDPRPipeline
from src.dataset import BaseDepthDataset, DatasetMode, get_dataset
from src.dataset.mixed_sampler import MixedBatchSampler
from src.trainer import get_evaluator_cls
from src.util.config_util import (
    find_value_in_omegaconf,
    recursive_load_config,
)


if "__main__" == __name__:
    t_start = datetime.now()
    print(f"start at {t_start}")

    # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser(description="Evaluate your model!")
    # 评估相关参数
    parser.add_argument(
        "--config",
        type=str,
        default="config/eval.yaml",
        help="Path to config file."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default='checkpoint/depth/CDPR-polCNN',
        help="Path to the checkpoint directory for loading model weights."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default='output/output_per_sample',
        help="Directory to save evaluation results."
    )
    parser.add_argument(
        "--base_data_dir",
        type=str,
        default='.',
        help="Path to base data directory containing dataset."
    )
    parser.add_argument(
        "--no_cuda",
        action="store_true",
        help="Do not use CUDA even if it is available."
    )

    args = parser.parse_args()
    output_dir = args.output_dir
    # 除非在parser中设定了base_data_dir和base_ckpt_dir，否则读取环境变量下的路径
    # ！！！这里后续将路径改到parser里提前设定好
    # base_data_dir = (
    #     args.base_data_dir
    #     if args.base_data_dir is not None
    #     else os.environ["BASE_DATA_DIR"]
    # )
    # base_ckpt_dir = (
    #     args.base_ckpt_dir
    #     if args.base_ckpt_dir is not None
    #     else os.environ["BASE_CKPT_DIR"]
    # )

    # 配置参数
    checkpoint_path = args.checkpoint
    output_dir = args.output_dir
    base_data_dir = args.base_data_dir
    cfg = recursive_load_config(args.config)

    # 设备配置
    cuda_avail = torch.cuda.is_available() and not args.no_cuda
    device = torch.device("cuda" if cuda_avail else "cpu")
    logging.info(f"Device: {device}")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # -------------------- Data --------------------
    # 加载数据集配置
    cfg_data = cfg.dataset

    # 加载验证数据集
    val_loaders = []
    for _val_dic in cfg_data.val:
        _val_dataset = get_dataset(
            _val_dic,
            base_data_dir=base_data_dir,
            mode=DatasetMode.EVAL
        )
        _val_loader = DataLoader(
            dataset=_val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )
        val_loaders.append(_val_loader)

    # -------------------- Model --------------------
    # 从预训练检查点加载模型
    pipeline_cls_map = {
        "CDPRPipeline": CDPRPipeline,
        "CDPRDepthPipeline": CDPRDepthPipeline,
    }
    model = pipeline_cls_map[cfg.pipeline.name].from_pretrained(
        checkpoint_path, torch_dtype=torch.float16
    )

    # -------------------- Evaluator --------------------
    evalutaor_cls = get_evaluator_cls(cfg.evaluator.name)
    logging.debug(f"Evaluator: {evalutaor_cls}")
    evaluator = evalutaor_cls(
        cfg=cfg,
        model=model,
        val_dataloader=val_loaders,
        device=device,
        checkpoint_path=checkpoint_path,
        seed=cfg.validation.init_seed,
        out_dir_eval=output_dir,
    )
    evaluator.evaluate()

    logging.info(f"Evaluation finished. Results saved to {output_dir}")
