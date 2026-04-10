# Author: Bingxin Ke
# Last modified: 2024-05-17

from .cdpr_depth_trainer_v1 import CDPRDepthTrainer
from .cdpr_evaluator import CDPREvaluator
from .cdpr_trainer import CDPRTrainer


trainer_cls_name_dict = {
    "CDPRDepthTrainer": CDPRDepthTrainer,
    "CDPRTrainer": CDPRTrainer,
}

evaluator_cls_name_dict = {
    "CDPREvaluator": CDPREvaluator,
}

def get_trainer_cls(trainer_name):
    return trainer_cls_name_dict[trainer_name]

def get_evaluator_cls(evaluator_name):
    return evaluator_cls_name_dict[evaluator_name]
