import os
import re
import numpy as np
import logging

logs = set()


def init_log(name, level=logging.INFO):
    if (name, level) in logs:
        return
    logs.add((name, level))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    if "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        logger.addFilter(lambda record: rank == 0)
    else:
        rank = 0
    format_str = "[%(asctime)s][%(levelname)8s] %(message)s"
    formatter = logging.Formatter(format_str)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


import torch


def compare_model_weights(model1, model2, output_file=None):
    print("Start Compare!")
    file_handle = open(output_file, "a") if output_file else None

    # Get the state dict of two models
    # state_dict1 = (
    #     model1.module.state_dict()
    #     if isinstance(model1, torch.nn.DataParallel)
    #     else model1.state_dict()
    # )
    # state_dict2 = (
    #     model2.module.state_dict()
    #     if isinstance(model2, torch.nn.DataParallel)
    #     else model2.state_dict()
    # )
    # from model.epde.utils import clean_pretrained_weight
    # state_dict2 = clean_pretrained_weight(state_dict2)
    # state_dict1 = clean_pretrained_weight(state_dict1)

    state_dict1 = model1
    state_dict2 = model2
    keys1 = set(state_dict1.keys())
    keys2 = set(state_dict2.keys())

    missing_in_model1 = keys2 - keys1  # Keys present in model2 but not in model1
    missing_in_model2 = keys1 - keys2  # Keys present in model1 but not in model2

    all_match = True

    if missing_in_model1 or missing_in_model2:
        all_match = False
        mismatch_info = "State dict keys do not match.\n"
        print(mismatch_info)
        if file_handle:
            file_handle.write(mismatch_info)

        if missing_in_model1:
            mismatch_info = f"Keys missing in model1: {missing_in_model1}\n"
            print(mismatch_info)
            if file_handle:
                file_handle.write(mismatch_info)

        if missing_in_model2:
            mismatch_info = f"Keys missing in model2: {missing_in_model2}\n"
            print(mismatch_info)
            if file_handle:
                file_handle.write(mismatch_info)
        
    common_keys = keys1.intersection(keys2)
    for key in common_keys:
        if not torch.allclose(state_dict1[key], state_dict2[key]):
            all_match = False
            mismatch_info = f"Weight mismatch found at layer: {key}\n"
            print(mismatch_info)
            if file_handle:
                file_handle.write(mismatch_info)
                file_handle.write(f"Model 1 tensor: {state_dict1[key]}\n")
                file_handle.write(f"Model 2 tensor: {state_dict2[key]}\n")
                file_handle.write("-" * 80 + "\n")
        else:
            file_handle.write(f"Key {key} is matched between model1 and model2\n")

    if all_match:
            print("All weights match.")
    return all_match
