import argparse
import logging
import os
import pprint
import random

import warnings
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from dataset.eventscape_align import EventScape_Align

from model.depth_anything_v2.dpt_align import DepthAnythingV2
from model.depth_anything_v2.dpt_align_lora import DepthAnythingV2_lora

from util.dist_helper import setup_distributed
from util.loss import FeatureCosLoss, L1_Loss
from util.metric import eval_disparity
from util.utils import init_log

import loralib as lora

model_configs = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {
        "encoder": "vitb",
        "features": 128,
        "out_channels": [96, 192, 384, 768],
    },
    "vitl": {
        "encoder": "vitl",
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
    },
    "vitg": {
        "encoder": "vitg",
        "features": 384,
        "out_channels": [1536, 1536, 1536, 1536],
    },
}


parser = argparse.ArgumentParser(
    description="Depth Anything V2 for Metric Depth Estimation"
)

parser.add_argument(
    "--encoder", default="vitl", choices=["vits", "vitb", "vitl", "vitg"]
)
parser.add_argument(
    "--dataset",
    choices=["mvsec", "eventscape", "eventscape_align"],
)

parser.add_argument("--img-size", default=518, type=int)
parser.add_argument("--epochs", default=40, type=int)
parser.add_argument("--bs", default=2, type=int)
parser.add_argument("--lr", default=0.000005, type=float)
# parser.add_argument("--pretrained-from", type=str)
parser.add_argument("--load-from", type=str)
parser.add_argument("--save-path", type=str, required=True)
parser.add_argument("--local-rank", default=0, type=int)
parser.add_argument("--port", default=None, type=int)
parser.add_argument("--event_voxel_chans", default=5, type=int)
parser.add_argument("--return-feature", action="store_true")


def get_dataloader(args):
    size = (args.img_size, args.img_size)
    trainset = EventScape_Align(
        "dataset/splits/eventscape/train.txt",
        "train",
        size=size,
    )

    trainsampler = torch.utils.data.distributed.DistributedSampler(
        trainset, shuffle=True
    )
    trainloader = DataLoader(
        trainset,
        batch_size=args.bs,
        pin_memory=True,
        num_workers=4,
        drop_last=True,
        sampler=trainsampler,
    )

    valset = EventScape_Align(
        "dataset/splits/eventscape/val_1k.txt",
        "val",
        size=size,
    )

    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(
        valset,
        batch_size=1,
        pin_memory=True,
        num_workers=4,
        drop_last=True,
        sampler=valsampler,
    )

    return trainloader, valloader


def main():
    args = parser.parse_args()

    warnings.simplefilter("ignore", np.RankWarning)

    logger = init_log("global", logging.INFO)
    logger.propagate = 0

    rank, world_size = setup_distributed(port=args.port)

    if rank == 0:
        all_args = {**vars(args), "ngpus": world_size}
        logger.info(f"{pprint.pformat(all_args)}\n")
        writer = SummaryWriter(args.save_path)

    cudnn.enabled = True
    cudnn.benchmark = True

    # Data Loader
    trainloader, valloader = get_dataloader(args=args)

    local_rank = int(os.environ["LOCAL_RANK"])

    teacher_model = DepthAnythingV2(
        **{
            **model_configs[args.encoder],
            "return_feature": args.return_feature,
        }
    )
    student_model = DepthAnythingV2_lora(
        **{
            **model_configs[args.encoder],
            "return_feature": args.return_feature,
        }
    )

    checkpoint = torch.load(args.load_from, map_location="cpu")
    if "model" in checkpoint.keys():
        checkpoint = checkpoint["model"]
        checkpoint = {
            (key[7:] if key.startswith("module.") else key): value
            for key, value in checkpoint.items()
        }

    teacher_model.eval()
    student_model.eval()
    teacher_model.load_state_dict(checkpoint, strict=False)
    student_model.load_state_dict(checkpoint, strict=False)
    print(f"Model weights load from {args.load_from} successfully!")

    # Train student models in parallel
    student_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(student_model)
    student_model.cuda(local_rank)
    student_model = torch.nn.parallel.DistributedDataParallel(
        student_model,
        device_ids=[local_rank],
        broadcast_buffers=False,
        output_device=local_rank,
        find_unused_parameters=True,
    )

    # Place teacher on the same device and set teacher to eval mode
    teacher_model = teacher_model.cuda(local_rank)
    teacher_model.eval()

    # Train student by LoRA
    lora.mark_only_lora_as_trainable(student_model)
    for name, param in student_model.named_parameters():
        # Add cls_token, patch_embed, pos_embed for train
        if "pretrained" in name and "blocks" not in name:
            param.requires_grad = True
        if "pretrained" not in name:
            param.requires_grad = False

    feature_loss = FeatureCosLoss(beta=0.2).cuda(local_rank)
    l1_loss = L1_Loss(log_normalized=False).cuda(local_rank)

    # Configure optimizer to include only trainable parameters
    optimizer = AdamW(
        [
            {
                "params": [
                    param
                    for name, param in student_model.named_parameters()
                    if "pretrained" in name and "lora" not in name
                ],
                "lr": args.lr,
            },
            {
                "params": [
                    param
                    for name, param in student_model.named_parameters()
                    if "pretrained" not in name
                ],
                "lr": args.lr * 10.0,
            },
            {
                "params": [
                    param
                    for name, param in student_model.named_parameters()
                    if "lora" in name
                ],
                "lr": args.lr * 10.0,
            },
        ],
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )

    total_iters = args.epochs * len(trainloader)

    previous_best = {
        "d1": 0,
        "d2": 0,
        "d3": 0,
        "abs_rel": 100,
        "sq_rel": 100,
        "rmse": 100,
        "rmse_log": 100,
        "log10": 100,
        "silog": 100,
    }

    if rank == 0:
        # Log module names and trainable parameter counts
        for name, param in student_model.named_parameters():
            if param.requires_grad:
                logger.info(f"Module: {name}, Trainable Parameters: {param.numel()}")

        total_trainable_params = sum(
            p.numel() for p in student_model.parameters() if p.requires_grad
        )
        logger.info(f"Total Trainable Parameters: {total_trainable_params}")

    for epoch in range(args.epochs):
        if rank == 0:
            logger.info(
                "===========> Epoch: {:}/{:}, d1: {:.3f}, d2: {:.3f}, d3: {:.3f}".format(
                    epoch,
                    args.epochs,
                    previous_best["d1"],
                    previous_best["d2"],
                    previous_best["d3"],
                )
            )
            logger.info(
                "===========> Epoch: {:}/{:}, abs_rel: {:.3f}, sq_rel: {:.3f}, rmse: {:.3f}, rmse_log: {:.3f}, "
                "log10: {:.3f}, silog: {:.3f}".format(
                    epoch,
                    args.epochs,
                    previous_best["abs_rel"],
                    previous_best["sq_rel"],
                    previous_best["rmse"],
                    previous_best["rmse_log"],
                    previous_best["log10"],
                    previous_best["silog"],
                )
            )

        trainloader.sampler.set_epoch(epoch + 1)

        student_model.train()

        for i, sample in enumerate(trainloader):
            optimizer.zero_grad()

            img, voxel = (
                sample["image"].cuda(),
                sample["event_voxel"].cuda(),
            )

            if random.random() < 0.5:
                img = img.flip(-1)
                voxel = voxel.flip(-1)

            # Teacher output
            with torch.no_grad():
                teacher_pred, teacher_features = teacher_model(img)
            student_pred, student_features = student_model(voxel)

            valid_mask = torch.ones_like(student_pred, dtype=torch.bool)
            fea_loss = feature_loss(student_features, teacher_features)
            l1 = l1_loss(student_pred, teacher_pred, valid_mask)

            total_loss = fea_loss + l1
            total_loss.backward()
            optimizer.step()

            iters = epoch * len(trainloader) + i

            lr = args.lr * (1 - iters / total_iters) ** 0.9

            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * 10.0
            optimizer.param_groups[2]["lr"] = lr * 10.0

            if rank == 0:
                writer.add_scalar("train/loss", total_loss.item(), iters)
                writer.add_scalar("train/l1_loss", l1.item(), iters)
                writer.add_scalar("train/feature_loss", fea_loss.item(), iters)

            if rank == 0 and i % 100 == 0:
                logger.info(
                    "Iter: {}/{}, LR: {:.7f}, Loss: {:.3f}, l1_loss: {:.3f}, fea_loss: {:.3f}".format(
                        i,
                        len(trainloader),
                        optimizer.param_groups[0]["lr"],
                        total_loss.item(),
                        l1.item(),
                        fea_loss.item(),
                    )
                )

        student_model.eval()

        results = {
            "d1": torch.tensor([0.0]).cuda(),
            "d2": torch.tensor([0.0]).cuda(),
            "d3": torch.tensor([0.0]).cuda(),
            "abs_rel": torch.tensor([0.0]).cuda(),
            "sq_rel": torch.tensor([0.0]).cuda(),
            "rmse": torch.tensor([0.0]).cuda(),
            "rmse_log": torch.tensor([0.0]).cuda(),
            "log10": torch.tensor([0.0]).cuda(),
            "silog": torch.tensor([0.0]).cuda(),
        }

        nsamples = torch.tensor([0.0]).cuda()

        for i, sample in enumerate(valloader):

            img, voxel = (
                sample["image"].cuda(),
                sample["event_voxel"].cuda(),
            )

            with torch.no_grad():
                student_pred, _ = student_model(voxel)
                teacher_pred, _ = teacher_model(img)

            valid_mask = torch.ones_like(student_pred, dtype=bool)
            student_pred = student_pred[valid_mask]
            teacher_pred = teacher_pred[valid_mask]

            cur_results = eval_disparity(student_pred, teacher_pred)

            for k in results.keys():
                results[k] += cur_results[k]
            nsamples += 1

        torch.distributed.barrier()

        for k in results.keys():
            dist.reduce(results[k], dst=0)
        dist.reduce(nsamples, dst=0)

        if rank == 0:
            logger.info(
                "=========================================================================================="
            )
            logger.info(
                "{:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}".format(
                    *tuple(results.keys())
                )
            )
            logger.info(
                "{:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}".format(
                    *tuple([(v / nsamples).item() for v in results.values()])
                )
            )
            logger.info(
                "=========================================================================================="
            )
            print()

            for name, metric in results.items():
                writer.add_scalar(f"eval/{name}", (metric / nsamples).item(), epoch)

        cur_results = {}
        for k in results.keys():
            cur_results[k] = (results[k] / nsamples).item()

        for k in results.keys():
            if k in ["d1", "d2", "d3"]:
                previous_best[k] = max(previous_best[k], (results[k] / nsamples).item())
            else:
                previous_best[k] = min(previous_best[k], (results[k] / nsamples).item())

        if rank == 0:
            checkpoint = {
                "model": student_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "previous_best": previous_best,
            }
            torch.save(checkpoint, os.path.join(args.save_path, "latest.pth"))


if __name__ == "__main__":
    main()
