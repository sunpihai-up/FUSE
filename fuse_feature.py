import argparse
import logging
from math import fabs
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
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from dataset.dense import Dense
from dataset.mvsec import MVSEC
from dataset.eventscape import EventScape
from dataset.eventscape_align import EventScape_Align
from dataset.eventscape_fuse import EventScape_Fuse, EventScape_Fuse_Cor

from model.depth_anything_v2.dpt_align import DepthAnythingV2
from model.depth_anything_v2.dpt_align_lora import DepthAnythingV2_lora
from model.epde_modal import EPDE

from util.dist_helper import setup_distributed
from util.loss import SiLogLoss, FeatureCosLoss, MixedLoss, F1_Loss
from util.metric import eval_depth, eval_depth_ori, eval_disparity
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
    choices=[
        "mvsec",
        "eventscape",
        "eventscape_align",
        "eventscape_fuse",
        "eventscape_fuse_cor",
    ],
)

parser.add_argument("--img-size", default=518, type=int)
parser.add_argument("--epochs", default=40, type=int)
parser.add_argument("--bs", default=2, type=int)
parser.add_argument("--lr", default=0.000005, type=float)
parser.add_argument("--max-depth", type=int)

parser.add_argument("--load-from", type=str)
parser.add_argument("--prompt-encoder-pretrained", type=str)
parser.add_argument("--depth-anything-pretrained", type=str)

parser.add_argument("--save-path", type=str, required=True)
parser.add_argument("--local-rank", default=0, type=int)
parser.add_argument("--port", default=None, type=int)
parser.add_argument("--event_voxel_chans", default=5, type=int)
parser.add_argument("--return-feature", action="store_true")
parser.add_argument("--half", action="store_true")


def get_dataloader(args):
    size = (args.img_size, args.img_size)

    if args.dataset == "mvsec":
        trainset = MVSEC(
            "dataset/splits/mvsec/train.txt",
            "train",
            normalized_d=args.normalized_depth,
            size=size,
        )
    elif args.dataset == "eventscape":
        trainset = EventScape(
            "dataset/splits/eventscape/train.txt",
            "train",
            normalized_d=args.normalized_depth,
            size=size,
        )
    elif args.dataset == "eventscape_align":
        trainset = EventScape_Align(
            "dataset/splits/eventscape/train.txt",
            "train",
            size=size,
        )
    elif args.dataset == "eventscape_fuse":
        trainset = EventScape_Fuse(
            "dataset/splits/eventscape/train.txt",
            "train",
            size=size,
        )
    elif args.dataset == "eventscape_fuse_cor":
        trainset = EventScape_Fuse_Cor(
            "dataset/splits/eventscape/train.txt",
            "train",
            size=size,
        )
    else:
        raise NotImplementedError

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

    if args.dataset == "dense":
        valset = Dense("dataset/splits/dense/val.txt", "val", size=size)
    elif args.dataset == "mvsec":
        valset = MVSEC(
            "./dataset/splits/mvsec/outdoor_night1_val.txt",
            "val",
            normalized_d=args.normalized_depth,
            size=size,
        )
    elif args.dataset == "eventscape":
        valset = EventScape(
            "./dataset/splits/eventscape/val_1k.txt",
            "val",
            normalized_d=args.normalized_depth,
            size=size,
        )
    elif args.dataset == "eventscape_align":
        valset = EventScape_Align(
            "dataset/splits/eventscape/val_1k.txt",
            "val",
            size=size,
        )
    elif args.dataset == "eventscape_fuse":
        valset = EventScape_Fuse(
            "dataset/splits/eventscape/val_1k.txt",
            "val",
            size=size,
        )
    elif args.dataset == "eventscape_fuse_cor":
        valset = EventScape_Fuse_Cor(
            "dataset/splits/eventscape/val_1k.txt",
            "val",
            size=size,
        )
    else:
        raise NotImplementedError

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
    student_model = EPDE(
        model_name=args.encoder,
        max_depth=args.max_depth,
        event_voxel_chans=args.event_voxel_chans,
        return_feature=args.return_feature,
        prompt_encoder_pretrained=args.prompt_encoder_pretrained,
        depth_anything_pretrained=args.depth_anything_pretrained,
    )

    checkpoint = torch.load(args.load_from, map_location="cpu")
    if "model" in checkpoint.keys():
        checkpoint = checkpoint["model"]
        checkpoint = {
            (key[7:] if key.startswith("module.") else key): value
            for key, value in checkpoint.items()
        }
    teacher_model.load_state_dict(checkpoint, strict=False)
    print(f"Model weights load from {args.load_from} successfully!")

    if args.half:
        student_model = student_model.half()
        teacher_model = teacher_model.half()

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

    for name, param in student_model.named_parameters():
        # Make module about feature fusion and read out trainable
        param.requires_grad = "prompt_fuse" in name or "read_out" in name

    # criterion = SiLogLoss().cuda(local_rank)
    criterion = MixedLoss(log_normalize=True).cuda(local_rank)
    feature_loss = FeatureCosLoss(beta=0).cuda(local_rank)
    l1_loss = F1_Loss(log_normalized=True).cuda(local_rank)

    # Configure optimizer to include only trainable parameters
    optimizer = AdamW(
        [
            {
                "params": [
                    param
                    for name, param in student_model.named_parameters()
                    if "encoder" in name and param.requires_grad
                ],
                "lr": args.lr,
            },
            {
                "params": [
                    param
                    for name, param in student_model.named_parameters()
                    if "encoder" not in name and param.requires_grad
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

            img, img_voxel = (
                sample["image"].cuda(),
                sample["input"].cuda(),
            )

            if args.half:
                img = img.half()
                img_voxel = img_voxel.half()
            
            if random.random() < 0.5:
                img = img.flip(-1)
                img_voxel = img_voxel.flip(-1)

            # img = img[0].cpu().numpy().transpose(1, 2, 0)
            # img_cor = img_voxel[0][:3,].cpu().numpy().transpose(1, 2, 0)
            # voxel = img_voxel[0][3:,].cpu().numpy().transpose(1, 2, 0)
            # print(img.shape, voxel.shape, img_cor.shape)

            # import cv2
            # img = (img - img.min()) / (img.max() - img.min()) * 255.0
            # img = img.astype(np.uint8)
            # cv2.imwrite(f"{i}_img.png", img)

            # img_cor = (img_cor - img_cor.min()) / (img_cor.max() - img_cor.min()) * 255.0
            # img_cor = img_cor.astype(np.uint8)
            # cv2.imwrite(f"{i}_img_cor.png", img_cor)

            # voxel = (voxel - voxel.min()) / (voxel.max() - voxel.min()) * 255.0
            # voxel = voxel.astype(np.uint8)
            # cv2.imwrite(f"{i}_voxel.png", voxel)

            # if i >= 5:
            #     exit()
            # continue

            # Teacher output
            with torch.no_grad():
                teacher_pred, teacher_features = teacher_model(img)
            student_pred, student_features = student_model(img_voxel)


            valid_mask = torch.ones_like(student_pred, dtype=torch.bool)
            loss, si_loss, grad_loss = criterion(student_pred, teacher_pred, valid_mask)
            fea_loss = feature_loss(student_features, teacher_features)
            l1 = l1_loss(student_pred, teacher_pred, valid_mask)
            # print(student_pred.shape, teacher_pred.shape)
            # print(student_pred.min(), teacher_pred.min())
            # print(student_pred.max(), teacher_pred.max())
            # print(student_features[0].shape, teacher_features[0].shape)
            # print("*****************************************************")
            # print(f"l1: {l1}, fea_loss: {fea_loss}, si_loss: {si_loss}, grad_loss: {grad_loss}")
            # if i >= 5:
            #     exit()
            
            # total_loss = loss + fea_loss
            # total_loss = si_loss
            total_loss = l1 + fea_loss
            # total_loss = l1
            # print(total_loss)
            total_loss.backward()
            optimizer.step()

            iters = epoch * len(trainloader) + i

            lr = args.lr * (1 - iters / total_iters) ** 0.9

            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * 10.0
            # optimizer.param_groups[2]["lr"] = lr * 10.0

            if rank == 0:
                writer.add_scalar("train/loss", loss.item(), iters)
                writer.add_scalar("train/si_loss", si_loss.item(), iters)
                writer.add_scalar("train/grad_loss", grad_loss.item(), iters)
                writer.add_scalar("train/feature_loss", fea_loss.item(), iters)
                writer.add_scalar("train/feature_loss", total_loss.item(), iters)

            if rank == 0 and i % 100 == 0:
                logger.info(
                    "Iter: {}/{}, LR: {:.7f}, Loss: {:.3f}, si_loss: {:.3f}, grad_loss: {:.3f}, fea_loss: {:.3f}".format(
                        i,
                        len(trainloader),
                        optimizer.param_groups[0]["lr"],
                        total_loss.item(),
                        si_loss.item(),
                        grad_loss.item(),
                        fea_loss.item(),
                    )
                )

            if iters % 2000 == 0 and iters >= 2000 and rank == 0:
                checkpoint = {
                    "model": student_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "previous_best": previous_best,
                }
                torch.save(checkpoint, os.path.join(args.save_path, "latest.pth"))

            # if i >= 500:
            #     break
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

        save_metric = ["d1", "abs_rel", "rmse"]
        nsamples = torch.tensor([0.0]).cuda()

        for i, sample in enumerate(valloader):

            img, img_voxel = (
                sample["image"].cuda(),
                sample["input"].cuda(),
            )

            if args.half:
                img = img.half()
                img_voxel = img_voxel.half()

            with torch.no_grad():
                student_pred, _ = student_model(img_voxel)
                teacher_pred, _ = teacher_model(img)

            # print(student_pred.shape, teacher_pred.shape)
            # print(student_pred.min(), teacher_pred.min())
            # print(student_pred.max(), teacher_pred.max())
            # diff = abs(student_pred - teacher_pred)
            # print("*********** Diff **************")
            # print(diff.min(), diff.max(), diff.mean())

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

        if rank == 0 and (epoch + 1) % 20 == 0:
            checkpoint = {
                "model": student_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "previous_best": previous_best,
            }
            torch.save(checkpoint, os.path.join(args.save_path, f"{epoch}.pth"))

        cur_results = {}
        for k in results.keys():
            cur_results[k] = (results[k] / nsamples).item()

        if rank == 0:
            for k in save_metric:
                flag = False
                if k in ["d1", "d2", "d3"] and cur_results[k] > previous_best[k]:
                    flag = True
                if k not in ["d1", "d2", "d3"] and cur_results[k] < previous_best[k]:
                    flag = True

                if flag:
                    # Delate previous checkpoint
                    for file in os.listdir(args.save_path):
                        if file.startswith(k) and file.endswith(".pth"):
                            os.remove(os.path.join(args.save_path, file))
                    torch.save(
                        student_model.state_dict(),
                        os.path.join(
                            args.save_path, f"{k}-{cur_results[k]}-{epoch}.pth"
                        ),
                    )

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
