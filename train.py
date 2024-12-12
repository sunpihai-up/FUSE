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

from model.epde.epde import EPDE
from util.dist_helper import setup_distributed
from util.loss import SiLogLoss
from util.metric import eval_depth
from util.utils import init_log


parser = argparse.ArgumentParser(
    description="Depth Anything V2 for Metric Depth Estimation"
)

parser.add_argument(
    "--encoder", default="vitl", choices=["vits", "vitb", "vitl", "vitg"]
)
parser.add_argument(
    "--dataset",
    default="dense",
    choices=["dense", "mvsec", "eventscape"],
)
parser.add_argument("--img-size", default=518, type=int)
parser.add_argument("--epochs", default=40, type=int)
parser.add_argument("--bs", default=2, type=int)
parser.add_argument("--lr", default=0.000005, type=float)
parser.add_argument("--depth-anything-pretrained", type=str)
parser.add_argument("--pretrained-from", type=str)
parser.add_argument("--save-path", type=str, required=True)
parser.add_argument("--local-rank", default=0, type=int)
parser.add_argument("--port", default=None, type=int)
parser.add_argument(
    "--finetune-mode",
    choices=["foundation_all_frozen", "foundation_encoders_frozen", "nothing_frozen"],
    default="foundation_all_frozen",
    type=str,
    help="Specify which parameters to freeze during fine-tuning. Options are: "
    "'foundation_all_frozen' (freeze all foundation params), "
    "'foundation_encoders_frozen' (freeze only encoder params in foundation), "
    "'nothing_frozen' (train all params).",
)


def main():
    args = parser.parse_args()

    warnings.simplefilter("ignore", np.RankWarning)

    logger = init_log("global", logging.INFO)
    logger.propagate = 0

    rank, world_size = setup_distributed(port=args.port)

    if rank == 0:
        all_args = {**vars(args), "ngpus": world_size}
        logger.info("{}\n".format(pprint.pformat(all_args)))
        writer = SummaryWriter(args.save_path)

    cudnn.enabled = True
    cudnn.benchmark = True

    size = (args.img_size, args.img_size)
    if args.dataset == "dense":
        trainset = Dense("dataset/splits/dense/train.txt", "train", size=size)
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

    local_rank = int(os.environ["LOCAL_RANK"])

    # Instantiate Model
    model = EPDE(
        model_name=args.encoder,
        dataset=args.dataset,
        depth_anything_pretrained=args.depth_anything_pretrained,
    )

    if args.pretrained_from:
        model.load_state_dict(
            {
                k: v
                for k, v in torch.load(args.pretrained_from, map_location="cpu").items()
                if "pretrained" in k
            },
            strict=False,
        )

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        broadcast_buffers=False,
        output_device=local_rank,
        find_unused_parameters=True,
    )

    criterion = SiLogLoss().cuda(local_rank)

    # Handling frozen parameters
    if args.finetune_mode == "foundation_all_frozen":
        for name, param in model.named_parameters():
            if 'foundation' in name:
                param.requires_grad = False
    elif args.finetune_mode == "foundation_encoders_frozen":
        for name, param in model.named_parameters():
            if 'foundation.pretrained' in name:
                param.requires_grad = False
    print(f'The freezing mode of weights is: {args.finetune_mode}')

    # Configure optimizer to include only trainable parameters
    optimizer = AdamW(
        [
            {
                "params": [
                    param
                    for name, param in model.named_parameters()
                    if "pretrained" in name and param.requires_grad
                ],
                "lr": args.lr,
            },
            {
                "params": [
                    param
                    for name, param in model.named_parameters()
                    if "pretrained" not in name and param.requires_grad
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
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info(f"Module: {name}, Trainable Parameters: {param.numel()}")

        # Optional: Total trainable parameters
        total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
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

        model.train()
        total_loss = 0

        for i, sample in enumerate(trainloader):
            optimizer.zero_grad()

            img, depth, valid_mask = (
                # sample["image"].cuda(),
                sample["input"].cuda(),
                sample["depth"].cuda(),
                sample["valid_mask"].cuda(),
            )

            if random.random() < 0.5:
                img = img.flip(-1)
                depth = depth.flip(-1)
                valid_mask = valid_mask.flip(-1)

            pred = model(img)

            loss = criterion(
                pred,
                depth,
                valid_mask,
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            iters = epoch * len(trainloader) + i

            lr = args.lr * (1 - iters / total_iters) ** 0.9

            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * 10.0

            if rank == 0:
                writer.add_scalar("train/loss", loss.item(), iters)

            if rank == 0 and i % 100 == 0:
                logger.info(
                    "Iter: {}/{}, LR: {:.7f}, Loss: {:.3f}".format(
                        i,
                        len(trainloader),
                        optimizer.param_groups[0]["lr"],
                        loss.item(),
                    )
                )

        model.eval()

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

            img, depth, valid_mask = (
                # sample["image"].cuda().float(),
                sample["input"].cuda().float(),
                sample["depth"].cuda()[0],
                sample["valid_mask"].cuda()[0],
            )

            with torch.no_grad():
                pred = model(img)
                pred = F.interpolate(
                    pred[:, None], depth.shape[-2:], mode="bilinear", align_corners=True
                )[0, 0]

            if valid_mask.sum() < 10:
                continue

            cur_results = eval_depth(pred[valid_mask], depth[valid_mask])

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

        if rank == 0 and (epoch + 1) % 20 == 0 and epoch > 50:
            checkpoint = {
                "model": model.state_dict(),
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
                    checkpoint = {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "previous_best": previous_best,
                    }
                    torch.save(
                        checkpoint,
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
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "previous_best": previous_best,
            }
            torch.save(checkpoint, os.path.join(args.save_path, "latest.pth"))


if __name__ == "__main__":
    main()
