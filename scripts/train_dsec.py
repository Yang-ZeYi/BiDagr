# avoid matlab error on server
import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

import torch
import tqdm
from pathlib import Path
import argparse

from torch_geometric.data import DataLoader

from dagr.utils.logging import Checkpointer, LocalMetricsLogger, set_up_logging_directory, log_hparams
from dagr.utils.buffers import DetectionBuffer
from dagr.utils.args import FLAGS
from dagr.utils.learning_rate_scheduler import LRSchedule

from dagr.data.augment import Augmentations
from dagr.utils.buffers import format_data
from dagr.data.dsec_data import DSEC

from dagr.model.networks.dagr import DAGR
from dagr.model.networks.ema import ModelEMA


def gradients_broken(model):
    valid_gradients = True
    for name, param in model.named_parameters():
        if param.grad is not None:
            # valid_gradients = not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
            valid_gradients = not (torch.isnan(param.grad).any())
            if not valid_gradients:
                break
    return not valid_gradients

def fix_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            param.grad = torch.nan_to_num(param.grad, nan=0.0)


def train(loader: DataLoader,
          model: torch.nn.Module,
          ema: ModelEMA,
          scheduler: torch.optim.lr_scheduler.LambdaLR,
          optimizer: torch.optim.Optimizer,
          args: argparse.ArgumentParser,
          epoch: int = 0,
          run_name="",
          logger: LocalMetricsLogger = None):

    model.train()

    # ============ Update current epoch (for warmup) ============
    # Plan D: Gating
    if hasattr(model, 'head') and hasattr(model.head, 'current_epoch'):
        model.head.current_epoch = epoch

    # Plan A: Feedback
    if hasattr(model, 'backbone') and hasattr(model.backbone, 'current_epoch'):
        model.backbone.current_epoch = epoch
    # ===========================================================

    for i, data in enumerate(tqdm.tqdm(loader, desc=f"Training {run_name} Epoch {epoch}")):
        data = data.cuda(non_blocking=True)
        data = format_data(data)

        optimizer.zero_grad(set_to_none=True)

        model_outputs = model(data)

        loss_dict = {k: v for k, v in model_outputs.items() if "loss" in k}
        loss = loss_dict.pop("total_loss")

        loss.backward()

        torch.nn.utils.clip_grad_value_(model.parameters(), args.clip)

        fix_gradients(model)

        optimizer.step()
        scheduler.step()

        ema.update(model)

        training_logs = {f"training/loss/{k}": v for k, v in loss_dict.items()}
        if logger is not None:
            global_step = epoch * len(loader) + i
            logger.log({"training/loss": loss.item(), "training/lr": scheduler.get_last_lr()[-1], **training_logs},
                       step=global_step, epoch=epoch)

        # ============ Log gating info (Plan D) ============
        if args.use_bidirectional_gating and i % 100 == 0:
            if hasattr(model, 'head'):
                gate_logs = {}

                # Scale 1 gating info
                if hasattr(model.head, 'gate_info_scale1') and model.head.gate_info_scale1:
                    info1 = model.head.gate_info_scale1
                    if info1.get('gate_applied', False):
                        gate_logs.update({
                            'training/gate_scale1/strength': info1.get('gate_strength', 0),
                            'training/gate_scale1/avg_value': info1.get('avg_gate_value', 0.5),
                            'training/gate_scale1/num_nodes': info1.get('num_nodes', 0),
                            'training/gate_scale1/gate_min': info1.get('gate_min', 0),
                            'training/gate_scale1/gate_max': info1.get('gate_max', 1),
                            'training/gate_scale1/gate_std': info1.get('gate_std', 0),
                        })

                # Scale 2 gating info
                if hasattr(model.head, 'gate_info_scale2') and model.head.gate_info_scale2:
                    info2 = model.head.gate_info_scale2
                    if info2.get('gate_applied', False):
                        gate_logs.update({
                            'training/gate_scale2/strength': info2.get('gate_strength', 0),
                            'training/gate_scale2/avg_value': info2.get('avg_gate_value', 0.5),
                            'training/gate_scale2/num_nodes': info2.get('num_nodes', 0),
                        })

                if gate_logs:
                    if logger is not None:
                        global_step = epoch * len(loader) + i
                        logger.log(gate_logs, step=global_step, epoch=epoch)
        # ====================================================

        # ============ Log feedback info (Plan A) ============
        if args.use_bidirectional_feedback and i % 100 == 0:
            if hasattr(model, 'backbone') and hasattr(model.backbone, 'feedback_info'):
                info = model.backbone.feedback_info

                if info.get('feedback_applied', False):
                    feedback_logs = {
                        'training/feedback/strength': info.get('feedback_strength', 0),
                        'training/feedback/avg_norm': info.get('avg_feedback_norm', 0),
                        'training/feedback/current_norm': info.get('current_feedback_norm', 0),
                        'training/feedback/active_cells': info.get('active_cells', 0),
                        'training/feedback/active_ratio': info.get('active_cells_ratio', 0),
                        'training/feedback/num_nodes': info.get('num_gnn_nodes', 0),
                        'training/feedback/num_calls': info.get('num_forward_calls', 0),
                    }
                    if logger is not None:
                        global_step = epoch * len(loader) + i
                        logger.log(feedback_logs, step=global_step, epoch=epoch)
        # ====================================================

def run_test(loader: DataLoader,
         model: torch.nn.Module,
         dry_run_steps: int=-1,
         dataset="gen1"):

    model.eval()

    mapcalc = DetectionBuffer(height=loader.dataset.height, width=loader.dataset.width, classes=loader.dataset.classes)

    for i, data in enumerate(tqdm.tqdm(loader)):
        data = data.cuda()
        data = format_data(data)

        detections, targets = model(data)
        if i % 10 == 0:
            torch.cuda.empty_cache()

        mapcalc.update(detections, targets, dataset, data.height[0], data.width[0])

        if dry_run_steps > 0 and i == dry_run_steps:
            break

    torch.cuda.empty_cache()

    return mapcalc

if __name__ == '__main__':
    import torch_geometric
    import random
    import numpy as np

    seed = 42
    torch_geometric.seed.seed_everything(seed)
    torch.random.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    args = FLAGS()

    output_directory = set_up_logging_directory(args.dataset, args.task, args.output_directory, exp_name=args.exp_name, use_wandb=False)
    train_logger = LocalMetricsLogger(output_directory, jsonl_name="train_metrics.jsonl", csv_name="train_metrics.csv")
    eval_logger = LocalMetricsLogger(output_directory, jsonl_name="eval_metrics.jsonl", csv_name="eval_metrics.csv")
    log_hparams(args, output_directory=output_directory, use_wandb=False)

    # ============ Print gating configuration (Plan D) ============
    if args.use_bidirectional_gating:
        print("\n" + "="*60)
        print("BIDIRECTIONAL GATING (Plan D) ENABLED")
        print("="*60)
        print(f"  Gate Type: {args.gate_type}")
        print(f"  Hidden Dim: {args.gate_hidden_dim}")
        print(f"  Strength Init: {args.gate_strength_init}")
        print(f"  Warmup Epochs: {args.gate_warmup_epochs}")
        if args.gate_type == 'spatial':
            print(f"  Spatial Bins: {args.gate_spatial_bins}")
        print(f"  Use in Eval: {args.use_gating_in_eval}")
        print("="*60 + "\n")
    # =============================================================

    # ============ Print feedback configuration (Plan A) ============
    if args.use_bidirectional_feedback:
        print("\n" + "="*60)
        print("COARSE-GRAINED FEEDBACK (Plan A) ENABLED")
        print("="*60)
        print(f"  Hidden Dim: {args.feedback_hidden_dim}")
        print(f"  Strength Init: {args.feedback_strength_init}")
        print(f"  Warmup Epochs: {args.feedback_warmup_epochs}")
        print(f"  Grid Size: {args.feedback_grid_size}")
        print(f"  Use in Eval: {args.use_feedback_in_eval}")
        print("="*60 + "\n")
    # =============================================================

    if not args.use_bidirectional_gating and not args.use_bidirectional_feedback:
        print("\n[INFO] No bidirectional mechanism enabled (baseline mode)\n")

    augmentations = Augmentations(args)

    print("init datasets")
    dataset_path = args.dataset_directory / args.dataset

    train_dataset = DSEC(root=dataset_path, split="train", transform=augmentations.transform_training, debug=False,
                         min_bbox_diag=15, min_bbox_height=10)
    test_dataset = DSEC(root=dataset_path, split="val", transform=augmentations.transform_testing, debug=False,
                        min_bbox_diag=15, min_bbox_height=10)

    train_loader = DataLoader(train_dataset, follow_batch=['bbox', 'bbox0'], batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    num_iters_per_epoch = len(train_loader)

    sampler = np.random.permutation(np.arange(len(test_dataset)))
    test_loader = DataLoader(test_dataset, sampler=sampler, follow_batch=['bbox', 'bbox0'], batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=True)

    print("init net")
    # load a dummy sample to get height, width
    model = DAGR(args, height=test_dataset.height, width=test_dataset.width)

    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    print(f"Training with {num_params} number of parameters.")

    # ============ Count gating/feedback module parameters ============
    if args.use_bidirectional_gating:
        gate_params = 0
        for name, param in model.named_parameters():
            if 'gate' in name.lower():
                gate_params += np.prod(param.size())
        print(f"Gate module parameters: {gate_params} ({100*gate_params/num_params:.2f}% of total)")

    if args.use_bidirectional_feedback:
        feedback_params = 0
        for name, param in model.named_parameters():
            if 'feedback' in name.lower():
                feedback_params += np.prod(param.size())
        print(f"Feedback module parameters: {feedback_params} ({100*feedback_params/num_params:.2f}% of total)")
    # =================================================================

    model = model.cuda()
    ema = ModelEMA(model)

    nominal_batch_size = 64
    lr = args.l_r * np.sqrt(args.batch_size) / np.sqrt(nominal_batch_size)
    optimizer = torch.optim.AdamW(list(model.parameters()), lr=lr, weight_decay=args.weight_decay)

    lr_func = LRSchedule(warmup_epochs=.3,
                         num_iters_per_epoch=num_iters_per_epoch,
                         tot_num_epochs=args.tot_num_epochs)

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lr_func)

    checkpointer = Checkpointer(output_directory=output_directory,
                                model=model, optimizer=optimizer,
                                scheduler=lr_scheduler, ema=ema,
                                args=args, logger=eval_logger)

    start_epoch = 0
    if "resume_checkpoint" in args:
        start_epoch = checkpointer.restore_checkpoint(args.resume_checkpoint, best=False)
        print(f"Resume from checkpoint at epoch {start_epoch}")
    else:
        start_epoch = checkpointer.restore_if_existing(output_directory, resume_from_best=False)

    with torch.no_grad():
        mapcalc = run_test(test_loader, ema.ema, dry_run_steps=2, dataset=args.dataset)
        mapcalc.compute()

    print("starting to train")
    for epoch in range(start_epoch, args.tot_num_epochs):
        train(train_loader, model, ema, lr_scheduler, optimizer, args, epoch=epoch, run_name=output_directory.name, logger=train_logger)
        checkpointer.checkpoint(epoch, name=f"last_model")

        with torch.no_grad():
            mapcalc = run_test(test_loader, ema.ema, dataset=args.dataset)
            metrics = mapcalc.compute()
            checkpointer.process(metrics, epoch)
