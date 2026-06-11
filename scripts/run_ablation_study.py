"""
Ablation study script: evaluate the performance of different module combinations.
"""
import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

import torch
import wandb
import numpy as np
from torch_geometric.data import DataLoader

from dagr.utils.args import FLAGS
from dagr.data.dsec_data import DSEC
from dagr.data.augment import Augmentations
from dagr.model.networks.dagr_v2 import DAGR_V2
from dagr.model.networks.ema import ModelEMA
from dagr.utils.buffers import DetectionBuffer
from dagr.utils.buffers import format_data
import tqdm


def run_test(loader, model, dataset="dsec"):
    """Run evaluation and return metrics."""
    model.eval()
    mapcalc = DetectionBuffer(
        height=loader.dataset.height,
        width=loader.dataset.width,
        classes=loader.dataset.classes
    )

    for i, data in enumerate(tqdm.tqdm(loader, desc="Testing")):
        data = data.cuda()
        data = format_data(data)

        detections, targets = model(data)
        mapcalc.update(detections, targets, dataset, data.height[0], data.width[0])

        if i % 10 == 0:
            torch.cuda.empty_cache()

    metrics = mapcalc.compute()
    return metrics


def run_ablation_experiment(args, test_loader, experiment_config):
    """
    Run a single ablation experiment.

    Args:
        args: Argument namespace
        test_loader: Test data loader
        experiment_config: dict, experiment configuration
            {
                'name': str,
                'use_bidirectional_fusion': bool,
                'use_spatial_attention': bool,
                'use_temporal_attention': bool,
            }
    """
    print(f"\n{'='*60}")
    print(f"Running Experiment: {experiment_config['name']}")
    print(f"{'='*60}")
    print(f"Config:")
    for k, v in experiment_config.items():
        if k != 'name':
            print(f"  {k}: {v}")

    # Update args
    args.use_bidirectional_fusion = experiment_config['use_bidirectional_fusion']
    args.use_spatial_attention = experiment_config['use_spatial_attention']
    args.use_temporal_attention = experiment_config['use_temporal_attention']

    # Create model
    model = DAGR_V2(args, height=test_loader.dataset.height, width=test_loader.dataset.width)
    model = model.cuda()
    ema = ModelEMA(model)

    # Load checkpoint (if available)
    if hasattr(args, 'checkpoint') and args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint)
        try:
            ema.ema.load_state_dict(checkpoint['ema'])
        except:
            print("Warning: Failed to load checkpoint, using random initialization")

    # Run evaluation
    with torch.no_grad():
        metrics = run_test(test_loader, ema.ema, dataset=args.dataset)

    print(f"\nResults for {experiment_config['name']}:")
    print(f"  mAP: {metrics.get('mAP', 0.0):.4f}")
    if 'mAP_50' in metrics:
        print(f"  mAP@0.5: {metrics.get('mAP_50', 0.0):.4f}")

    return metrics


if __name__ == '__main__':
    import torch_geometric
    import random

    seed = 42
    torch_geometric.seed.seed_everything(seed)
    torch.random.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    args = FLAGS()

    # Initialize wandb
    wandb.init(project="dagr-ablation", name="ablation_study")

    # Load test dataset
    print("Loading test dataset...")
    dataset_path = args.dataset_directory / args.dataset
    test_dataset = DSEC(
        root=dataset_path,
        split="val",
        transform=Augmentations.transform_testing,
        debug=False,
        min_bbox_diag=15,
        min_bbox_height=10
    )

    sampler = np.random.permutation(np.arange(len(test_dataset)))
    test_loader = DataLoader(
        test_dataset,
        sampler=sampler,
        follow_batch=['bbox', 'bbox0'],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=True
    )

    # Define ablation experiment configurations
    ablation_configs = [
        {
            'name': 'E1_Baseline',
            'use_bidirectional_fusion': False,
            'use_spatial_attention': False,
            'use_temporal_attention': False,
        },
        {
            'name': 'E2_BidirectionalOnly',
            'use_bidirectional_fusion': True,
            'use_spatial_attention': False,
            'use_temporal_attention': False,
        },
        {
            'name': 'E3_BidirectionalWithGate',
            'use_bidirectional_fusion': True,
            'use_spatial_attention': False,
            'use_temporal_attention': False,
        },
        {
            'name': 'E4_SpatialAttentionOnly',
            'use_bidirectional_fusion': False,
            'use_spatial_attention': True,
            'use_temporal_attention': False,
        },
        {
            'name': 'E5_TemporalAttentionOnly',
            'use_bidirectional_fusion': False,
            'use_spatial_attention': False,
            'use_temporal_attention': True,
        },
        {
            'name': 'E6_BothAttentions',
            'use_bidirectional_fusion': False,
            'use_spatial_attention': True,
            'use_temporal_attention': True,
        },
        {
            'name': 'E7_Full',
            'use_bidirectional_fusion': True,
            'use_spatial_attention': True,
            'use_temporal_attention': True,
        },
    ]

    # Run all experiments
    all_results = {}
    for config in ablation_configs:
        metrics = run_ablation_experiment(args, test_loader, config)
        all_results[config['name']] = metrics

        # Log to wandb
        wandb.log({
            f"ablation/{config['name']}/mAP": metrics.get('mAP', 0.0),
            f"ablation/{config['name']}/mAP_50": metrics.get('mAP_50', 0.0),
        })

    # Print summary
    print(f"\n{'='*80}")
    print("ABLATION STUDY SUMMARY")
    print(f"{'='*80}")
    print(f"{'Experiment':<30} {'mAP':<10} {'mAP@0.5':<10}")
    print(f"{'-'*80}")

    for name, metrics in all_results.items():
        print(f"{name:<30} {metrics.get('mAP', 0.0):<10.4f} {metrics.get('mAP_50', 0.0):<10.4f}")

    print(f"{'='*80}\n")

    # Save results
    import json
    with open('ablation_results.json', 'w') as f:
        json.dump({k: {mk: float(mv) for mk, mv in v.items()} for k, v in all_results.items()}, f, indent=2)

    print("Results saved to ablation_results.json")
    wandb.finish()
