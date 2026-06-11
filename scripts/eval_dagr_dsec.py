# eval_dagr_dsec.py
# Run inference and evaluation with a trained DAGR model on DSEC
# (supports original / Plan A / Plan D / Plan A + Plan D)
import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Avoid CUDA ordering issues on some servers

import torch
import random
import numpy as np
import torch_geometric

from torch_geometric.data import DataLoader

from dagr.utils.args import FLAGS
from dagr.data.dsec_data import DSEC
from dagr.data.augment import Augmentations
from dagr.model.networks.dagr import DAGR
from dagr.utils.logging import set_up_logging_directory, log_hparams
from dagr.utils.testing import run_test_with_visualization


# ================= Utility: Load weights from checkpoint ================= #

def load_weights_from_ckpt(model, ckpt_path, use_ema=True, strict=True):
    """
    Load weights from a checkpoint into the model:
    - Prefer 'ema' dictionary (if present)
    - Otherwise use 'model' dictionary
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    print(f"\n[INFO] Loading checkpoint from: {ckpt_path}")
    print("[INFO] Top-level keys:", list(ckpt.keys()))

    if use_ema and "ema" in ckpt:
        state_dict = ckpt["ema"]
        print("[INFO] Using EMA weights: ckpt['ema']")
    elif "model" in ckpt:
        state_dict = ckpt["model"]
        print("[INFO] Using normal model weights: ckpt['model']")
    else:
        raise ValueError("Cannot find 'ema' or 'model' key in checkpoint. Please check the checkpoint structure.")

    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    print("[INFO] load_state_dict completed")
    print("  Number of missing keys:", len(missing))
    print("  Number of unexpected keys:", len(unexpected))
    if missing:
        print("  Some missing keys:", missing[:20])
    if unexpected:
        print("  Some unexpected keys:", unexpected[:20])

    return model


# =============================== Main Logic =============================== #

if __name__ == '__main__':
    # ----- Fix random seed -----
    seed = 42
    torch_geometric.seed.seed_everything(seed)
    torch.random.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # ----- Parse command-line arguments -----
    args = FLAGS()

    # --checkpoint is required
    if not hasattr(args, "checkpoint") or args.checkpoint is None:
        raise ValueError("Please provide --checkpoint=path/to/ckpt.pth on the command line")

    # Set up output directory (mainly for compatibility with the existing logging mechanism)
    output_directory = set_up_logging_directory(args.dataset, args.task, args.output_directory)
    print(f"[INFO] Output directory: {output_directory}")
    log_hparams(args)

    # ----- Construct dataset and DataLoader -----
    print("[INFO] Init datasets ...")
    augmentations = Augmentations(args)
    # Consistent with training: root = args.dataset_directory / args.dataset
    dataset_path = args.dataset_directory / args.dataset

    # Default: use 'val' for evaluation; change to split='test' if a 'test' split is available
    test_dataset = DSEC(
        root=dataset_path,
        split="test",  # Change to "test" to evaluate on the test set
        transform=augmentations.transform_testing,
        debug=False,
        min_bbox_diag=15,
        min_bbox_height=10
    )

    # Randomly shuffle the order (similar to the training script)
    sampler = np.random.permutation(np.arange(len(test_dataset)))
    test_loader = DataLoader(
        test_dataset,
        sampler=sampler,
        follow_batch=['bbox', 'bbox0'],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False
    )

    # ----- Build model -----
    print("[INFO] Init network ...")
    # Note: height / width are read from the dataset, consistent with training
    model = DAGR(args, height=test_dataset.height, width=test_dataset.width).cuda()

    # Print parameter count for verification of the desired architecture (original / Plan A / Plan D / Plan A+D)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Model parameters: {num_params}")

    # ----- Load weights -----
    model = load_weights_from_ckpt(model, args.checkpoint, use_ema=True, strict=False)
    model.eval()

    # If DAGR has a cache_luts precomputation function, it can be called here:
    #   if hasattr(model, "cache_luts") and hasattr(args, "radius"):
    #       model.cache_luts(radius=args.radius, height=test_dataset.height, width=test_dataset.width)

    # ----- Inference + Evaluation (with visualization) -----
    print("[INFO] Start inference & evaluation ...")
    with torch.no_grad():
        # This function is from the existing project: dagr.utils.testing.run_test_with_visualization
        metrics = run_test_with_visualization(test_loader, model, dataset=args.dataset)

    print("\n[RESULT] Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    if 'mAP' in metrics:
        print(f"\n[RESULT] mAP = {metrics['mAP']}")
