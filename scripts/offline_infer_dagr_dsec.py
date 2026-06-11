# scripts/offline_infer_dagr_dsec.py

import argparse
from pathlib import Path
from types import SimpleNamespace

import torch
import numpy as np

# PyG DataLoader
from torch_geometric.data import DataLoader

# Add src to Python path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

# Imports based on the provided source/test scripts
from dagr.model.networks.dagr import DAGR          # DAGR network
from dagr.data.dsec_data import DSEC               # DSEC dataset
from dagr.data.augment import Augmentations        # Same as run_test.py
from dagr.utils.buffers import format_data         # Same as testing.py


EXPERIMENT_DIRS = {
    "dsec_s_50": "dsec_s_50",
    "gating_global_warmup5": "gating_global_warmup5",
    "plan_a_feedback_coarse": "plan_a_feedback_coarse",
    "plan_d_plus_a_combined": "plan_d_plus_a_combined",
}


# ------------------------- Argument Parsing ------------------------- #
def parse_args():
    parser = argparse.ArgumentParser("Offline inference for DAGR on DSEC")

    parser.add_argument(
        "--experiment",
        type=str,
        default="all",
        choices=list(EXPERIMENT_DIRS.keys()) + ["all"],
        help="Name of the experiment to test, or 'all' to test all four experiments sequentially",
    )
    parser.add_argument(
        "--logs-root",
        type=str,
        default="logs/dsec/detection",
        help="Root path containing the experiment subdirectories (default matches the current layout)",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Checkpoint filename (under the corresponding experiment directory). "
             "If not specified, will preferentially look for best_model*.pth, otherwise any .pth file",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        required=True,
        help="DSEC dataset root directory (containing train/val/test subdirectories)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Which split to use: train / val / test",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Inference batch size (for sanity check, 1 is sufficient)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of DataLoader workers",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=2,
        help="Maximum number of batches to run per model for sanity check",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Inference device, e.g., cuda:0 or cpu",
    )
    return parser.parse_args()


# ------------------------- Model Utilities ------------------------- #
def _coerce_args(args_obj):
    """
    Convert the saved args from checkpoint to a Namespace-like object.
    """
    if isinstance(args_obj, SimpleNamespace):
        return args_obj
    if isinstance(args_obj, dict):
        return SimpleNamespace(**args_obj)
    if hasattr(args_obj, "__dict__"):
        return args_obj
    # fallback
    return SimpleNamespace(**vars(args_obj))


def build_model_from_checkpoint(ckpt_path: Path, dataset: DSEC, device: torch.device) -> torch.nn.Module:
    print(f"\n[INFO] Loading checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, torch.nn.Module):
        model = ckpt.to(device)
        model.eval()
        print("[INFO] Checkpoint is a nn.Module, using it directly.")
        return model

    if not isinstance(ckpt, dict):
        raise RuntimeError(f"Unknown checkpoint format type={type(ckpt)}")

    # 1) Retrieve the training args (if saved)
    args = None
    if "args" in ckpt:
        args = _coerce_args(ckpt["args"])
        print("[INFO] Found 'args' in checkpoint.")
    else:
        # If args were not saved during training, use a safe fallback configuration
        # You may need to adjust this based on your actual training FLAGS
        print("[WARN] No 'args' found in checkpoint; using minimal default args. "
              "If you see shape mismatch errors, please edit this part.")
        args = SimpleNamespace()
        args.yolo_stem_width = 1.0
        args.num_scales = 2
        args.use_image = True
        args.batch_size = 1
        args.no_events = False
        args.pretrain_cnn = False

        # Gating defaults: disabled to avoid AttributeError
        args.use_bidirectional_gating = False
        args.use_gating_in_eval = True
        args.gate_type = "global"
        args.gate_hidden_dim = 64
        args.gate_strength_init = 0.1
        args.gate_spatial_bins = (8, 8)
        args.gate_warmup_epochs = 0
        args.radius = 3  # Adjust this to match your training radius

    # Fallback attributes for older checkpoints that may lack these fields
    if not hasattr(args, "batch_size"):
        args.batch_size = 1
    if not hasattr(args, "use_bidirectional_gating"):
        args.use_bidirectional_gating = False
    if not hasattr(args, "use_gating_in_eval"):
        args.use_gating_in_eval = True
    if not hasattr(args, "gate_type"):
        args.gate_type = "global"
    if not hasattr(args, "gate_hidden_dim"):
        args.gate_hidden_dim = 64
    if not hasattr(args, "gate_strength_init"):
        args.gate_strength_init = 0.1
    if not hasattr(args, "gate_spatial_bins"):
        args.gate_spatial_bins = (8, 8)
    if not hasattr(args, "gate_warmup_epochs"):
        args.gate_warmup_epochs = 0
    if not hasattr(args, "radius"):
        args.radius = 3

    # 2) Instantiate DAGR
    model = DAGR(args=args, height=dataset.height, width=dataset.width)  # Consistent with run_test
    model.to(device)

    # If the head has current_epoch and the checkpoint has epoch, set it for gate warmup logic
    if hasattr(model.head, "current_epoch") and "epoch" in ckpt:
        model.head.current_epoch = ckpt["epoch"]

    # 3) Parse state_dict (prefer 'ema', then 'model')
    state_dict = None
    for key in ["ema", "model", "state_dict", "net"]:
        if key in ckpt and isinstance(ckpt[key], dict):
            state_dict = ckpt[key]
            print(f"[INFO] Using state_dict from ckpt['{key}']")
            break

    if state_dict is None:
        # Try to use tensor-like items in the checkpoint as state_dict
        tensor_items = {k: v for k, v in ckpt.items() if isinstance(v, torch.Tensor)}
        if tensor_items:
            state_dict = tensor_items
            print("[INFO] Using tensor-only items in checkpoint as state_dict.")
        else:
            raise RuntimeError("Could not find a suitable state_dict in checkpoint.")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[WARN] Missing keys in state_dict: {missing[:10]}{' ...' if len(missing) > 10 else ''}")
    if unexpected:
        print(f"[WARN] Unexpected keys in state_dict: {unexpected[:10]}{' ...' if len(unexpected) > 10 else ''}")

    # 4) Cache LUTs (same as run_test)
    if hasattr(model, "cache_luts"):
        try:
            model.cache_luts(radius=args.radius, height=dataset.height, width=dataset.width)
            print(f"[INFO] Cached LUTs with radius={args.radius}, "
                  f"height={dataset.height}, width={dataset.width}")
        except Exception as e:
            print(f"[WARN] cache_luts failed: {e}")

    model.eval()
    return model


# ------------------------- Build DSEC Dataset ------------------------- #
def build_dsec_dataset(dataset_root: Path, split: str) -> DSEC:
    """
    Build the DSEC dataset using the provided DSEC class.
    Referenced from run_test.py usage:
      DSEC(args.dataset_directory, "test", Augmentations.transform_testing, debug=False,
           min_bbox_diag=15, min_bbox_height=10)
    Assumes dataset_root is the DSEC root directory (containing train/val/test),
    and split specifies the subdirectory.
    """
    dataset_directory = dataset_root  # DSEC internally locates content based on split
    dataset = DSEC(
        dataset_directory,
        split,
        Augmentations.transform_testing,
        debug=False,
        min_bbox_diag=15,
        min_bbox_height=10,
    )
    print(f"[INFO] Built DSEC dataset: split={split}, len={len(dataset)}, "
          f"height={dataset.height}, width={dataset.width}")
    return dataset


# ------------------------- Inference Loop ------------------------- #
@torch.no_grad()
def offline_inference(
    exp_name: str,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_samples: int,
):
    print("\n" + "=" * 80)
    print(f"[INFO] Experiment: {exp_name}")
    print(f"[INFO]   Num batches to test (max): {num_samples}")

    model.eval()
    processed = 0

    for i, data in enumerate(loader):
        if processed >= num_samples:
            break

        # PyG Batch supports .to(device)
        data = data.to(device)

        # Consistent with testing.py: format_data first, then clone and feed to model
        formatted = format_data(data)
        detections, targets = model(formatted.clone())

        print(f"\n[Sample {i}]")
        # Print basic detection info to verify forward pass works
        if isinstance(detections, (list, tuple)):
            print(f"  detections: list/tuple, len={len(detections)}")
            if len(detections) > 0 and isinstance(detections[0], dict):
                keys = list(detections[0].keys())
                print(f"  first detection keys: {keys}")
        elif isinstance(detections, dict):
            print(f"  detections: dict, keys={list(detections.keys())}")
            for k, v in detections.items():
                if isinstance(v, torch.Tensor):
                    print(f"    {k}: tensor shape={tuple(v.shape)}")
        else:
            print(f"  detections type={type(detections)}")

        processed += 1

    print(f"\n[INFO] Offline inference finished for {exp_name}, "
          f"processed {processed} batch(es).")


# ------------------------- Checkpoint Search ------------------------- #
def find_checkpoint(exp_dir: Path, ckpt_name: str | None) -> Path:
    if ckpt_name is not None:
        ckpt_path = exp_dir / ckpt_name
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint '{ckpt_name}' not found in {exp_dir}")
        return ckpt_path

    # Preferentially look for best_model*.pth
    bests = sorted(exp_dir.glob("best_model*.pth"))
    if bests:
        print(f"[INFO] Using checkpoint: {bests[-1].name}")
        return bests[-1]

    # Otherwise look for any .pth file
    ckpts = sorted(exp_dir.glob("*.pth"))
    if ckpts:
        print(f"[INFO] Using checkpoint: {ckpts[-1].name}")
        return ckpts[-1]

    raise FileNotFoundError(f"No .pth checkpoint found in {exp_dir}")


# ------------------------- main ------------------------- #
def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    dataset_root = Path(args.dataset_root)
    assert dataset_root.exists(), f"dataset_root not found: {dataset_root}"

    # Build dataset once, then create a separate DataLoader for each experiment
    dataset = build_dsec_dataset(dataset_root, args.split)

    exp_list = list(EXPERIMENT_DIRS.keys()) if args.experiment == "all" else [args.experiment]

    for exp_name in exp_list:
        exp_dir = Path(args.logs_root) / EXPERIMENT_DIRS[exp_name]
        if not exp_dir.is_dir():
            print(f"[WARN] Experiment directory not found, skip: {exp_dir}")
            continue

        ckpt_path = find_checkpoint(exp_dir, args.ckpt)

        # Each experiment gets its own DataLoader (to avoid DataLoader being exhausted)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            follow_batch=['bbox', 'bbox0'],
            num_workers=args.num_workers,
            drop_last=False,
        )

        model = build_model_from_checkpoint(ckpt_path, dataset, device)
        offline_inference(exp_name, model, loader, device, args.num_samples)


if __name__ == "__main__":
    main()
