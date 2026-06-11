# BiDagr: From Unidirectional to Bidirectional Synergy Graph-Guided Event Detection via Semantics-Guided Feedback and Adaptive Gating

## Description

BiDagr is an extension of the [DAGR (Delay-Aware Graph Routing)](https://github.com/uzh-rpg/dagr) framework for event-camera object detection. It introduces two novel bidirectional mechanisms to enhance the synergy between Graph Neural Networks (GNN) and Convolutional Neural Networks (CNN):

- **Coarse-Grained Feedback**: A GNN-to-CNN feedback mechanism that generates dense, coarse-grained modulation signals from sparse GNN node features back to CNN feature maps, enabling bidirectional information flow.
- **Adaptive Gating**: A lightweight event-activity gating mechanism that modulates CNN features based on global or spatially-aware GNN node statistics, adaptively enhancing or suppressing feature channels.

These mechanisms can be used independently or combined (Plan A+D) to achieve improved detection performance on event-camera data.

## Dataset Information

This project uses the **DSEC (Dynamic Stereo Event Camera)** dataset for training and evaluation.

- **Dataset**: DSEC — a high-resolution event-camera dataset for object detection
- **Splits**: train / val / test
- **Input**: Event camera data (spatiotemporal event streams)
- **Task**: Object detection in driving scenarios

Please download the DSEC dataset from the [official DSEC repository](https://github.com/uzh-rpg/dsec) and organize it as follows:

```
<data_root>/
  dsec/
    train/
    val/
    test/
```

## Code Information

### Project Structure

```
BiDagr-main/
├── README.md                          # This file
├── setup.py                           # Package setup with CUDA extensions
├── install_env.sh                     # Environment installation script
├── download_and_install_dependencies.sh
├── download_example_data.sh
├── config/                            # YAML configuration files
│   ├── dagr-s-dsec.yaml              # Baseline (no bidirectional mechanisms)
│   ├── dagr-s-dsec-gating.yaml       # Plan D: Gating only
│   ├── dagr-s-dsec-feedback.yaml     # Plan A: Feedback only
│   ├── dagr-s-dsec-combine.yaml      # Plan A+D: Combined
│   ├── dagr-s-dsec-50.yaml           # Baseline with ResNet-50
│   ├── dagr-s-dsec-gating-50.yaml    # Plan D with ResNet-50
│   ├── dagr-s-dsec-feedback-50.yaml  # Plan A with ResNet-50
│   └── dagr-s-dsec-combine-50.yaml   # Plan A+D with ResNet-50
├── scripts/                           # Training, evaluation, and utility scripts
│   ├── train_dsec.py                  # Main training script
│   ├── eval_dagr_dsec.py             # Evaluation script
│   ├── offline_infer_dagr_dsec.py    # Offline inference script
│   ├── run_ablation_study.py         # Ablation study script
│   ├── plot_dsec_detection_logs.py   # Plot training logs
│   ├── visualize_detections.py       # Visualize detection results
│   └── ...
└── src/
    └── dagr/                          # Main Python package
        ├── model/
        │   ├── layers/
        │   │   ├── feedback.py        # Plan A: Coarse GNN-to-CNN feedback
        │   │   ├── gating.py          # Plan D: Event activity gating
        │   │   ├── spline_conv.py     # Spline convolution layers
        │   │   ├── ev_tgn.py          # Event temporal graph network
        │   │   └── components.py      # Shared components
        │   └── networks/
        │       ├── dagr.py            # Main DAGR network
        │       ├── net.py             # Network backbone
        │       └── ema.py             # Exponential moving average
        ├── data/
        │   ├── dsec_data.py           # DSEC dataset loader
        │   └── augment.py             # Data augmentation
        ├── graph/
        │   └── ev_graph.py            # Event graph construction (with CUDA)
        ├── adaptive/
        │   └── sampler.py             # Adaptive event sampling
        ├── utils/
        │   ├── args.py                # Argument parsing
        │   ├── buffers.py             # Detection buffers
        │   ├── logging.py             # Logging utilities
        │   └── learning_rate_scheduler.py
        └── visualization/
            └── bbox_viz.py            # Bounding box visualization
```

### Key Components

| Module | File | Description |
|--------|------|-------------|
| Coarse Feedback| `src/dagr/model/layers/feedback.py` | Aggregates sparse GNN nodes into a coarse grid, generates additive feedback signals to CNN feature maps |
| Adaptive Gating| `src/dagr/model/layers/gating.py` | Global and spatial-aware gating mechanisms that modulate CNN features based on GNN statistics |
| Training Script | `scripts/train_dsec.py` | End-to-end training with support for Feedback, Gating, and their combination |
| Evaluation Script | `scripts/eval_dagr_dsec.py` | Inference and mAP evaluation on DSEC dataset |
| Ablation Study | `scripts/run_ablation_study.py` | Systematic evaluation of different module combinations |

## Usage Instructions

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/Yang-ZeYi/BiDagr.git
cd BiDagr

# Install dependencies
bash install_env.sh

# Install the package with CUDA extensions
pip install -e .
```

### 2. Data Preparation

Download the DSEC dataset and place it under your data directory. Update the `dataset_directory` path in the config YAML files.

### 3. Training

```bash
# Train baseline (no bidirectional mechanisms)
python scripts/train_dsec.py --config config/dagr-s-dsec.yaml

# Train with Plan D (Gating only)
python scripts/train_dsec.py --config config/dagr-s-dsec-gating.yaml

# Train with Plan A (Feedback only)
python scripts/train_dsec.py --config config/dagr-s-dsec-feedback.yaml

# Train with Plan A+D (Combined)
python scripts/train_dsec.py --config config/dagr-s-dsec-combine.yaml
```

### 4. Evaluation

```bash
# Evaluate a trained model
python scripts/eval_dagr_dsec.py --config config/dagr-s-dsec-gating.yaml \
    --checkpoint path/to/checkpoint.pth

# Offline inference across multiple experiments
python scripts/offline_infer_dagr_dsec.py \
    --dataset-root /path/to/dsec \
    --logs-root logs/dsec/detection \
    --experiment all
```

### 5. Ablation Study

```bash
# Run ablation study
python scripts/run_ablation_study.py --config config/dagr-s-dsec.yaml
```

## Requirements

### Hardware

- NVIDIA GPU with CUDA support (tested on NVIDIA GPUs with CUDA 11.x)
- Sufficient GPU memory for training (recommended: >= 8 GB)

### Software

- **Operating System**: Linux (Ubuntu 18.04 or later recommended)
- **Python**: 3.8+
- **PyTorch**: 1.10+
- **PyTorch Geometric**: 2.0+
- **CUDA Toolkit**: 11.x
- **Other dependencies**: numpy, scipy, opencv-python, tqdm, pyyaml, wandb (optional, for logging)

### Python Dependencies

```
torch >= 1.10
torch_geometric >= 2.0
numpy
scipy
opencv-python
tqdm
pyyaml
matplotlib
```

## Methodology

### Architecture Overview

BiDagr extends the original DAGR architecture with two bidirectional mechanisms:

1. **Coarse-Grained Feedback**:
   - Aggregates sparse GNN node features into a coarse spatial grid (e.g., 16×12)
   - Encodes cell features using an MLP with BatchNorm and Dropout
   - Upsamples to CNN feature map resolution via transposed convolutions
   - Applies additive modulation with learnable strength parameter

2. **Adaptive Gating**:
   - **Global gating**: Pools all GNN node features globally, generates per-channel gating weights via a small MLP
   - **Spatial gating**: Assigns nodes to spatial bins, generates spatially varying gating maps
   - Modulates CNN features with gated values in the range [0.5, 1.5]

3. **Warmup Strategy**: Both mechanisms support epoch-based warmup to stabilize early training.

### Data Preprocessing

- Event data is converted to graph representations using CUDA-accelerated event graph construction
- Adaptive event sampling adjusts the number of events based on scene complexity
- Standard augmentation techniques (random flip, zoom, translation) are applied during training

## Citations

If you use this code in your research, please cite:

```bibtex
@article{dagr2023,
  title={DAGR: Delay-Aware Graph Routing for Event Camera Detection},
  author={Schaaf, Cedric and others},
  journal={arXiv preprint},
  year={2023}
}
```

## License

This project is based on the [DAGR](https://github.com/uzh-rpg/dagr) framework from the Robotics and Perception Group at the University of Zurich. Please refer to the original repository for license terms.

## Acknowledgments

- Original DAGR implementation: [uzh-rpg/dagr](https://github.com/uzh-rpg/dagr)
- DSEC dataset: [uzh-rpg/dsec](https://github.com/uzh-rpg/dsec)
