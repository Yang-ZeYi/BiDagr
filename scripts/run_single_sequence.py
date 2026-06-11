import os
import json
import random
import numpy as np
import torch

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

from pathlib import Path
from torch_geometric.data import DataLoader

from dagr.utils.args import FLAGS
from dagr.utils.logging import set_up_logging_directory, log_hparams
from dagr.utils.buffers import format_data

from dagr.data.augment import Augmentations
from dagr.data.dsec_data import DSEC

from dagr.model.networks.dagr import DAGR
from dagr.model.networks.ema import ModelEMA


def main():
    import torch_geometric

    seed = 42
    torch_geometric.seed.seed_everything(seed)
    torch.random.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    args = FLAGS()

    output_directory = set_up_logging_directory(args.dataset, args.task, args.output_directory)
    log_hparams(args)

    sequence_name = getattr(args, "sequence", None)
    assert sequence_name is not None, "args.sequence must be provided for single-sequence inference"

    test_dataset = DSEC(
        root=args.dataset_directory,
        split="test",
        transform=Augmentations.transform_testing,
        debug=False,
        min_bbox_diag=15,
        min_bbox_height=10,
        only_perfect_tracks=False,
        no_eval=True
    )

    test_loader = DataLoader(
        test_dataset,
        follow_batch=['bbox', 'bbox0'],
        batch_size=1,
        shuffle=False,
        num_workers=0,
        drop_last=False
    )

    model = DAGR(args, height=test_dataset.height, width=test_dataset.width)
    model = model.cuda()
    ema = ModelEMA(model)

    assert "checkpoint" in args
    checkpoint = torch.load(args.checkpoint)
    ema.ema.load_state_dict(checkpoint['ema'])
    ema.ema.cache_luts(radius=args.radius, height=test_dataset.height, width=test_dataset.width)
    ema.ema.eval()

    images = {}
    annotations = []
    categories = []
    image_id_map = {}
    next_image_id = 1
    next_ann_id = 1

    for idx, cls_name in enumerate(test_dataset.classes):
        categories.append({
            "id": idx + 1,
            "name": cls_name
        })

    with torch.no_grad():
        for _, data in enumerate(test_loader):
            data = data.cuda(non_blocking=True)
            data = format_data(data)

            seqs = data.sequence
            times = data.t1

            if len(seqs) == 0:
                continue

            seq = seqs[0]
            if seq != sequence_name:
                continue

            detections, _ = ema.ema(data.clone())

            det = detections[0]
            boxes = det["boxes"]
            scores = det["scores"]
            labels = det["labels"]

            t_val = int(times[0].item())
            key = (seq, t_val)

            if key not in image_id_map:
                image_id_map[key] = next_image_id
                file_name = f"{seq}_t{t_val}.png"
                images[next_image_id] = {
                    "id": next_image_id,
                    "file_name": file_name,
                    "height": int(test_dataset.height),
                    "width": int(test_dataset.width)
                }
                next_image_id += 1

            image_id = image_id_map[key]

            if boxes.numel() == 0:
                continue

            for i in range(boxes.shape[0]):
                x1 = float(boxes[i, 0].item())
                y1 = float(boxes[i, 1].item())
                x2 = float(boxes[i, 2].item())
                y2 = float(boxes[i, 3].item())
                w = x2 - x1
                h = y2 - y1

                score = float(scores[i].item())
                label = int(labels[i].item()) + 1

                annotations.append({
                    "id": next_ann_id,
                    "image_id": image_id,
                    "category_id": label,
                    "bbox": [x1, y1, w, h],
                    "score": score,
                    "sequence": seq,
                    "time_us": t_val
                })
                next_ann_id += 1

    coco_result = {
        "info": {
            "description": "DAGR single-sequence detections in COCO format",
            "sequence": sequence_name
        },
        "images": list(images.values()),
        "annotations": annotations,
        "categories": categories
    }

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    output_file = output_directory / f"coco_detections_{sequence_name}.json"

    with open(output_file, "w") as f:
        json.dump(coco_result, f)


if __name__ == "__main__":
    main()
