from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import contextlib
from pycocotools.coco import COCO
from detectron2.evaluation.fast_eval_api import COCOeval_opt as COCOeval
#from detectron2.evaluation.fast_eval_api import COCOeval

import numpy as np
from typing import List, Dict, Tuple
from torch import Tensor

BBOX_DTYPE = np.dtype({'names':['t','x','y','w','h','class_id','track_id','class_confidence'], 'formats':['<i8','<f4','<f4','<f4','<f4','<u4','<u4','<f4'], 'offsets':[0,8,12,16,20,24,28,32], 'itemsize':40})

def _convert_to_coco_format(gt_boxes_list: List[Dict[str, Tensor]],
                       dt_boxes_list: List[Dict[str, Tensor]],
                       classes: str=("car", "pedestrian"),
                       height: int=240,
                       width: int=304,
                       time_tol: int=50000) -> Tuple[Dict, Dict]:
    """
    Compute detection KPIs on list of boxes in the numpy format, using the COCO python API
    https://github.com/cocodataset/cocoapi
    KPIs are only computed on timestamps where there is actual at least one box
    (fully empty frames are not considered)
    :param gt_boxes_list: list of numpy array for GT boxes (one per file)
    :param dt_boxes_list: list of numpy array for detected boxes
    :param classes: iterable of classes names
    :param height: int for box size statistics
    :param width: int for box size statistics
    :param time_tol: int size of the temporal window in micro seconds to look for a detection around a gt box
    """
    flattened_gt = []
    flattened_dt = []
    for gt_boxes, dt_boxes in zip(gt_boxes_list, dt_boxes_list):
        gt_boxes = _to_prophesee(gt_boxes)
        dt_boxes = _to_prophesee(dt_boxes)

        assert np.all(gt_boxes['t'][1:] >= gt_boxes['t'][:-1])
        assert np.all(dt_boxes['t'][1:] >= dt_boxes['t'][:-1])

        all_ts = np.unique(gt_boxes['t'])

        gt_win, dt_win = _match_times(all_ts, gt_boxes, dt_boxes, time_tol)
        flattened_gt = flattened_gt + gt_win
        flattened_dt = flattened_dt + dt_win


    num_detections = sum([d.size for d in flattened_dt])
    if num_detections == 0:
        # Corner case at the very beginning of the training.
        print('no detections for evaluation found.')
        return None

    categories = [{"id": id + 1, "name": class_name, "supercategory": "none"}
                  for id, class_name in enumerate(classes)]

    return _to_coco_format(flattened_gt, flattened_dt, categories, height=height, width=width), len(flattened_gt)



def evaluate_detection(gt_boxes_list: List[Dict[str, Tensor]],
                       dt_boxes_list: List[Dict[str, Tensor]],
                       classes: str=("car", "pedestrian"),
                       height: int=240,
                       width: int=304,
                       time_tol: int=50000) -> Dict[str, float]:
    """
    Compute detection KPIs on list of boxes in the numpy format, using the COCO python API
    https://github.com/cocodataset/cocoapi
    KPIs are only computed on timestamps where there is actual at least one box
    (fully empty frames are not considered)
    :param gt_boxes_list: list of numpy array for GT boxes (one per file)
    :param dt_boxes_list: list of numpy array for detected boxes
    :param classes: iterable of classes names
    :param height: int for box size statistics
    :param width: int for box size statistics
    :param time_tol: int size of the temporal window in micro seconds to look for a detection around a gt box
    """
    output = _convert_to_coco_format(gt_boxes_list,
                                     dt_boxes_list,
                                     classes,
                                     height,
                                     width,
                                     time_tol)

    if output is None:
        out_keys = ('AP', 'AP_50', 'AP_75', 'AP_90', 'AP_S', 'AP_M', 'AP_L', 'AR',
                    'precision', 'recall', 'F1')
        metrics = {k: 0 for k in out_keys}
        for class_name in classes:
            metrics[f"AP_class/{class_name}"] = 0
        return metrics
    else:
        (dataset, results), num_gts = output
        return _coco_eval(dataset, results, num_gts, classes)

def _to_prophesee(det: Dict[str, Tensor]):
    num_bboxes = len(det['boxes'])
    out = np.zeros(shape=(num_bboxes,), dtype=BBOX_DTYPE)
    det = {k: v.cpu().numpy() for k, v in det.items()}
    x1, y1, x2, y2 = det['boxes'].T
    out["x"] = x1
    out["y"] = y1
    out["w"] = x2-x1
    out["h"] = y2-y1
    out["class_id"] = det["labels"]
    out["class_confidence"] = det.get("scores", np.ones(shape=(num_bboxes,), dtype="float32"))
    return out

def _match_times(all_ts, gt_boxes, dt_boxes, time_tol):
    """
    match ground truth boxes and ground truth detections at all timestamps using a specified tolerance
    return a list of boxes vectors
    """
    gt_size = len(gt_boxes)
    dt_size = len(dt_boxes)

    windowed_gt = []
    windowed_dt = []

    low_gt, high_gt = 0, 0
    low_dt, high_dt = 0, 0
    for ts in all_ts:

        while low_gt < gt_size and gt_boxes[low_gt]['t'] < ts:
            low_gt += 1
        # the high index is at least as big as the low one
        high_gt = max(low_gt, high_gt)
        while high_gt < gt_size and gt_boxes[high_gt]['t'] <= ts:
            high_gt += 1

        # detection are allowed to be inside a window around the right detection timestamp
        low = ts - time_tol
        high = ts + time_tol
        while low_dt < dt_size and dt_boxes[low_dt]['t'] < low:
            low_dt += 1
        # the high index is at least as big as the low one
        high_dt = max(low_dt, high_dt)
        while high_dt < dt_size and dt_boxes[high_dt]['t'] <= high:
            high_dt += 1

        windowed_gt.append(gt_boxes[low_gt:high_gt])
        windowed_dt.append(dt_boxes[low_dt:high_dt])

    return windowed_gt, windowed_dt


def _coco_eval(dataset, results, num_gts, class_names):
    """simple helper function wrapping around COCO's Python API
    :params:  gts iterable of numpy boxes for the ground truth
    :params:  detections iterable of numpy boxes for the detections
    :params:  height int
    :params:  width int
    :params:  labelmap iterable of class labels
    """


    # Meaning: https://cocodataset.org/#detection-eval
    out_keys = ('AP', 'AP_50', 'AP_75', 'AP_S', 'AP_M', 'AP_L')
    out_dict = {k: 0.0 for k in out_keys}


    coco_gt = COCO()
    coco_gt.dataset = dataset
    coco_gt.createIndex()
    coco_pred = coco_gt.loadRes(results)

    coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
    coco_eval.params.imgIds = np.arange(1, num_gts + 1, dtype=int)
    coco_eval.evaluate()
    coco_eval.accumulate()

    with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
        # info: https://stackoverflow.com/questions/8391411/how-to-block-calls-to-print
        coco_eval.summarize()
    for idx, key in enumerate(out_keys):
        out_dict[key] = coco_eval.stats[idx]

    # Extra metrics from COCOeval internals
    iou_thrs = coco_eval.params.iouThrs
    precision = coco_eval.eval["precision"]
    recall = coco_eval.eval["recall"]
    area_idx = 0  # all
    max_det_idx = len(coco_eval.params.maxDets) - 1  # typically 100

    def _iou_index(iou):
        idx = np.where(np.isclose(iou_thrs, iou))[0]
        return int(idx[0]) if len(idx) else None

    def _mean_precision_at_iou(iou):
        t = _iou_index(iou)
        if t is None:
            return 0.0
        p = precision[t, :, :, area_idx, max_det_idx]
        p = p[p > -1]
        return float(np.mean(p)) if p.size else 0.0

    def _mean_recall_at_iou(iou):
        t = _iou_index(iou)
        if t is None:
            return 0.0
        r = recall[t, :, area_idx, max_det_idx]
        r = r[r > -1]
        return float(np.mean(r)) if r.size else 0.0

    out_dict["AP_90"] = _mean_precision_at_iou(0.90)
    out_dict["AR"] = float(coco_eval.stats[8])
    out_dict["precision"] = _mean_precision_at_iou(0.50)
    out_dict["recall"] = _mean_recall_at_iou(0.50)
    p_val = out_dict["precision"]
    r_val = out_dict["recall"]
    out_dict["F1"] = 2 * p_val * r_val / (p_val + r_val) if (p_val + r_val) > 0 else 0.0

    # Per-class AP (IoU=0.5:0.95)
    num_classes = precision.shape[2]
    for class_idx in range(num_classes):
        class_name = class_names[class_idx] if class_idx < len(class_names) else f"class_{class_idx}"
        p = precision[:, :, class_idx, area_idx, max_det_idx]
        p = p[p > -1]
        out_dict[f"AP_class/{class_name}"] = float(np.mean(p)) if p.size else 0.0

    return out_dict



def _to_coco_format(gts, detections, categories, height=240, width=304):
    """
    utilitary function producing our data in a COCO usable format
    """
    annotations = []
    results = []
    images = []

    # to dictionary
    for image_id, (gt, pred) in enumerate(zip(gts, detections)):
        im_id = image_id + 1

        images.append(
            {"date_captured": "2019",
             "file_name": "n.a",
             "id": im_id,
             "license": 1,
             "url": "",
             "height": height,
             "width": width})

        for bbox in gt:
            x1, y1 = bbox['x'], bbox['y']
            w, h = bbox['w'], bbox['h']
            area = w * h

            annotation = {
                "area": float(area),
                "iscrowd": False,
                "image_id": im_id,
                "bbox": [x1, y1, w, h],
                "category_id": int(bbox['class_id']) + 1,
                "id": len(annotations) + 1
            }
            annotations.append(annotation)

        for bbox in pred:

            image_result = {
                'image_id': im_id,
                'category_id': int(bbox['class_id']) + 1,
                'score': float(bbox['class_confidence']),
                'bbox': [bbox['x'], bbox['y'], bbox['w'], bbox['h']],
            }
            results.append(image_result)

    dataset = {"info": {},
               "licenses": [],
               "type": 'instances',
               "images": images,
               "annotations": annotations,
               "categories": categories}
    return dataset, results
