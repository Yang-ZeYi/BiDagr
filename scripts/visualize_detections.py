import cv2
import argparse

from pathlib import Path
import numpy as np

# from dsec_det.directory import DSECDirectory
# from dsec_det.io import extract_from_h5_by_timewindow, extract_image_by_index, load_start_and_end_time
# from dsec_det.preprocessing import compute_index

from dagr.visualization.bbox_viz import draw_bbox_on_img
from dagr.visualization.event_viz import draw_events_on_image

import numpy as np
from functools import lru_cache


class BaseDirectory:
    def __init__(self, root):
        self.root = root


class DSECDirectory:
    def __init__(self, root):
        self.root = root
        self.images = ImageDirectory(root / "images")
        self.events = EventDirectory(root / "events")
        self.tracks = TracksDirectory(root / "object_detections")


class ImageDirectory(BaseDirectory):
    @property
    @lru_cache(maxsize=1)
    def timestamps(self):
        return np.genfromtxt(self.root / "timestamps.txt", dtype="int64")

    @property
    @lru_cache(maxsize=1)
    def image_files_rectified(self):
        return sorted(list((self.root / "left/rectified").glob("*.png")))

    @property
    @lru_cache(maxsize=1)
    def image_files_distorted(self):
        return sorted(list((self.root / "left/distorted").glob("*.png")))


class EventDirectory(BaseDirectory):
    @property
    @lru_cache(maxsize=1)
    def event_file(self):
        return self.root / "left/events.h5"


class TracksDirectory(BaseDirectory):
    @property
    @lru_cache(maxsize=1)
    def tracks(self):
        return np.load(self.root / "left/tracks.npy")


#io.py

import hdf5plugin
import yaml
import h5py
import numpy as np
import math
from pathlib import Path
import filecmp


# from https://stackoverflow.com/questions/4187564/recursively-compare-two-directories-to-ensure-they-have-the-same-files-and-subdi
class dircmp(filecmp.dircmp):
    """
    Compare the content of dir1 and dir2. In contrast with filecmp.dircmp, this
    subclass compares the content of files with the same path.
    """
    def phase3(self):
        """
        Find out differences between common files.
        Ensure we are using content comparison with shallow=False.
        """
        fcomp = filecmp.cmpfiles(self.left, self.right, self.common_files,
                                 shallow=False)
        self.same_files, self.diff_files, self.funny_files = fcomp

def compare_dirs(dir1: Path, dir2: Path):
    """
    Compare two directory trees content.
    Return False if they differ, True is they are the same.
    """
    compared = dircmp(dir1, dir2)
    if (compared.left_only or compared.right_only or compared.diff_files
        or compared.funny_files):
        return False
    for subdir in compared.common_dirs:
        if not compare_dirs(dir1 / subdir, dir2 / subdir):
            return False
    return True

def _extract_from_h5_by_index(filehandle, ev_start_idx: int, ev_end_idx: int):
    events = filehandle['events']
    x = events['x']
    y = events['y']
    p = events['p']
    t = events['t']

    x_new = x[ev_start_idx:ev_end_idx]
    y_new = y[ev_start_idx:ev_end_idx]
    p_new = p[ev_start_idx:ev_end_idx]
    t_new = t[ev_start_idx:ev_end_idx].astype("int64") + filehandle["t_offset"][()]

    output = {
        'p': p_new,
        't': t_new,
        'x': x_new,
        'y': y_new,
    }
    return output


def get_num_events(h5file):
    with h5py.File(str(h5file), 'r') as h5f:
        return len(h5f['events/t'])

def extract_from_h5_by_index(h5file, ev_start_idx: int, ev_end_idx: int):
    with h5py.File(str(h5file), 'r') as h5f:
        return _extract_from_h5_by_index(h5f, ev_start_idx, ev_end_idx)

def extract_from_h5_by_timewindow(h5file, t_min_us: int, t_max_us: int):
    with h5py.File(str(h5file), 'r') as h5f:
        ms2idx = np.asarray(h5f['ms_to_idx'], dtype='int64')
        t_offset = h5f['t_offset'][()]

        events = h5f['events']
        t = events['t']

        t_ev_start_us = t_min_us - t_offset
        #assert t_ev_start_us >= t[0], (t_ev_start_us, t[0])
        t_ev_start_ms = t_ev_start_us // 1000
        ms2idx_start_idx = t_ev_start_ms
        ev_start_idx = ms2idx[ms2idx_start_idx]

        t_ev_end_us = t_max_us - t_offset
        #assert t_ev_end_us <= t[-1], (t_ev_end_us, t[-1])
        t_ev_end_ms = math.floor(t_ev_end_us / 1000)
        ms2idx_end_idx = np.clip(t_ev_end_ms, 0, len(ms2idx)-1)
        ev_end_idx = ms2idx[ms2idx_end_idx]

        return _extract_from_h5_by_index(h5f, ev_start_idx, ev_end_idx)

def h5_file_to_dict(h5_file: Path):
    with h5py.File(h5_file) as fh:
        return {k: fh[k][()] for k in fh.keys()}

def yaml_file_to_dict(yaml_file: Path):
    with yaml_file.open() as fh:
        return yaml.load(fh, Loader=yaml.UnsafeLoader)


#preprocessing

import numpy as np


def compute_img_idx_to_track_idx(t_track, t_image):
    x, counts = np.unique(t_track, return_counts=True)
    i, j = (x.reshape((-1,1)) == t_image.reshape((1,-1))).nonzero()
    deltas = np.zeros_like(t_image)

    deltas[j] = counts[i]

    idx = np.concatenate([np.array([0]), deltas]).cumsum()
    return np.stack([idx[:-1], idx[1:]], axis=-1).astype("uint64")

if __name__ == '__main__':
    parser = argparse.ArgumentParser("""Visualization script to show bounding boxes""")
    parser.add_argument("--detections_folder", help="Path to folder with detections.", type=Path)
    parser.add_argument("--dataset_directory", help="Path to DSEC folder including which split.", type=Path, default="/home/yang/work/dagr/data/dsec/test/")
    parser.add_argument("--vis_time_step_us", help="Number of microseconds to step each iteration.", type=int, default=1000)
    parser.add_argument("--event_time_window_us", help="Length of sliding event time window for visualization.", type=int, default=5000)
    parser.add_argument("--sequence", help="Sequence to visualize. Must be an official DSEC sequence e.g. zurich_city_13_b", default="zurich_city_13_b", type=str)
    parser.add_argument("--write_to_output", help="Whether to save images in folder ${detections_folder}/visualization. Otherwise, just cv2.imshow is used.", action="store_true")
    args = parser.parse_args()

    assert args.dataset_directory.exists()
    assert args.vis_time_step_us > 0
    assert args.event_time_window_us > 0

    if args.write_to_output:
        assert (args.detections_folder / f"detections_{args.sequence}.npy").exists()
        assert args.detections_folder.exists()
        output_path = args.detections_folder / "visualization"
        output_path.mkdir(parents=True, exist_ok=True)

    dsec_directory = DSECDirectory(args.dataset_directory / args.sequence)

    t0, t1 = load_start_and_end_time(dsec_directory)

    vis_timestamps = np.arange(t0, t1, step=args.vis_time_step_us)
    step_index_to_image_index = compute_index(dsec_directory.images.timestamps, vis_timestamps)

    show_detections = args.detections_folder is not None

    if not show_detections:
        print("Did not specifiy detections. Just showing events and images.")

    if show_detections:
        detections_file = args.detections_folder / f"detections_{args.sequence}.npy"
        detections = np.load(detections_file)
        detection_timestamps = np.unique(detections['t'])
        step_index_to_boxes_index = compute_index(detection_timestamps, vis_timestamps)

    scale = 2

    for step, t in enumerate(vis_timestamps):

        # find most recent image
        image_index = step_index_to_image_index[step]
        image = extract_image_by_index(dsec_directory.images.image_files_distorted, image_index)

        # find events within time window [image_timestamps, t]
        events = extract_from_h5_by_timewindow(dsec_directory.events.event_file, t-args.event_time_window_us, t)
        image = draw_events_on_image(image, events['x'], events['y'], events['p'])

        if show_detections:
            # find most recent bounding boxes
            boxes_index = step_index_to_boxes_index[step]
            boxes_timestamp = detection_timestamps[boxes_index]
            boxes = detections[detections['t'] == boxes_timestamp]

            # draw them on one image
            scale = 2
            image = draw_bbox_on_img(image, scale*boxes['x'], scale*boxes['y'], scale*boxes['w'], scale*boxes["h"],
                                     boxes["class_id"], boxes['class_confidence'], conf=0.3, nms=0.65)

        if args.write_to_output:
            cv2.imwrite(str(output_path / ("%06d.png" % step)), image)
        else:
            cv2.imshow("DSEC Det: Visualization", image)
            cv2.waitKey(3)

