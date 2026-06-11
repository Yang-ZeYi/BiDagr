"""
Adaptive DSEC Dataset Wrapper

Wraps the original DSEC dataset to enable adaptive event sampling.
Maintains full compatibility with the original dataset interface.
"""

import numpy as np
import torch
from pathlib import Path
from typing import Optional, Callable

from dagr.data.dsec_data import DSEC, interpolate_tracks, tracks_to_array
from dagr.data.utils import to_data
from dagr.data.dsec_utils import filter_small_bboxes
from .sampler import AdaptiveSampler, FixedSampler


class AdaptiveDSEC(DSEC):
    """
    Adaptive version of DSEC dataset with dynamic event sampling.
    
    Inherits all functionality from DSEC and adds adaptive sampling capability.
    """
    
    def __init__(self,
                 root: Path,
                 split: str,
                 transform: Optional[Callable] = None,
                 debug=False,
                 min_bbox_diag=0,
                 min_bbox_height=0,
                 scale=2,
                 cropped_height=430,
                 only_perfect_tracks=False,
                 demo=False,
                 no_eval=False,
                 # Adaptive sampling parameters
                 enable_adaptive=True,
                 sampler_config=None):
        """
        Args:
            (same as DSEC base class)
            enable_adaptive: If True, use adaptive sampling; if False, use fixed sampling
            sampler_config: Dictionary with sampler parameters
        """
        # Initialize parent class
        super().__init__(
            root=root,
            split=split,
            transform=transform,
            debug=debug,
            min_bbox_diag=min_bbox_diag,
            min_bbox_height=min_bbox_height,
            scale=scale,
            cropped_height=cropped_height,
            only_perfect_tracks=only_perfect_tracks,
            demo=demo,
            no_eval=no_eval
        )
        
        self.enable_adaptive = enable_adaptive
        
        # Initialize sampler
        if sampler_config is None:
            sampler_config = {}
        
        # Set default values
        default_config = {
            'base_num_us': 50000,
            'min_num_us': 10000,
            'max_num_us': 200000,
            'complexity_weight': 0.5,
            'motion_weight': 0.5,
            'smoothing_alpha': 0.7,
            'use_ema': True,
            'width': self.width,
            'height': self.height,
            'time_window': self.time_window
        }
        
        # Update with user config
        default_config.update(sampler_config)
        
        if enable_adaptive:
            self.sampler = AdaptiveSampler(**default_config)
        else:
            # Use fixed sampler for baseline
            self.sampler = FixedSampler(num_us=default_config['base_num_us'])
        
        # Cache for the last computed interval (for logging)
        self.last_adaptive_interval = None
        self.last_adaptive_details = None
        
    def __getitem__(self, idx):
        """
        Override parent's __getitem__ to add adaptive sampling.
        """
        # Get the directory and index info
        dataset, image_index_pairs, track_masks, rel_idx = self.rel_index(idx)
        image_index_0, image_index_1 = image_index_pairs[rel_idx]
        image_ts_0, image_ts_1_original = dataset.images.timestamps[[image_index_0, image_index_1]]
        
        # Get detections
        detections_0 = self.dataset.get_tracks(image_index_0, mask=track_masks, 
                                               directory_name=dataset.root.name)
        detections_1_original = self.dataset.get_tracks(image_index_1, mask=track_masks, 
                                                       directory_name=dataset.root.name)
        
        detections_0 = self.preprocess_detections(detections_0)
        detections_1_original = self.preprocess_detections(detections_1_original)
        
        # Get image
        image_0 = self.dataset.get_image(image_index_0, directory_name=dataset.root.name)
        image_0 = self.preprocess_image(image_0)
        
        # Get ALL events first (to compute adaptive interval)
        events_full = self.dataset.get_events(image_index_0, directory_name=dataset.root.name)
        
        # Determine the actual timestamp to use
        image_ts_1 = image_ts_1_original
        detections_1 = detections_1_original
        
        if not self.enable_adaptive:
            # Use fixed interval from parent class
            if self.num_us >= 0:
                image_ts_1 = image_ts_0 + self.num_us
                # Clip to original timestamp
                if image_ts_1 > image_ts_1_original:
                    image_ts_1 = image_ts_1_original
                elif not self.no_eval and len(detections_0) > 0 and len(detections_1_original) > 0:
                    # Only interpolate if we have valid detections in both frames
                    # and track IDs match
                    if self._can_interpolate(detections_0, detections_1_original):
                        detections_1 = interpolate_tracks(detections_0, detections_1_original, image_ts_1)
            
            self.last_adaptive_interval = self.num_us if self.num_us >= 0 else (image_ts_1_original - image_ts_0)
            
        else:
            # Adaptive sampling: compute interval based on event characteristics
            events_for_analysis = self._denormalize_events_for_analysis(events_full)
            
            # Compute adaptive interval
            adaptive_interval = self.sampler.compute_sampling_interval(events_for_analysis)
            
            # Store for logging
            self.last_adaptive_interval = adaptive_interval
            self.last_adaptive_details = self.sampler.get_detailed_decision(events_for_analysis)
            
            # Apply adaptive interval
            image_ts_1 = image_ts_0 + adaptive_interval
            
            # Clip to original timestamp (can't go beyond the next frame)
            if image_ts_1 > image_ts_1_original:
                image_ts_1 = image_ts_1_original
            elif not self.no_eval and len(detections_0) > 0 and len(detections_1_original) > 0:
                # Interpolate tracks only if possible
                if self._can_interpolate(detections_0, detections_1_original):
                    detections_1 = interpolate_tracks(detections_0, detections_1_original, image_ts_1)
        
        # Filter events up to image_ts_1
        events = {k: v[events_full['t'] < image_ts_1] for k, v in events_full.items()}
        
        # Preprocess events (normalize timestamps, etc.)
        events = self.preprocess_events(events)
        
        # Convert to PyG Data object
        data = to_data(
            **events, 
            bbox=tracks_to_array(detections_1), 
            bbox0=tracks_to_array(detections_0), 
            t0=image_ts_0, 
            t1=image_ts_1,
            width=self.width, 
            height=self.height, 
            time_window=self.time_window,
            image=image_0, 
            sequence=str(dataset.root.name)
        )
        
        # Apply transforms
        if self.transform is not None:
            data = self.transform(data)
        
        # Filter small bboxes
        mask = filter_small_bboxes(data.bbox[:, 2], data.bbox[:, 3], 
                                   self.min_bbox_height, self.min_bbox_diag)
        data.bbox = data.bbox[mask]
        
        mask = filter_small_bboxes(data.bbox0[:, 2], data.bbox0[:, 3], 
                                   self.min_bbox_height, self.min_bbox_diag)
        data.bbox0 = data.bbox0[mask]
        
        # Add adaptive sampling metadata to data object
        data.adaptive_interval = self.last_adaptive_interval
        if self.enable_adaptive and self.last_adaptive_details is not None:
            data.complexity_score = self.last_adaptive_details['complexity_score']
            data.motion_score = self.last_adaptive_details['motion_score']
            data.combined_score = self.last_adaptive_details['combined_score']
        
        return data
    
    def _can_interpolate(self, detections_0, detections_1):
        """
        Check if we can safely interpolate between two detection frames.
        
        Interpolation is only valid if:
        1. Both frames have the same number of detections
        2. Track IDs match between frames
        """
        if len(detections_0) != len(detections_1):
            return False
        
        if len(detections_0) == 0:
            return True  # Empty is fine
        
        # Check if track_id field exists
        if 'track_id' not in detections_0.dtype.names or 'track_id' not in detections_1.dtype.names:
            # If no track_id, assume they don't match
            return False
        
        # Sort by track_id and compare
        track_ids_0 = np.sort(detections_0['track_id'])
        track_ids_1 = np.sort(detections_1['track_id'])
        
        return np.array_equal(track_ids_0, track_ids_1)
    
    def _denormalize_events_for_analysis(self, events: dict) -> dict:
        """
        Denormalize events from DSEC format for complexity/motion analysis.
        
        Args:
            events: Dictionary with 'x', 'y', 't', 'p' in DSEC format
        
        Returns:
            Dictionary with denormalized events for analysis
        """
        # Safe conversion to numpy arrays
        x = events['x'] if isinstance(events['x'], np.ndarray) else (
            events['x'].numpy() if hasattr(events['x'], 'numpy') else np.array(events['x'])
        )
        y = events['y'] if isinstance(events['y'], np.ndarray) else (
            events['y'].numpy() if hasattr(events['y'], 'numpy') else np.array(events['y'])
        )
        t = events['t'] if isinstance(events['t'], np.ndarray) else (
            events['t'].numpy() if hasattr(events['t'], 'numpy') else np.array(events['t'])
        )
        p = events['p'] if isinstance(events['p'], np.ndarray) else (
            events['p'].numpy() if hasattr(events['p'], 'numpy') else np.array(events['p'])
        )
        
        # Ensure correct data types
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        t = t.astype(np.int64)
        
        return {
            'x': x,
            'y': y,
            't': t,
            'p': p
        }
    
    def get_sampler_stats(self) -> dict:
        """Get statistics from the sampler."""
        return self.sampler.get_stats()
    
    def reset_sampler_stats(self):
        """Reset sampler statistics."""
        self.sampler.reset_stats()
    
    def get_last_adaptive_details(self) -> dict:
        """Get detailed information about the last adaptive decision."""
        if self.last_adaptive_details is None:
            return {}
        return self.last_adaptive_details
    
    def set_adaptive_mode(self, enable: bool):
        """
        Enable or disable adaptive sampling on the fly.
        
        Args:
            enable: If True, use adaptive sampling; if False, use fixed
        """
        if enable and not isinstance(self.sampler, AdaptiveSampler):
            # Switch to adaptive
            config = {
                'base_num_us': self.sampler.num_us if hasattr(self.sampler, 'num_us') else 50000,
                'width': self.width,
                'height': self.height,
                'time_window': self.time_window
            }
            self.sampler = AdaptiveSampler(**config)
            self.enable_adaptive = True
        elif not enable and not isinstance(self.sampler, FixedSampler):
            # Switch to fixed
            num_us = self.sampler.base_num_us
            self.sampler = FixedSampler(num_us=num_us)
            self.enable_adaptive = False