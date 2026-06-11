"""
Adaptive Event Sampling Strategy

Dynamically adjusts sampling intervals based on scene complexity and motion.
Uses exponential moving average for temporal smoothing to avoid oscillations.

References:
- Gehrig, D., et al. "Asynchronous, Photometric Feature Tracking." IROS 2020.
- Rebecq, H., et al. "High Speed and High Dynamic Range Video." TPAMI 2019.
"""

import numpy as np
from .complexity_estimator import ComplexityEstimator
from .motion_estimator import MotionEstimator


class AdaptiveSampler:
    """
    Adaptive sampling strategy that adjusts event window size based on scene characteristics.
    """
    
    def __init__(self, 
                 base_num_us=50000,
                 min_num_us=10000,
                 max_num_us=200000,
                 complexity_weight=0.5,
                 motion_weight=0.5,
                 width=640,
                 height=480,
                 time_window=1000000,
                 smoothing_alpha=0.7,
                 use_ema=True):
        """
        Args:
            base_num_us: Base sampling interval in microseconds
            min_num_us: Minimum sampling interval (for fast motion/complex scenes)
            max_num_us: Maximum sampling interval (for static scenes)
            complexity_weight: Weight for complexity score in decision
            motion_weight: Weight for motion score in decision
            width: Image width
            height: Image height
            time_window: Total time window for events (microseconds)
            smoothing_alpha: EMA smoothing factor (0=no smoothing, 1=full smoothing)
            use_ema: Whether to use exponential moving average for smoothing
        """
        self.base_num_us = base_num_us
        self.min_num_us = min_num_us
        self.max_num_us = max_num_us
        self.width = width
        self.height = height
        self.time_window = time_window
        self.use_ema = use_ema
        self.smoothing_alpha = smoothing_alpha
        
        # Normalize weights
        total_weight = complexity_weight + motion_weight
        self.complexity_weight = complexity_weight / total_weight
        self.motion_weight = motion_weight / total_weight
        
        # Initialize estimators
        self.complexity_estimator = ComplexityEstimator(
            spatial_bins=32,
            temporal_bins=10,
            width=width,
            height=height
        )
        
        self.motion_estimator = MotionEstimator(
            grid_size=16,
            time_window_us=50000,
            spatial_radius=3,
            min_events_per_grid=10,
            width=width,
            height=height
        )
        
        # EMA state for smoothing
        self.ema_score = None
        self.ema_interval = None
        
        # Statistics tracking
        self.stats = {
            'total_calls': 0,
            'avg_complexity': 0.0,
            'avg_motion': 0.0,
            'avg_interval': 0.0
        }
        
    def compute_sampling_interval(self, events: dict) -> int:
        """
        Compute adaptive sampling interval based on event stream characteristics.
        
        Args:
            events: Dictionary with keys:
                - 'x': array of x coordinates (denormalized pixels)
                - 'y': array of y coordinates (denormalized pixels)
                - 't': array of timestamps (microseconds, absolute or normalized)
                - 'p': array of polarities
        
        Returns:
            sampling_interval: int, time interval in microseconds
        """
        # Handle empty events
        if len(events['t']) == 0:
            return self.max_num_us
        
        # Compute scene characteristics
        complexity_score = self.complexity_estimator.estimate(events)
        motion_score = self.motion_estimator.estimate(events)
        
        # Weighted combination
        combined_score = (self.complexity_weight * complexity_score + 
                         self.motion_weight * motion_score)
        
        # Apply exponential moving average for temporal smoothing
        if self.use_ema:
            if self.ema_score is None:
                self.ema_score = combined_score
            else:
                self.ema_score = (self.smoothing_alpha * self.ema_score + 
                                 (1 - self.smoothing_alpha) * combined_score)
            combined_score = self.ema_score
        
        # Map score to sampling interval using inverse relationship
        # High score (complex/fast) -> short interval (frequent sampling)
        # Low score (simple/slow) -> long interval (sparse sampling)
        
        # Use exponential mapping for smoother transitions
        # interval = max_interval * exp(-k * score)
        # where k is chosen such that score=1 gives min_interval
        
        k = np.log(self.max_num_us / self.min_num_us)
        interval = self.max_num_us * np.exp(-k * combined_score)
        
        # Clip to bounds
        interval = np.clip(interval, self.min_num_us, self.max_num_us)
        interval = int(interval)
        
        # Apply EMA smoothing to interval as well for stability
        if self.use_ema:
            if self.ema_interval is None:
                self.ema_interval = interval
            else:
                self.ema_interval = (self.smoothing_alpha * self.ema_interval + 
                                    (1 - self.smoothing_alpha) * interval)
            interval = int(self.ema_interval)
        
        # Update statistics
        self._update_stats(complexity_score, motion_score, interval)
        
        return interval
    
    def _update_stats(self, complexity: float, motion: float, interval: int):
        """Update running statistics for monitoring."""
        n = self.stats['total_calls']
        
        # Incremental mean update
        self.stats['avg_complexity'] = (n * self.stats['avg_complexity'] + complexity) / (n + 1)
        self.stats['avg_motion'] = (n * self.stats['avg_motion'] + motion) / (n + 1)
        self.stats['avg_interval'] = (n * self.stats['avg_interval'] + interval) / (n + 1)
        self.stats['total_calls'] = n + 1
    
    def get_stats(self) -> dict:
        """Get current statistics."""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset statistics tracking."""
        self.stats = {
            'total_calls': 0,
            'avg_complexity': 0.0,
            'avg_motion': 0.0,
            'avg_interval': 0.0
        }
        self.ema_score = None
        self.ema_interval = None
    
    def get_detailed_decision(self, events: dict) -> dict:
        """
        Get detailed breakdown of the sampling decision (for analysis/debugging).
        
        Returns:
            Dictionary with all intermediate values
        """
        if len(events['t']) == 0:
            return {
                'complexity_score': 0.0,
                'motion_score': 0.0,
                'combined_score': 0.0,
                'ema_score': self.ema_score if self.ema_score is not None else 0.0,
                'sampling_interval': self.max_num_us,
                'interval_reduction_percent': 0.0
            }
        
        complexity_score = self.complexity_estimator.estimate(events)
        motion_score = self.motion_estimator.estimate(events)
        combined_score = (self.complexity_weight * complexity_score + 
                         self.motion_weight * motion_score)
        
        # Get complexity details
        complexity_details = self.complexity_estimator.get_detailed_metrics(events)
        motion_details = self.motion_estimator.get_detailed_metrics(events)
        
        # Compute interval
        k = np.log(self.max_num_us / self.min_num_us)
        interval = self.max_num_us * np.exp(-k * combined_score)
        interval = int(np.clip(interval, self.min_num_us, self.max_num_us))
        
        # Calculate reduction percentage
        reduction = 100.0 * (1.0 - interval / self.base_num_us)
        
        return {
            'complexity_score': float(complexity_score),
            'motion_score': float(motion_score),
            'combined_score': float(combined_score),
            'ema_score': float(self.ema_score) if self.ema_score is not None else None,
            'sampling_interval': interval,
            'base_interval': self.base_num_us,
            'interval_reduction_percent': float(reduction),
            'complexity_details': complexity_details,
            'motion_details': motion_details,
            'num_events': len(events['t'])
        }


class FixedSampler:
    """
    Baseline sampler with fixed interval (for ablation studies).
    """
    
    def __init__(self, num_us=50000):
        """
        Args:
            num_us: Fixed sampling interval in microseconds
        """
        self.num_us = num_us
        self.stats = {
            'total_calls': 0,
            'avg_interval': num_us
        }
    
    def compute_sampling_interval(self, events: dict) -> int:
        """Always return fixed interval."""
        self.stats['total_calls'] += 1
        return self.num_us
    
    def get_stats(self) -> dict:
        return self.stats.copy()
    
    def reset_stats(self):
        self.stats['total_calls'] = 0
    
    def get_detailed_decision(self, events: dict) -> dict:
        """Return minimal info for compatibility."""
        return {
            'complexity_score': None,
            'motion_score': None,
            'combined_score': None,
            'sampling_interval': self.num_us,
            'interval_reduction_percent': 0.0,
            'num_events': len(events['t']) if len(events['t']) > 0 else 0
        }