"""
Scene Complexity Estimator for Event-based Data

Based on established metrics from event-based vision literature:
1. Event density (Mueggler et al., IJRR 2017)
2. Spatial entropy (Shannon entropy on spatial distribution)
3. Temporal variance (Gallego et al., CVPR 2018)
"""

import numpy as np
from scipy.stats import entropy


class ComplexityEstimator:
    """
    Estimates scene complexity from event streams using multiple metrics.
    
    References:
    - Mueggler, E., et al. "The event-camera dataset and simulator." IJRR 2017.
    - Gallego, G., et al. "Event-based vision: A survey." CVPR 2018.
    """
    
    def __init__(self, 
                 spatial_bins=32, 
                 temporal_bins=10,
                 density_weight=0.4,
                 spatial_entropy_weight=0.3,
                 temporal_variance_weight=0.3,
                 width=640,
                 height=480):
        """
        Args:
            spatial_bins: Number of spatial bins for histogram (default: 32x32 grid)
            temporal_bins: Number of temporal bins for variance calculation
            density_weight: Weight for event density metric
            spatial_entropy_weight: Weight for spatial entropy metric
            temporal_variance_weight: Weight for temporal variance metric
            width: Image width
            height: Image height
        """
        self.spatial_bins = spatial_bins
        self.temporal_bins = temporal_bins
        self.width = width
        self.height = height
        
        # Normalize weights
        total_weight = density_weight + spatial_entropy_weight + temporal_variance_weight
        self.weights = np.array([
            density_weight / total_weight,
            spatial_entropy_weight / total_weight,
            temporal_variance_weight / total_weight
        ])
        
        # Normalization constants (learned from DSEC dataset statistics)
        # These are approximate values based on typical event rates
        self.density_norm = 0.1  # events per pixel
        self.entropy_norm = np.log(spatial_bins * spatial_bins)  # max entropy
        self.variance_norm = 1000.0  # typical variance in event counts
        
    def estimate(self, events: dict) -> float:
        """
        Estimate scene complexity from event stream.
        
        Args:
            events: Dictionary with keys:
                - 'x': np.ndarray [N], x coordinates (denormalized pixel values)
                - 'y': np.ndarray [N], y coordinates (denormalized pixel values)
                - 't': np.ndarray [N], timestamps in microseconds
                - 'p': np.ndarray [N, 1], polarity
        
        Returns:
            complexity_score: float in [0, 1], higher means more complex scene
        """
        if len(events['t']) == 0:
            return 0.0
        
        # Extract event data
        x = events['x'] if isinstance(events['x'], np.ndarray) else events['x'].numpy()
        y = events['y'] if isinstance(events['y'], np.ndarray) else events['y'].numpy()
        t = events['t'] if isinstance(events['t'], np.ndarray) else events['t'].numpy()
        
        # Compute individual metrics
        density_score = self._compute_density(x, y)
        entropy_score = self._compute_spatial_entropy(x, y)
        variance_score = self._compute_temporal_variance(t)
        
        # Weighted combination
        scores = np.array([density_score, entropy_score, variance_score])
        complexity = np.dot(self.weights, scores)
        
        # Clip to [0, 1]
        return float(np.clip(complexity, 0.0, 1.0))
    
    def _compute_density(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute normalized event density (events per pixel).
        Higher density indicates more visual activity.
        """
        num_events = len(x)
        num_pixels = self.width * self.height
        density = num_events / num_pixels
        
        # Normalize to [0, 1]
        normalized = density / self.density_norm
        return min(normalized, 1.0)
    
    def _compute_spatial_entropy(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute Shannon entropy of spatial event distribution.
        Higher entropy means events are more uniformly distributed (complex scene).
        Lower entropy means events are concentrated (simple/sparse scene).
        """
        # Create 2D histogram
        hist, _, _ = np.histogram2d(
            x, y,
            bins=[self.spatial_bins, self.spatial_bins],
            range=[[0, self.width], [0, self.height]]
        )
        
        # Flatten and normalize to probability distribution
        hist_flat = hist.flatten()
        hist_flat = hist_flat[hist_flat > 0]  # Remove empty bins
        
        if len(hist_flat) == 0:
            return 0.0
        
        prob_dist = hist_flat / hist_flat.sum()
        
        # Compute Shannon entropy
        spatial_entropy = entropy(prob_dist, base=2)
        
        # Normalize by maximum possible entropy
        normalized = spatial_entropy / self.entropy_norm
        return min(normalized, 1.0)
    
    def _compute_temporal_variance(self, t: np.ndarray) -> float:
        """
        Compute variance in event counts across temporal bins.
        Higher variance indicates non-uniform temporal activity (fast motion/changes).
        """
        if len(t) < 2:
            return 0.0
        
        # Divide time into bins
        t_min, t_max = t.min(), t.max()
        if t_max - t_min == 0:
            return 0.0
        
        # Count events in each temporal bin
        bin_edges = np.linspace(t_min, t_max, self.temporal_bins + 1)
        counts, _ = np.histogram(t, bins=bin_edges)
        
        # Compute coefficient of variation (normalized variance)
        mean_count = counts.mean()
        if mean_count == 0:
            return 0.0
        
        std_count = counts.std()
        cv = std_count / mean_count  # Coefficient of variation
        
        # Normalize
        normalized = cv / np.sqrt(self.temporal_bins)  # Typical CV range
        return min(normalized, 1.0)
    
    def get_detailed_metrics(self, events: dict) -> dict:
        """
        Get detailed breakdown of all complexity metrics (for analysis/debugging).
        
        Returns:
            Dictionary with individual metric scores and final complexity
        """
        if len(events['t']) == 0:
            return {
                'density': 0.0,
                'spatial_entropy': 0.0,
                'temporal_variance': 0.0,
                'complexity': 0.0
            }
        
        x = events['x'] if isinstance(events['x'], np.ndarray) else events['x'].numpy()
        y = events['y'] if isinstance(events['y'], np.ndarray) else events['y'].numpy()
        t = events['t'] if isinstance(events['t'], np.ndarray) else events['t'].numpy()
        
        density = self._compute_density(x, y)
        spatial_ent = self._compute_spatial_entropy(x, y)
        temporal_var = self._compute_temporal_variance(t)
        
        scores = np.array([density, spatial_ent, temporal_var])
        complexity = np.dot(self.weights, scores)
        
        return {
            'density': float(density),
            'spatial_entropy': float(spatial_ent),
            'temporal_variance': float(temporal_var),
            'complexity': float(np.clip(complexity, 0.0, 1.0)),
            'num_events': len(x)
        }