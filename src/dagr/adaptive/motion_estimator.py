"""
Motion Speed Estimator for Event-based Data

Based on established optical flow and motion estimation methods:
1. Local Plane Fitting (Benosman et al., Neural Computation 2012)
2. Time Surface based motion (Lagorce et al., TPAMI 2017)
3. Spatio-temporal gradient (Lucas-Kanade style for events)

References:
- Benosman, R., et al. "Asynchronous frameless event-based optical flow." Neural Computation 2012.
- Lagorce, X., et al. "HOTS: A hierarchy of event-based time-surfaces." TPAMI 2017.
- Gallego, G., et al. "A unifying contrast maximization framework." CVPR 2018.
"""

import numpy as np
from collections import defaultdict


class MotionEstimator:
    """
    Estimates motion speed from event streams using spatio-temporal gradients.
    """
    
    def __init__(self, 
                 grid_size=16,
                 time_window_us=50000,
                 spatial_radius=3,
                 min_events_per_grid=10,
                 width=640,
                 height=480):
        """
        Args:
            grid_size: Divide image into grid_size x grid_size cells
            time_window_us: Time window for local motion estimation (microseconds)
            spatial_radius: Spatial radius for local neighborhood (pixels)
            min_events_per_grid: Minimum events needed per grid cell for valid estimate
            width: Image width
            height: Image height
        """
        self.grid_size = grid_size
        self.time_window_us = time_window_us
        self.spatial_radius = spatial_radius
        self.min_events_per_grid = min_events_per_grid
        self.width = width
        self.height = height
        
        # Grid dimensions
        self.grid_width = width / grid_size
        self.grid_height = height / grid_size
        
        # Normalization constant (typical motion speed in pixels/second)
        # Based on DSEC dataset: car motion ~100 pixels/sec, pedestrian ~50 pixels/sec
        self.speed_norm = 100.0  # pixels per second
        
    def estimate(self, events: dict) -> float:
        """
        Estimate normalized motion speed from event stream.
        
        Args:
            events: Dictionary with keys:
                - 'x': np.ndarray [N], x coordinates in pixels
                - 'y': np.ndarray [N], y coordinates in pixels
                - 't': np.ndarray [N], timestamps in microseconds
                - 'p': np.ndarray [N, 1], polarity
        
        Returns:
            motion_score: float in [0, 1], higher means faster motion
        """
        if len(events['t']) < 10:  # Need minimum events for motion estimation
            return 0.0
        
        # Extract and convert event data
        x = events['x'] if isinstance(events['x'], np.ndarray) else events['x'].numpy()
        y = events['y'] if isinstance(events['y'], np.ndarray) else events['y'].numpy()
        t = events['t'] if isinstance(events['t'], np.ndarray) else events['t'].numpy()
        p = events['p'] if isinstance(events['p'], np.ndarray) else events['p'].numpy()
        
        # Ensure p is 1D
        if len(p.shape) > 1:
            p = p.flatten()
        
        # Compute motion in each grid cell
        grid_motions = self._compute_grid_motions(x, y, t, p)
        
        if len(grid_motions) == 0:
            return 0.0
        
        # Aggregate: use 75th percentile to be robust to outliers
        motion_magnitudes = np.array(grid_motions)
        motion_speed = np.percentile(motion_magnitudes, 75)
        
        # Normalize to [0, 1]
        normalized = motion_speed / self.speed_norm
        return float(np.clip(normalized, 0.0, 1.0))
    
    def _compute_grid_motions(self, x: np.ndarray, y: np.ndarray, 
                              t: np.ndarray, p: np.ndarray) -> list:
        """
        Compute motion magnitude in each grid cell using local plane fitting.
        
        Returns:
            List of motion magnitudes (pixels/second) for each valid grid cell
        """
        grid_motions = []
        
        # Divide events into grid cells
        grid_events = self._partition_into_grid(x, y, t, p)
        
        for (gx, gy), cell_events in grid_events.items():
            if len(cell_events['t']) < self.min_events_per_grid:
                continue
            
            # Estimate motion in this cell
            motion_mag = self._estimate_local_motion(
                cell_events['x'], 
                cell_events['y'], 
                cell_events['t'],
                cell_events['p']
            )
            
            if motion_mag is not None and motion_mag > 0:
                grid_motions.append(motion_mag)
        
        return grid_motions
    
    def _partition_into_grid(self, x: np.ndarray, y: np.ndarray, 
                            t: np.ndarray, p: np.ndarray) -> dict:
        """
        Partition events into spatial grid cells.
        
        Returns:
            Dictionary mapping (grid_x, grid_y) to event data
        """
        grid_dict = defaultdict(lambda: {'x': [], 'y': [], 't': [], 'p': []})
        
        # Compute grid indices
        grid_x = np.floor(x / self.grid_width).astype(int)
        grid_y = np.floor(y / self.grid_height).astype(int)
        
        # Clip to valid range
        grid_x = np.clip(grid_x, 0, self.grid_size - 1)
        grid_y = np.clip(grid_y, 0, self.grid_size - 1)
        
        # Partition events
        for i in range(len(x)):
            gx, gy = grid_x[i], grid_y[i]
            grid_dict[(gx, gy)]['x'].append(x[i])
            grid_dict[(gx, gy)]['y'].append(y[i])
            grid_dict[(gx, gy)]['t'].append(t[i])
            grid_dict[(gx, gy)]['p'].append(p[i])
        
        # Convert lists to arrays
        result = {}
        for key, val in grid_dict.items():
            result[key] = {
                'x': np.array(val['x']),
                'y': np.array(val['y']),
                't': np.array(val['t']),
                'p': np.array(val['p'])
            }
        
        return result
    
    def _estimate_local_motion(self, x: np.ndarray, y: np.ndarray, 
                               t: np.ndarray, p: np.ndarray) -> float:
        """
        Estimate motion magnitude using spatio-temporal gradient method.
        
        This is a simplified version of the plane fitting approach from:
        Benosman et al., "Asynchronous frameless event-based optical flow", 2012
        
        The method fits a plane to the event cloud in space-time:
        t = ax + by + c
        Then velocity is v = [a, b], and speed = ||v||
        
        Returns:
            Motion magnitude in pixels/second, or None if estimation fails
        """
        if len(t) < 5:  # Need minimum points for fitting
            return None
        
        # Sort by time to use recent events only
        sort_idx = np.argsort(t)
        x = x[sort_idx]
        y = y[sort_idx]
        t = t[sort_idx]
        p = p[sort_idx]
        
        # Use only recent events within time window
        t_max = t[-1]
        mask = t >= (t_max - self.time_window_us)
        x = x[mask]
        y = y[mask]
        t = t[mask]
        p = p[mask]
        
        if len(t) < 5:
            return None
        
        # Normalize time to seconds
        t_sec = (t - t[0]) / 1e6  # Convert microseconds to seconds
        
        # Build design matrix for plane fitting: t = ax + by + c
        # A @ [a, b, c]^T = t
        A = np.column_stack([x, y, np.ones(len(x))])
        b = t_sec
        
        try:
            # Solve least squares: min ||A @ params - t||^2
            params, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            
            if rank < 3:  # Degenerate case
                return None
            
            # Extract velocity components (a, b) = (dt/dx, dt/dy)
            # But we want (dx/dt, dy/dt), so invert
            dt_dx, dt_dy = params[0], params[1]
            
            # Velocity in pixels/second
            # If dt/dx is small, motion in x is large
            # v_x = 1 / dt_dx (approximately, for small motion)
            
            # More robust: estimate speed from spatial spread over time
            spatial_extent = np.sqrt(np.var(x) + np.var(y))
            temporal_extent = t_sec[-1] - t_sec[0]
            
            if temporal_extent < 1e-6:  # Avoid division by zero
                return None
            
            # Speed = spatial displacement / time
            speed = spatial_extent / temporal_extent  # pixels per second
            
            # Alternative: use gradient magnitudes
            gradient_mag = np.sqrt(dt_dx**2 + dt_dy**2)
            if gradient_mag > 1e-6:
                speed_from_gradient = 1.0 / gradient_mag
                # Use geometric mean of both estimates
                speed = np.sqrt(speed * speed_from_gradient)
            
            return float(speed)
            
        except np.linalg.LinAlgError:
            return None
    
    def get_detailed_metrics(self, events: dict) -> dict:
        """
        Get detailed motion analysis (for debugging/visualization).
        
        Returns:
            Dictionary with motion statistics
        """
        if len(events['t']) < 10:
            return {
                'motion_score': 0.0,
                'num_active_grids': 0,
                'mean_motion': 0.0,
                'max_motion': 0.0
            }
        
        x = events['x'] if isinstance(events['x'], np.ndarray) else events['x'].numpy()
        y = events['y'] if isinstance(events['y'], np.ndarray) else events['y'].numpy()
        t = events['t'] if isinstance(events['t'], np.ndarray) else events['t'].numpy()
        p = events['p'] if isinstance(events['p'], np.ndarray) else events['p'].numpy()
        
        if len(p.shape) > 1:
            p = p.flatten()
        
        grid_motions = self._compute_grid_motions(x, y, t, p)
        
        if len(grid_motions) == 0:
            return {
                'motion_score': 0.0,
                'num_active_grids': 0,
                'mean_motion': 0.0,
                'max_motion': 0.0
            }
        
        motion_array = np.array(grid_motions)
        motion_score = np.percentile(motion_array, 75)
        normalized = motion_score / self.speed_norm
        
        return {
            'motion_score': float(np.clip(normalized, 0.0, 1.0)),
            'num_active_grids': len(grid_motions),
            'mean_motion': float(motion_array.mean()),
            'max_motion': float(motion_array.max()),
            'median_motion': float(np.median(motion_array)),
            'p75_motion': float(motion_score)
        }