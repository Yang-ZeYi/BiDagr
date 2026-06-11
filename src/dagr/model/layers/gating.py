"""
Ultra-lightweight gating mechanism for GNN->CNN bidirectional feedback
Implements global and spatial-aware gating strategies
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class EventActivityGating(nn.Module):
    """
    Lightweight event activity gating (global version).

    Core idea:
    1. Extract global statistics from GNN node features
    2. Use a small MLP to generate per-channel gating weights
    3. Modulate CNN features (enhance or suppress certain channels)

    Computational complexity: ~2M FLOPS (negligible compared to the original DAGR's 0.015M/event)
    """

    def __init__(self,
                 gnn_channels=128,
                 cnn_channels=256,
                 hidden_dim=64,
                 gate_strength_init=0.1,
                 learnable_strength=True):
        """
        Args:
            gnn_channels: GNN node feature dimension
            cnn_channels: Number of CNN feature channels
            hidden_dim: MLP hidden layer dimension
            gate_strength_init: Initial gating strength
            learnable_strength: Whether the gating strength is learnable
        """
        super().__init__()

        self.gnn_channels = gnn_channels
        self.cnn_channels = cnn_channels

        # Project node features to a global representation
        self.node_encoder = nn.Sequential(
            nn.Linear(gnn_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),  # Light dropout to prevent overfitting
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Generate per-channel gating weights
        self.gate_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, cnn_channels),
            nn.Sigmoid()  # Output gating values in [0, 1]
        )

        # Learnable gating strength (controls modulation magnitude)
        if learnable_strength:
            self.gate_strength = nn.Parameter(torch.tensor(gate_strength_init))
        else:
            self.register_buffer('gate_strength', torch.tensor(gate_strength_init))

        # Statistics (for monitoring and debugging)
        self.register_buffer('avg_gate_value', torch.tensor(0.5))
        self.register_buffer('num_updates', torch.tensor(0))

    def forward(self, cnn_features, gnn_node_features, update=True):
        """
        Args:
            cnn_features: (B, C_cnn, H, W) - CNN feature map
            gnn_node_features: (N_nodes, C_gnn) - GNN node features
            update: bool - whether to actually apply gating

        Returns:
            modulated_features: (B, C_cnn, H, W) - gated CNN features
            gate_info: dict - gating information
        """
        # Handle edge cases
        if not update:
            return cnn_features, {'gate_applied': False, 'reason': 'disabled'}

        # Check input
        if gnn_node_features is None:
            return cnn_features, {'gate_applied': False, 'reason': 'no_input'}

        # If a Data object is passed, extract .x
        if hasattr(gnn_node_features, 'x'):
            gnn_node_features = gnn_node_features.x

        # Ensure it is a tensor
        if not torch.is_tensor(gnn_node_features):
            return cnn_features, {'gate_applied': False, 'reason': 'invalid_type'}

        # Check dimensions
        if gnn_node_features.dim() != 2:
            return cnn_features, {'gate_applied': False, 'reason': 'wrong_dim'}

        # Check if there are nodes
        if len(gnn_node_features) == 0:
            return cnn_features, {'gate_applied': False, 'reason': 'no_nodes'}

        # Check channel count
        if gnn_node_features.shape[1] != self.gnn_channels:
            return cnn_features, {'gate_applied': False, 'reason': 'channel_mismatch'}

        device = cnn_features.device
        B, C, H, W = cnn_features.shape

        # Ensure tensors are on the correct device
        if gnn_node_features.device != device:
            gnn_node_features = gnn_node_features.to(device)

        # 1. Global pooling of GNN node features
        global_feat = gnn_node_features.mean(dim=0, keepdim=True)  # (1, C_gnn)

        # 2. Expand to batch dimension
        global_feat = global_feat.expand(B, -1)  # (B, C_gnn)

        # 3. Encode to hidden space
        hidden = self.node_encoder(global_feat)  # (B, hidden_dim)

        # 4. Generate per-channel gating weights
        gate_weights = self.gate_generator(hidden)  # (B, C_cnn)

        # 5. Apply gating modulation
        gate_weights = gate_weights.view(B, self.cnn_channels, 1, 1)
        gate_modulation = 1.0 + self.gate_strength * (gate_weights - 0.5) * 2.0
        gate_modulation = torch.clamp(gate_modulation, min=0.5, max=1.5)

        modulated_features = cnn_features.detach() * gate_modulation

        # 6. Update statistics
        if self.training:
            with torch.no_grad():
                self.avg_gate_value = 0.99 * self.avg_gate_value + 0.01 * gate_weights.mean()
                self.num_updates += 1

        # 7. Return debugging info
        gate_info = {
            'gate_applied': True,
            'gate_strength': self.gate_strength.item(),
            'avg_gate_value': self.avg_gate_value.item(),
            'num_nodes': len(gnn_node_features),
            'gate_min': gate_weights.min().item(),
            'gate_max': gate_weights.max().item(),
            'gate_std': gate_weights.std().item()
        }

        return modulated_features, gate_info


class SpatialEventActivityGating(nn.Module):
    """
    Spatially-aware event activity gating (enhanced version).

    Compared to the global version, this generates a spatially varying gating map.
    Higher computational cost but greater expressiveness.

    Computational complexity: ~40M FLOPS
    """

    def __init__(self,
                 gnn_channels=128,
                 cnn_channels=256,
                 spatial_bins=(8, 8),  # Coarse grid resolution
                 hidden_dim=64,
                 gate_strength_init=0.1):
        """
        Args:
            spatial_bins: (bin_h, bin_w) spatial grid resolution
        """
        super().__init__()

        self.gnn_channels = gnn_channels
        self.cnn_channels = cnn_channels
        self.spatial_bins = spatial_bins
        self.num_bins = spatial_bins[0] * spatial_bins[1]

        # Encode node features for each spatial bin
        self.node_encoder = nn.Sequential(
            nn.Linear(gnn_channels, hidden_dim),
            nn.ReLU(inplace=True)
        )

        # Generate gating for each bin
        self.gate_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, cnn_channels),
            nn.Sigmoid()
        )

        self.gate_strength = nn.Parameter(torch.tensor(gate_strength_init))

        # Statistics
        self.register_buffer('avg_gate_value', torch.tensor(0.5))
        self.register_buffer('active_bins_ratio', torch.tensor(0.5))

    def forward(self, cnn_features, gnn_node_data, update=True):
        """
        Args:
            cnn_features: (B, C_cnn, H, W)
            gnn_node_data: Data object or dict containing:
                - pos: (N, 3) normalized positions [x, y, t] in [0, 1]
                - x: (N, C_gnn) node features
                - batch: (N,) batch indices
        """
        if not update or gnn_node_data is None:
            return cnn_features, {'gate_applied': False}

        B, C, H, W = cnn_features.shape
        device = cnn_features.device

        # Parse GNN node data
        if hasattr(gnn_node_data, 'pos'):
            positions = gnn_node_data.pos[:, :2]  # (N, 2) only x, y
            features = gnn_node_data.x  # (N, C_gnn)
        elif isinstance(gnn_node_data, dict):
            positions = gnn_node_data['pos']
            features = gnn_node_data['feat']
        else:
            # If only features are passed, fall back to global gating
            return EventActivityGating.forward(self, cnn_features, gnn_node_data, update)

        if len(positions) == 0:
            return cnn_features, {'gate_applied': False, 'reason': 'no_nodes'}

        # 1. Assign nodes to spatial bins
        bin_h, bin_w = self.spatial_bins

        # Positions are already normalized to [0, 1]
        bin_indices_h = (positions[:, 1] * bin_h).long().clamp(0, bin_h - 1)
        bin_indices_w = (positions[:, 0] * bin_w).long().clamp(0, bin_w - 1)
        bin_indices = bin_indices_h * bin_w + bin_indices_w  # (N,) flattened indices

        # 2. Pool node features within each bin
        bin_features = torch.zeros(self.num_bins, features.shape[1],
                                   device=device, dtype=features.dtype)
        bin_counts = torch.zeros(self.num_bins, device=device)

        # Use scatter_add for efficient aggregation
        bin_features.scatter_add_(0, bin_indices.unsqueeze(1).expand_as(features), features)
        bin_counts.scatter_add_(0, bin_indices, torch.ones_like(bin_indices, dtype=torch.float))

        # Normalize (avoid division by zero)
        bin_counts = bin_counts.clamp(min=1.0)
        bin_features = bin_features / bin_counts.unsqueeze(1)

        # 3. Generate gating for each bin
        encoded = self.node_encoder(bin_features)  # (num_bins, hidden_dim)
        gate_weights = self.gate_generator(encoded)  # (num_bins, C_cnn)

        # 4. Reshape into spatial gating map
        gate_map = gate_weights.view(bin_h, bin_w, C)  # (bin_h, bin_w, C_cnn)
        gate_map = gate_map.permute(2, 0, 1).unsqueeze(0)  # (1, C_cnn, bin_h, bin_w)

        # 5. Upsample to feature map resolution
        gate_map_upsampled = F.interpolate(
            gate_map,
            size=(H, W),
            mode='bilinear',
            align_corners=False
        )  # (1, C_cnn, H, W)

        # 6. Apply gating
        gate_modulation = 1.0 + self.gate_strength * (gate_map_upsampled - 0.5) * 2.0
        gate_modulation = torch.clamp(gate_modulation, min=0.5, max=1.5)

        modulated_features = cnn_features.detach() * gate_modulation

        # 7. Update statistics
        if self.training:
            with torch.no_grad():
                self.avg_gate_value = 0.99 * self.avg_gate_value + 0.01 * gate_map.mean()
                active_bins = (bin_counts > 1).sum().float() / self.num_bins
                self.active_bins_ratio = 0.99 * self.active_bins_ratio + 0.01 * active_bins

        gate_info = {
            'gate_applied': True,
            'gate_map': gate_map_upsampled.detach().cpu(),
            'gate_strength': self.gate_strength.item(),
            'active_bins': (bin_counts > 1).sum().item(),
            'active_bins_ratio': self.active_bins_ratio.item(),
            'avg_gate_value': self.avg_gate_value.item(),
            'num_nodes': len(positions)
        }

        return modulated_features, gate_info


def create_gating_module(args):
    """
    Factory function: create a gating module based on configuration.
    """
    gate_type = getattr(args, 'gate_type', 'global')
    gnn_channels = int(args.net_stem_width * 128)  # Compute based on configuration

    if gate_type == 'global':
        return EventActivityGating(
            gnn_channels=gnn_channels,
            cnn_channels=256,  # Adjust based on actual CNN output
            hidden_dim=getattr(args, 'gate_hidden_dim', 64),
            gate_strength_init=getattr(args, 'gate_strength_init', 0.1),
            learnable_strength=True
        )
    elif gate_type == 'spatial':
        return SpatialEventActivityGating(
            gnn_channels=gnn_channels,
            cnn_channels=256,
            spatial_bins=getattr(args, 'gate_spatial_bins', (8, 8)),
            hidden_dim=getattr(args, 'gate_hidden_dim', 64),
            gate_strength_init=getattr(args, 'gate_strength_init', 0.1)
        )
    else:
        raise ValueError(f"Unknown gate_type: {gate_type}")
