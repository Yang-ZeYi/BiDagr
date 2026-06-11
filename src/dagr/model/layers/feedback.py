"""
Coarse-grained GNN-to-CNN feedback mechanism
Implements feature-level bidirectional information flow from GNN to CNN
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data


class CoarseGNNToCNNFeedback(nn.Module):
    """
    Generate coarse-grained dense feedback from sparse GNN nodes to CNN feature maps.

    Core design:
    1. Aggregate sparse GNN nodes into a coarse grid (to avoid computational overhead of direct densification)
    2. Use mean pooling within each grid cell to aggregate node features
    3. Upsample the coarse grid to CNN feature map resolution via a lightweight network
    4. Generate an additive modulation signal to enhance/suppress CNN features

    Key characteristics:
    - Coarse granularity: uses a 16x12 grid (vs. 480x640 original image) to reduce computation
    - Additive feedback: preserves original CNN features, only adds a modulation signal
    - Learnable strength: controls feedback magnitude via the feedback_strength parameter
    """

    def __init__(self,
                 gnn_channels=128,           # GNN node feature dimension
                 cnn_channels=256,           # Number of CNN feature channels
                 grid_size=(16, 12),         # Coarse grid size (W, H)
                 hidden_dim=128,             # Hidden layer dimension
                 feedback_strength=0.1):     # Initial feedback strength
        """
        Args:
            gnn_channels: Number of channels in GNN node features
            cnn_channels: Number of channels in CNN feature maps
            grid_size: Coarse grid resolution (grid_w, grid_h)
            hidden_dim: MLP hidden layer dimension
            feedback_strength: Initial strength of the feedback signal (learnable)
        """
        super().__init__()

        self.gnn_channels = gnn_channels
        self.cnn_channels = cnn_channels
        self.grid_size = grid_size
        self.num_cells = grid_size[0] * grid_size[1]

        # ========== Module 1: Cell Feature Encoder ==========
        # Encode aggregated GNN features into the hidden space
        self.cell_encoder = nn.Sequential(
            nn.Linear(gnn_channels, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )

        # ========== Module 2: Feedback Feature Generator ==========
        # Generate feedback signal from encoded features
        self.feedback_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, cnn_channels),
            nn.Tanh()  # Constrain output range to [-1, 1]
        )

        # ========== Module 3: Upsampling Network ==========
        # Upsample the coarse grid to CNN feature map resolution
        # Uses transposed convolution to progressively increase resolution

        # Stage 1: grid_size -> grid_size*2
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(
                cnn_channels, cnn_channels // 2,
                kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(cnn_channels // 2),
            nn.ReLU(inplace=True)
        )

        # Stage 2: grid_size*2 -> grid_size*4 (approximate target resolution)
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(
                cnn_channels // 2, cnn_channels // 4,
                kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(cnn_channels // 4),
            nn.ReLU(inplace=True)
        )

        # Stage 3: Fine-tune to target channel count
        self.upsample3 = nn.Sequential(
            nn.Conv2d(
                cnn_channels // 4, cnn_channels,
                kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(cnn_channels),
            nn.Tanh()  # Constrain output range
        )

        # ========== Learnable Parameters ==========
        # Feedback strength: controls the global magnitude of the feedback signal
        self.feedback_strength = nn.Parameter(torch.tensor(feedback_strength))

        # ========== Statistics (for monitoring and debugging) ==========
        self.register_buffer('avg_feedback_norm', torch.tensor(0.0))
        self.register_buffer('active_cells_ratio', torch.tensor(0.0))
        self.register_buffer('num_forward_calls', torch.tensor(0))

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def aggregate_nodes_to_grid(self, gnn_data, batch_idx):
        """
        Aggregate sparse GNN nodes from a single batch sample into a coarse grid.

        Args:
            gnn_data: Data object
            batch_idx: int, index of the current batch item

        Returns:
            grid_features: (num_cells, C_gnn)
            cell_counts: (num_cells,)
        """
        device = gnn_data.x.device

        # Select nodes belonging to the current batch item
        if gnn_data.batch is not None:
            mask = (gnn_data.batch == batch_idx)
            positions = gnn_data.pos[mask, :2]
            features = gnn_data.x[mask]
        else:
            positions = gnn_data.pos[:, :2]
            features = gnn_data.x

        if len(positions) == 0:
            return torch.zeros(self.num_cells, self.gnn_channels, device=device), \
                   torch.zeros(self.num_cells, device=device)

        grid_w, grid_h = self.grid_size
        cell_x = (positions[:, 0] * grid_w).long().clamp(0, grid_w - 1)
        cell_y = (positions[:, 1] * grid_h).long().clamp(0, grid_h - 1)
        cell_indices = cell_y * grid_w + cell_x

        grid_features = torch.zeros(self.num_cells, features.shape[1], device=device, dtype=features.dtype)
        cell_counts = torch.zeros(self.num_cells, device=device, dtype=torch.float)

        grid_features.scatter_add_(0, cell_indices.unsqueeze(1).expand(-1, features.shape[1]), features)
        cell_counts.scatter_add_(0, cell_indices, torch.ones(len(cell_indices), device=device, dtype=torch.float))

        cell_counts_safe = cell_counts.clamp(min=1.0)
        grid_features = grid_features / cell_counts_safe.unsqueeze(1)

        return grid_features, cell_counts

    def forward(self, gnn_data, cnn_features, update=True):
        """
        Generate feedback signal from GNN to CNN.

        Args:
            gnn_data: Data object, GNN nodes after pool3
                - x: (N, C_gnn) node features
                - pos: (N, 3) node positions
            cnn_features: (B, C_cnn, H, W) - image_feat[3]
            update: bool - whether to actually apply feedback (for ablation studies)

        Returns:
            feedback: (B, C_cnn, H, W) - additive feedback signal
            info: dict - debugging and monitoring information
        """
        # ========== Edge case handling ==========
        if not update or gnn_data is None or len(gnn_data.x) == 0:
            return torch.zeros_like(cnn_features), {
                'feedback_applied': False,
                'reason': 'disabled' if not update else 'no_nodes'
            }

        B, C, H, W = cnn_features.shape
        device = cnn_features.device
        grid_w, grid_h = self.grid_size

        per_batch_feedbacks = []
        total_cell_counts = torch.zeros(self.num_cells, device=device)

        for b in range(B):
            grid_features, cell_counts = self.aggregate_nodes_to_grid(gnn_data, b)
            total_cell_counts = total_cell_counts + cell_counts

            encoded = self.cell_encoder(grid_features)  # (num_cells, hidden_dim)
            feedback_features = self.feedback_generator(encoded)  # (num_cells, C_cnn)

            feedback_map = feedback_features.view(grid_h, grid_w, C)
            feedback_map = feedback_map.permute(2, 0, 1).unsqueeze(0)  # (1, C, H_grid, W_grid)

            x = self.upsample1(feedback_map)
            x = self.upsample2(x)
            feedback_upsampled = self.upsample3(x)  # (1, C, H_approx, W_approx)

            if feedback_upsampled.shape[2:] != (H, W):
                feedback_upsampled = F.interpolate(feedback_upsampled, size=(H, W), mode='bilinear', align_corners=False)

            per_batch_feedbacks.append(feedback_upsampled)

        feedback = self.feedback_strength * torch.cat(per_batch_feedbacks, dim=0)  # (B, C, H, W)

        avg_cell_counts = total_cell_counts / B

        if self.training:
            with torch.no_grad():
                feedback_norm = feedback.abs().mean()
                active_ratio = (avg_cell_counts > 0).float().mean()
                self.avg_feedback_norm = 0.99 * self.avg_feedback_norm + 0.01 * feedback_norm
                self.active_cells_ratio = 0.99 * self.active_cells_ratio + 0.01 * active_ratio
                self.num_forward_calls += 1

        info = {
            'feedback_applied': True,
            'feedback_strength': self.feedback_strength.item(),
            'avg_feedback_norm': self.avg_feedback_norm.item(),
            'current_feedback_norm': feedback.abs().mean().item(),
            'active_cells': (avg_cell_counts > 0).sum().item(),
            'total_cells': self.num_cells,
            'active_cells_ratio': self.active_cells_ratio.item(),
            'num_gnn_nodes': len(gnn_data.x),
            'grid_size': self.grid_size,
            'feedback_shape': tuple(feedback.shape),
            'num_forward_calls': self.num_forward_calls.item()
        }

        return feedback, info


def create_feedback_module(args, gnn_channels, cnn_channels, height, width):
    """
    Factory function: create a feedback module based on configuration.

    Args:
        args: Configuration object
        gnn_channels: Number of GNN feature channels
        cnn_channels: Number of CNN feature channels
        height: Image height
        width: Image width

    Returns:
        CoarseGNNToCNNFeedback instance
    """
    # Compute appropriate grid size
    # Target: approximately 1/30 to 1/50 of the original image resolution
    grid_w = max(8, width // 40)   # 640/40 = 16
    grid_h = max(8, height // 40)  # 480/40 = 12

    # Use configuration value if specified
    if hasattr(args, 'feedback_grid_size'):
        if isinstance(args.feedback_grid_size, tuple):
            grid_w, grid_h = args.feedback_grid_size
        elif isinstance(args.feedback_grid_size, str):
            grid_w, grid_h = map(int, args.feedback_grid_size.split('x'))

    feedback_module = CoarseGNNToCNNFeedback(
        gnn_channels=gnn_channels,
        cnn_channels=cnn_channels,
        grid_size=(grid_w, grid_h),
        hidden_dim=getattr(args, 'feedback_hidden_dim', 128),
        feedback_strength=getattr(args, 'feedback_strength_init', 0.1)
    )

    return feedback_module
