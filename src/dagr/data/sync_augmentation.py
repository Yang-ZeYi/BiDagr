"""
Synchronization Error Augmentation for TAT-DAGR
时间同步误差数据增强工具
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional
from torch_geometric.data import Data


class SyncErrorAugmentation:
    """
    时间同步误差数据增强
    模拟传感器之间的时间同步误差
    """
    
    def __init__(
        self,
        max_offset: float = 20.0,
        offset_distribution: str = 'uniform',
        augment_prob: float = 0.8,
        compensate_labels: bool = True
    ):
        """
        Args:
            max_offset: 最大偏移量（毫秒）
            offset_distribution: 偏移分布类型 ('uniform', 'gaussian')
            augment_prob: 应用增强的概率
            compensate_labels: 是否根据偏移补偿标签位置
        """
        self.max_offset = max_offset * 1000.0  # 转换为微秒
        self.offset_distribution = offset_distribution
        self.augment_prob = augment_prob
        self.compensate_labels = compensate_labels
    
    def __call__(self, data: Data) -> Tuple[Data, float]:
        """
        对数据应用同步误差增强
        
        Args:
            data: torch_geometric.data.Data对象
                包含: pos (事件坐标), x (事件特征), bbox (标签)
        
        Returns:
            augmented_data: 增强后的数据
            true_offset: 真实偏移量（微秒）
        """
        # 按概率决定是否应用增强
        if np.random.rand() > self.augment_prob:
            return data, 0.0
        
        # 生成随机偏移量
        true_offset = self._generate_offset()
        
        # 复制数据（避免修改原始数据）
        augmented_data = data.clone()
        
        # 修改事件时间戳
        if hasattr(augmented_data, 'pos') and augmented_data.pos is not None:
            # pos可能包含 [x, y] 或 [x, y, t]
            if augmented_data.pos.shape[1] >= 3:
                augmented_data.pos[:, 2] += true_offset
        
        # 如果事件特征中包含时间信息
        if hasattr(augmented_data, 'x') and augmented_data.x is not None:
            # 假设特征格式为 [t, p] 或其他包含时间的格式
            # 根据实际数据格式调整
            pass
        
        # 补偿标签位置（根据物体运动）
        if self.compensate_labels and hasattr(augmented_data, 'bbox'):
            augmented_data.bbox = self._adjust_labels_for_offset(
                augmented_data.bbox,
                true_offset,
                augmented_data
            )
        
        return augmented_data, true_offset
    
    def _generate_offset(self) -> float:
        """
        生成随机时间偏移
        
        Returns:
            offset: 时间偏移（微秒）
        """
        if self.offset_distribution == 'uniform':
            # 均匀分布
            offset = np.random.uniform(-self.max_offset, self.max_offset)
        
        elif self.offset_distribution == 'gaussian':
            # 高斯分布（标准差为max_offset/3，使99.7%在范围内）
            offset = np.random.normal(0, self.max_offset / 3.0)
            offset = np.clip(offset, -self.max_offset, self.max_offset)
        
        else:
            raise ValueError(f"Unknown distribution: {self.offset_distribution}")
        
        return float(offset)
    
    def _adjust_labels_for_offset(
        self,
        bbox: torch.Tensor,
        offset: float,
        data: Data
    ) -> torch.Tensor:
        """
        根据时间偏移调整标签位置（运动补偿）
        
        Args:
            bbox: (N, 5) [x, y, w, h, class]
            offset: 时间偏移（微秒）
            data: 完整数据对象
        
        Returns:
            adjusted_bbox: 调整后的标签
        """
        if len(bbox) == 0:
            return bbox
        
        # 估计物体速度（简单方案：从bbox0到bbox的变化）
        if hasattr(data, 'bbox0') and data.bbox0 is not None and len(data.bbox0) > 0:
            # 计算位置变化
            if hasattr(data, 't0') and hasattr(data, 't1'):
                dt = data.t1 - data.t0  # 时间间隔（微秒）
                
                if dt > 0 and len(data.bbox0) == len(bbox):
                    # 匹配bbox和bbox0（假设顺序一致或通过track_id匹配）
                    dx = bbox[:, 0] - data.bbox0[:, 0]
                    dy = bbox[:, 1] - data.bbox0[:, 1]
                    
                    # 估计速度（像素/微秒）
                    vx = dx / dt
                    vy = dy / dt
                    
                    # 根据偏移调整位置
                    adjusted_bbox = bbox.clone()
                    adjusted_bbox[:, 0] += vx * offset
                    adjusted_bbox[:, 1] += vy * offset
                    
                    return adjusted_bbox
        
        # 如果无法估计速度，返回原始bbox
        return bbox
    
    def shift_event_timestamps(
        self,
        events: torch.Tensor,
        offset: float
    ) -> torch.Tensor:
        """
        时间戳平移（独立函数）
        
        Args:
            events: (N, 4) [x, y, t, p]
            offset: 时间偏移（微秒）
        
        Returns:
            shifted_events: 平移后的事件
        """
        shifted_events = events.clone()
        shifted_events[:, 2] += offset
        return shifted_events
    
    def visualize_augmentation(
        self,
        original_data: Data,
        augmented_data: Data,
        offset: float
    ) -> Dict:
        """
        可视化增强效果
        
        Args:
            original_data: 原始数据
            augmented_data: 增强后数据
            offset: 应用的偏移量
        
        Returns:
            vis_info: 可视化信息字典
        """
        vis_info = {
            'offset_ms': offset / 1000.0,
            'num_events': len(augmented_data.pos) if hasattr(augmented_data, 'pos') else 0,
            'num_labels': len(augmented_data.bbox) if hasattr(augmented_data, 'bbox') else 0,
        }
        
        # 比较事件时间戳范围
        if hasattr(original_data, 'pos') and original_data.pos.shape[1] >= 3:
            vis_info['original_time_range'] = (
                original_data.pos[:, 2].min().item(),
                original_data.pos[:, 2].max().item()
            )
            vis_info['augmented_time_range'] = (
                augmented_data.pos[:, 2].min().item(),
                augmented_data.pos[:, 2].max().item()
            )
        
        # 比较标签位置
        if hasattr(original_data, 'bbox') and hasattr(augmented_data, 'bbox'):
            if len(original_data.bbox) > 0 and len(augmented_data.bbox) > 0:
                pos_diff = (augmented_data.bbox[:, :2] - original_data.bbox[:, :2]).abs().mean()
                vis_info['avg_label_displacement'] = pos_diff.item()
        
        return vis_info


class AdaptiveSyncAugmentation(SyncErrorAugmentation):
    """
    自适应同步误差增强
    根据物体速度自适应调整增强强度
    """
    
    def __init__(
        self,
        max_offset: float = 20.0,
        offset_distribution: str = 'uniform',
        augment_prob: float = 0.8,
        velocity_adaptive: bool = True
    ):
        super().__init__(max_offset, offset_distribution, augment_prob)
        self.velocity_adaptive = velocity_adaptive
    
    def __call__(self, data: Data) -> Tuple[Data, float]:
        """
        自适应增强
        
        Args:
            data: 输入数据
        
        Returns:
            augmented_data: 增强后数据
            true_offset: 真实偏移
        """
        if np.random.rand() > self.augment_prob:
            return data, 0.0
        
        # 估计平均物体速度
        avg_velocity = self._estimate_average_velocity(data)
        
        # 根据速度调整偏移量
        if self.velocity_adaptive:
            # 快速物体 -> 更大偏移
            velocity_factor = min(2.0, 1.0 + avg_velocity / 50.0)  # 归一化速度
            adjusted_max_offset = self.max_offset * velocity_factor
        else:
            adjusted_max_offset = self.max_offset
        
        # 临时修改max_offset
        original_max = self.max_offset
        self.max_offset = adjusted_max_offset
        
        # 调用父类方法
        augmented_data, true_offset = super().__call__(data)
        
        # 恢复原始max_offset
        self.max_offset = original_max
        
        return augmented_data, true_offset
    
    def _estimate_average_velocity(self, data: Data) -> float:
        """
        估计平均物体速度
        
        Args:
            data: 数据对象
        
        Returns:
            avg_velocity: 平均速度（像素/秒）
        """
        if not hasattr(data, 'bbox') or not hasattr(data, 'bbox0'):
            return 0.0
        
        if len(data.bbox) == 0 or len(data.bbox0) == 0:
            return 0.0
        
        if not hasattr(data, 't0') or not hasattr(data, 't1'):
            return 0.0
        
        dt = data.t1 - data.t0  # 微秒
        if dt <= 0:
            return 0.0
        
        # 计算位置变化
        if len(data.bbox) == len(data.bbox0):
            dx = data.bbox[:, 0] - data.bbox0[:, 0]
            dy = data.bbox[:, 1] - data.bbox0[:, 1]
            
            # 速度（像素/微秒）
            velocities = torch.sqrt(dx**2 + dy**2) / dt
            
            # 转换为像素/秒
            avg_velocity = velocities.mean().item() * 1e6
            
            return avg_velocity
        
        return 0.0


def apply_sync_augmentation_to_batch(
    batch: Data,
    augmenter: SyncErrorAugmentation
) -> Tuple[Data, torch.Tensor]:
    """
    对batch数据应用同步误差增强
    
    Args:
        batch: 批量数据
        augmenter: 增强器
    
    Returns:
        augmented_batch: 增强后的批量数据
        true_offsets: 每个样本的真实偏移 (B,)
    """
    # 获取batch size
    if hasattr(batch, 'num_graphs'):
        batch_size = batch.num_graphs
    else:
        batch_size = 1
    
    # 对每个样本应用增强
    true_offsets = []
    
    # 注意：由于torch_geometric的Data是批量格式
    # 这里需要特殊处理
    # 简化版本：对整个batch应用相同偏移
    augmented_batch, offset = augmenter(batch)
    true_offsets = torch.tensor([offset] * batch_size)
    
    return augmented_batch, true_offsets