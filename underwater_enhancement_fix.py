import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import random
import os
import json
from collections import deque
from detectron2.data.transforms import Augmentation
from fvcore.transforms.transform import NoOpTransform, Transform

class ParameterScheduler:
    """参数自动调整调度器，根据损失历史动态调整参数 - 优化版本"""
    
    def __init__(
        self,
        initial_params=None,
        history_size=5,  # 减少历史大小
        adjust_rate=0.01,
        min_adjust_threshold=0.01,  # 提高调整阈值
        param_bounds=None,
        adjust_interval=50  # 每50步调整一次参数
    ):
        """
        初始化参数调度器
        
        Args:
            initial_params: 初始参数字典，包含depth_mean, depth_std, light_mean, light_std等
            history_size: 保存的历史损失数量
            adjust_rate: 参数调整率
            min_adjust_threshold: 最小调整阈值，损失变化低于此值时不调整
            param_bounds: 参数边界字典，格式为{param_name: (min_val, max_val)}
            adjust_interval: 参数调整间隔步数
        """
        # 默认初始参数
        self.params = {
            'depth_mean': 0.5,
            'depth_std': 0.1,
            'light_mean': 0.6,
            'light_std': 0.1,
            'trans_factor': 0.3,
            'depth_factor': 0.05,
            'light_factor': 0.05,
            'brightness': 0.8
        } if initial_params is None else initial_params
        
        # 默认参数边界
        self.param_bounds = {
            'depth_mean': (0.3, 0.7),
            'depth_std': (0.05, 0.2),
            'light_mean': (0.4, 0.8),
            'light_std': (0.05, 0.2),
            'trans_factor': (0.1, 0.5),
            'depth_factor': (0.01, 0.2),
            'light_factor': (0.01, 0.2),
            'brightness': (0.6, 1.0)
        } if param_bounds is None else param_bounds
        
        self.history_size = history_size
        self.adjust_rate = adjust_rate
        self.min_adjust_threshold = min_adjust_threshold
        self.adjust_interval = adjust_interval
        
        # 损失历史记录 - 使用列表替代deque，只在需要时修剪
        self.loss_history = []
        self.param_history = []  # 只在参数变化时记录
        
        # 最佳参数和对应的损失
        self.best_loss = float('inf')
        self.best_params = self.params.copy()
        
        # 参数调整方向 - 简化动量机制
        self.adjust_direction = {param: 0 for param in self.params}
        
        # 训练步数计数
        self.steps = 0
    
    def update_loss(self, loss):
        """
        更新损失历史并根据需要调整参数
        
        Args:
            loss: 当前损失值
            
        Returns:
            bool: 是否调整了参数
        """
        self.loss_history.append(loss)
        if len(self.loss_history) > self.history_size:
            self.loss_history.pop(0)  # 保持历史大小不变
            
        self.steps += 1
        
        # 记录最佳参数
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_params = self.params.copy()
            
        # 降低参数调整频率，从每10步调整一次改为每50步调整一次
        if self.steps % self.adjust_interval != 0:
            return False
            
        # 计算损失趋势 - 简化计算
        loss_trend = self._calculate_loss_trend_simple()
        
        # 损失变化太小，不调整
        if abs(loss_trend) < self.min_adjust_threshold:
            return False
            
        # 根据损失趋势调整参数 - 使用简化的调整策略
        self._adjust_parameters_simple(loss_trend)
        
        # 记录参数历史
        self.param_history.append(self.params.copy())
        
        return True
    
    def _calculate_loss_trend_simple(self):
        """
        计算损失趋势的简化版本
        
        Returns:
            float: 损失趋势值
        """
        if len(self.loss_history) < 2:
            return 0
            
        # 只比较最近和最早的损失
        recent_loss = self.loss_history[-1]
        earliest_loss = self.loss_history[0]
        
        # 返回损失趋势（正值表示损失增加）
        return (recent_loss - earliest_loss) / max(earliest_loss, 1e-6)
    
    def _adjust_parameters_simple(self, loss_trend):
        """
        简化的参数调整策略
        
        Args:
            loss_trend: 损失趋势值
        """
        # 计算基础调整量
        base_adjust = self.adjust_rate * (1.0 + abs(loss_trend) * 2)
        
        # 只调整部分参数，而不是所有参数
        params_to_adjust = list(self.params.keys())
        if len(params_to_adjust) > 2:  # 如果参数超过2个，随机选择2个进行调整
            params_to_adjust = random.sample(params_to_adjust, 2)
            
        for param in params_to_adjust:
            value = self.params[param]
            
            # 根据损失趋势决定调整方向
            if loss_trend > 0:  # 损失增加，反转方向
                direction = -1 if self.adjust_direction[param] > 0 else 1
            else:  # 损失减少，保持方向
                direction = self.adjust_direction[param] if self.adjust_direction[param] != 0 else random.choice([-1, 1])
            
            # 应用调整
            new_value = value + base_adjust * direction
            
            # 确保参数在有效范围内
            if param in self.param_bounds:
                min_val, max_val = self.param_bounds[param]
                new_value = max(min_val, min(max_val, new_value))
            
            # 更新参数值和调整方向
            self.params[param] = new_value
            self.adjust_direction[param] = direction
    
    def get_params(self):
        """获取当前参数"""
        return self.params
    
    def get_best_params(self):
        """获取历史最佳参数"""
        return self.best_params
    
    def save_params(self, file_path):
        """保存参数到文件 - 简化版本，只保存最关键的信息"""
        save_data = {
            'best_params': self.best_params,
            'best_loss': float(self.best_loss),
            'steps': self.steps
        }
        
        with open(file_path, 'w') as f:
            json.dump(save_data, f, indent=2)
    
    def load_params(self, file_path):
        """从文件加载参数"""
        if not os.path.exists(file_path):
            return False
            
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            # 只加载最重要的参数
            self.best_params = data['best_params']
            self.best_loss = data['best_loss']
            self.steps = data.get('steps', 0)
            
            # 使用最佳参数作为当前参数
            self.params = self.best_params.copy()
            return True
        except Exception as e:
            print(f"加载参数失败: {e}")
            return False

class DynamicTransmissionEstimator(nn.Module):
    """动态传输图估计模块 - 简化版本"""
    def __init__(self, in_channels=3, hidden_channels=32):  # 减少通道数从64到32
        super(DynamicTransmissionEstimator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1)  # 移除一层卷积
        
        # 环境参数估计网络 - 简化版本
        self.env_conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.env_conv2 = nn.Conv2d(hidden_channels, 2, kernel_size=3, padding=1)  # 移除一层卷积
        
        # 自适应权重网络
        self.weight_conv = nn.Conv2d(2, 1, kernel_size=3, padding=1)  # 只使用一层卷积
        
        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        # 基础传输图估计 - 简化流程
        base_trans = F.relu(self.conv1(x))
        base_trans = torch.sigmoid(self.conv2(base_trans))
        
        # 环境参数估计 - 简化流程
        env_params = F.relu(self.env_conv1(x))
        env_params = torch.sigmoid(self.env_conv2(env_params))
        
        # 计算自适应权重 - 简化流程
        weights = torch.sigmoid(self.weight_conv(env_params))
        
        # 动态调整传输图
        dynamic_trans = base_trans * weights
        
        return dynamic_trans, env_params

class AdaptiveEnhancement(nn.Module):
    """自适应图像增强模块 - 带参数自动调整功能"""
    def __init__(self, param_scheduler=None, params_path=None):
        super(AdaptiveEnhancement, self).__init__()
        self.trans_estimator = DynamicTransmissionEstimator()
    
        # 创建或加载参数调度器
        if param_scheduler is None:
            self.param_scheduler = ParameterScheduler()
            # 尝试从文件加载参数
            if params_path is not None and os.path.exists(params_path):
                self.param_scheduler.load_params(params_path)
        else:
            self.param_scheduler = param_scheduler
        
        # 从调度器获取当前参数
        params = self.param_scheduler.get_params()
        
        # 设置理想参数目标值
        self.ideal_depth_mean = params['depth_mean']
        self.ideal_depth_std = params['depth_std']
        self.ideal_light_mean = params['light_mean']
        self.ideal_light_std = params['light_std']
        
        # 设置增强因子
        self.trans_factor = params['trans_factor']
        self.depth_factor = params['depth_factor']
        self.light_factor = params['light_factor']
        self.brightness = params['brightness']
        
        # 损失缩放因子，使损失保持在与检测损失相同的数量级
        self.loss_scale = 0.1
        
        # 参数自动保存路径
        self.params_path = params_path
        
        # 是否处于训练模式
        self.training_mode = True
    
    def update_parameters(self):
        """从参数调度器更新参数"""
        params = self.param_scheduler.get_params()
        
        self.ideal_depth_mean = params['depth_mean']
        self.ideal_depth_std = params['depth_std']
        self.ideal_light_mean = params['light_mean']
        self.ideal_light_std = params['light_std']
        
        self.trans_factor = params['trans_factor']
        self.depth_factor = params['depth_factor']
        self.light_factor = params['light_factor']
        self.brightness = params['brightness']
    
    def train(self, mode=True):
        """设置训练模式"""
        super().train(mode)
        self.training_mode = mode
        return self
    
    def forward(self, x, return_loss=False):
        # 估计动态传输图
        trans_map, env_params = self.trans_estimator(x)
        
        # 基于传输图的图像增强 - 使用动态参数
        # 增加底数防止除以接近0的值导致过亮
        enhanced = x / (trans_map * self.trans_factor + (1 - self.trans_factor))
        
        # 应用环境自适应调整 - 使用动态参数
        depth_factor = env_params[:, 0:1, :, :] * self.depth_factor
        light_factor = env_params[:, 1:2, :, :] * self.light_factor
        
        # 根据环境参数调整增强结果
        enhanced = enhanced * (1 + depth_factor * 0.1) * (1 + light_factor * 0.1)
        
        # 整体亮度控制 - 使用动态参数
        enhanced = enhanced * self.brightness
        
        # 确保值在合理范围内
        enhanced = torch.clamp(enhanced, 0, 1)
        
        if return_loss:
            # 计算环境参数损失
            loss = self.calculate_enhancement_loss(env_params, trans_map)
            
            # 如果处于训练模式，更新参数
            if self.training_mode:
                # 更新参数调度器
                params_updated = self.param_scheduler.update_loss(loss.item())
                
                # 如果参数已更新，则更新模型参数
                if params_updated:
                    self.update_parameters()
                    
                    # 每100步保存一次参数
                    if self.params_path is not None and self.param_scheduler.steps % 100 == 0:
                        self.param_scheduler.save_params(self.params_path)
            
            return enhanced, loss
        
        return enhanced
    
    def calculate_enhancement_loss(self, env_params, trans_map):
        """
        计算环境参数的损失，使其向理想值靠拢
        
        Args:
            env_params: 环境参数张量，形状为[B, 2, H, W]
            trans_map: 传输图张量，形状为[B, 1, H, W]
            
        Returns:
            loss: 环境参数损失
        """
        # 提取深度和光照参数
        depth_params = env_params[:, 0:1, :, :]  # [B, 1, H, W]
        light_params = env_params[:, 1:2, :, :]  # [B, 1, H, W]
        
        # 计算参数统计特性
        depth_mean = depth_params.mean()
        depth_std = depth_params.std()
        light_mean = light_params.mean()
        light_std = light_params.std()
        
        # 计算与理想值的差异损失
        depth_mean_loss = F.mse_loss(depth_mean, torch.tensor(self.ideal_depth_mean, device=depth_mean.device))
        depth_std_loss = F.mse_loss(depth_std, torch.tensor(self.ideal_depth_std, device=depth_std.device))
        light_mean_loss = F.mse_loss(light_mean, torch.tensor(self.ideal_light_mean, device=light_mean.device))
        light_std_loss = F.mse_loss(light_std, torch.tensor(self.ideal_light_std, device=light_std.device))
        
        # 传输图平滑度损失 - 鼓励传输图的平滑性
        trans_dx = trans_map[:, :, :, 1:] - trans_map[:, :, :, :-1]
        trans_dy = trans_map[:, :, 1:, :] - trans_map[:, :, :-1, :]
        trans_smooth_loss = torch.mean(torch.abs(trans_dx)) + torch.mean(torch.abs(trans_dy))
        
        # 组合损失
        total_loss = (depth_mean_loss + depth_std_loss + light_mean_loss + light_std_loss + trans_smooth_loss)
        
        # 缩放损失使其与检测损失在同一数量级
        return total_loss * self.loss_scale
    
    def get_current_params(self):
        """获取当前参数"""
        return self.param_scheduler.get_params()
    
    def get_best_params(self):
        """获取最佳参数"""
        return self.param_scheduler.get_best_params()
    
    def save_params(self, path=None):
        """保存参数到文件"""
        save_path = path or self.params_path
        if save_path is not None:
            self.param_scheduler.save_params(save_path)
            return True
        return False

class UnderwaterEnhancementTransform(Transform):
    """将水下图像增强应用到图像的Transform类"""
    
    def __init__(self, params_path=None):
        """
        初始化水下图像增强变换
        
        Args:
            params_path: 参数文件路径，如果提供则从文件加载参数
        """
        super().__init__()
        self.model = AdaptiveEnhancement(params_path=params_path)
        # 使用CPU处理
        self.device = torch.device("cpu")
        self.model = self.model.to(self.device)
        # 确保模型处于评估模式
        self.model.eval()
    
    def apply_image(self, img):
        """
        应用水下增强到图像
        Args:
            img: numpy array, 格式为HWC, RGB, 范围为0-255的uint8
        Returns:
            numpy array: 增强后的图像，相同格式
        """
        # 转换为PyTorch张量
        img_tensor = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
        img_tensor = img_tensor / 255.0  # 归一化到0-1
        img_tensor = img_tensor.to(self.device)
        
        # 应用增强
        with torch.no_grad():
            enhanced = self.model(img_tensor)
        
        # 转换回numpy格式
        enhanced = enhanced.squeeze(0).permute(1, 2, 0).cpu().numpy()
        enhanced = (enhanced * 255.0).clip(0, 255).astype(np.uint8)
        
        return enhanced
    
    def apply_coords(self, coords):
        """对坐标不做任何改变"""
        return coords

class UnderwaterEnhancement(Augmentation):
    """水下图像增强的增强类"""
    
    def __init__(self, prob=0.5, params_path=None):
        """
        Args:
            prob: 应用此增强的概率
            params_path: 参数文件路径
        """
        super().__init__()
        self.prob = prob
        self.params_path = params_path
    
    def get_transform(self, image):
        """
        Args:
            image: numpy array, 格式为HWC
        Returns:
            Transform 或 NoOpTransform
        """
        if random.random() < self.prob:
            return UnderwaterEnhancementTransform(params_path=self.params_path)
        return NoOpTransform()

# 用于训练过程中的水下增强模块
class UnderwaterEnhancementModule(nn.Module):
    """用于训练过程中的水下增强模块，返回增强后的图像和损失"""
    
    def __init__(self, loss_weight=0.3, params_path=None):
        """
        初始化水下增强模块
        
        Args:
            loss_weight: 水下增强损失在总损失中的权重，默认为0.3（30%）
            params_path: 参数文件路径，如果提供则从文件加载参数
        """
        super(UnderwaterEnhancementModule, self).__init__()
        self.enhancement_model = AdaptiveEnhancement(params_path=params_path)
        self.loss_weight = loss_weight
        self.params_path = params_path
    
    def forward(self, images):
        """
        前向传播，增强图像并计算损失
        
        Args:
            images: 输入图像张量，形状为[B, C, H, W]，范围为0-1
            
        Returns:
            enhanced_images: 增强后的图像张量
            weighted_loss: 加权后的增强损失
        """
        enhanced_images, enhancement_loss = self.enhancement_model(images, return_loss=True)
        
        # 应用损失权重
        weighted_loss = enhancement_loss * self.loss_weight
        
        return enhanced_images, weighted_loss
    
    def get_current_params(self):
        """获取当前参数"""
        return self.enhancement_model.get_current_params()
    
    def get_best_params(self):
        """获取最佳参数"""
        return self.enhancement_model.get_best_params()
    
    def save_params(self, path=None):
        """保存参数到文件"""
        return self.enhancement_model.save_params(path or self.params_path)

def build_underwater_enhancement_module(cfg):
    """
    构建水下图像增强模块和损失函数
    
    Args:
        cfg: 配置对象或字典
        
    Returns:
        module: 水下图像增强模块
        loss_fn: 水下图像增强损失函数
    """
    # 确定参数文件路径
    params_path = None
    if hasattr(cfg, 'OUTPUT_DIR'):
        params_path = os.path.join(cfg.OUTPUT_DIR, 'underwater_params.json')
    elif isinstance(cfg, dict) and 'OUTPUT_DIR' in cfg:
        params_path = os.path.join(cfg['OUTPUT_DIR'], 'underwater_params.json')
    
    # 确定损失权重
    loss_weight = 0.3  # 默认值
    if hasattr(cfg, 'UNDERWATER_ENHANCEMENT_WEIGHT'):
        loss_weight = cfg.UNDERWATER_ENHANCEMENT_WEIGHT
    elif isinstance(cfg, dict) and 'UNDERWATER_ENHANCEMENT_WEIGHT' in cfg:
        loss_weight = cfg['UNDERWATER_ENHANCEMENT_WEIGHT']
    
    # 创建增强模块
    enhancement_module = UnderwaterEnhancementModule(
        loss_weight=loss_weight,
        params_path=params_path
    )
    
    # 损失函数是模块的一部分，直接返回模块本身
    return enhancement_module, enhancement_module.forward
