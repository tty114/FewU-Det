import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import sys
from typing import Tuple
import math
import numbers
from detectron2.data.transforms import Augmentation
from fvcore.transforms.transform import NoOpTransform, Transform
from custom_transforms import Custom_Erase, Shear

__all__ = [
    "UnderwaterEnhancement",
	"BBox_Erase",
	"RandomShear",
]

# 设置随机种子
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)
torch.backends.cudnn.benchmark = False

class RandomShear(Augmentation):
    def __init__(self, shear_factor=0.2):
        super().__init__()
        assert -1 <= shear_factor <= 1
        self.shear_factor = random.uniform(-shear_factor, shear_factor)

    def get_transform(self, image):
        img_center = np.array(image.shape[:2])[::-1]/2
        img_center = np.hstack((img_center, img_center))
        M = np.array([[1, abs(self.shear_factor), 0],[0,1,0]])
        nW =  image.shape[1] + abs(self.shear_factor*image.shape[0])
        return Shear(shear_factor=self.shear_factor, img_center=img_center, M=M, nW=nW, w=image.shape[1])


class BBox_Erase(Augmentation):
    """
    This method returns a copy of this image, with erasing applied to a random bounding box.
    """

    def __init__(self, scale=(0.02, 0.33), ratio=(0.3, 3.3)):
        """
        Args:
            n_max : Number of bounding boxes to apply the transform to.
        """
        super().__init__()
        # self.n_max = n_max
        assert isinstance(scale, tuple) and isinstance(ratio, tuple)
        # TODO:Put a check on values
        self.scale = scale
        self.ratio = ratio
        self.value = 0
        self.inplace = False
        # self._init(locals())

    def get_params(self, img, bbox):
        """Get parameters for ``erase`` for a random erasing.
        Args:
            img (Tensor): Tensor image of size (C, H, W) to be erased.
            scale: range of proportion of erased area against input image.
            ratio: range of aspect ratio of erased area.
        Returns:
            tuple: params (i, j, h, w, v) to be passed to ``erase`` for random erasing.
        """
        img_c, img_h, img_w = img.shape[0], int(bbox[3]-bbox[1]), int(bbox[2]-bbox[0])
        area = img_h * img_w

        for attempt in range(10):
            erase_area = random.uniform(self.scale[0], self.scale[1]) * area
            aspect_ratio = random.uniform(self.ratio[0], self.ratio[1])

            h = int(round(math.sqrt(erase_area * aspect_ratio)))
            w = int(round(math.sqrt(erase_area / aspect_ratio)))
            if h < img_h and w < img_w:
                i = random.randint(int(bbox[1]), int(bbox[1]) + img_h - h)
                j = random.randint(int(bbox[0]), int(bbox[0]) + img_w - w)
                if isinstance(self.value, numbers.Number):
                    v = self.value
                elif isinstance(self.value, torch._six.string_classes):
                    v = torch.empty([img_c, h, w], dtype=torch.float32).normal_()
                elif isinstance(self.value, (list, tuple)):
                    v = torch.tensor(self.value, dtype=torch.float32).view(-1, 1, 1).expand(-1, h, w)
                return i, j, h, w, v

        # Return original image
        return 0, 0, img_h, img_w, img

    def get_transform(self, image, boxes):
        if len(boxes) == 0:
            return NoOpTransform()
        ind = np.random.randint(0, len(boxes))
        x, y, h, w, v = self.get_params(image, boxes[ind])
        return Custom_Erase(x, y, h, w, v, self.inplace)

class DynamicTransmissionEstimator(nn.Module):
    """动态传输图估计模块"""
    def __init__(self, in_channels=3, hidden_channels=64):
        super(DynamicTransmissionEstimator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1)
        
        # 环境参数估计网络
        self.env_conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.env_conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.env_conv3 = nn.Conv2d(hidden_channels, 2, kernel_size=3, padding=1)  # 估计水深和光照参数
        
        # 自适应权重网络
        self.weight_conv1 = nn.Conv2d(2, hidden_channels, kernel_size=3, padding=1)
        self.weight_conv2 = nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1)
        
        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        # 基础传输图估计
        base_trans = F.relu(self.conv1(x))
        base_trans = F.relu(self.conv2(base_trans))
        base_trans = torch.sigmoid(self.conv3(base_trans))
        
        # 环境参数估计
        env_params = F.relu(self.env_conv1(x))
        env_params = F.relu(self.env_conv2(env_params))
        env_params = torch.sigmoid(self.env_conv3(env_params))
        
        # 计算自适应权重
        weights = F.relu(self.weight_conv1(env_params))
        weights = torch.sigmoid(self.weight_conv2(weights))
        
        # 动态调整传输图
        dynamic_trans = base_trans * weights
        
        return dynamic_trans, env_params

class AdaptiveEnhancement(nn.Module):
    """自适应图像增强模块"""
    def __init__(self):
        super(AdaptiveEnhancement, self).__init__()
        self.trans_estimator = DynamicTransmissionEstimator()
    
    def forward(self, x):
        # 估计动态传输图
        trans_map, env_params = self.trans_estimator(x)
        
        # 基于传输图的图像增强 - 大幅降低增强强度
        # 增加底数防止除以接近0的值导致过亮
        enhanced = x / (trans_map * 0.3 + 0.7)  # 降低传输图影响
        
        # 应用环境自适应调整 - 极大降低因子影响
        depth_factor = env_params[:, 0:1, :, :] * 0.2  # 水深因子，降低到原来的5%
        light_factor = env_params[:, 1:2, :, :] * 0.2 # 光照因子，降低到原来的5%
        
        # 根据环境参数调整增强结果 - 极大降低增强强度
        enhanced = enhanced * (1 + depth_factor * 0.1) * (1 + light_factor * 0.1)
        
        # 整体亮度控制 - 降低整体亮度
        enhanced = enhanced * 0.8  # 整体降低20%亮度
        
        # 确保值在合理范围内
        enhanced = torch.clamp(enhanced, 0, 1)
        
        return enhanced

class UnderwaterEnhancementTransform(Transform):
    """将水下图像增强应用到图像的Transform类"""
    
    def __init__(self):
        """
        初始化水下图像增强变换
        """
        super().__init__()
        self.model = AdaptiveEnhancement()
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
    
    def __init__(self, prob=0.5):
        """
        Args:
            prob: 应用此增强的概率
        """
        super().__init__()
        self.prob = prob
    
    def get_transform(self, image):
        """
        Args:
            image: numpy array, 格式为HWC
        Returns:
            Transform 或 NoOpTransform
        """
        if random.random() < self.prob:
            return UnderwaterEnhancementTransform()
        return NoOpTransform()