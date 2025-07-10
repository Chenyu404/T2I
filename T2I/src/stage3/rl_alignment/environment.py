"""
文本-图像对齐强化学习环境
定义智能体与环境的交互接口
"""

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    # 备用方案：使用gym
    try:
        import gym
        from gym import spaces
    except ImportError:
        # 如果都没有，创建简单的替代
        class MockGym:
            class Env:
                def __init__(self):
                    pass
                def reset(self, seed=None, options=None):
                    return None, {}
                def step(self, action):
                    return None, 0, False, False, {}
                def render(self, mode='human'):
                    pass
                def close(self):
                    pass

        class MockSpaces:
            class Box:
                def __init__(self, low, high, shape=None, dtype=None):
                    self.low = low
                    self.high = high
                    self.shape = shape
                    self.dtype = dtype

        gym = MockGym()
        spaces = MockSpaces()
import numpy as np
import torch
from PIL import Image
from typing import Dict, Tuple, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class TextImageAlignmentEnv(gym.Env):
    """
    文本-图像对齐强化学习环境
    
    智能体的任务是识别和定位图像中的幻觉区域，
    并提供相应的纠正建议
    """
    
    def __init__(self, 
                 config: Dict,
                 hallucination_detector=None,
                 reward_function=None):
        """
        初始化环境
        
        Args:
            config: 环境配置
            hallucination_detector: 幻觉检测器
            reward_function: 奖励函数
        """
        super().__init__()
        
        self.config = config
        self.hallucination_detector = hallucination_detector
        self.reward_function = reward_function
        
        # 环境参数
        self.max_steps = config['environment']['max_steps']
        self.reward_threshold = config['environment']['reward_threshold']
        
        # 状态空间：图像特征 + 文本特征 + 当前检测状态
        # 简化为固定维度的特征向量
        self.image_feature_dim = 512
        self.text_feature_dim = 512
        self.detection_state_dim = 64
        
        state_dim = self.image_feature_dim + self.text_feature_dim + self.detection_state_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(state_dim,), dtype=np.float32
        )
        
        # 动作空间：定位坐标 + 幻觉类型 + 置信度
        # 动作包括：[x1, y1, x2, y2, hallucination_type, confidence]
        # 坐标归一化到[0,1]，幻觉类型为离散值，置信度为[0,1]
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0, 4.0, 1.0]),
            dtype=np.float32
        )
        
        # 幻觉类型映射
        self.hallucination_types = {
            0: "semantic_inconsistency",
            1: "factual_error", 
            2: "object_hallucination",
            3: "attribute_error",
            4: "spatial_error"
        }
        
        # 当前状态
        self.current_image = None
        self.current_text = None
        self.current_step = 0
        self.episode_reward = 0.0
        self.detected_regions = []
        
        logger.info("TextImageAlignmentEnv initialized")
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        重置环境
        
        Args:
            seed: 随机种子
            options: 重置选项，应包含 'image' 和 'text'
            
        Returns:
            初始观察和信息字典
        """
        super().reset(seed=seed)
        
        if options is None:
            raise ValueError("Options must contain 'image' and 'text'")
        
        self.current_image = options['image']
        self.current_text = options['text']
        self.current_step = 0
        self.episode_reward = 0.0
        self.detected_regions = []
        
        # 获取初始状态
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        执行动作
        
        Args:
            action: 动作向量 [x1, y1, x2, y2, hallucination_type, confidence]
            
        Returns:
            观察、奖励、是否结束、是否截断、信息字典
        """
        self.current_step += 1
        
        # 解析动作
        x1, y1, x2, y2, h_type, confidence = action
        
        # 确保坐标有效
        x1, y1, x2, y2 = np.clip([x1, y1, x2, y2], 0.0, 1.0)
        h_type = int(np.clip(h_type, 0, 4))
        confidence = np.clip(confidence, 0.0, 1.0)
        
        # 构建检测结果
        detection = {
            'bbox': [x1, y1, x2, y2],
            'hallucination_type': self.hallucination_types[h_type],
            'confidence': confidence,
            'step': self.current_step
        }
        
        self.detected_regions.append(detection)
        
        # 计算奖励
        reward = self._calculate_reward(detection)
        self.episode_reward += reward
        
        # 检查是否结束
        terminated = self._is_terminated()
        truncated = self.current_step >= self.max_steps
        
        # 获取新的观察
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """
        获取当前观察
        
        Returns:
            观察向量
        """
        # 提取图像特征（简化实现）
        image_features = self._extract_image_features(self.current_image)
        
        # 提取文本特征（简化实现）
        text_features = self._extract_text_features(self.current_text)
        
        # 构建检测状态特征
        detection_state = self._get_detection_state()
        
        # 拼接所有特征
        observation = np.concatenate([
            image_features,
            text_features, 
            detection_state
        ])
        
        return observation.astype(np.float32)
    
    def _extract_image_features(self, image: Image.Image) -> np.ndarray:
        """
        提取图像特征（简化实现）
        
        Args:
            image: 输入图像
            
        Returns:
            图像特征向量
        """
        # 这里应该使用预训练的视觉模型提取特征
        # 简化实现：返回随机特征
        return np.random.randn(self.image_feature_dim)
    
    def _extract_text_features(self, text: str) -> np.ndarray:
        """
        提取文本特征（简化实现）
        
        Args:
            text: 输入文本
            
        Returns:
            文本特征向量
        """
        # 这里应该使用预训练的语言模型提取特征
        # 简化实现：基于文本长度和内容的简单特征
        features = np.zeros(self.text_feature_dim)
        
        # 文本长度特征
        features[0] = len(text) / 100.0
        
        # 词汇复杂度特征（简化）
        words = text.split()
        features[1] = len(words) / 50.0
        
        # 其余特征随机填充（实际应用中应使用真实的文本编码）
        features[2:] = np.random.randn(self.text_feature_dim - 2) * 0.1
        
        return features
    
    def _get_detection_state(self) -> np.ndarray:
        """
        获取当前检测状态特征
        
        Returns:
            检测状态特征向量
        """
        state = np.zeros(self.detection_state_dim)
        
        # 已检测区域数量
        state[0] = len(self.detected_regions) / 10.0
        
        # 当前步数
        state[1] = self.current_step / self.max_steps
        
        # 累积奖励
        state[2] = self.episode_reward / 10.0
        
        # 最近检测的置信度（如果有）
        if self.detected_regions:
            state[3] = self.detected_regions[-1]['confidence']
        
        # 其余状态特征
        state[4:] = np.random.randn(self.detection_state_dim - 4) * 0.1
        
        return state
    
    def _calculate_reward(self, detection: Dict) -> float:
        """
        计算奖励
        
        Args:
            detection: 检测结果
            
        Returns:
            奖励值
        """
        if self.reward_function:
            return self.reward_function.calculate_reward(
                self.current_image,
                self.current_text,
                detection,
                self.detected_regions[:-1]  # 之前的检测结果
            )
        else:
            # 简化的奖励函数
            base_reward = detection['confidence']
            
            # 区域大小惩罚（避免检测过大区域）
            x1, y1, x2, y2 = detection['bbox']
            area = (x2 - x1) * (y2 - y1)
            area_penalty = -area if area > 0.5 else 0
            
            # 重叠惩罚（避免重复检测）
            overlap_penalty = self._calculate_overlap_penalty(detection)
            
            return base_reward + area_penalty + overlap_penalty
    
    def _calculate_overlap_penalty(self, detection: Dict) -> float:
        """
        计算重叠惩罚
        
        Args:
            detection: 当前检测结果
            
        Returns:
            重叠惩罚值
        """
        if not self.detected_regions[:-1]:  # 排除当前检测
            return 0.0
        
        current_bbox = detection['bbox']
        penalty = 0.0
        
        for prev_detection in self.detected_regions[:-1]:
            prev_bbox = prev_detection['bbox']
            iou = self._calculate_iou(current_bbox, prev_bbox)
            if iou > 0.3:  # 重叠阈值
                penalty -= iou * 0.5
        
        return penalty
    
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """
        计算两个边界框的IoU
        
        Args:
            bbox1: 边界框1 [x1, y1, x2, y2]
            bbox2: 边界框2 [x1, y1, x2, y2]
            
        Returns:
            IoU值
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # 计算交集
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # 计算并集
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _is_terminated(self) -> bool:
        """
        检查是否达到终止条件
        
        Returns:
            是否终止
        """
        # 如果累积奖励达到阈值，认为任务完成
        return self.episode_reward >= self.reward_threshold
    
    def _get_info(self) -> Dict:
        """
        获取环境信息
        
        Returns:
            信息字典
        """
        return {
            'step': self.current_step,
            'episode_reward': self.episode_reward,
            'detected_regions': len(self.detected_regions),
            'current_text': self.current_text
        }
    
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """
        渲染环境（可选实现）
        
        Args:
            mode: 渲染模式
            
        Returns:
            渲染结果
        """
        # 这里可以实现可视化检测结果
        pass
    
    def close(self):
        """关闭环境"""
        pass
