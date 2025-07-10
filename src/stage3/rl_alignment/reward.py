"""
奖励函数
为强化学习智能体设计奖励机制
"""

import numpy as np
import torch
from PIL import Image
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class RewardFunction:
    """
    幻觉检测与对齐的奖励函数
    """
    
    def __init__(self, config: Dict, hallucination_detector=None, clip_model=None):
        """
        初始化奖励函数
        
        Args:
            config: 奖励配置
            hallucination_detector: 幻觉检测器
            clip_model: CLIP模型用于语义一致性评估
        """
        self.config = config
        self.hallucination_detector = hallucination_detector
        self.clip_model = clip_model
        
        reward_config = config['reward']
        
        # 奖励权重
        self.semantic_consistency_weight = reward_config['semantic_consistency_weight']
        self.detail_accuracy_weight = reward_config['detail_accuracy_weight']
        self.overall_quality_weight = reward_config['overall_quality_weight']
        self.hallucination_penalty = reward_config['hallucination_penalty']
        
        logger.info("RewardFunction initialized")
    
    def calculate_reward(self, 
                        image: Image.Image,
                        text: str,
                        detection: Dict,
                        previous_detections: List[Dict]) -> float:
        """
        计算奖励
        
        Args:
            image: 输入图像
            text: 输入文本
            detection: 当前检测结果
            previous_detections: 之前的检测结果
            
        Returns:
            奖励值
        """
        # 语义一致性奖励
        semantic_reward = self._calculate_semantic_consistency_reward(
            image, text, detection
        )
        
        # 细节准确性奖励
        detail_reward = self._calculate_detail_accuracy_reward(
            image, text, detection
        )
        
        # 整体质量奖励
        quality_reward = self._calculate_overall_quality_reward(
            image, text, detection
        )
        
        # 重叠惩罚
        overlap_penalty = self._calculate_overlap_penalty(
            detection, previous_detections
        )
        
        # 幻觉检测奖励
        hallucination_reward = self._calculate_hallucination_detection_reward(
            image, text, detection
        )
        
        # 综合奖励
        total_reward = (
            self.semantic_consistency_weight * semantic_reward +
            self.detail_accuracy_weight * detail_reward +
            self.overall_quality_weight * quality_reward +
            overlap_penalty +
            hallucination_reward
        )
        
        return float(total_reward)
    
    def _calculate_semantic_consistency_reward(self, 
                                             image: Image.Image,
                                             text: str,
                                             detection: Dict) -> float:
        """
        计算语义一致性奖励
        
        Args:
            image: 输入图像
            text: 输入文本
            detection: 检测结果
            
        Returns:
            语义一致性奖励
        """
        try:
            # 提取检测区域
            bbox = detection['bbox']
            region = self._extract_region(image, bbox)
            
            # 使用CLIP计算区域与文本的语义相似度
            if self.clip_model:
                similarity = self._calculate_clip_similarity(region, text)
                
                # 根据幻觉类型调整奖励
                h_type = detection['hallucination_type']
                if h_type in ['semantic_inconsistency', 'factual_error']:
                    # 对于语义不一致，低相似度应该得到高奖励（正确识别了不一致）
                    reward = 1.0 - similarity
                else:
                    # 对于其他类型，高相似度得到高奖励
                    reward = similarity
                
                return reward * detection['confidence']
            else:
                # 简化实现：基于置信度
                return detection['confidence'] * 0.5
                
        except Exception as e:
            logger.warning(f"Error calculating semantic consistency reward: {e}")
            return 0.0
    
    def _calculate_detail_accuracy_reward(self, 
                                        image: Image.Image,
                                        text: str,
                                        detection: Dict) -> float:
        """
        计算细节准确性奖励
        
        Args:
            image: 输入图像
            text: 输入文本
            detection: 检测结果
            
        Returns:
            细节准确性奖励
        """
        try:
            # 检查边界框的合理性
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            
            # 边界框大小合理性
            area = (x2 - x1) * (y2 - y1)
            size_reward = 1.0 if 0.01 <= area <= 0.5 else 0.5
            
            # 边界框形状合理性（避免过于细长）
            width = x2 - x1
            height = y2 - y1
            aspect_ratio = max(width, height) / (min(width, height) + 1e-8)
            shape_reward = 1.0 if aspect_ratio <= 5.0 else 0.5
            
            # 位置合理性（避免边缘检测）
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            position_reward = 1.0 if 0.1 <= center_x <= 0.9 and 0.1 <= center_y <= 0.9 else 0.7
            
            # 综合细节奖励
            detail_reward = (size_reward + shape_reward + position_reward) / 3.0
            
            return detail_reward * detection['confidence']
            
        except Exception as e:
            logger.warning(f"Error calculating detail accuracy reward: {e}")
            return 0.0
    
    def _calculate_overall_quality_reward(self, 
                                        image: Image.Image,
                                        text: str,
                                        detection: Dict) -> float:
        """
        计算整体质量奖励
        
        Args:
            image: 输入图像
            text: 输入文本
            detection: 检测结果
            
        Returns:
            整体质量奖励
        """
        try:
            # 置信度奖励
            confidence_reward = detection['confidence']
            
            # 幻觉类型的合理性
            h_type = detection['hallucination_type']
            type_reward = self._evaluate_hallucination_type_relevance(text, h_type)
            
            # 综合质量奖励
            quality_reward = (confidence_reward + type_reward) / 2.0
            
            return quality_reward
            
        except Exception as e:
            logger.warning(f"Error calculating overall quality reward: {e}")
            return 0.0
    
    def _calculate_overlap_penalty(self, 
                                 detection: Dict,
                                 previous_detections: List[Dict]) -> float:
        """
        计算重叠惩罚
        
        Args:
            detection: 当前检测结果
            previous_detections: 之前的检测结果
            
        Returns:
            重叠惩罚
        """
        if not previous_detections:
            return 0.0
        
        current_bbox = detection['bbox']
        penalty = 0.0
        
        for prev_detection in previous_detections:
            prev_bbox = prev_detection['bbox']
            iou = self._calculate_iou(current_bbox, prev_bbox)
            
            if iou > 0.3:  # 重叠阈值
                penalty -= iou * 0.5
        
        return penalty
    
    def _calculate_hallucination_detection_reward(self, 
                                                image: Image.Image,
                                                text: str,
                                                detection: Dict) -> float:
        """
        计算幻觉检测奖励
        
        Args:
            image: 输入图像
            text: 输入文本
            detection: 检测结果
            
        Returns:
            幻觉检测奖励
        """
        try:
            if self.hallucination_detector:
                # 使用训练好的幻觉检测器验证检测结果
                region = self._extract_region(image, detection['bbox'])
                
                # 简化实现：假设检测器返回幻觉概率
                hallucination_prob = self._get_hallucination_probability(region, text)
                
                # 如果检测到的区域确实有幻觉，给予奖励
                if hallucination_prob > 0.5:
                    return detection['confidence'] * hallucination_prob
                else:
                    return self.hallucination_penalty * detection['confidence']
            else:
                # 简化实现：基于启发式规则
                return self._heuristic_hallucination_reward(text, detection)
                
        except Exception as e:
            logger.warning(f"Error calculating hallucination detection reward: {e}")
            return 0.0
    
    def _extract_region(self, image: Image.Image, bbox: List[float]) -> Image.Image:
        """
        从图像中提取区域
        
        Args:
            image: 输入图像
            bbox: 边界框 [x1, y1, x2, y2] (归一化坐标)
            
        Returns:
            提取的区域图像
        """
        width, height = image.size
        x1, y1, x2, y2 = bbox
        
        # 转换为像素坐标
        x1_px = int(x1 * width)
        y1_px = int(y1 * height)
        x2_px = int(x2 * width)
        y2_px = int(y2 * height)
        
        # 确保坐标有效
        x1_px = max(0, min(x1_px, width))
        y1_px = max(0, min(y1_px, height))
        x2_px = max(x1_px, min(x2_px, width))
        y2_px = max(y1_px, min(y2_px, height))
        
        return image.crop((x1_px, y1_px, x2_px, y2_px))
    
    def _calculate_clip_similarity(self, image: Image.Image, text: str) -> float:
        """
        计算CLIP相似度
        
        Args:
            image: 图像
            text: 文本
            
        Returns:
            相似度分数
        """
        # 这里应该实现真正的CLIP相似度计算
        # 简化实现：返回随机值
        return np.random.random()
    
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """
        计算IoU
        
        Args:
            bbox1: 边界框1
            bbox2: 边界框2
            
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
    
    def _evaluate_hallucination_type_relevance(self, text: str, h_type: str) -> float:
        """
        评估幻觉类型的相关性
        
        Args:
            text: 输入文本
            h_type: 幻觉类型
            
        Returns:
            相关性分数
        """
        # 简化实现：基于文本内容的启发式规则
        text_lower = text.lower()
        
        if h_type == "factual_error":
            # 如果文本包含事实性内容，factual_error更相关
            factual_keywords = ['year', 'location', 'name', 'date', 'number']
            if any(keyword in text_lower for keyword in factual_keywords):
                return 0.8
        
        elif h_type == "semantic_inconsistency":
            # 如果文本描述复杂，语义不一致更可能
            if len(text.split()) > 10:
                return 0.7
        
        elif h_type == "object_hallucination":
            # 如果文本提到具体对象
            object_keywords = ['person', 'car', 'house', 'animal', 'tree']
            if any(keyword in text_lower for keyword in object_keywords):
                return 0.6
        
        return 0.5  # 默认相关性
    
    def _get_hallucination_probability(self, image: Image.Image, text: str) -> float:
        """
        获取幻觉概率（简化实现）
        
        Args:
            image: 图像
            text: 文本
            
        Returns:
            幻觉概率
        """
        # 这里应该使用训练好的幻觉检测器
        # 简化实现：返回随机概率
        return np.random.random()
    
    def _heuristic_hallucination_reward(self, text: str, detection: Dict) -> float:
        """
        启发式幻觉奖励
        
        Args:
            text: 输入文本
            detection: 检测结果
            
        Returns:
            启发式奖励
        """
        # 基于简单规则的奖励
        confidence = detection['confidence']
        h_type = detection['hallucination_type']
        
        # 根据幻觉类型给予不同的基础奖励
        type_rewards = {
            'semantic_inconsistency': 0.6,
            'factual_error': 0.8,
            'object_hallucination': 0.7,
            'attribute_error': 0.5,
            'spatial_error': 0.4
        }
        
        base_reward = type_rewards.get(h_type, 0.5)
        return base_reward * confidence
