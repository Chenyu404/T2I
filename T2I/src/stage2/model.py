"""
幻觉检测模型
基于多模态特征的幻觉检测分类器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPConfig
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class MultiModalFusion(nn.Module):
    """多模态特征融合模块"""
    
    def __init__(self, 
                 text_dim: int = 512,
                 image_dim: int = 512,
                 fusion_dim: int = 512,
                 fusion_method: str = "concat"):
        """
        初始化融合模块
        
        Args:
            text_dim: 文本特征维度
            image_dim: 图像特征维度
            fusion_dim: 融合后特征维度
            fusion_method: 融合方法 (concat, attention, cross_attention)
        """
        super().__init__()
        
        self.fusion_method = fusion_method
        self.fusion_dim = fusion_dim
        
        if fusion_method == "concat":
            self.fusion_layer = nn.Linear(text_dim + image_dim, fusion_dim)
        
        elif fusion_method == "attention":
            self.text_proj = nn.Linear(text_dim, fusion_dim)
            self.image_proj = nn.Linear(image_dim, fusion_dim)
            self.attention = nn.MultiheadAttention(fusion_dim, num_heads=8)
            
        elif fusion_method == "cross_attention":
            self.text_proj = nn.Linear(text_dim, fusion_dim)
            self.image_proj = nn.Linear(image_dim, fusion_dim)
            self.cross_attention = nn.MultiheadAttention(fusion_dim, num_heads=8)
            
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
    
    def forward(self, text_features: torch.Tensor, image_features: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            text_features: 文本特征 [batch_size, text_dim]
            image_features: 图像特征 [batch_size, image_dim]
            
        Returns:
            融合后的特征 [batch_size, fusion_dim]
        """
        if self.fusion_method == "concat":
            # 简单拼接
            combined = torch.cat([text_features, image_features], dim=-1)
            fused = self.fusion_layer(combined)
            
        elif self.fusion_method == "attention":
            # 注意力融合
            text_proj = self.text_proj(text_features).unsqueeze(1)  # [B, 1, D]
            image_proj = self.image_proj(image_features).unsqueeze(1)  # [B, 1, D]
            
            # 将文本和图像特征作为序列
            features = torch.cat([text_proj, image_proj], dim=1)  # [B, 2, D]
            features = features.transpose(0, 1)  # [2, B, D]
            
            attended, _ = self.attention(features, features, features)
            fused = attended.mean(dim=0)  # [B, D]
            
        elif self.fusion_method == "cross_attention":
            # 交叉注意力融合
            text_proj = self.text_proj(text_features).unsqueeze(1)  # [B, 1, D]
            image_proj = self.image_proj(image_features).unsqueeze(1)  # [B, 1, D]
            
            text_proj = text_proj.transpose(0, 1)  # [1, B, D]
            image_proj = image_proj.transpose(0, 1)  # [1, B, D]
            
            # 文本查询图像
            text_attended, _ = self.cross_attention(text_proj, image_proj, image_proj)
            # 图像查询文本
            image_attended, _ = self.cross_attention(image_proj, text_proj, text_proj)
            
            # 平均融合
            fused = (text_attended + image_attended).squeeze(0) / 2  # [B, D]
        
        return fused


class HallucinationClassifier(nn.Module):
    """幻觉检测分类器"""
    
    def __init__(self, 
                 input_dim: int = 512,
                 hidden_dims: list = [256, 128],
                 num_classes: int = 2,
                 dropout: float = 0.3,
                 activation: str = "relu"):
        """
        初始化分类器
        
        Args:
            input_dim: 输入特征维度
            hidden_dims: 隐藏层维度列表
            num_classes: 分类数量
            dropout: Dropout率
            activation: 激活函数
        """
        super().__init__()
        
        # 激活函数
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()
        
        # 构建分类器层
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self.activation,
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征 [batch_size, input_dim]
            
        Returns:
            分类logits [batch_size, num_classes]
        """
        return self.classifier(x)


class HallucinationDetectionModel(nn.Module):
    """
    幻觉检测模型
    基于CLIP的多模态幻觉检测模型
    """
    
    def __init__(self, config: Dict):
        """
        初始化模型
        
        Args:
            config: 模型配置
        """
        super().__init__()
        
        self.config = config
        model_config = config['model']
        
        # 加载基础CLIP模型
        clip_model_name = model_config['clip']['model_name']

        # 安全加载模型，避免 torch.load 安全问题
        try:
            # 尝试使用 safetensors 格式
            self.clip_model = CLIPModel.from_pretrained(
                clip_model_name,
                use_safetensors=True
            )
            logger.info(f"Loaded CLIP model with safetensors: {clip_model_name}")
        except Exception as e:
            logger.warning(f"Failed to load with safetensors: {e}")
            try:
                # 回退到标准加载，但设置安全参数
                self.clip_model = CLIPModel.from_pretrained(
                    clip_model_name,
                    torch_dtype=torch.float32,
                    trust_remote_code=False
                )
                logger.info(f"Loaded CLIP model with standard method: {clip_model_name}")
            except Exception as e2:
                logger.error(f"Failed to load CLIP model: {e2}")
                # 创建一个最小的模拟模型
                logger.warning("Creating mock CLIP model for testing")
                self.clip_model = self._create_mock_clip_model()
                self.mock_clip = True

        # 初始化模拟标志
        if not hasattr(self, 'mock_clip'):
            self.mock_clip = False
        
        # 是否冻结CLIP参数
        if model_config['clip']['freeze_backbone']:
            for param in self.clip_model.parameters():
                param.requires_grad = False
            logger.info("CLIP backbone frozen")
        
        # 获取特征维度
        clip_config = self.clip_model.config
        text_dim = clip_config.text_config.hidden_size
        image_dim = clip_config.vision_config.hidden_size
        
        # 多模态融合模块
        fusion_config = model_config['fusion']
        self.fusion = MultiModalFusion(
            text_dim=text_dim,
            image_dim=image_dim,
            fusion_dim=fusion_config['fusion_dim'],
            fusion_method=fusion_config['method']
        )
        
        # 分类器
        classifier_config = model_config['classifier']
        self.classifier = HallucinationClassifier(
            input_dim=fusion_config['fusion_dim'],
            hidden_dims=classifier_config['hidden_dims'],
            num_classes=classifier_config['num_classes'],
            dropout=classifier_config['dropout'],
            activation=classifier_config['activation']
        )
        
        logger.info("HallucinationDetectionModel initialized")
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            input_ids: 文本token IDs [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            pixel_values: 图像像素值 [batch_size, 3, H, W]
            
        Returns:
            模型输出字典
        """
        # 获取CLIP特征
        if self.mock_clip:
            # 使用模拟CLIP模型
            text_features = self.clip_model.get_text_features(input_ids, attention_mask)
            image_features = self.clip_model.get_image_features(pixel_values)
        else:
            # 使用真实CLIP模型
            clip_outputs = self.clip_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                return_dict=True
            )

            # 提取文本和图像特征
            text_features = clip_outputs.text_embeds  # [batch_size, text_dim]
            image_features = clip_outputs.image_embeds  # [batch_size, image_dim]
        
        # 多模态融合
        fused_features = self.fusion(text_features, image_features)
        
        # 分类
        logits = self.classifier(fused_features)
        
        return {
            'logits': logits,
            'text_features': text_features,
            'image_features': image_features,
            'fused_features': fused_features
        }
    
    def predict(self, 
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                pixel_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        预测函数
        
        Args:
            input_ids: 文本token IDs
            attention_mask: 注意力掩码
            pixel_values: 图像像素值
            
        Returns:
            预测概率和预测标签
        """
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask, pixel_values)
            logits = outputs['logits']
            
            # 计算概率
            probs = F.softmax(logits, dim=-1)
            
            # 预测标签
            preds = torch.argmax(logits, dim=-1)
            
            return probs, preds
    
    def get_feature_importance(self, 
                              input_ids: torch.Tensor,
                              attention_mask: torch.Tensor,
                              pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        获取特征重要性（用于可解释性分析）
        
        Args:
            input_ids: 文本token IDs
            attention_mask: 注意力掩码
            pixel_values: 图像像素值
            
        Returns:
            特征重要性字典
        """
        # 启用梯度计算
        input_ids.requires_grad_(True)
        pixel_values.requires_grad_(True)
        
        outputs = self.forward(input_ids, attention_mask, pixel_values)
        logits = outputs['logits']
        
        # 计算梯度
        # 对于二分类，我们关注正类（有幻觉）的梯度
        positive_logits = logits[:, 1]
        positive_logits.sum().backward()
        
        # 获取梯度作为重要性指标
        text_importance = input_ids.grad.abs().sum(dim=-1)  # [batch_size]
        image_importance = pixel_values.grad.abs().sum(dim=(1, 2, 3))  # [batch_size]
        
        return {
            'text_importance': text_importance,
            'image_importance': image_importance,
            'text_features': outputs['text_features'],
            'image_features': outputs['image_features']
        }

    def _create_mock_clip_model(self):
        """
        创建模拟 CLIP 模型，用于测试和避免加载问题
        """
        class MockCLIPModel(nn.Module):
            def __init__(self):
                super().__init__()
                # 创建简单的模拟组件
                self.text_model = nn.Sequential(
                    nn.Embedding(50000, 512),  # 词汇表大小和嵌入维度
                    nn.TransformerEncoder(
                        nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True),
                        num_layers=2
                    ),
                    nn.Linear(512, 512)
                )

                self.vision_model = nn.Sequential(
                    nn.Conv2d(3, 64, 7, stride=2, padding=3),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((7, 7)),
                    nn.Flatten(),
                    nn.Linear(64 * 7 * 7, 512)
                )

                self.text_projection = nn.Linear(512, 512)
                self.visual_projection = nn.Linear(512, 512)

                logger.info("Created mock CLIP model for testing")

            def get_text_features(self, input_ids, attention_mask=None):
                # 简单的文本特征提取
                batch_size = input_ids.size(0)
                # 截断或填充到合理长度
                if input_ids.size(1) > 77:
                    input_ids = input_ids[:, :77]
                elif input_ids.size(1) < 77:
                    pad_size = 77 - input_ids.size(1)
                    padding = torch.zeros(batch_size, pad_size, dtype=input_ids.dtype, device=input_ids.device)
                    input_ids = torch.cat([input_ids, padding], dim=1)

                # 模拟文本编码
                text_embeds = self.text_model[0](input_ids)  # Embedding
                text_embeds = text_embeds.mean(dim=1)  # 简单平均池化
                text_features = self.text_projection(text_embeds)
                return text_features

            def get_image_features(self, pixel_values):
                # 简单的图像特征提取
                image_features = self.vision_model(pixel_values)
                image_features = self.visual_projection(image_features)
                return image_features

        return MockCLIPModel()
