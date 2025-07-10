"""
阶段二训练脚本
训练幻觉检测模型
"""

import os
import json
import logging
import yaml
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import numpy as np
from datetime import datetime
from tqdm import tqdm

from .model import HallucinationDetectionModel
from .dataset import create_data_loaders

try:
    from safetensors.torch import save_file as safetensors_save
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False

logger = logging.getLogger(__name__)


class Stage2Trainer:
    """
    阶段二训练器
    """
    
    def __init__(self, config_path: str = "config/stage2_config.yaml"):
        """
        初始化训练器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        # 设置设备
        self.device = self.config['device']
        if self.device == 'cuda' and not torch.cuda.is_available():
            self.device = 'cpu'
            logger.warning("CUDA not available, using CPU")
        
        # 创建输出目录
        self.output_dir = self.config['output']['model_save_dir']
        self.results_dir = self.config['output']['results_dir']
        self.log_dir = self.config['output']['log_dir']
        
        for dir_path in [self.output_dir, self.results_dir, self.log_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # 初始化模型
        self.model = HallucinationDetectionModel(self.config).to(self.device)
        
        # 初始化数据加载器
        self.train_loader, self.val_loader, self.test_loader = create_data_loaders(self.config)
        
        # 初始化优化器和调度器
        self._init_optimizer_and_scheduler()
        
        # 初始化损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # 训练状态
        self.current_epoch = 0
        self.best_val_score = 0.0
        self.train_history = []
        self.val_history = []
        
        logger.info("Stage2Trainer initialized successfully")
    
    def _load_config(self) -> Dict:
        """加载配置文件"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    
    def _init_optimizer_and_scheduler(self):
        """初始化优化器和学习率调度器"""
        training_config = self.config['training']
        
        # 优化器
        if training_config['optimizer'] == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=training_config['learning_rate'],
                weight_decay=training_config['weight_decay']
            )
        elif training_config['optimizer'] == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=training_config['learning_rate'],
                weight_decay=training_config['weight_decay']
            )
        else:
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=training_config['learning_rate'],
                weight_decay=training_config['weight_decay'],
                momentum=0.9
            )
        
        # 学习率调度器
        total_steps = len(self.train_loader) * training_config['num_epochs']
        warmup_steps = training_config['warmup_steps']
        
        if training_config['scheduler'] == 'cosine':
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
        elif training_config['scheduler'] == 'linear':
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
        else:
            self.scheduler = None
    
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        progress_bar = tqdm(self.train_loader, desc=f"Training Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # 移动数据到设备
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            pixel_values = batch['pixel_values'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # 前向传播
            outputs = self.model(input_ids, attention_mask, pixel_values)
            logits = outputs['logits']
            
            # 计算损失
            loss = self.criterion(logits, labels)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            if self.config['training']['gradient_clip_norm'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['gradient_clip_norm']
                )
            
            self.optimizer.step()
            
            if self.scheduler:
                self.scheduler.step()
            
            # 统计
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss / (batch_idx + 1):.4f}'
            })
        
        # 计算指标
        avg_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted'
        )
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def validate(self) -> Dict[str, float]:
        """验证模型"""
        self.model.eval()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                # 移动数据到设备
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                pixel_values = batch['pixel_values'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # 前向传播
                outputs = self.model(input_ids, attention_mask, pixel_values)
                logits = outputs['logits']
                
                # 计算损失
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                
                # 预测
                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(logits, dim=-1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # 计算指标
        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted'
        )
        
        # 计算AUC
        try:
            all_probs = np.array(all_probs)
            auc = roc_auc_score(all_labels, all_probs[:, 1])
        except:
            auc = 0.0
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
    
    def train(self):
        """完整训练流程"""
        logger.info("Starting training...")
        
        num_epochs = self.config['training']['num_epochs']
        early_stopping_config = self.config['training']['early_stopping']
        
        best_score = 0.0
        patience_counter = 0
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch + 1
            
            logger.info(f"Epoch {self.current_epoch}/{num_epochs}")
            
            # 训练
            train_metrics = self.train_epoch()
            self.train_history.append(train_metrics)
            
            # 验证
            val_metrics = self.validate()
            self.val_history.append(val_metrics)
            
            # 记录日志
            logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, "
                       f"Acc: {train_metrics['accuracy']:.4f}, "
                       f"F1: {train_metrics['f1']:.4f}")
            logger.info(f"Val - Loss: {val_metrics['loss']:.4f}, "
                       f"Acc: {val_metrics['accuracy']:.4f}, "
                       f"F1: {val_metrics['f1']:.4f}, "
                       f"AUC: {val_metrics['auc']:.4f}")
            
            # 检查是否是最佳模型
            current_score = val_metrics[early_stopping_config['monitor']]
            
            if early_stopping_config['mode'] == 'max':
                is_best = current_score > best_score
            else:
                is_best = current_score < best_score
            
            if is_best:
                best_score = current_score
                patience_counter = 0
                self.save_model('best_model.pth')
                logger.info(f"New best model saved with {early_stopping_config['monitor']}: {best_score:.4f}")
            else:
                patience_counter += 1
            
            # 早停检查
            if patience_counter >= early_stopping_config['patience']:
                logger.info(f"Early stopping triggered after {patience_counter} epochs without improvement")
                break
            
            # 保存最新模型
            if self.config['output']['save_last']:
                self.save_model('last_model.pth')
        
        logger.info("Training completed!")
        
        # 保存训练历史
        self.save_training_history()
    
    def save_model(self, filename: str):
        """保存模型"""
        model_path = os.path.join(self.output_dir, filename)

        # 准备保存数据
        save_data = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'epoch': self.current_epoch,
            'config': self.config,
            'train_history': self.train_history,
            'val_history': self.val_history
        }

        # 尝试使用safetensors格式保存
        if SAFETENSORS_AVAILABLE and filename.endswith('.pth'):
            try:
                # 保存模型权重为safetensors格式
                safetensors_path = model_path.replace('.pth', '.safetensors')
                safetensors_save(self.model.state_dict(), safetensors_path)

                # 保存其他数据为json格式
                metadata_path = model_path.replace('.pth', '_metadata.json')
                metadata = {
                    'epoch': self.current_epoch,
                    'config': self.config,
                    'train_history': self.train_history,
                    'val_history': self.val_history
                }
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)

                logger.info(f"Model saved to {safetensors_path} (safetensors format)")
                logger.info(f"Metadata saved to {metadata_path}")
                return
            except Exception as e:
                logger.warning(f"Failed to save with safetensors: {e}, falling back to torch.save")

        # 回退到传统的torch.save方法（需要torch>=2.6）
        try:
            torch.save(save_data, model_path, _use_new_zipfile_serialization=False)
            logger.info(f"Model saved to {model_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            logger.error("Please upgrade PyTorch to version 2.6+ or install safetensors")
            raise
    
    def save_training_history(self):
        """保存训练历史"""
        history = {
            'train_history': self.train_history,
            'val_history': self.val_history,
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }
        
        history_path = os.path.join(self.results_dir, 'training_history.json')
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Training history saved to {history_path}")
