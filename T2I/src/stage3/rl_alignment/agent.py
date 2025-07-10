"""
强化学习智能体
实现PPO算法的幻觉检测与定位智能体
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
import numpy as np
from typing import Dict, Tuple, List
import logging

try:
    from safetensors.torch import save_file as safetensors_save, load_file as safetensors_load
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False

logger = logging.getLogger(__name__)


class PolicyNetwork(nn.Module):
    """策略网络"""
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 hidden_dims: List[int] = [256, 128],
                 activation: str = "relu"):
        """
        初始化策略网络
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            hidden_dims: 隐藏层维度
            activation: 激活函数
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 激活函数
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()
        
        # 构建网络层
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self.activation,
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        self.shared_layers = nn.Sequential(*layers)
        
        # 连续动作的均值和标准差输出
        # 动作：[x1, y1, x2, y2, confidence] (5维连续)
        self.continuous_mean = nn.Linear(prev_dim, 5)
        self.continuous_log_std = nn.Linear(prev_dim, 5)
        
        # 离散动作的输出（幻觉类型）
        self.discrete_head = nn.Linear(prev_dim, 5)  # 5种幻觉类型
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            state: 状态张量 [batch_size, state_dim]
            
        Returns:
            连续动作分布参数、离散动作logits
        """
        features = self.shared_layers(state)
        
        # 连续动作（坐标和置信度）
        continuous_mean = torch.sigmoid(self.continuous_mean(features))  # 限制到[0,1]
        continuous_log_std = self.continuous_log_std(features)
        continuous_log_std = torch.clamp(continuous_log_std, -20, 2)
        
        # 离散动作（幻觉类型）
        discrete_logits = self.discrete_head(features)
        
        return continuous_mean, continuous_log_std, discrete_logits
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取动作
        
        Args:
            state: 状态张量
            deterministic: 是否确定性采样
            
        Returns:
            动作和对数概率
        """
        continuous_mean, continuous_log_std, discrete_logits = self.forward(state)
        
        if deterministic:
            # 确定性动作
            continuous_action = continuous_mean
            discrete_action = torch.argmax(discrete_logits, dim=-1)
        else:
            # 随机采样
            continuous_std = torch.exp(continuous_log_std)
            continuous_dist = Normal(continuous_mean, continuous_std)
            continuous_action = continuous_dist.sample()
            continuous_action = torch.clamp(continuous_action, 0, 1)
            
            discrete_dist = Categorical(logits=discrete_logits)
            discrete_action = discrete_dist.sample()
        
        # 合并动作
        action = torch.cat([continuous_action, discrete_action.unsqueeze(-1).float()], dim=-1)
        
        # 计算对数概率
        log_prob = self.get_log_prob(state, action)
        
        return action, log_prob
    
    def get_log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        计算动作的对数概率
        
        Args:
            state: 状态张量
            action: 动作张量 [batch_size, 6]
            
        Returns:
            对数概率
        """
        continuous_mean, continuous_log_std, discrete_logits = self.forward(state)
        
        # 分离连续和离散动作
        continuous_action = action[:, :5]
        discrete_action = action[:, 5].long()
        
        # 连续动作的对数概率
        continuous_std = torch.exp(continuous_log_std)
        continuous_dist = Normal(continuous_mean, continuous_std)
        continuous_log_prob = continuous_dist.log_prob(continuous_action).sum(dim=-1)
        
        # 离散动作的对数概率
        discrete_dist = Categorical(logits=discrete_logits)
        discrete_log_prob = discrete_dist.log_prob(discrete_action)
        
        return continuous_log_prob + discrete_log_prob


class ValueNetwork(nn.Module):
    """价值网络"""
    
    def __init__(self, 
                 state_dim: int,
                 hidden_dims: List[int] = [256, 128],
                 activation: str = "relu"):
        """
        初始化价值网络
        
        Args:
            state_dim: 状态维度
            hidden_dims: 隐藏层维度
            activation: 激活函数
        """
        super().__init__()
        
        # 激活函数
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()
        
        # 构建网络层
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self.activation,
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            state: 状态张量 [batch_size, state_dim]
            
        Returns:
            状态价值 [batch_size, 1]
        """
        return self.network(state)


class AlignmentAgent:
    """
    幻觉检测与对齐智能体
    基于PPO算法
    """
    
    def __init__(self, config: Dict, state_dim: int, action_dim: int, device: str = "cuda"):
        """
        初始化智能体
        
        Args:
            config: 配置字典
            state_dim: 状态维度
            action_dim: 动作维度
            device: 计算设备
        """
        self.config = config
        self.device = device
        
        agent_config = config['agent']
        
        # 初始化网络
        self.policy_net = PolicyNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=agent_config['policy_network']['hidden_dims'],
            activation=agent_config['policy_network']['activation']
        ).to(device)
        
        self.value_net = ValueNetwork(
            state_dim=state_dim,
            hidden_dims=agent_config['value_network']['hidden_dims'],
            activation=agent_config['value_network']['activation']
        ).to(device)
        
        # 优化器
        training_config = config['training']
        self.policy_optimizer = torch.optim.Adam(
            self.policy_net.parameters(), 
            lr=training_config['learning_rate']
        )
        self.value_optimizer = torch.optim.Adam(
            self.value_net.parameters(), 
            lr=training_config['learning_rate']
        )
        
        # PPO参数
        self.clip_range = training_config['clip_range']
        self.ent_coef = training_config['ent_coef']
        self.vf_coef = training_config['vf_coef']
        
        logger.info("AlignmentAgent initialized")
    
    def get_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float]:
        """
        获取动作
        
        Args:
            state: 状态数组
            deterministic: 是否确定性采样
            
        Returns:
            动作和对数概率
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob = self.policy_net.get_action(state_tensor, deterministic)
        
        return action.cpu().numpy()[0], log_prob.cpu().numpy()[0]
    
    def get_value(self, state: np.ndarray) -> float:
        """
        获取状态价值
        
        Args:
            state: 状态数组
            
        Returns:
            状态价值
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            value = self.value_net(state_tensor)
        
        return value.cpu().numpy()[0][0]
    
    def update(self, batch_data: Dict) -> Dict[str, float]:
        """
        更新网络参数
        
        Args:
            batch_data: 批次数据
            
        Returns:
            训练指标
        """
        states = torch.FloatTensor(batch_data['states']).to(self.device)
        actions = torch.FloatTensor(batch_data['actions']).to(self.device)
        old_log_probs = torch.FloatTensor(batch_data['log_probs']).to(self.device)
        returns = torch.FloatTensor(batch_data['returns']).to(self.device)
        advantages = torch.FloatTensor(batch_data['advantages']).to(self.device)
        
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 计算当前策略的对数概率和价值
        current_log_probs = self.policy_net.get_log_prob(states, actions)
        current_values = self.value_net(states).squeeze()
        
        # PPO策略损失
        ratio = torch.exp(current_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # 熵损失（鼓励探索）
        entropy_loss = -self._calculate_entropy(states).mean()
        
        # 价值损失
        value_loss = F.mse_loss(current_values, returns)
        
        # 总损失
        total_loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss
        
        # 更新策略网络
        self.policy_optimizer.zero_grad()
        policy_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
        self.policy_optimizer.step()
        
        # 更新价值网络
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
        self.value_optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def _calculate_entropy(self, states: torch.Tensor) -> torch.Tensor:
        """
        计算策略熵
        
        Args:
            states: 状态张量
            
        Returns:
            熵值
        """
        continuous_mean, continuous_log_std, discrete_logits = self.policy_net.forward(states)
        
        # 连续动作熵
        continuous_std = torch.exp(continuous_log_std)
        continuous_entropy = 0.5 * torch.log(2 * np.pi * np.e * continuous_std**2).sum(dim=-1)
        
        # 离散动作熵
        discrete_probs = F.softmax(discrete_logits, dim=-1)
        discrete_entropy = -(discrete_probs * torch.log(discrete_probs + 1e-8)).sum(dim=-1)
        
        return continuous_entropy + discrete_entropy
    
    def save(self, path: str):
        """保存模型"""
        save_data = {
            'policy_net': self.policy_net.state_dict(),
            'value_net': self.value_net.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'value_optimizer': self.value_optimizer.state_dict(),
            'config': self.config
        }

        # 尝试使用safetensors格式保存
        if SAFETENSORS_AVAILABLE and path.endswith('.pth'):
            try:
                import json
                import os

                # 保存模型权重为safetensors格式
                safetensors_path = path.replace('.pth', '.safetensors')
                model_weights = {
                    'policy_net': self.policy_net.state_dict(),
                    'value_net': self.value_net.state_dict()
                }
                # 展平嵌套字典
                flat_weights = {}
                for net_name, state_dict in model_weights.items():
                    for key, value in state_dict.items():
                        flat_weights[f"{net_name}.{key}"] = value

                safetensors_save(flat_weights, safetensors_path)

                # 保存其他数据为json格式
                metadata_path = path.replace('.pth', '_metadata.json')
                metadata = {
                    'config': self.config
                }
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)

                logger.info(f"Agent saved to {safetensors_path} (safetensors format)")
                return
            except Exception as e:
                logger.warning(f"Failed to save with safetensors: {e}, falling back to torch.save")

        # 回退到传统的torch.save方法
        try:
            torch.save(save_data, path, _use_new_zipfile_serialization=False)
            logger.info(f"Agent saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save agent: {e}")
            logger.error("Please upgrade PyTorch to version 2.6+ or install safetensors")
            raise

    def load(self, path: str):
        """加载模型"""
        # 尝试加载safetensors格式
        safetensors_path = path.replace('.pth', '.safetensors')
        if SAFETENSORS_AVAILABLE and os.path.exists(safetensors_path):
            try:
                import json
                import os

                # 加载模型权重
                flat_weights = safetensors_load(safetensors_path)

                # 重构嵌套字典
                policy_weights = {}
                value_weights = {}
                for key, value in flat_weights.items():
                    if key.startswith('policy_net.'):
                        policy_weights[key[11:]] = value  # 移除'policy_net.'前缀
                    elif key.startswith('value_net.'):
                        value_weights[key[10:]] = value   # 移除'value_net.'前缀

                self.policy_net.load_state_dict(policy_weights)
                self.value_net.load_state_dict(value_weights)

                logger.info(f"Agent loaded from {safetensors_path} (safetensors format)")
                return
            except Exception as e:
                logger.warning(f"Failed to load safetensors format: {e}, trying torch.load")

        # 回退到torch.load
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=True)

            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.value_net.load_state_dict(checkpoint['value_net'])
            if 'policy_optimizer' in checkpoint:
                self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
            if 'value_optimizer' in checkpoint:
                self.value_optimizer.load_state_dict(checkpoint['value_optimizer'])

            logger.info(f"Agent loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load agent: {e}")
            logger.error("Please upgrade PyTorch to version 2.6+ or use safetensors format")
            raise
