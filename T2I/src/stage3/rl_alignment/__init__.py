"""
强化学习对齐方法
针对语义细节幻觉，设计基于强化学习的对齐方法
"""

from .environment import TextImageAlignmentEnv
from .agent import AlignmentAgent
from .reward import RewardFunction
from .trainer import RLTrainer

__all__ = [
    'TextImageAlignmentEnv',
    'AlignmentAgent', 
    'RewardFunction',
    'RLTrainer'
]
