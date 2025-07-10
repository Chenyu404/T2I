"""
多模态检索增强模块
针对事实性幻觉，引入多模态检索增强机制
"""

from .knowledge_base import KnowledgeBase

# 其他模块将在后续实现
# from .retriever import MultiModalRetriever
# from .augmentor import ContextAugmentor
# from .fact_checker import FactChecker

__all__ = [
    'KnowledgeBase',
    # 'MultiModalRetriever',
    # 'ContextAugmentor',
    # 'FactChecker'
]
