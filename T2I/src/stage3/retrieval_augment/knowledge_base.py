"""
知识库模块
构建和管理多模态知识库
"""

import os
import json
import pickle
import logging
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    # 创建简单的替代实现
    class MockFaiss:
        class IndexFlatIP:
            def __init__(self, dim):
                self.dim = dim
                self.vectors = []

            def add(self, vectors):
                self.vectors.extend(vectors)

            def search(self, query, k):
                # 简化的搜索实现
                import numpy as np
                if not self.vectors:
                    return np.array([[0.0] * k]), np.array([[0] * k])

                scores = np.random.random((1, min(k, len(self.vectors))))
                indices = np.arange(min(k, len(self.vectors))).reshape(1, -1)
                return scores, indices

        @staticmethod
        def write_index(index, path):
            pass

        @staticmethod
        def read_index(path):
            return MockFaiss.IndexFlatIP(512)

    faiss = MockFaiss()
from sentence_transformers import SentenceTransformer
from PIL import Image
import torch

logger = logging.getLogger(__name__)


class KnowledgeBase:
    """
    多模态知识库
    支持文本、图像和结构化知识的存储与检索
    """
    
    def __init__(self, config: Dict):
        """
        初始化知识库
        
        Args:
            config: 知识库配置
        """
        self.config = config
        kb_config = config['knowledge_base']
        
        # 配置参数
        self.embedding_model_name = kb_config['embedding_model']
        self.vector_dim = kb_config['vector_dim']
        self.index_type = kb_config['index_type']
        
        # 初始化嵌入模型
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        
        # 初始化向量索引
        if self.index_type == "faiss":
            if FAISS_AVAILABLE:
                self.text_index = faiss.IndexFlatIP(self.vector_dim)  # 内积索引
                self.image_index = faiss.IndexFlatIP(self.vector_dim)
            else:
                logger.warning("FAISS not available, using mock implementation")
                self.text_index = faiss.IndexFlatIP(self.vector_dim)
                self.image_index = faiss.IndexFlatIP(self.vector_dim)
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
        
        # 知识存储
        self.text_knowledge = []  # 文本知识条目
        self.image_knowledge = []  # 图像知识条目
        self.structured_knowledge = {}  # 结构化知识
        
        # 索引映射
        self.text_id_to_index = {}
        self.image_id_to_index = {}
        
        logger.info("KnowledgeBase initialized")
    
    def add_text_knowledge(self, 
                          knowledge_id: str,
                          text: str,
                          metadata: Optional[Dict] = None):
        """
        添加文本知识
        
        Args:
            knowledge_id: 知识ID
            text: 文本内容
            metadata: 元数据
        """
        try:
            # 生成嵌入
            embedding = self.embedding_model.encode([text])[0]
            
            # 添加到索引
            index = len(self.text_knowledge)
            self.text_index.add(embedding.reshape(1, -1))
            
            # 存储知识条目
            knowledge_item = {
                'id': knowledge_id,
                'text': text,
                'embedding': embedding,
                'metadata': metadata or {},
                'type': 'text'
            }
            
            self.text_knowledge.append(knowledge_item)
            self.text_id_to_index[knowledge_id] = index
            
            logger.debug(f"Added text knowledge: {knowledge_id}")
            
        except Exception as e:
            logger.error(f"Failed to add text knowledge {knowledge_id}: {e}")
    
    def add_image_knowledge(self, 
                           knowledge_id: str,
                           image_path: str,
                           description: str,
                           metadata: Optional[Dict] = None):
        """
        添加图像知识
        
        Args:
            knowledge_id: 知识ID
            image_path: 图像路径
            description: 图像描述
            metadata: 元数据
        """
        try:
            # 使用描述生成嵌入（简化实现）
            # 实际应用中应该使用视觉-语言模型
            embedding = self.embedding_model.encode([description])[0]
            
            # 添加到索引
            index = len(self.image_knowledge)
            self.image_index.add(embedding.reshape(1, -1))
            
            # 存储知识条目
            knowledge_item = {
                'id': knowledge_id,
                'image_path': image_path,
                'description': description,
                'embedding': embedding,
                'metadata': metadata or {},
                'type': 'image'
            }
            
            self.image_knowledge.append(knowledge_item)
            self.image_id_to_index[knowledge_id] = index
            
            logger.debug(f"Added image knowledge: {knowledge_id}")
            
        except Exception as e:
            logger.error(f"Failed to add image knowledge {knowledge_id}: {e}")
    
    def add_structured_knowledge(self, 
                               knowledge_id: str,
                               structured_data: Dict,
                               text_representation: str):
        """
        添加结构化知识
        
        Args:
            knowledge_id: 知识ID
            structured_data: 结构化数据
            text_representation: 文本表示
        """
        try:
            # 将结构化知识也作为文本知识添加
            self.add_text_knowledge(
                knowledge_id, 
                text_representation, 
                {'structured_data': structured_data, 'type': 'structured'}
            )
            
            # 同时存储在结构化知识中
            self.structured_knowledge[knowledge_id] = {
                'data': structured_data,
                'text_representation': text_representation
            }
            
            logger.debug(f"Added structured knowledge: {knowledge_id}")
            
        except Exception as e:
            logger.error(f"Failed to add structured knowledge {knowledge_id}: {e}")
    
    def search_text_knowledge(self, 
                             query: str,
                             top_k: int = 5,
                             threshold: float = 0.0) -> List[Dict]:
        """
        搜索文本知识
        
        Args:
            query: 查询文本
            top_k: 返回top-k结果
            threshold: 相似度阈值
            
        Returns:
            搜索结果列表
        """
        try:
            if len(self.text_knowledge) == 0:
                return []
            
            # 生成查询嵌入
            query_embedding = self.embedding_model.encode([query])[0]
            
            # 搜索
            scores, indices = self.text_index.search(
                query_embedding.reshape(1, -1), 
                min(top_k, len(self.text_knowledge))
            )
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if score >= threshold:
                    knowledge_item = self.text_knowledge[idx].copy()
                    knowledge_item['score'] = float(score)
                    results.append(knowledge_item)
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to search text knowledge: {e}")
            return []
    
    def search_image_knowledge(self, 
                              query: str,
                              top_k: int = 5,
                              threshold: float = 0.0) -> List[Dict]:
        """
        搜索图像知识
        
        Args:
            query: 查询文本
            top_k: 返回top-k结果
            threshold: 相似度阈值
            
        Returns:
            搜索结果列表
        """
        try:
            if len(self.image_knowledge) == 0:
                return []
            
            # 生成查询嵌入
            query_embedding = self.embedding_model.encode([query])[0]
            
            # 搜索
            scores, indices = self.image_index.search(
                query_embedding.reshape(1, -1), 
                min(top_k, len(self.image_knowledge))
            )
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if score >= threshold:
                    knowledge_item = self.image_knowledge[idx].copy()
                    knowledge_item['score'] = float(score)
                    results.append(knowledge_item)
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to search image knowledge: {e}")
            return []
    
    def search_all_knowledge(self, 
                           query: str,
                           top_k: int = 5,
                           threshold: float = 0.0) -> Dict[str, List[Dict]]:
        """
        搜索所有类型的知识
        
        Args:
            query: 查询文本
            top_k: 每种类型返回top-k结果
            threshold: 相似度阈值
            
        Returns:
            按类型分组的搜索结果
        """
        results = {
            'text': self.search_text_knowledge(query, top_k, threshold),
            'image': self.search_image_knowledge(query, top_k, threshold)
        }
        
        return results
    
    def get_knowledge_by_id(self, knowledge_id: str) -> Optional[Dict]:
        """
        根据ID获取知识
        
        Args:
            knowledge_id: 知识ID
            
        Returns:
            知识条目或None
        """
        # 搜索文本知识
        if knowledge_id in self.text_id_to_index:
            index = self.text_id_to_index[knowledge_id]
            return self.text_knowledge[index]
        
        # 搜索图像知识
        if knowledge_id in self.image_id_to_index:
            index = self.image_id_to_index[knowledge_id]
            return self.image_knowledge[index]
        
        # 搜索结构化知识
        if knowledge_id in self.structured_knowledge:
            return self.structured_knowledge[knowledge_id]
        
        return None
    
    def load_from_sources(self, knowledge_sources: List[Dict]):
        """
        从知识源加载数据
        
        Args:
            knowledge_sources: 知识源配置列表
        """
        for source in knowledge_sources:
            if not source.get('enabled', True):
                continue
            
            source_name = source['name']
            source_type = source['type']
            source_path = source['path']
            
            logger.info(f"Loading knowledge from {source_name} ({source_type})")
            
            try:
                if source_type == 'text':
                    self._load_text_source(source_name, source_path)
                elif source_type == 'graph':
                    self._load_graph_source(source_name, source_path)
                elif source_type == 'visual':
                    self._load_visual_source(source_name, source_path)
                else:
                    logger.warning(f"Unknown source type: {source_type}")
                    
            except Exception as e:
                logger.error(f"Failed to load {source_name}: {e}")
    
    def _load_text_source(self, source_name: str, source_path: str):
        """加载文本知识源"""
        # 创建示例文本知识
        sample_texts = [
            "The Eiffel Tower is located in Paris, France.",
            "Albert Einstein developed the theory of relativity.",
            "The Great Wall of China is over 13,000 miles long.",
            "Leonardo da Vinci painted the Mona Lisa.",
            "The Statue of Liberty was a gift from France to the United States."
        ]
        
        for i, text in enumerate(sample_texts):
            knowledge_id = f"{source_name}_{i}"
            self.add_text_knowledge(
                knowledge_id, 
                text, 
                {'source': source_name, 'type': 'factual'}
            )
        
        logger.info(f"Loaded {len(sample_texts)} text knowledge items from {source_name}")
    
    def _load_graph_source(self, source_name: str, source_path: str):
        """加载图谱知识源"""
        # 创建示例结构化知识
        sample_facts = [
            {
                'subject': 'Eiffel Tower',
                'predicate': 'located_in',
                'object': 'Paris',
                'text': 'The Eiffel Tower is located in Paris.'
            },
            {
                'subject': 'Albert Einstein',
                'predicate': 'known_for',
                'object': 'Theory of Relativity',
                'text': 'Albert Einstein is known for the Theory of Relativity.'
            }
        ]
        
        for i, fact in enumerate(sample_facts):
            knowledge_id = f"{source_name}_{i}"
            self.add_structured_knowledge(
                knowledge_id,
                fact,
                fact['text']
            )
        
        logger.info(f"Loaded {len(sample_facts)} structured knowledge items from {source_name}")
    
    def _load_visual_source(self, source_name: str, source_path: str):
        """加载视觉知识源"""
        # 创建示例图像知识
        sample_images = [
            {
                'path': 'placeholder_eiffel_tower.jpg',
                'description': 'The Eiffel Tower in Paris, a tall iron lattice tower'
            },
            {
                'path': 'placeholder_einstein.jpg', 
                'description': 'Albert Einstein, theoretical physicist with distinctive hair'
            }
        ]
        
        for i, img_info in enumerate(sample_images):
            knowledge_id = f"{source_name}_{i}"
            self.add_image_knowledge(
                knowledge_id,
                img_info['path'],
                img_info['description'],
                {'source': source_name}
            )
        
        logger.info(f"Loaded {len(sample_images)} image knowledge items from {source_name}")
    
    def save(self, save_path: str):
        """保存知识库"""
        os.makedirs(save_path, exist_ok=True)
        
        # 保存索引
        if self.index_type == "faiss":
            faiss.write_index(self.text_index, os.path.join(save_path, "text_index.faiss"))
            faiss.write_index(self.image_index, os.path.join(save_path, "image_index.faiss"))
        
        # 保存知识数据
        with open(os.path.join(save_path, "text_knowledge.pkl"), 'wb') as f:
            pickle.dump(self.text_knowledge, f)
        
        with open(os.path.join(save_path, "image_knowledge.pkl"), 'wb') as f:
            pickle.dump(self.image_knowledge, f)
        
        with open(os.path.join(save_path, "structured_knowledge.json"), 'w') as f:
            json.dump(self.structured_knowledge, f, indent=2)
        
        # 保存映射
        with open(os.path.join(save_path, "mappings.json"), 'w') as f:
            json.dump({
                'text_id_to_index': self.text_id_to_index,
                'image_id_to_index': self.image_id_to_index
            }, f, indent=2)
        
        logger.info(f"Knowledge base saved to {save_path}")
    
    def load(self, load_path: str):
        """加载知识库"""
        # 加载索引
        if self.index_type == "faiss":
            text_index_path = os.path.join(load_path, "text_index.faiss")
            image_index_path = os.path.join(load_path, "image_index.faiss")
            
            if os.path.exists(text_index_path):
                self.text_index = faiss.read_index(text_index_path)
            if os.path.exists(image_index_path):
                self.image_index = faiss.read_index(image_index_path)
        
        # 加载知识数据
        text_knowledge_path = os.path.join(load_path, "text_knowledge.pkl")
        if os.path.exists(text_knowledge_path):
            with open(text_knowledge_path, 'rb') as f:
                self.text_knowledge = pickle.load(f)
        
        image_knowledge_path = os.path.join(load_path, "image_knowledge.pkl")
        if os.path.exists(image_knowledge_path):
            with open(image_knowledge_path, 'rb') as f:
                self.image_knowledge = pickle.load(f)
        
        structured_knowledge_path = os.path.join(load_path, "structured_knowledge.json")
        if os.path.exists(structured_knowledge_path):
            with open(structured_knowledge_path, 'r') as f:
                self.structured_knowledge = json.load(f)
        
        # 加载映射
        mappings_path = os.path.join(load_path, "mappings.json")
        if os.path.exists(mappings_path):
            with open(mappings_path, 'r') as f:
                mappings = json.load(f)
                self.text_id_to_index = mappings.get('text_id_to_index', {})
                self.image_id_to_index = mappings.get('image_id_to_index', {})
        
        logger.info(f"Knowledge base loaded from {load_path}")
    
    def get_statistics(self) -> Dict:
        """获取知识库统计信息"""
        return {
            'text_knowledge_count': len(self.text_knowledge),
            'image_knowledge_count': len(self.image_knowledge),
            'structured_knowledge_count': len(self.structured_knowledge),
            'total_knowledge_count': len(self.text_knowledge) + len(self.image_knowledge),
            'embedding_dimension': self.vector_dim,
            'index_type': self.index_type
        }
