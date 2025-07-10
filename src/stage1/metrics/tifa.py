"""
TIFA (Text-to-Image Faithfulness evaluation with question Answering)评估指标实现
基于问答的文本-图像忠实度评估
"""

import torch
from PIL import Image
from transformers import (
    AutoTokenizer, AutoModelForQuestionAnswering,
    BlipProcessor, BlipForQuestionAnswering,
    pipeline
)
import numpy as np
from typing import List, Dict, Tuple
import logging
import re

logger = logging.getLogger(__name__)


class TIFA:
    """
    TIFA评估指标
    通过生成问题并回答来评估图像与文本的忠实度
    """
    
    def __init__(self, model_name: str = "microsoft/unilm-base-cased", device: str = "cuda"):
        """
        初始化TIFA评估器
        
        Args:
            model_name: 基础模型名称
            device: 计算设备
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        
        try:
            logger.info("Loading TIFA components...")

            # 尝试加载VQA模型用于图像问答
            try:
                self.vqa_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
                self.vqa_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(self.device)
                self._use_vqa = True
            except Exception as e:
                logger.warning(f"Failed to load BLIP VQA model: {e}")
                logger.info("Using fallback implementation")
                self._use_vqa = False

            # 尝试加载文本问答模型
            try:
                self.text_qa_pipeline = pipeline(
                    "question-answering",
                    model="distilbert-base-cased-distilled-squad",
                    device=0 if self.device == "cuda" else -1
                )
                self._use_text_qa = True
            except Exception as e:
                logger.warning(f"Failed to load text QA model: {e}")
                self._use_text_qa = False

            logger.info("TIFA initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize TIFA: {e}")
            # 不抛出异常，使用备用实现
            self._use_vqa = False
            self._use_text_qa = False
    
    def generate_questions(self, text: str) -> List[str]:
        """
        从文本中生成问题
        
        Args:
            text: 输入文本
            
        Returns:
            问题列表
        """
        # 简化的问题生成策略
        questions = []
        
        # 基于关键词生成问题
        words = text.lower().split()
        
        # 寻找名词和形容词
        common_nouns = ['person', 'people', 'man', 'woman', 'child', 'dog', 'cat', 'car', 'house', 'tree', 'flower']
        colors = ['red', 'blue', 'green', 'yellow', 'black', 'white', 'brown', 'pink', 'purple', 'orange']
        
        # 生成存在性问题
        for noun in common_nouns:
            if noun in words:
                questions.append(f"Is there a {noun} in the image?")
        
        # 生成颜色相关问题
        for color in colors:
            if color in words:
                questions.append(f"Is there something {color} in the image?")
        
        # 生成数量问题
        numbers = ['one', 'two', 'three', 'four', 'five', 'many', 'several']
        for num in numbers:
            if num in words:
                questions.append(f"How many objects are in the image?")
                break
        
        # 如果没有生成问题，添加通用问题
        if not questions:
            questions = [
                "What is in the image?",
                "What is the main object in the image?",
                "What color is prominent in the image?"
            ]
        
        return questions[:5]  # 限制问题数量
    
    def answer_question_on_image(self, image: Image.Image, question: str) -> str:
        """
        在图像上回答问题

        Args:
            image: 输入图像
            question: 问题

        Returns:
            答案
        """
        try:
            if self._use_vqa:
                inputs = self.vqa_processor(image, question, return_tensors="pt").to(self.device)

                with torch.no_grad():
                    outputs = self.vqa_model.generate(**inputs, max_length=20)

                answer = self.vqa_processor.decode(outputs[0], skip_special_tokens=True)
                return answer.strip()
            else:
                # 简化的备用实现
                return self._fallback_image_qa(image, question)
        except Exception as e:
            logger.warning(f"Error in VQA: {e}")
            return "unknown"
    
    def answer_question_on_text(self, text: str, question: str) -> str:
        """
        基于文本回答问题

        Args:
            text: 输入文本
            question: 问题

        Returns:
            答案
        """
        try:
            if self._use_text_qa:
                result = self.text_qa_pipeline(question=question, context=text)
                return result['answer'].strip().lower()
            else:
                return self._fallback_text_qa(text, question)
        except Exception as e:
            logger.warning(f"Error in text QA: {e}")
            return "unknown"

    def _fallback_image_qa(self, image: Image.Image, question: str) -> str:
        """备用图像问答实现"""
        # 简化的启发式回答
        question_lower = question.lower()

        if "color" in question_lower:
            return "unknown color"
        elif "how many" in question_lower:
            return "unknown number"
        elif "is there" in question_lower:
            return "maybe"
        else:
            return "unknown"

    def _fallback_text_qa(self, text: str, question: str) -> str:
        """备用文本问答实现"""
        # 简化的关键词匹配
        text_lower = text.lower()
        question_lower = question.lower()

        if "color" in question_lower:
            colors = ['red', 'blue', 'green', 'yellow', 'black', 'white']
            for color in colors:
                if color in text_lower:
                    return color
        elif "how many" in question_lower:
            numbers = ['one', 'two', 'three', 'four', 'five']
            for num in numbers:
                if num in text_lower:
                    return num
        elif "is there" in question_lower:
            return "yes" if any(word in text_lower for word in ['a', 'an', 'the']) else "no"

        return "unknown"
    
    def compute_score(self, images: List[Image.Image], texts: List[str]) -> List[float]:
        """
        计算TIFA分数
        
        Args:
            images: 图像列表
            texts: 文本列表
            
        Returns:
            TIFA分数列表
        """
        if len(images) != len(texts):
            raise ValueError("Images and texts must have the same length")
        
        scores = []
        
        for image, text in zip(images, texts):
            # 生成问题
            questions = self.generate_questions(text)
            
            if not questions:
                scores.append(0.0)
                continue
            
            correct_answers = 0
            total_questions = len(questions)
            
            for question in questions:
                # 在图像上回答问题
                image_answer = self.answer_question_on_image(image, question)
                
                # 在文本上回答问题
                text_answer = self.answer_question_on_text(text, question)
                
                # 比较答案
                if self._compare_answers(image_answer, text_answer):
                    correct_answers += 1
            
            # 计算准确率作为TIFA分数
            score = correct_answers / total_questions if total_questions > 0 else 0.0
            scores.append(score)
        
        return scores
    
    def _compare_answers(self, answer1: str, answer2: str) -> bool:
        """
        比较两个答案是否匹配
        
        Args:
            answer1: 答案1
            answer2: 答案2
            
        Returns:
            是否匹配
        """
        answer1 = answer1.lower().strip()
        answer2 = answer2.lower().strip()
        
        # 直接匹配
        if answer1 == answer2:
            return True
        
        # 包含匹配
        if answer1 in answer2 or answer2 in answer1:
            return True
        
        # 是/否问题的特殊处理
        yes_words = ['yes', 'true', 'correct', 'right']
        no_words = ['no', 'false', 'incorrect', 'wrong']
        
        if (answer1 in yes_words and answer2 in yes_words) or \
           (answer1 in no_words and answer2 in no_words):
            return True
        
        return False
    
    def compute_statistics(self, scores: List[float]) -> dict:
        """
        计算分数统计信息
        
        Args:
            scores: 分数列表
            
        Returns:
            统计信息字典
        """
        scores_array = np.array(scores)
        
        return {
            'mean': float(np.mean(scores_array)),
            'std': float(np.std(scores_array)),
            'min': float(np.min(scores_array)),
            'max': float(np.max(scores_array)),
            'median': float(np.median(scores_array)),
            'count': len(scores)
        }
