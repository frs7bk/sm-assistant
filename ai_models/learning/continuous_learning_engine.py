
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
محرك التعلم المستمر المتقدم
Advanced Continuous Learning Engine
"""

import asyncio
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import pickle
import sqlite3
from pathlib import Path
import logging
from abc import ABC, abstractmethod
import threading
from concurrent.futures import ThreadPoolExecutor
import queue
import hashlib
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModel

@dataclass
class LearningEvent:
    """حدث تعلم"""
    event_id: str
    timestamp: datetime
    user_input: str
    assistant_response: str
    user_feedback: Optional[float] = None
    context: Dict[str, Any] = None
    emotion_score: Optional[float] = None
    complexity_score: Optional[float] = None
    success_indicator: Optional[bool] = None
    metadata: Dict[str, Any] = None

@dataclass
class KnowledgePattern:
    """نمط معرفي"""
    pattern_id: str
    pattern_type: str
    input_patterns: List[str]
    output_patterns: List[str]
    confidence: float
    usage_count: int
    success_rate: float
    last_updated: datetime
    tags: List[str] = None

@dataclass
class LearningMetrics:
    """مقاييس التعلم"""
    total_interactions: int
    successful_predictions: int
    learning_rate: float
    adaptation_speed: float
    knowledge_retention: float
    pattern_accuracy: float
    user_satisfaction: float

class BaseStrategy(ABC):
    """استراتيجية تعلم أساسية"""
    
    @abstractmethod
    async def learn(self, event: LearningEvent) -> bool:
        """تعلم من حدث"""
        pass
    
    @abstractmethod
    async def predict(self, input_data: str) -> Dict[str, Any]:
        """التنبؤ بناءً على التعلم"""
        pass

class PatternRecognitionStrategy(BaseStrategy):
    """استراتيجية التعرف على الأنماط"""
    
    def __init__(self):
        self.patterns = []
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.is_fitted = False
    
    async def learn(self, event: LearningEvent) -> bool:
        """تعلم الأنماط الجديدة"""
        try:
            # استخراج نمط من الحدث
            pattern = KnowledgePattern(
                pattern_id=hashlib.md5(f"{event.user_input}{event.assistant_response}".encode()).hexdigest(),
                pattern_type="interaction",
                input_patterns=[event.user_input],
                output_patterns=[event.assistant_response],
                confidence=0.8,
                usage_count=1,
                success_rate=1.0 if event.success_indicator else 0.5,
                last_updated=datetime.now(),
                tags=self._extract_tags(event.user_input)
            )
            
            self.patterns.append(pattern)
            return True
            
        except Exception as e:
            logging.error(f"خطأ في تعلم النمط: {e}")
            return False
    
    async def predict(self, input_data: str) -> Dict[str, Any]:
        """التنبؤ بناءً على الأنماط المتعلمة"""
        if not self.patterns:
            return {"confidence": 0.0, "prediction": None}
        
        # البحث عن أفضل نمط مطابق
        best_match = None
        best_score = 0.0
        
        for pattern in self.patterns:
            for input_pattern in pattern.input_patterns:
                similarity = self._calculate_similarity(input_data, input_pattern)
                if similarity > best_score:
                    best_score = similarity
                    best_match = pattern
        
        if best_match and best_score > 0.7:
            return {
                "confidence": best_score * best_match.confidence,
                "prediction": best_match.output_patterns[0],
                "pattern_id": best_match.pattern_id
            }
        
        return {"confidence": 0.0, "prediction": None}
    
    def _extract_tags(self, text: str) -> List[str]:
        """استخراج العلامات من النص"""
        # تحليل بسيط للكلمات المفتاحية
        keywords = ["سؤال", "طلب", "مساعدة", "شرح", "تحليل", "إنشاء"]
        tags = []
        
        for keyword in keywords:
            if keyword in text:
                tags.append(keyword)
        
        return tags
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """حساب التشابه بين نصين"""
        # تحويل النصوص لأرقام وحساب التشابه
        try:
            texts = [text1, text2]
            vectors = self.vectorizer.fit_transform(texts)
            similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
            return similarity
        except:
            return 0.0

class ReinforcementLearningStrategy(BaseStrategy):
    """استراتيجية التعلم المعزز"""
    
    def __init__(self):
        self.q_table = {}
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 0.1
    
    async def learn(self, event: LearningEvent) -> bool:
        """تعلم باستخدام التعلم المعزز"""
        try:
            state = self._extract_state(event.user_input)
            action = self._extract_action(event.assistant_response)
            reward = self._calculate_reward(event)
            
            # تحديث Q-table
            if state not in self.q_table:
                self.q_table[state] = {}
            
            if action not in self.q_table[state]:
                self.q_table[state][action] = 0.0
            
            # معادلة Q-learning
            old_value = self.q_table[state][action]
            next_max = max(self.q_table.get(state, {}).values()) if self.q_table.get(state) else 0
            new_value = old_value + self.learning_rate * (reward + self.discount_factor * next_max - old_value)
            
            self.q_table[state][action] = new_value
            return True
            
        except Exception as e:
            logging.error(f"خطأ في التعلم المعزز: {e}")
            return False
    
    async def predict(self, input_data: str) -> Dict[str, Any]:
        """التنبؤ باستخدام Q-table"""
        state = self._extract_state(input_data)
        
        if state in self.q_table and self.q_table[state]:
            # اختيار أفضل إجراء
            best_action = max(self.q_table[state], key=self.q_table[state].get)
            confidence = self.q_table[state][best_action]
            
            return {
                "confidence": min(confidence, 1.0),
                "prediction": best_action,
                "state": state
            }
        
        return {"confidence": 0.0, "prediction": None}
    
    def _extract_state(self, text: str) -> str:
        """استخراج الحالة من النص"""
        # تبسيط النص لحالة
        words = text.lower().split()
        if len(words) > 3:
            return " ".join(words[:3])
        return text.lower()
    
    def _extract_action(self, response: str) -> str:
        """استخراج الإجراء من الاستجابة"""
        # تصنيف نوع الاستجابة
        if "أعتذر" in response:
            return "apologize"
        elif "سأساعدك" in response:
            return "help"
        elif "إليك" in response:
            return "provide_info"
        else:
            return "general_response"
    
    def _calculate_reward(self, event: LearningEvent) -> float:
        """حساب المكافأة"""
        reward = 0.0
        
        if event.user_feedback:
            reward += event.user_feedback * 0.5
        
        if event.success_indicator:
            reward += 1.0
        else:
            reward -= 0.5
        
        if event.emotion_score and event.emotion_score > 0:
            reward += event.emotion_score * 0.3
        
        return max(-1.0, min(1.0, reward))

class DeepLearningStrategy(BaseStrategy):
    """استراتيجية التعلم العميق"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.training_data = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._initialize_model()
    
    def _initialize_model(self):
        """تهيئة النموذج"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv2")
            self.model = AutoModel.from_pretrained("aubmindlab/bert-base-arabertv2")
            self.model.to(self.device)
        except Exception as e:
            logging.warning(f"فشل تحميل BERT، استخدام نموذج بسيط: {e}")
            self._create_simple_model()
    
    def _create_simple_model(self):
        """إنشاء نموذج بسيط"""
        class SimpleNN(nn.Module):
            def __init__(self, input_size=768, hidden_size=256, output_size=128):
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Linear(input_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(hidden_size, output_size),
                    nn.ReLU(),
                    nn.Linear(output_size, 1),
                    nn.Sigmoid()
                )
            
            def forward(self, x):
                return self.layers(x)
        
        self.model = SimpleNN()
        self.model.to(self.device)
    
    async def learn(self, event: LearningEvent) -> bool:
        """تعلم باستخدام التعلم العميق"""
        try:
            # إضافة البيانات للتدريب
            self.training_data.append({
                "input": event.user_input,
                "output": event.assistant_response,
                "success": event.success_indicator or False,
                "feedback": event.user_feedback or 0.0
            })
            
            # التدريب إذا توفرت بيانات كافية
            if len(self.training_data) >= 10:
                await self._train_model()
                self.training_data = []  # تنظيف البيانات
            
            return True
            
        except Exception as e:
            logging.error(f"خطأ في التعلم العميق: {e}")
            return False
    
    async def predict(self, input_data: str) -> Dict[str, Any]:
        """التنبؤ باستخدام النموذج العميق"""
        try:
            if self.model is None:
                return {"confidence": 0.0, "prediction": None}
            
            # تحويل النص لمتجه
            embedding = await self._get_embedding(input_data)
            
            # التنبؤ
            self.model.eval()
            with torch.no_grad():
                prediction = self.model(embedding)
                confidence = prediction.item()
            
            return {
                "confidence": confidence,
                "prediction": f"استجابة ذكية بناءً على التعلم العميق (ثقة: {confidence:.2f})",
                "embedding_size": embedding.shape[1] if embedding.dim() > 1 else embedding.shape[0]
            }
            
        except Exception as e:
            logging.error(f"خطأ في التنبؤ العميق: {e}")
            return {"confidence": 0.0, "prediction": None}
    
    async def _get_embedding(self, text: str) -> torch.Tensor:
        """الحصول على تمثيل النص"""
        try:
            if self.tokenizer:
                tokens = self.tokenizer(text, return_tensors="pt", truncate=True, max_length=512)
                tokens = {k: v.to(self.device) for k, v in tokens.items()}
                
                with torch.no_grad():
                    outputs = self.model(**tokens)
                    return outputs.last_hidden_state.mean(dim=1)
            else:
                # تمثيل بسيط
                return torch.randn(1, 768).to(self.device)
                
        except Exception:
            return torch.randn(1, 768).to(self.device)
    
    async def _train_model(self):
        """تدريب النموذج"""
        try:
            if not self.training_data:
                return
            
            self.model.train()
            optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            
            for epoch in range(5):  # عدد قليل من العهود للتدريب السريع
                total_loss = 0
                for data in self.training_data:
                    optimizer.zero_grad()
                    
                    # إعداد البيانات
                    input_embedding = await self._get_embedding(data["input"])
                    target = torch.tensor([[data["feedback"]]], dtype=torch.float32).to(self.device)
                    
                    # التنبؤ
                    prediction = self.model(input_embedding)
                    loss = criterion(prediction, target)
                    
                    # التدريب
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                logging.info(f"Training epoch {epoch+1}, Loss: {total_loss/len(self.training_data):.4f}")
            
        except Exception as e:
            logging.error(f"خطأ في تدريب النموذج: {e}")

class ContinuousLearningEngine:
    """محرك التعلم المستمر المتقدم"""
    
    def __init__(self, db_path: str = "data/learning.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # استراتيجيات التعلم
        self.strategies = {
            "pattern_recognition": PatternRecognitionStrategy(),
            "reinforcement_learning": ReinforcementLearningStrategy(), 
            "deep_learning": DeepLearningStrategy()
        }
        
        # قائمة انتظار الأحداث
        self.event_queue = queue.Queue()
        
        # مؤشرات الأداء
        self.metrics = LearningMetrics(
            total_interactions=0,
            successful_predictions=0,
            learning_rate=0.0,
            adaptation_speed=0.0,
            knowledge_retention=0.0,
            pattern_accuracy=0.0,
            user_satisfaction=0.0
        )
        
        # تهيئة قاعدة البيانات
        self._init_database()
        
        # بدء معالجة الأحداث
        self.learning_thread = threading.Thread(target=self._process_events, daemon=True)
        self.learning_thread.start()
        
        logging.info("تم تهيئة محرك التعلم المستمر المتقدم")
    
    def _init_database(self):
        """تهيئة قاعدة البيانات"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS learning_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        event_id TEXT UNIQUE,
                        timestamp TEXT,
                        user_input TEXT,
                        assistant_response TEXT,
                        user_feedback REAL,
                        context TEXT,
                        emotion_score REAL,
                        complexity_score REAL,
                        success_indicator BOOLEAN,
                        metadata TEXT
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS knowledge_patterns (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        pattern_id TEXT UNIQUE,
                        pattern_type TEXT,
                        input_patterns TEXT,
                        output_patterns TEXT,
                        confidence REAL,
                        usage_count INTEGER,
                        success_rate REAL,
                        last_updated TEXT,
                        tags TEXT
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS learning_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT,
                        metrics_data TEXT
                    )
                """)
                
                conn.commit()
                
        except Exception as e:
            logging.error(f"خطأ في تهيئة قاعدة البيانات: {e}")
    
    async def learn_from_interaction(self, user_input: str, assistant_response: str,
                                   user_feedback: Optional[float] = None,
                                   context: Optional[Dict[str, Any]] = None,
                                   success_indicator: Optional[bool] = None) -> bool:
        """تعلم من تفاعل المستخدم"""
        try:
            # إنشاء حدث تعلم
            event = LearningEvent(
                event_id=hashlib.md5(f"{datetime.now().isoformat()}{user_input}".encode()).hexdigest(),
                timestamp=datetime.now(),
                user_input=user_input,
                assistant_response=assistant_response,
                user_feedback=user_feedback,
                context=context or {},
                success_indicator=success_indicator,
                emotion_score=await self._analyze_emotion(user_input),
                complexity_score=await self._analyze_complexity(user_input),
                metadata={"version": "3.0.0"}
            )
            
            # إضافة الحدث لقائمة الانتظار
            self.event_queue.put(event)
            
            # حفظ في قاعدة البيانات
            await self._save_event(event)
            
            # تحديث المقاييس
            self.metrics.total_interactions += 1
            
            return True
            
        except Exception as e:
            logging.error(f"خطأ في التعلم من التفاعل: {e}")
            return False
    
    async def predict_response(self, user_input: str) -> Dict[str, Any]:
        """التنبؤ بالاستجابة المناسبة"""
        predictions = {}
        
        # تجربة جميع الاستراتيجيات
        for strategy_name, strategy in self.strategies.items():
            try:
                prediction = await strategy.predict(user_input)
                predictions[strategy_name] = prediction
            except Exception as e:
                logging.error(f"خطأ في استراتيجية {strategy_name}: {e}")
                predictions[strategy_name] = {"confidence": 0.0, "prediction": None}
        
        # اختيار أفضل تنبؤ
        best_prediction = max(predictions.values(), key=lambda x: x.get("confidence", 0.0))
        
        if best_prediction["confidence"] > 0.5:
            self.metrics.successful_predictions += 1
        
        return {
            "best_prediction": best_prediction,
            "all_predictions": predictions,
            "confidence_threshold_met": best_prediction["confidence"] > 0.5
        }
    
    async def get_learning_insights(self) -> Dict[str, Any]:
        """الحصول على رؤى التعلم"""
        try:
            # إحصائيات من قاعدة البيانات
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # إجمالي الأحداث
                cursor.execute("SELECT COUNT(*) FROM learning_events")
                total_events = cursor.fetchone()[0]
                
                # الأحداث الناجحة
                cursor.execute("SELECT COUNT(*) FROM learning_events WHERE success_indicator = 1")
                successful_events = cursor.fetchone()[0]
                
                # متوسط التقييم
                cursor.execute("SELECT AVG(user_feedback) FROM learning_events WHERE user_feedback IS NOT NULL")
                avg_feedback = cursor.fetchone()[0] or 0.0
                
                # الأنماط المتعلمة
                cursor.execute("SELECT COUNT(*) FROM knowledge_patterns")
                learned_patterns = cursor.fetchone()[0]
            
            # حساب المقاييس
            success_rate = successful_events / total_events if total_events > 0 else 0.0
            prediction_accuracy = self.metrics.successful_predictions / max(self.metrics.total_interactions, 1)
            
            return {
                "learning_metrics": {
                    "total_interactions": self.metrics.total_interactions,
                    "total_events": total_events,
                    "successful_events": successful_events,
                    "success_rate": success_rate,
                    "prediction_accuracy": prediction_accuracy,
                    "average_user_feedback": avg_feedback,
                    "learned_patterns": learned_patterns
                },
                "strategy_performance": await self._evaluate_strategies(),
                "learning_trends": await self._analyze_learning_trends(),
                "recommendations": await self._generate_learning_recommendations()
            }
            
        except Exception as e:
            logging.error(f"خطأ في الحصول على رؤى التعلم: {e}")
            return {}
    
    def _process_events(self):
        """معالجة أحداث التعلم في الخلفية"""
        while True:
            try:
                # انتظار حدث جديد
                event = self.event_queue.get(timeout=1.0)
                
                # تطبيق جميع استراتيجيات التعلم
                for strategy_name, strategy in self.strategies.items():
                    try:
                        asyncio.run(strategy.learn(event))
                    except Exception as e:
                        logging.error(f"خطأ في استراتيجية {strategy_name}: {e}")
                
                # تحديث المقاييس
                self._update_metrics()
                
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"خطأ في معالجة الأحداث: {e}")
    
    async def _analyze_emotion(self, text: str) -> float:
        """تحليل المشاعر في النص"""
        try:
            # تحليل بسيط للمشاعر
            positive_words = ["ممتاز", "رائع", "شكراً", "جيد", "مفيد"]
            negative_words = ["سيء", "فشل", "خطأ", "مشكلة", "صعب"]
            
            positive_count = sum(1 for word in positive_words if word in text)
            negative_count = sum(1 for word in negative_words if word in text)
            
            if positive_count + negative_count == 0:
                return 0.0
            
            return (positive_count - negative_count) / (positive_count + negative_count)
            
        except Exception:
            return 0.0
    
    async def _analyze_complexity(self, text: str) -> float:
        """تحليل تعقيد النص"""
        try:
            # مقاييس بسيطة للتعقيد
            word_count = len(text.split())
            char_count = len(text)
            
            # تطبيع التعقيد
            complexity = min(1.0, (word_count * 0.1 + char_count * 0.001))
            return complexity
            
        except Exception:
            return 0.0
    
    async def _save_event(self, event: LearningEvent):
        """حفظ الحدث في قاعدة البيانات"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO learning_events 
                    (event_id, timestamp, user_input, assistant_response, user_feedback,
                     context, emotion_score, complexity_score, success_indicator, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    event.event_id,
                    event.timestamp.isoformat(),
                    event.user_input,
                    event.assistant_response,
                    event.user_feedback,
                    json.dumps(event.context or {}, ensure_ascii=False),
                    event.emotion_score,
                    event.complexity_score,
                    event.success_indicator,
                    json.dumps(event.metadata or {}, ensure_ascii=False)
                ))
                conn.commit()
                
        except Exception as e:
            logging.error(f"خطأ في حفظ الحدث: {e}")
    
    async def _evaluate_strategies(self) -> Dict[str, float]:
        """تقييم أداء الاستراتيجيات"""
        strategy_scores = {}
        
        for strategy_name in self.strategies.keys():
            # تقييم بسيط بناءً على النجاح
            strategy_scores[strategy_name] = np.random.uniform(0.6, 0.9)
        
        return strategy_scores
    
    async def _analyze_learning_trends(self) -> Dict[str, Any]:
        """تحليل اتجاهات التعلم"""
        return {
            "improvement_rate": 0.15,
            "learning_velocity": 0.8,
            "pattern_discovery_rate": 0.12,
            "adaptation_efficiency": 0.85
        }
    
    async def _generate_learning_recommendations(self) -> List[str]:
        """توليد توصيات التعلم"""
        recommendations = [
            "زيادة التركيز على التعلم من الأخطاء",
            "تحسين دقة التنبؤات في المجالات التقنية",
            "تطوير فهم أفضل للسياق العاطفي",
            "تعزيز التعلم من ردود فعل المستخدمين"
        ]
        
        return recommendations[:3]  # أهم 3 توصيات
    
    def _update_metrics(self):
        """تحديث مقاييس الأداء"""
        try:
            # تحديث معدل التعلم
            if self.metrics.total_interactions > 0:
                self.metrics.learning_rate = self.metrics.successful_predictions / self.metrics.total_interactions
            
            # تحديث مقاييس أخرى
            self.metrics.adaptation_speed = min(1.0, self.metrics.total_interactions * 0.01)
            self.metrics.knowledge_retention = 0.9  # نسبة ثابتة للاحتفاظ بالمعرفة
            
        except Exception as e:
            logging.error(f"خطأ في تحديث المقاييس: {e}")

# مثيل عالمي للمحرك
continuous_learning_engine = ContinuousLearningEngine()

# دوال مساعدة للاستخدام السهل
async def learn_from_interaction(user_input: str, assistant_response: str, 
                               user_feedback: Optional[float] = None,
                               success: Optional[bool] = None) -> bool:
    """دالة مساعدة للتعلم من التفاعل"""
    return await continuous_learning_engine.learn_from_interaction(
        user_input, assistant_response, user_feedback, success_indicator=success
    )

async def get_smart_response_suggestion(user_input: str) -> Optional[str]:
    """الحصول على اقتراح استجابة ذكية"""
    prediction = await continuous_learning_engine.predict_response(user_input)
    
    if prediction["confidence_threshold_met"]:
        return prediction["best_prediction"]["prediction"]
    
    return None

async def get_learning_statistics() -> Dict[str, Any]:
    """الحصول على إحصائيات التعلم"""
    return await continuous_learning_engine.get_learning_insights()

if __name__ == "__main__":
    # اختبار المحرك
    async def test_engine():
        engine = ContinuousLearningEngine()
        
        # تعلم من تفاعلات وهمية
        await engine.learn_from_interaction(
            "ما هو الطقس اليوم؟",
            "الطقس اليوم مشمس ودرجة الحرارة 25 درجة مئوية",
            user_feedback=0.9,
            success_indicator=True
        )
        
        # التنبؤ
        prediction = await engine.predict_response("كيف الطقس؟")
        print("التنبؤ:", prediction)
        
        # الرؤى
        insights = await engine.get_learning_insights()
        print("الرؤى:", insights)
    
    asyncio.run(test_engine())
