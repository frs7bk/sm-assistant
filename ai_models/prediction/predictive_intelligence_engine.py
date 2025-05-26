
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
نظام الذكاء التنبؤي والتوصيات الذكية
Predictive Intelligence and Smart Recommendations Engine
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import json
import sqlite3
from dataclasses import dataclass, asdict
import threading
import queue
from collections import defaultdict, deque
import pickle
import hashlib

@dataclass
class Prediction:
    """تنبؤ"""
    id: str
    user_id: str
    prediction_type: str
    predicted_value: Any
    confidence: float
    probability: float
    context: Dict[str, Any]
    features_used: List[str]
    timestamp: datetime
    expires_at: datetime
    accuracy_score: Optional[float] = None

@dataclass
class Recommendation:
    """توصية"""
    id: str
    user_id: str
    recommendation_type: str
    title: str
    description: str
    action: str
    priority: float
    relevance_score: float
    context: Dict[str, Any]
    timestamp: datetime
    expires_at: datetime
    is_personalized: bool = True

class PredictiveIntelligenceEngine:
    """محرك الذكاء التنبؤي والتوصيات"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # قاعدة بيانات التنبؤات
        self.predictions_db_path = Path("data/predictions/predictions.db")
        self.predictions_db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_predictions_database()
        
        # نماذج التنبؤ
        self.prediction_models = {
            "behavior": None,        # نموذج التنبؤ بالسلوك
            "preferences": None,     # نموذج التنبؤ بالتفضيلات
            "needs": None,          # نموذج التنبؤ بالاحتياجات
            "mood": None,           # نموذج التنبؤ بالحالة المزاجية
            "usage": None,          # نموذج التنبؤ بأنماط الاستخدام
            "content": None         # نموذج التوصية بالمحتوى
        }
        
        # بيانات المستخدمين
        self.user_profiles = {}
        self.user_behavior_history = defaultdict(lambda: deque(maxlen=1000))
        self.user_interaction_patterns = defaultdict(dict)
        
        # نماذج التعلم الآلي
        self.ml_models_path = Path("data/models/prediction")
        self.ml_models_path.mkdir(parents=True, exist_ok=True)
        
        # إعدادات التنبؤ
        self.prediction_settings = {
            "prediction_horizon": 24,    # ساعات
            "confidence_threshold": 0.7,
            "update_frequency": 3600,    # ثواني
            "max_predictions_per_user": 20,
            "recommendation_limit": 10,
            "personalization_level": 0.8
        }
        
        # قائمة انتظار التنبؤات
        self.prediction_queue = queue.Queue()
        self.prediction_worker = None
        
        # إحصائيات التنبؤ
        self.prediction_stats = {
            "total_predictions": 0,
            "accurate_predictions": 0,
            "recommendations_generated": 0,
            "user_engagement": 0.0,
            "model_accuracy": {}
        }
        
        # أنواع التنبؤات المدعومة
        self.supported_predictions = {
            "next_action": "التنبؤ بالإجراء التالي",
            "mood_change": "التنبؤ بتغيير المزاج",
            "content_preference": "التنبؤ بتفضيلات المحتوى",
            "usage_pattern": "التنبؤ بنمط الاستخدام",
            "optimal_timing": "التنبؤ بالتوقيت الأمثل",
            "feature_adoption": "التنبؤ بتبني الميزات"
        }
        
        self._start_prediction_worker()
        self._initialize_prediction_models()
    
    def _init_predictions_database(self):
        """تهيئة قاعدة بيانات التنبؤات"""
        conn = sqlite3.connect(self.predictions_db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                prediction_type TEXT NOT NULL,
                predicted_value TEXT,
                confidence REAL NOT NULL,
                probability REAL NOT NULL,
                context TEXT,
                features_used TEXT,
                timestamp REAL NOT NULL,
                expires_at REAL NOT NULL,
                accuracy_score REAL
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS recommendations (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                recommendation_type TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT,
                action TEXT,
                priority REAL NOT NULL,
                relevance_score REAL NOT NULL,
                context TEXT,
                timestamp REAL NOT NULL,
                expires_at REAL NOT NULL,
                is_personalized INTEGER DEFAULT 1,
                interaction_count INTEGER DEFAULT 0,
                effectiveness_score REAL
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id TEXT PRIMARY KEY,
                profile_data TEXT NOT NULL,
                last_updated REAL NOT NULL,
                preferences TEXT,
                behavior_summary TEXT
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_predictions_user ON predictions(user_id, timestamp);
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_recommendations_user ON recommendations(user_id, priority);
        """)
        
        conn.commit()
        conn.close()
    
    def _initialize_prediction_models(self):
        """تهيئة نماذج التنبؤ"""
        try:
            # تحميل النماذج المحفوظة أو إنشاء نماذج جديدة
            self._load_or_create_behavior_model()
            self._load_or_create_preferences_model()
            self._load_or_create_needs_model()
            self._load_or_create_mood_model()
            self._load_or_create_usage_model()
            self._load_or_create_content_model()
            
            self.logger.info("✅ تم تهيئة جميع نماذج التنبؤ")
            
        except Exception as e:
            self.logger.error(f"خطأ في تهيئة نماذج التنبؤ: {e}")
    
    def _load_or_create_behavior_model(self):
        """تحميل أو إنشاء نموذج التنبؤ بالسلوك"""
        try:
            model_path = self.ml_models_path / "behavior_model.pkl"
            
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    self.prediction_models["behavior"] = pickle.load(f)
                self.logger.info("تم تحميل نموذج التنبؤ بالسلوك")
            else:
                # إنشاء نموذج جديد
                self.prediction_models["behavior"] = self._create_behavior_model()
                self.logger.info("تم إنشاء نموذج التنبؤ بالسلوك")
                
        except Exception as e:
            self.logger.error(f"خطأ في نموذج السلوك: {e}")
            self.prediction_models["behavior"] = None
    
    def _create_behavior_model(self):
        """إنشاء نموذج التنبؤ بالسلوك"""
        # محاكاة نموذج تعلم آلي متقدم
        class BehaviorPredictionModel:
            def __init__(self):
                self.features = [
                    "time_of_day", "day_of_week", "session_duration",
                    "interaction_frequency", "last_action", "mood_state"
                ]
                self.trained = False
            
            def predict(self, features):
                # محاكاة تنبؤ
                actions = ["search", "create", "analyze", "communicate", "learn"]
                probabilities = np.random.dirichlet([1, 1, 1, 1, 1])
                return {action: prob for action, prob in zip(actions, probabilities)}
            
            def update(self, features, actual_outcome):
                # محاكاة تحديث النموذج
                self.trained = True
        
        return BehaviorPredictionModel()
    
    def _load_or_create_preferences_model(self):
        """تحميل أو إنشاء نموذج التنبؤ بالتفضيلات"""
        # مشابه للنموذج السابق
        self.prediction_models["preferences"] = self._create_preferences_model()
        
    def _create_preferences_model(self):
        """إنشاء نموذج التنبؤ بالتفضيلات"""
        class PreferencesPredictionModel:
            def predict(self, user_history, content_features):
                # محاكاة التنبؤ بالتفضيلات
                preferences = {
                    "content_type": np.random.choice(["educational", "entertainment", "productivity"]),
                    "interaction_style": np.random.choice(["detailed", "brief", "interactive"]),
                    "difficulty_level": np.random.uniform(0.3, 0.9)
                }
                return preferences
        
        return PreferencesPredictionModel()
    
    def _load_or_create_needs_model(self):
        """تحميل أو إنشاء نموذج التنبؤ بالاحتياجات"""
        self.prediction_models["needs"] = self._create_needs_model()
    
    def _create_needs_model(self):
        """إنشاء نموذج التنبؤ بالاحتياجات"""
        class NeedsPredictionModel:
            def predict(self, user_context, time_context):
                needs = {
                    "information": np.random.uniform(0.2, 0.8),
                    "entertainment": np.random.uniform(0.1, 0.6),
                    "productivity": np.random.uniform(0.3, 0.9),
                    "social_interaction": np.random.uniform(0.1, 0.5),
                    "learning": np.random.uniform(0.2, 0.7)
                }
                return needs
        
        return NeedsPredictionModel()
    
    def _load_or_create_mood_model(self):
        """تحميل أو إنشاء نموذج التنبؤ بالحالة المزاجية"""
        self.prediction_models["mood"] = self._create_mood_model()
    
    def _create_mood_model(self):
        """إنشاء نموذج التنبؤ بالحالة المزاجية"""
        class MoodPredictionModel:
            def predict(self, current_mood, context_factors):
                mood_transitions = {
                    "happiness": {"happiness": 0.7, "neutral": 0.2, "sadness": 0.1},
                    "sadness": {"sadness": 0.5, "neutral": 0.3, "happiness": 0.2},
                    "neutral": {"neutral": 0.6, "happiness": 0.25, "sadness": 0.15}
                }
                
                current = current_mood.get("primary_emotion", "neutral")
                transitions = mood_transitions.get(current, mood_transitions["neutral"])
                
                return transitions
        
        return MoodPredictionModel()
    
    def _load_or_create_usage_model(self):
        """تحميل أو إنشاء نموذج التنبؤ بأنماط الاستخدام"""
        self.prediction_models["usage"] = self._create_usage_model()
    
    def _create_usage_model(self):
        """إنشاء نموذج التنبؤ بأنماط الاستخدام"""
        class UsagePredictionModel:
            def predict(self, historical_usage, time_features):
                usage_patterns = {
                    "peak_hours": [9, 14, 20],  # الساعات الأكثر نشاطاً
                    "session_duration": np.random.uniform(15, 120),  # دقائق
                    "feature_usage": {
                        "text_chat": np.random.uniform(0.6, 0.9),
                        "voice_commands": np.random.uniform(0.2, 0.5),
                        "image_analysis": np.random.uniform(0.1, 0.4)
                    }
                }
                return usage_patterns
        
        return UsagePredictionModel()
    
    def _load_or_create_content_model(self):
        """تحميل أو إنشاء نموذج التوصية بالمحتوى"""
        self.prediction_models["content"] = self._create_content_model()
    
    def _create_content_model(self):
        """إنشاء نموذج التوصية بالمحتوى"""
        class ContentRecommendationModel:
            def recommend(self, user_preferences, available_content):
                # محاكاة توصيات المحتوى
                recommendations = [
                    {
                        "type": "tutorial",
                        "title": "تعلم استخدام الميزات المتقدمة",
                        "relevance": np.random.uniform(0.7, 0.95)
                    },
                    {
                        "type": "tip",
                        "title": "نصيحة لتحسين الإنتاجية",
                        "relevance": np.random.uniform(0.6, 0.8)
                    }
                ]
                return recommendations
        
        return ContentRecommendationModel()
    
    async def predict_user_behavior(self, user_id: str, context: Dict[str, Any] = None) -> List[Prediction]:
        """التنبؤ بسلوك المستخدم"""
        try:
            if not self.prediction_models["behavior"]:
                return []
            
            current_time = datetime.now()
            user_profile = await self._get_user_profile(user_id)
            user_history = list(self.user_behavior_history[user_id])
            
            # استخراج الميزات
            features = self._extract_behavioral_features(user_profile, user_history, context)
            
            # التنبؤ
            behavior_predictions = self.prediction_models["behavior"].predict(features)
            
            predictions = []
            for action, probability in behavior_predictions.items():
                if probability > self.prediction_settings["confidence_threshold"]:
                    prediction_id = hashlib.md5(
                        f"{user_id}_{action}_{current_time}".encode()
                    ).hexdigest()
                    
                    prediction = Prediction(
                        id=prediction_id,
                        user_id=user_id,
                        prediction_type="next_action",
                        predicted_value=action,
                        confidence=probability,
                        probability=probability,
                        context=context or {},
                        features_used=list(features.keys()),
                        timestamp=current_time,
                        expires_at=current_time + timedelta(hours=self.prediction_settings["prediction_horizon"])
                    )
                    
                    predictions.append(prediction)
            
            # حفظ التنبؤات
            for prediction in predictions:
                await self._save_prediction(prediction)
            
            self.prediction_stats["total_predictions"] += len(predictions)
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"خطأ في التنبؤ بالسلوك: {e}")
            return []
    
    async def predict_user_needs(self, user_id: str, context: Dict[str, Any] = None) -> List[Prediction]:
        """التنبؤ باحتياجات المستخدم"""
        try:
            if not self.prediction_models["needs"]:
                return []
            
            user_profile = await self._get_user_profile(user_id)
            current_time = datetime.now()
            
            # تحليل السياق الزمني
            time_context = {
                "hour": current_time.hour,
                "day_of_week": current_time.weekday(),
                "is_weekend": current_time.weekday() >= 5
            }
            
            # التنبؤ بالاحتياجات
            needs_scores = self.prediction_models["needs"].predict(user_profile, time_context)
            
            predictions = []
            for need, score in needs_scores.items():
                if score > 0.5:
                    prediction_id = hashlib.md5(
                        f"{user_id}_{need}_{current_time}".encode()
                    ).hexdigest()
                    
                    prediction = Prediction(
                        id=prediction_id,
                        user_id=user_id,
                        prediction_type="user_need",
                        predicted_value=need,
                        confidence=score,
                        probability=score,
                        context={**context or {}, **time_context},
                        features_used=["user_profile", "time_context"],
                        timestamp=current_time,
                        expires_at=current_time + timedelta(hours=6)
                    )
                    
                    predictions.append(prediction)
            
            for prediction in predictions:
                await self._save_prediction(prediction)
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"خطأ في التنبؤ بالاحتياجات: {e}")
            return []
    
    async def generate_smart_recommendations(self, user_id: str, context: Dict[str, Any] = None) -> List[Recommendation]:
        """توليد التوصيات الذكية"""
        try:
            current_time = datetime.now()
            user_profile = await self._get_user_profile(user_id)
            
            # الحصول على التنبؤات الحديثة
            behavior_predictions = await self.predict_user_behavior(user_id, context)
            needs_predictions = await self.predict_user_needs(user_id, context)
            
            recommendations = []
            
            # توصيات بناءً على التنبؤ بالسلوك
            for prediction in behavior_predictions:
                if prediction.predicted_value == "learn":
                    rec = await self._create_learning_recommendation(user_id, prediction, context)
                    recommendations.append(rec)
                elif prediction.predicted_value == "create":
                    rec = await self._create_creative_recommendation(user_id, prediction, context)
                    recommendations.append(rec)
                elif prediction.predicted_value == "analyze":
                    rec = await self._create_analysis_recommendation(user_id, prediction, context)
                    recommendations.append(rec)
            
            # توصيات بناءً على التنبؤ بالاحتياجات
            for prediction in needs_predictions:
                if prediction.predicted_value == "productivity":
                    rec = await self._create_productivity_recommendation(user_id, prediction, context)
                    recommendations.append(rec)
                elif prediction.predicted_value == "entertainment":
                    rec = await self._create_entertainment_recommendation(user_id, prediction, context)
                    recommendations.append(rec)
            
            # ترتيب التوصيات حسب الأولوية
            recommendations.sort(key=lambda x: x.priority, reverse=True)
            
            # الحد من عدد التوصيات
            recommendations = recommendations[:self.prediction_settings["recommendation_limit"]]
            
            # حفظ التوصيات
            for recommendation in recommendations:
                await self._save_recommendation(recommendation)
            
            self.prediction_stats["recommendations_generated"] += len(recommendations)
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"خطأ في توليد التوصيات: {e}")
            return []
    
    async def _create_learning_recommendation(self, user_id: str, prediction: Prediction, 
                                           context: Dict[str, Any]) -> Recommendation:
        """إنشاء توصية تعليمية"""
        
        learning_topics = [
            "تعلم ميزات الذكاء الاصطناعي الجديدة",
            "كيفية تحسين استخدام المساعد الذكي",
            "نصائح للاستفادة القصوى من التحليل العاطفي",
            "استراتيجيات التفاعل الفعال مع الذكاء الاصطناعي"
        ]
        
        topic = np.random.choice(learning_topics)
        
        rec_id = hashlib.md5(f"{user_id}_learning_{datetime.now()}".encode()).hexdigest()
        
        return Recommendation(
            id=rec_id,
            user_id=user_id,
            recommendation_type="learning",
            title=f"تعلم: {topic}",
            description=f"بناءً على نشاطك الأخير، قد تكون مهتماً بـ {topic}",
            action="start_tutorial",
            priority=prediction.confidence * 0.8,
            relevance_score=prediction.confidence,
            context=context or {},
            timestamp=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=24)
        )
    
    async def _create_creative_recommendation(self, user_id: str, prediction: Prediction,
                                            context: Dict[str, Any]) -> Recommendation:
        """إنشاء توصية إبداعية"""
        
        creative_suggestions = [
            "جرب إنشاء مشروع جديد باستخدام الذكاء الاصطناعي",
            "اكتشف الأدوات الإبداعية المتقدمة",
            "شارك في تحدي إبداعي مع المجتمع",
            "استخدم ميزات التصميم الذكية الجديدة"
        ]
        
        suggestion = np.random.choice(creative_suggestions)
        
        rec_id = hashlib.md5(f"{user_id}_creative_{datetime.now()}".encode()).hexdigest()
        
        return Recommendation(
            id=rec_id,
            user_id=user_id,
            recommendation_type="creative",
            title=f"إبداع: {suggestion}",
            description=f"إطلق إبداعك مع {suggestion}",
            action="open_creative_tools",
            priority=prediction.confidence * 0.9,
            relevance_score=prediction.confidence,
            context=context or {},
            timestamp=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=12)
        )
    
    async def _create_analysis_recommendation(self, user_id: str, prediction: Prediction,
                                            context: Dict[str, Any]) -> Recommendation:
        """إنشاء توصية تحليلية"""
        
        analysis_options = [
            "حلل أداءك وتقدمك الشخصي",
            "استكشف أنماط استخدامك للمساعد",
            "راجع تحليل مشاعرك وحالتك النفسية",
            "احصل على رؤى متقدمة حول عاداتك"
        ]
        
        option = np.random.choice(analysis_options)
        
        rec_id = hashlib.md5(f"{user_id}_analysis_{datetime.now()}".encode()).hexdigest()
        
        return Recommendation(
            id=rec_id,
            user_id=user_id,
            recommendation_type="analysis",
            title=f"تحليل: {option}",
            description=f"احصل على رؤى قيمة من خلال {option}",
            action="open_analytics",
            priority=prediction.confidence * 0.7,
            relevance_score=prediction.confidence,
            context=context or {},
            timestamp=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=48)
        )
    
    async def _create_productivity_recommendation(self, user_id: str, prediction: Prediction,
                                                context: Dict[str, Any]) -> Recommendation:
        """إنشاء توصية للإنتاجية"""
        
        productivity_tips = [
            "نظم مهامك اليومية بذكاء",
            "استخدم ميزات الأتمتة لتوفير الوقت",
            "حسن من طريقة عملك مع النصائح الذكية",
            "استفد من التذكيرات الذكية والجدولة"
        ]
        
        tip = np.random.choice(productivity_tips)
        
        rec_id = hashlib.md5(f"{user_id}_productivity_{datetime.now()}".encode()).hexdigest()
        
        return Recommendation(
            id=rec_id,
            user_id=user_id,
            recommendation_type="productivity",
            title=f"إنتاجية: {tip}",
            description=f"حسن إنتاجيتك من خلال {tip}",
            action="setup_productivity_tools",
            priority=prediction.confidence * 0.85,
            relevance_score=prediction.confidence,
            context=context or {},
            timestamp=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=8)
        )
    
    async def _create_entertainment_recommendation(self, user_id: str, prediction: Prediction,
                                                 context: Dict[str, Any]) -> Recommendation:
        """إنشاء توصية ترفيهية"""
        
        entertainment_options = [
            "جرب الألعاب التفاعلية مع الذكاء الاصطناعي",
            "استكشف المحتوى الترفيهي المخصص لك",
            "شارك في الأنشطة الاجتماعية الافتراضية",
            "اكتشف المحتوى الإبداعي والملهم"
        ]
        
        option = np.random.choice(entertainment_options)
        
        rec_id = hashlib.md5(f"{user_id}_entertainment_{datetime.now()}".encode()).hexdigest()
        
        return Recommendation(
            id=rec_id,
            user_id=user_id,
            recommendation_type="entertainment",
            title=f"ترفيه: {option}",
            description=f"استمتع بوقتك مع {option}",
            action="explore_entertainment",
            priority=prediction.confidence * 0.6,
            relevance_score=prediction.confidence,
            context=context or {},
            timestamp=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=6)
        )
    
    def _extract_behavioral_features(self, user_profile: Dict[str, Any], 
                                   user_history: List[Any], 
                                   context: Dict[str, Any]) -> Dict[str, float]:
        """استخراج الميزات السلوكية للتنبؤ"""
        
        current_time = datetime.now()
        
        features = {
            "time_of_day": current_time.hour / 24.0,
            "day_of_week": current_time.weekday() / 6.0,
            "is_weekend": 1.0 if current_time.weekday() >= 5 else 0.0,
            "session_count_today": len([h for h in user_history 
                                      if h.get("date", "").startswith(current_time.strftime("%Y-%m-%d"))]),
            "avg_session_duration": user_profile.get("avg_session_duration", 30) / 120.0,
            "preferred_interaction_type": self._encode_interaction_type(
                user_profile.get("preferred_interaction", "text")
            ),
            "emotional_state": self._encode_emotional_state(
                context.get("current_emotion", "neutral")
            )
        }
        
        return features
    
    def _encode_interaction_type(self, interaction_type: str) -> float:
        """تشفير نوع التفاعل"""
        encoding = {"text": 0.0, "voice": 0.5, "multimodal": 1.0}
        return encoding.get(interaction_type, 0.0)
    
    def _encode_emotional_state(self, emotion: str) -> float:
        """تشفير الحالة العاطفية"""
        encoding = {
            "happiness": 1.0, "excitement": 0.8, "calmness": 0.6,
            "neutral": 0.5, "confusion": 0.4, "sadness": 0.2,
            "anger": 0.1, "fear": 0.0
        }
        return encoding.get(emotion, 0.5)
    
    async def _get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """الحصول على ملف المستخدم"""
        if user_id in self.user_profiles:
            return self.user_profiles[user_id]
        
        # تحميل من قاعدة البيانات
        try:
            conn = sqlite3.connect(self.predictions_db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT profile_data, preferences, behavior_summary 
                FROM user_profiles 
                WHERE user_id = ?
            """, (user_id,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                profile = json.loads(result[0])
                profile["preferences"] = json.loads(result[1]) if result[1] else {}
                profile["behavior_summary"] = json.loads(result[2]) if result[2] else {}
                
                self.user_profiles[user_id] = profile
                return profile
            
        except Exception as e:
            self.logger.error(f"خطأ في تحميل ملف المستخدم: {e}")
        
        # إنشاء ملف افتراضي
        default_profile = {
            "user_id": user_id,
            "created_at": datetime.now().isoformat(),
            "avg_session_duration": 30,
            "preferred_interaction": "text",
            "activity_level": "medium",
            "interests": [],
            "goals": []
        }
        
        self.user_profiles[user_id] = default_profile
        await self._save_user_profile(user_id, default_profile)
        
        return default_profile
    
    async def _save_user_profile(self, user_id: str, profile: Dict[str, Any]):
        """حفظ ملف المستخدم"""
        try:
            conn = sqlite3.connect(self.predictions_db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO user_profiles 
                (user_id, profile_data, last_updated, preferences, behavior_summary)
                VALUES (?, ?, ?, ?, ?)
            """, (
                user_id,
                json.dumps(profile, ensure_ascii=False),
                datetime.now().timestamp(),
                json.dumps(profile.get("preferences", {}), ensure_ascii=False),
                json.dumps(profile.get("behavior_summary", {}), ensure_ascii=False)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"خطأ في حفظ ملف المستخدم: {e}")
    
    async def _save_prediction(self, prediction: Prediction):
        """حفظ التنبؤ في قاعدة البيانات"""
        try:
            conn = sqlite3.connect(self.predictions_db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO predictions 
                (id, user_id, prediction_type, predicted_value, confidence, probability,
                 context, features_used, timestamp, expires_at, accuracy_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                prediction.id,
                prediction.user_id,
                prediction.prediction_type,
                json.dumps(prediction.predicted_value, ensure_ascii=False),
                prediction.confidence,
                prediction.probability,
                json.dumps(prediction.context, ensure_ascii=False),
                json.dumps(prediction.features_used, ensure_ascii=False),
                prediction.timestamp.timestamp(),
                prediction.expires_at.timestamp(),
                prediction.accuracy_score
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"خطأ في حفظ التنبؤ: {e}")
    
    async def _save_recommendation(self, recommendation: Recommendation):
        """حفظ التوصية في قاعدة البيانات"""
        try:
            conn = sqlite3.connect(self.predictions_db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO recommendations 
                (id, user_id, recommendation_type, title, description, action,
                 priority, relevance_score, context, timestamp, expires_at, is_personalized)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                recommendation.id,
                recommendation.user_id,
                recommendation.recommendation_type,
                recommendation.title,
                recommendation.description,
                recommendation.action,
                recommendation.priority,
                recommendation.relevance_score,
                json.dumps(recommendation.context, ensure_ascii=False),
                recommendation.timestamp.timestamp(),
                recommendation.expires_at.timestamp(),
                1 if recommendation.is_personalized else 0
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"خطأ في حفظ التوصية: {e}")
    
    def _start_prediction_worker(self):
        """بدء عامل التنبؤ في الخلفية"""
        def prediction_worker():
            while True:
                try:
                    task = self.prediction_queue.get(timeout=1)
                    if task is None:
                        break
                    
                    asyncio.run(self._process_prediction_task(task))
                    self.prediction_queue.task_done()
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    self.logger.error(f"خطأ في عامل التنبؤ: {e}")
        
        self.prediction_worker = threading.Thread(target=prediction_worker, daemon=True)
        self.prediction_worker.start()
    
    async def _process_prediction_task(self, task: Dict[str, Any]):
        """معالجة مهمة تنبؤ"""
        try:
            task_type = task.get("type")
            user_id = task.get("user_id")
            
            if task_type == "behavior_prediction":
                await self.predict_user_behavior(user_id)
            elif task_type == "needs_prediction":
                await self.predict_user_needs(user_id)
            elif task_type == "generate_recommendations":
                await self.generate_smart_recommendations(user_id)
                
        except Exception as e:
            self.logger.error(f"خطأ في معالجة مهمة التنبؤ: {e}")
    
    async def get_user_predictions(self, user_id: str, prediction_type: Optional[str] = None) -> List[Prediction]:
        """الحصول على تنبؤات المستخدم"""
        try:
            conn = sqlite3.connect(self.predictions_db_path)
            cursor = conn.cursor()
            
            if prediction_type:
                cursor.execute("""
                    SELECT * FROM predictions 
                    WHERE user_id = ? AND prediction_type = ? AND expires_at > ?
                    ORDER BY timestamp DESC
                """, (user_id, prediction_type, datetime.now().timestamp()))
            else:
                cursor.execute("""
                    SELECT * FROM predictions 
                    WHERE user_id = ? AND expires_at > ?
                    ORDER BY timestamp DESC
                """, (user_id, datetime.now().timestamp()))
            
            rows = cursor.fetchall()
            conn.close()
            
            predictions = []
            for row in rows:
                predictions.append(Prediction(
                    id=row[0],
                    user_id=row[1],
                    prediction_type=row[2],
                    predicted_value=json.loads(row[3]),
                    confidence=row[4],
                    probability=row[5],
                    context=json.loads(row[6]),
                    features_used=json.loads(row[7]),
                    timestamp=datetime.fromtimestamp(row[8]),
                    expires_at=datetime.fromtimestamp(row[9]),
                    accuracy_score=row[10]
                ))
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"خطأ في الحصول على التنبؤات: {e}")
            return []
    
    async def get_user_recommendations(self, user_id: str, recommendation_type: Optional[str] = None) -> List[Recommendation]:
        """الحصول على توصيات المستخدم"""
        try:
            conn = sqlite3.connect(self.predictions_db_path)
            cursor = conn.cursor()
            
            if recommendation_type:
                cursor.execute("""
                    SELECT * FROM recommendations 
                    WHERE user_id = ? AND recommendation_type = ? AND expires_at > ?
                    ORDER BY priority DESC, timestamp DESC
                """, (user_id, recommendation_type, datetime.now().timestamp()))
            else:
                cursor.execute("""
                    SELECT * FROM recommendations 
                    WHERE user_id = ? AND expires_at > ?
                    ORDER BY priority DESC, timestamp DESC
                """, (user_id, datetime.now().timestamp()))
            
            rows = cursor.fetchall()
            conn.close()
            
            recommendations = []
            for row in rows:
                recommendations.append(Recommendation(
                    id=row[0],
                    user_id=row[1],
                    recommendation_type=row[2],
                    title=row[3],
                    description=row[4],
                    action=row[5],
                    priority=row[6],
                    relevance_score=row[7],
                    context=json.loads(row[8]),
                    timestamp=datetime.fromtimestamp(row[9]),
                    expires_at=datetime.fromtimestamp(row[10]),
                    is_personalized=bool(row[11])
                ))
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"خطأ في الحصول على التوصيات: {e}")
            return []
    
    def get_prediction_statistics(self) -> Dict[str, Any]:
        """الحصول على إحصائيات التنبؤ"""
        accuracy_rate = 0.0
        if self.prediction_stats["total_predictions"] > 0:
            accuracy_rate = self.prediction_stats["accurate_predictions"] / self.prediction_stats["total_predictions"]
        
        return {
            "total_predictions": self.prediction_stats["total_predictions"],
            "accurate_predictions": self.prediction_stats["accurate_predictions"],
            "accuracy_rate": accuracy_rate,
            "recommendations_generated": self.prediction_stats["recommendations_generated"],
            "active_users": len(self.user_profiles),
            "supported_prediction_types": list(self.supported_predictions.keys()),
            "model_status": {
                model_name: "active" if model else "inactive" 
                for model_name, model in self.prediction_models.items()
            }
        }

# إنشاء مثيل عام
predictive_intelligence = PredictiveIntelligenceEngine()

def get_predictive_intelligence() -> PredictiveIntelligenceEngine:
    """الحصول على محرك الذكاء التنبؤي"""
    return predictive_intelligence

if __name__ == "__main__":
    # اختبار النظام
    async def test_predictive_intelligence():
        engine = get_predictive_intelligence()
        
        # اختبار التنبؤ بالسلوك
        behavior_predictions = await engine.predict_user_behavior(
            "test_user", 
            {"current_emotion": "happiness", "time_context": "morning"}
        )
        print(f"تنبؤات السلوك: {len(behavior_predictions)}")
        
        # اختبار التوصيات
        recommendations = await engine.generate_smart_recommendations("test_user")
        print(f"التوصيات: {len(recommendations)}")
        for rec in recommendations[:3]:
            print(f"- {rec.title} (أولوية: {rec.priority:.2f})")
        
        # عرض الإحصائيات
        stats = engine.get_prediction_statistics()
        print(f"الإحصائيات: {stats}")
    
    asyncio.run(test_predictive_intelligence())
