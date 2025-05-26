
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
نظام الذكاء العاطفي المتقدم
Advanced Emotional Intelligence Engine
"""

import asyncio
import logging
import numpy as np
import cv2
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import json
import sqlite3
from dataclasses import dataclass
import threading
import queue
from collections import defaultdict, deque

@dataclass
class EmotionalState:
    """حالة عاطفية"""
    primary_emotion: str
    intensity: float
    confidence: float
    secondary_emotions: Dict[str, float]
    context: Dict[str, Any]
    timestamp: datetime
    duration: Optional[float] = None

@dataclass
class EmotionalPattern:
    """نمط عاطفي"""
    user_id: str
    emotion_sequence: List[str]
    triggers: List[str]
    frequency: int
    avg_intensity: float
    context_factors: Dict[str, float]
    first_seen: datetime
    last_seen: datetime

class EmotionalIntelligenceEngine:
    """محرك الذكاء العاطفي المتقدم"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # قاعدة بيانات المشاعر
        self.emotions_db_path = Path("data/emotions/emotions.db")
        self.emotions_db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_emotions_database()
        
        # نماذج تحليل المشاعر
        self.emotion_models = {
            "facial": None,      # نموذج تحليل تعابير الوجه
            "voice": None,       # نموذج تحليل نبرة الصوت
            "text": None,        # نموذج تحليل النص العاطفي
            "behavioral": None   # نموذج تحليل السلوك
        }
        
        # خريطة المشاعر الأساسية
        self.emotion_map = {
            "happiness": {"color": "#FFD700", "energy": "high", "valence": "positive"},
            "sadness": {"color": "#4682B4", "energy": "low", "valence": "negative"},
            "anger": {"color": "#DC143C", "energy": "high", "valence": "negative"},
            "fear": {"color": "#8B0000", "energy": "high", "valence": "negative"},
            "surprise": {"color": "#FF69B4", "energy": "medium", "valence": "neutral"},
            "disgust": {"color": "#556B2F", "energy": "medium", "valence": "negative"},
            "neutral": {"color": "#808080", "energy": "low", "valence": "neutral"},
            "excitement": {"color": "#FF4500", "energy": "high", "valence": "positive"},
            "calmness": {"color": "#87CEEB", "energy": "low", "valence": "positive"},
            "confusion": {"color": "#DDA0DD", "energy": "medium", "valence": "negative"},
            "confidence": {"color": "#32CD32", "energy": "medium", "valence": "positive"},
            "anxiety": {"color": "#8B008B", "energy": "high", "valence": "negative"}
        }
        
        # أنماط المشاعر المكتشفة
        self.emotional_patterns = {}
        
        # تاريخ المشاعر للمستخدمين
        self.user_emotional_history = defaultdict(lambda: deque(maxlen=100))
        
        # إعدادات التحليل العاطفي
        self.analysis_settings = {
            "sensitivity": 0.7,
            "pattern_detection": True,
            "real_time_analysis": True,
            "multi_modal": True,
            "cultural_adaptation": True,
            "privacy_mode": False
        }
        
        # قائمة انتظار التحليل
        self.analysis_queue = queue.Queue()
        self.analysis_worker = None
        
        # إحصائيات التحليل العاطفي
        self.emotion_stats = {
            "total_analyses": 0,
            "patterns_detected": 0,
            "interventions_made": 0,
            "accuracy_score": 0.0
        }
        
        self._start_analysis_worker()
        self._initialize_emotion_models()
    
    def _init_emotions_database(self):
        """تهيئة قاعدة بيانات المشاعر"""
        conn = sqlite3.connect(self.emotions_db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS emotional_states (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                primary_emotion TEXT NOT NULL,
                intensity REAL NOT NULL,
                confidence REAL NOT NULL,
                secondary_emotions TEXT,
                context TEXT,
                timestamp REAL NOT NULL,
                duration REAL,
                source TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS emotional_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                pattern_type TEXT NOT NULL,
                pattern_data TEXT NOT NULL,
                frequency INTEGER DEFAULT 1,
                avg_intensity REAL,
                first_seen REAL NOT NULL,
                last_seen REAL NOT NULL,
                context_factors TEXT
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_user_timestamp ON emotional_states(user_id, timestamp);
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_user_patterns ON emotional_patterns(user_id, pattern_type);
        """)
        
        conn.commit()
        conn.close()
    
    def _initialize_emotion_models(self):
        """تهيئة نماذج تحليل المشاعر"""
        try:
            # تهيئة نموذج تحليل تعابير الوجه
            self._init_facial_emotion_model()
            
            # تهيئة نموذج تحليل الصوت
            self._init_voice_emotion_model()
            
            # تهيئة نموذج تحليل النص
            self._init_text_emotion_model()
            
            # تهيئة نموذج تحليل السلوك
            self._init_behavioral_model()
            
            self.logger.info("✅ تم تهيئة جميع نماذج التحليل العاطفي")
            
        except Exception as e:
            self.logger.error(f"خطأ في تهيئة نماذج المشاعر: {e}")
    
    def _init_facial_emotion_model(self):
        """تهيئة نموذج تحليل تعابير الوجه"""
        try:
            # محاكاة تحميل نموذج DeepFace أو FER
            self.emotion_models["facial"] = "FacialEmotionModel()"
            self.logger.info("✅ تم تحميل نموذج تحليل تعابير الوجه")
        except Exception as e:
            self.logger.error(f"فشل تحميل نموذج الوجه: {e}")
    
    def _init_voice_emotion_model(self):
        """تهيئة نموذج تحليل نبرة الصوت"""
        try:
            # محاكاة تحميل نموذج تحليل الصوت العاطفي
            self.emotion_models["voice"] = "VoiceEmotionModel()"
            self.logger.info("✅ تم تحميل نموذج تحليل نبرة الصوت")
        except Exception as e:
            self.logger.error(f"فشل تحميل نموذج الصوت: {e}")
    
    def _init_text_emotion_model(self):
        """تهيئة نموذج تحليل النص العاطفي"""
        try:
            # محاكاة تحميل نموذج BERT أو RoBERTa للمشاعر
            self.emotion_models["text"] = "TextEmotionModel()"
            self.logger.info("✅ تم تحميل نموذج تحليل النص العاطفي")
        except Exception as e:
            self.logger.error(f"فشل تحميل نموذج النص: {e}")
    
    def _init_behavioral_model(self):
        """تهيئة نموذج تحليل السلوك"""
        try:
            # محاكاة تحميل نموذج تحليل الأنماط السلوكية
            self.emotion_models["behavioral"] = "BehavioralEmotionModel()"
            self.logger.info("✅ تم تحميل نموذج تحليل السلوك")
        except Exception as e:
            self.logger.error(f"فشل تحميل نموذج السلوك: {e}")
    
    async def analyze_emotion_multimodal(
        self, 
        user_id: str, 
        text: Optional[str] = None,
        image: Optional[np.ndarray] = None,
        audio: Optional[np.ndarray] = None,
        behavioral_data: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> EmotionalState:
        """تحليل المشاعر متعدد الوسائط"""
        
        try:
            emotion_scores = {}
            confidence_scores = {}
            analysis_sources = []
            
            # تحليل النص
            if text and self.emotion_models["text"]:
                text_emotions = await self._analyze_text_emotion(text)
                emotion_scores.update(text_emotions)
                confidence_scores["text"] = 0.8
                analysis_sources.append("text")
            
            # تحليل الصورة/الوجه
            if image is not None and self.emotion_models["facial"]:
                facial_emotions = await self._analyze_facial_emotion(image)
                emotion_scores.update(facial_emotions)
                confidence_scores["facial"] = 0.9
                analysis_sources.append("facial")
            
            # تحليل الصوت
            if audio is not None and self.emotion_models["voice"]:
                voice_emotions = await self._analyze_voice_emotion(audio)
                emotion_scores.update(voice_emotions)
                confidence_scores["voice"] = 0.7
                analysis_sources.append("voice")
            
            # تحليل السلوك
            if behavioral_data and self.emotion_models["behavioral"]:
                behavioral_emotions = await self._analyze_behavioral_emotion(behavioral_data)
                emotion_scores.update(behavioral_emotions)
                confidence_scores["behavioral"] = 0.6
                analysis_sources.append("behavioral")
            
            # دمج النتائج
            final_emotions = self._fuse_emotion_results(emotion_scores, confidence_scores)
            
            # تحديد المشاعر الأساسية والثانوية
            primary_emotion = max(final_emotions.items(), key=lambda x: x[1])[0]
            intensity = final_emotions[primary_emotion]
            
            # إزالة المشاعر الأساسية من الثانوية
            secondary_emotions = {k: v for k, v in final_emotions.items() 
                                if k != primary_emotion and v > 0.1}
            
            # حساب الثقة الإجمالية
            overall_confidence = np.mean(list(confidence_scores.values())) if confidence_scores else 0.5
            
            # إنشاء الحالة العاطفية
            emotional_state = EmotionalState(
                primary_emotion=primary_emotion,
                intensity=intensity,
                confidence=overall_confidence,
                secondary_emotions=secondary_emotions,
                context=context or {},
                timestamp=datetime.now()
            )
            
            # حفظ الحالة العاطفية
            await self._save_emotional_state(user_id, emotional_state, analysis_sources)
            
            # إضافة لتاريخ المستخدم
            self.user_emotional_history[user_id].append(emotional_state)
            
            # كشف الأنماط العاطفية
            await self._detect_emotional_patterns(user_id)
            
            # تحديث الإحصائيات
            self.emotion_stats["total_analyses"] += 1
            
            self.logger.info(f"تحليل عاطفي للمستخدم {user_id}: {primary_emotion} ({intensity:.2f})")
            
            return emotional_state
            
        except Exception as e:
            self.logger.error(f"خطأ في التحليل العاطفي: {e}")
            
            # إرجاع حالة عاطفية افتراضية
            return EmotionalState(
                primary_emotion="neutral",
                intensity=0.5,
                confidence=0.0,
                secondary_emotions={},
                context=context or {},
                timestamp=datetime.now()
            )
    
    async def _analyze_text_emotion(self, text: str) -> Dict[str, float]:
        """تحليل المشاعر من النص"""
        # محاكاة تحليل نصي متقدم
        emotion_keywords = {
            "happiness": ["سعيد", "مبسوط", "فرحان", "مبهج", "رائع", "ممتاز"],
            "sadness": ["حزين", "متضايق", "مكتئب", "تعبان", "زعلان"],
            "anger": ["غاضب", "زعلان", "متنرفز", "مستاء", "غضبان"],
            "fear": ["خائف", "مرعوب", "قلقان", "متوتر", "خوف"],
            "surprise": ["مفاجأة", "صدمة", "مذهول", "متفاجئ"],
            "excitement": ["متحمس", "متشوق", "مثير", "نشيط"],
            "calmness": ["هادئ", "مرتاح", "مطمئن", "ساكن"],
            "confusion": ["محتار", "مشوش", "مبلبل", "لا أفهم"]
        }
        
        text_lower = text.lower()
        emotion_scores = defaultdict(float)
        
        for emotion, keywords in emotion_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    emotion_scores[emotion] += 0.3
        
        # تطبيع النتائج
        if emotion_scores:
            max_score = max(emotion_scores.values())
            emotion_scores = {k: v/max_score for k, v in emotion_scores.items()}
        else:
            emotion_scores["neutral"] = 1.0
        
        return dict(emotion_scores)
    
    async def _analyze_facial_emotion(self, image: np.ndarray) -> Dict[str, float]:
        """تحليل المشاعر من تعابير الوجه"""
        # محاكاة تحليل تعابير الوجه
        # في التطبيق الحقيقي، سيتم استخدام DeepFace أو نموذج مماثل
        
        emotion_scores = {
            "happiness": np.random.uniform(0.1, 0.9),
            "sadness": np.random.uniform(0.0, 0.3),
            "anger": np.random.uniform(0.0, 0.2),
            "fear": np.random.uniform(0.0, 0.1),
            "surprise": np.random.uniform(0.0, 0.4),
            "neutral": np.random.uniform(0.2, 0.6)
        }
        
        # تطبيع النتائج
        total = sum(emotion_scores.values())
        emotion_scores = {k: v/total for k, v in emotion_scores.items()}
        
        return emotion_scores
    
    async def _analyze_voice_emotion(self, audio: np.ndarray) -> Dict[str, float]:
        """تحليل المشاعر من نبرة الصوت"""
        # محاكاة تحليل نبرة الصوت
        # في التطبيق الحقيقي، سيتم استخدام librosa ونماذج تحليل الصوت
        
        # محاكاة استخراج الخصائص الصوتية
        pitch_variance = np.random.uniform(0.1, 1.0)
        energy_level = np.random.uniform(0.2, 1.0)
        speaking_rate = np.random.uniform(0.5, 1.5)
        
        emotion_scores = {}
        
        # ربط الخصائص بالمشاعر
        if energy_level > 0.7 and pitch_variance > 0.6:
            emotion_scores["excitement"] = 0.8
            emotion_scores["happiness"] = 0.6
        elif energy_level < 0.3 and pitch_variance < 0.4:
            emotion_scores["sadness"] = 0.7
            emotion_scores["calmness"] = 0.4
        elif pitch_variance > 0.8:
            emotion_scores["anger"] = 0.6
            emotion_scores["fear"] = 0.3
        else:
            emotion_scores["neutral"] = 0.6
        
        return emotion_scores
    
    async def _analyze_behavioral_emotion(self, behavioral_data: Dict[str, Any]) -> Dict[str, float]:
        """تحليل المشاعر من السلوك"""
        emotion_scores = {}
        
        # تحليل أنماط الاستخدام
        typing_speed = behavioral_data.get("typing_speed", 50)
        pause_duration = behavioral_data.get("pause_duration", 1.0)
        interaction_frequency = behavioral_data.get("interaction_frequency", 1.0)
        
        # ربط السلوك بالمشاعر
        if typing_speed > 80:
            emotion_scores["excitement"] = 0.6
            emotion_scores["anxiety"] = 0.4
        elif typing_speed < 30:
            emotion_scores["sadness"] = 0.5
            emotion_scores["calmness"] = 0.3
        
        if pause_duration > 3.0:
            emotion_scores["confusion"] = 0.4
            emotion_scores["thoughtfulness"] = 0.3
        
        if not emotion_scores:
            emotion_scores["neutral"] = 0.5
        
        return emotion_scores
    
    def _fuse_emotion_results(self, emotion_scores: Dict[str, float], 
                            confidence_scores: Dict[str, float]) -> Dict[str, float]:
        """دمج نتائج التحليل العاطفي من مصادر متعددة"""
        
        final_emotions = defaultdict(float)
        total_weight = sum(confidence_scores.values())
        
        if total_weight == 0:
            return {"neutral": 1.0}
        
        # دمج النتائج بناءً على الثقة
        for source, confidence in confidence_scores.items():
            weight = confidence / total_weight
            for emotion, score in emotion_scores.items():
                final_emotions[emotion] += score * weight
        
        # تطبيع النتائج النهائية
        total_score = sum(final_emotions.values())
        if total_score > 0:
            final_emotions = {k: v/total_score for k, v in final_emotions.items()}
        
        return dict(final_emotions)
    
    async def _save_emotional_state(self, user_id: str, state: EmotionalState, 
                                   sources: List[str]):
        """حفظ الحالة العاطفية في قاعدة البيانات"""
        try:
            conn = sqlite3.connect(self.emotions_db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO emotional_states 
                (user_id, primary_emotion, intensity, confidence, secondary_emotions, 
                 context, timestamp, duration, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                user_id,
                state.primary_emotion,
                state.intensity,
                state.confidence,
                json.dumps(state.secondary_emotions, ensure_ascii=False),
                json.dumps(state.context, ensure_ascii=False),
                state.timestamp.timestamp(),
                state.duration,
                ",".join(sources)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"خطأ في حفظ الحالة العاطفية: {e}")
    
    async def _detect_emotional_patterns(self, user_id: str):
        """كشف الأنماط العاطفية للمستخدم"""
        try:
            history = self.user_emotional_history[user_id]
            
            if len(history) < 3:
                return
            
            # تحليل الأنماط الزمنية
            emotions_sequence = [state.primary_emotion for state in list(history)[-10:]]
            
            # كشف التكرار في المشاعر
            emotion_counts = defaultdict(int)
            for emotion in emotions_sequence:
                emotion_counts[emotion] += 1
            
            # كشف الانتقالات العاطفية
            transitions = []
            for i in range(len(emotions_sequence) - 1):
                transition = f"{emotions_sequence[i]}->{emotions_sequence[i+1]}"
                transitions.append(transition)
            
            # حفظ الأنماط المكتشفة
            if len(transitions) >= 3:
                pattern = EmotionalPattern(
                    user_id=user_id,
                    emotion_sequence=emotions_sequence[-5:],
                    triggers=[],  # سيتم تحليلها لاحقاً
                    frequency=1,
                    avg_intensity=np.mean([state.intensity for state in list(history)[-5:]]),
                    context_factors={},
                    first_seen=datetime.now(),
                    last_seen=datetime.now()
                )
                
                self.emotional_patterns[user_id] = pattern
                self.emotion_stats["patterns_detected"] += 1
                
                self.logger.info(f"تم اكتشاف نمط عاطفي جديد للمستخدم {user_id}")
        
        except Exception as e:
            self.logger.error(f"خطأ في كشف الأنماط العاطفية: {e}")
    
    async def get_emotional_insights(self, user_id: str) -> Dict[str, Any]:
        """الحصول على رؤى عاطفية للمستخدم"""
        try:
            history = list(self.user_emotional_history[user_id])
            
            if not history:
                return {"message": "لا توجد بيانات عاطفية كافية"}
            
            # تحليل المشاعر الأكثر شيوعاً
            emotion_counts = defaultdict(int)
            intensity_sum = defaultdict(float)
            
            for state in history:
                emotion_counts[state.primary_emotion] += 1
                intensity_sum[state.primary_emotion] += state.intensity
            
            # حساب المتوسطات
            avg_intensities = {
                emotion: intensity_sum[emotion] / emotion_counts[emotion]
                for emotion in emotion_counts
            }
            
            # المشاعر الأكثر شيوعاً
            most_common_emotions = sorted(
                emotion_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5]
            
            # التغييرات الأخيرة
            recent_changes = []
            if len(history) >= 2:
                for i in range(len(history) - 1):
                    if history[i].primary_emotion != history[i+1].primary_emotion:
                        recent_changes.append({
                            "from": history[i].primary_emotion,
                            "to": history[i+1].primary_emotion,
                            "timestamp": history[i+1].timestamp.isoformat()
                        })
            
            # الحالة العاطفية الحالية
            current_state = history[-1] if history else None
            
            insights = {
                "current_emotion": {
                    "emotion": current_state.primary_emotion if current_state else "unknown",
                    "intensity": current_state.intensity if current_state else 0.0,
                    "confidence": current_state.confidence if current_state else 0.0
                },
                "most_common_emotions": most_common_emotions,
                "average_intensities": avg_intensities,
                "recent_emotional_changes": recent_changes[-5:],
                "emotional_stability": self._calculate_emotional_stability(history),
                "pattern_detected": user_id in self.emotional_patterns,
                "recommendations": self._generate_emotional_recommendations(history)
            }
            
            return insights
            
        except Exception as e:
            self.logger.error(f"خطأ في الحصول على الرؤى العاطفية: {e}")
            return {"error": str(e)}
    
    def _calculate_emotional_stability(self, history: List[EmotionalState]) -> float:
        """حساب الاستقرار العاطفي"""
        if len(history) < 2:
            return 0.5
        
        # حساب التباين في المشاعر
        emotions = [state.primary_emotion for state in history]
        unique_emotions = len(set(emotions))
        total_emotions = len(emotions)
        
        # حساب التباين في الشدة
        intensities = [state.intensity for state in history]
        intensity_variance = np.var(intensities)
        
        # حساب الاستقرار (كلما قل التباين، زاد الاستقرار)
        stability = 1 - (unique_emotions / total_emotions) - intensity_variance
        
        return max(0.0, min(1.0, stability))
    
    def _generate_emotional_recommendations(self, history: List[EmotionalState]) -> List[str]:
        """توليد توصيات عاطفية"""
        if not history:
            return ["لا توجد بيانات كافية للتوصيات"]
        
        recommendations = []
        current_state = history[-1]
        
        # توصيات بناءً على المشاعر الحالية
        if current_state.primary_emotion == "sadness" and current_state.intensity > 0.6:
            recommendations.append("قد تكون بحاجة إلى دعم عاطفي أو أنشطة مرحة")
            
        elif current_state.primary_emotion == "anger" and current_state.intensity > 0.7:
            recommendations.append("جرب تقنيات الاسترخاء والتنفس العميق")
            
        elif current_state.primary_emotion == "anxiety" and current_state.intensity > 0.5:
            recommendations.append("قد تساعدك تمارين التأمل والاسترخاء")
            
        elif current_state.primary_emotion == "happiness" and current_state.intensity > 0.8:
            recommendations.append("حالة رائعة! حاول الاستفادة من هذه الطاقة الإيجابية")
        
        # توصيات بناءً على الأنماط
        if len(history) >= 5:
            recent_emotions = [state.primary_emotion for state in history[-5:]]
            if recent_emotions.count("stress") >= 3:
                recommendations.append("لاحظت أنك تشعر بالتوتر مؤخراً، قد تحتاج لاستراحة")
        
        if not recommendations:
            recommendations.append("حافظ على نمط حياة صحي ومتوازن")
        
        return recommendations
    
    def _start_analysis_worker(self):
        """بدء عامل التحليل في الخلفية"""
        def analysis_worker():
            while True:
                try:
                    task = self.analysis_queue.get(timeout=1)
                    if task is None:
                        break
                    
                    # تنفيذ مهمة التحليل
                    asyncio.run(self._process_analysis_task(task))
                    self.analysis_queue.task_done()
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    self.logger.error(f"خطأ في عامل التحليل: {e}")
        
        self.analysis_worker = threading.Thread(target=analysis_worker, daemon=True)
        self.analysis_worker.start()
    
    async def _process_analysis_task(self, task: Dict[str, Any]):
        """معالجة مهمة تحليل عاطفي"""
        try:
            task_type = task.get("type")
            user_id = task.get("user_id")
            
            if task_type == "pattern_analysis":
                await self._detect_emotional_patterns(user_id)
            elif task_type == "intervention_check":
                await self._check_intervention_needed(user_id)
                
        except Exception as e:
            self.logger.error(f"خطأ في معالجة مهمة التحليل: {e}")
    
    async def _check_intervention_needed(self, user_id: str):
        """فحص ما إذا كان المستخدم يحتاج تدخل عاطفي"""
        try:
            history = list(self.user_emotional_history[user_id])
            
            if len(history) < 3:
                return
            
            recent_states = history[-3:]
            
            # فحص المشاعر السلبية المستمرة
            negative_emotions = ["sadness", "anger", "fear", "anxiety"]
            negative_count = sum(1 for state in recent_states 
                               if state.primary_emotion in negative_emotions 
                               and state.intensity > 0.6)
            
            if negative_count >= 2:
                self.emotion_stats["interventions_made"] += 1
                self.logger.info(f"تم اكتشاف حاجة لتدخل عاطفي للمستخدم {user_id}")
                
                # يمكن هنا إرسال إشعار أو اقتراح للمساعدة
                
        except Exception as e:
            self.logger.error(f"خطأ في فحص التدخل العاطفي: {e}")
    
    def get_emotion_statistics(self) -> Dict[str, Any]:
        """الحصول على إحصائيات التحليل العاطفي"""
        return {
            "total_analyses": self.emotion_stats["total_analyses"],
            "patterns_detected": self.emotion_stats["patterns_detected"],
            "interventions_made": self.emotion_stats["interventions_made"],
            "active_users": len(self.user_emotional_history),
            "supported_emotions": list(self.emotion_map.keys()),
            "analysis_modes": list(self.emotion_models.keys())
        }

# إنشاء مثيل عام
emotional_intelligence = EmotionalIntelligenceEngine()

def get_emotional_intelligence() -> EmotionalIntelligenceEngine:
    """الحصول على محرك الذكاء العاطفي"""
    return emotional_intelligence

if __name__ == "__main__":
    # اختبار النظام
    async def test_emotional_intelligence():
        engine = get_emotional_intelligence()
        
        # اختبار تحليل نصي
        result = await engine.analyze_emotion_multimodal(
            user_id="test_user",
            text="أشعر بالحزن اليوم، لست متأكداً من السبب",
            context={"source": "chat", "time_of_day": "evening"}
        )
        
        print(f"النتيجة: {result.primary_emotion} ({result.intensity:.2f})")
        
        # الحصول على رؤى
        insights = await engine.get_emotional_insights("test_user")
        print(f"الرؤى العاطفية: {insights}")
        
        # عرض الإحصائيات
        stats = engine.get_emotion_statistics()
        print(f"الإحصائيات: {stats}")
    
    asyncio.run(test_emotional_intelligence())
