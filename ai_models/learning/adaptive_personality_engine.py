
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
محرك تطوير الشخصية التكيفي
Adaptive Personality Development Engine
"""

import asyncio
import logging
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
import sqlite3
from enum import Enum
import pickle

class PersonalityTrait(Enum):
    """سمات الشخصية"""
    FRIENDLINESS = "friendliness"
    FORMALITY = "formality"
    ENTHUSIASM = "enthusiasm"
    PATIENCE = "patience"
    HUMOR = "humor"
    EMPATHY = "empathy"
    PRECISION = "precision"
    CREATIVITY = "creativity"

class CommunicationStyle(Enum):
    """أساليب التواصل"""
    CASUAL = "casual"
    PROFESSIONAL = "professional"
    ACADEMIC = "academic"
    FRIENDLY = "friendly"
    SUPPORTIVE = "supportive"
    DIRECT = "direct"
    ENCOURAGING = "encouraging"

@dataclass
class UserInteraction:
    """تفاعل المستخدم"""
    timestamp: datetime
    user_input: str
    assistant_response: str
    user_feedback: Optional[str]
    emotion_detected: str
    context: Dict[str, Any]
    satisfaction_score: float
    interaction_type: str

@dataclass
class PersonalityProfile:
    """ملف الشخصية"""
    traits: Dict[PersonalityTrait, float]  # 0-1 scale
    communication_style: CommunicationStyle
    preferred_topics: List[str]
    response_patterns: Dict[str, float]
    adaptation_rate: float
    last_updated: datetime

class AdaptivePersonalityEngine:
    """محرك تطوير الشخصية التكيفي"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # قاعدة البيانات
        self.db_path = Path("data/personality_engine.db")
        self.db_path.parent.mkdir(exist_ok=True)
        
        # ملف الشخصية الحالي
        self.personality_profile = PersonalityProfile(
            traits={trait: 0.5 for trait in PersonalityTrait},
            communication_style=CommunicationStyle.FRIENDLY,
            preferred_topics=[],
            response_patterns={},
            adaptation_rate=0.1,
            last_updated=datetime.now()
        )
        
        # سجل التفاعلات
        self.interaction_history: List[UserInteraction] = []
        
        # أنماط التعلم
        self.learning_patterns = {
            "positive_feedback_weight": 0.8,
            "negative_feedback_weight": 1.2,
            "recency_weight": 0.9,
            "consistency_threshold": 0.7
        }
        
        # قوالب الاستجابة
        self.response_templates = {
            CommunicationStyle.CASUAL: {
                "greeting": ["مرحبا!", "أهلاً وسهلاً!", "هاي!"],
                "acknowledgment": ["فهمت", "واضح", "تمام"],
                "enthusiasm": ["رائع!", "ممتاز!", "هذا جميل!"],
                "uncertainty": ["مش متأكد", "لست واثقاً", "ممكن"]
            },
            CommunicationStyle.PROFESSIONAL: {
                "greeting": ["أهلاً بك", "مرحباً بكم", "تحية طيبة"],
                "acknowledgment": ["مفهوم", "تم الاستلام", "تمت المراجعة"],
                "enthusiasm": ["ممتاز", "نتيجة إيجابية", "تطور جيد"],
                "uncertainty": ["غير محدد", "يتطلب مراجعة", "معلومات إضافية مطلوبة"]
            },
            CommunicationStyle.FRIENDLY: {
                "greeting": ["أهلاً صديقي!", "مرحبا يا صديق!", "سعيد برؤيتك!"],
                "acknowledgment": ["فهمتك تماماً", "أدرك ما تقصده", "واضح جداً"],
                "enthusiasm": ["هذا رائع حقاً!", "أحب هذا!", "ممتاز جداً!"],
                "uncertainty": ["لست متأكداً تماماً", "أحتاج للتفكير", "دعني أتحقق"]
            }
        }

    async def initialize(self):
        """تهيئة محرك الشخصية"""
        
        try:
            self.logger.info("🧠 تهيئة محرك تطوير الشخصية التكيفي...")
            
            # إنشاء قاعدة البيانات
            await self._initialize_database()
            
            # تحميل البيانات
            await self._load_personality_data()
            
            # تحليل الأنماط التاريخية
            await self._analyze_historical_patterns()
            
            self.logger.info("✅ تم تهيئة محرك الشخصية")
            
        except Exception as e:
            self.logger.error(f"❌ فشل تهيئة محرك الشخصية: {e}")

    async def _initialize_database(self):
        """تهيئة قاعدة البيانات"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # جدول التفاعلات
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                user_input TEXT NOT NULL,
                assistant_response TEXT NOT NULL,
                user_feedback TEXT,
                emotion_detected TEXT,
                context TEXT,
                satisfaction_score REAL,
                interaction_type TEXT
            )
        """)
        
        # جدول ملف الشخصية
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS personality_profile (
                id INTEGER PRIMARY KEY,
                traits TEXT NOT NULL,
                communication_style TEXT NOT NULL,
                preferred_topics TEXT,
                response_patterns TEXT,
                adaptation_rate REAL,
                last_updated TEXT NOT NULL
            )
        """)
        
        conn.commit()
        conn.close()

    async def _load_personality_data(self):
        """تحميل بيانات الشخصية"""
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # تحميل ملف الشخصية
            cursor.execute("SELECT * FROM personality_profile ORDER BY id DESC LIMIT 1")
            profile_row = cursor.fetchone()
            
            if profile_row:
                traits_data = json.loads(profile_row[1])
                traits = {PersonalityTrait(k): v for k, v in traits_data.items()}
                
                self.personality_profile = PersonalityProfile(
                    traits=traits,
                    communication_style=CommunicationStyle(profile_row[2]),
                    preferred_topics=json.loads(profile_row[3]) if profile_row[3] else [],
                    response_patterns=json.loads(profile_row[4]) if profile_row[4] else {},
                    adaptation_rate=profile_row[5],
                    last_updated=datetime.fromisoformat(profile_row[6])
                )
            
            # تحميل التفاعلات
            cursor.execute("SELECT * FROM interactions ORDER BY timestamp DESC LIMIT 1000")
            for row in cursor.fetchall():
                interaction = UserInteraction(
                    timestamp=datetime.fromisoformat(row[1]),
                    user_input=row[2],
                    assistant_response=row[3],
                    user_feedback=row[4],
                    emotion_detected=row[5],
                    context=json.loads(row[6]) if row[6] else {},
                    satisfaction_score=row[7],
                    interaction_type=row[8]
                )
                self.interaction_history.append(interaction)
            
            conn.close()
            
        except Exception as e:
            self.logger.error(f"خطأ في تحميل بيانات الشخصية: {e}")

    async def _analyze_historical_patterns(self):
        """تحليل الأنماط التاريخية"""
        
        if not self.interaction_history:
            return
        
        try:
            # تحليل أنماط الرضا
            satisfaction_scores = [i.satisfaction_score for i in self.interaction_history if i.satisfaction_score is not None]
            if satisfaction_scores:
                avg_satisfaction = np.mean(satisfaction_scores)
                
                if avg_satisfaction < 0.6:
                    # تحتاج لتحسين الشخصية
                    await self._adjust_personality_for_improvement()
            
            # تحليل المواضيع المفضلة
            topic_frequency = {}
            for interaction in self.interaction_history:
                context = interaction.context
                if "topic" in context:
                    topic = context["topic"]
                    topic_frequency[topic] = topic_frequency.get(topic, 0) + 1
            
            # تحديث المواضيع المفضلة
            sorted_topics = sorted(topic_frequency.items(), key=lambda x: x[1], reverse=True)
            self.personality_profile.preferred_topics = [topic for topic, _ in sorted_topics[:10]]
            
        except Exception as e:
            self.logger.error(f"خطأ في تحليل الأنماط التاريخية: {e}")

    async def _adjust_personality_for_improvement(self):
        """تعديل الشخصية للتحسين"""
        
        # تحليل التفاعلات السلبية
        negative_interactions = [
            i for i in self.interaction_history 
            if i.satisfaction_score is not None and i.satisfaction_score < 0.5
        ]
        
        if negative_interactions:
            # تحليل الأسباب المحتملة
            common_emotions = {}
            for interaction in negative_interactions:
                emotion = interaction.emotion_detected
                common_emotions[emotion] = common_emotions.get(emotion, 0) + 1
            
            # تعديل السمات بناءً على التحليل
            if "frustrated" in common_emotions:
                self.personality_profile.traits[PersonalityTrait.PATIENCE] += 0.1
                self.personality_profile.traits[PersonalityTrait.EMPATHY] += 0.1
            
            if "confused" in common_emotions:
                self.personality_profile.traits[PersonalityTrait.PRECISION] += 0.1
                
            if "bored" in common_emotions:
                self.personality_profile.traits[PersonalityTrait.ENTHUSIASM] += 0.1
                self.personality_profile.traits[PersonalityTrait.CREATIVITY] += 0.1
        
        # تطبيع القيم
        for trait in self.personality_profile.traits:
            self.personality_profile.traits[trait] = np.clip(
                self.personality_profile.traits[trait], 0.0, 1.0
            )

    async def record_interaction(
        self,
        user_input: str,
        assistant_response: str,
        emotion_detected: str,
        context: Dict[str, Any],
        user_feedback: Optional[str] = None,
        satisfaction_score: Optional[float] = None
    ):
        """تسجيل تفاعل جديد"""
        
        try:
            interaction = UserInteraction(
                timestamp=datetime.now(),
                user_input=user_input,
                assistant_response=assistant_response,
                user_feedback=user_feedback,
                emotion_detected=emotion_detected,
                context=context,
                satisfaction_score=satisfaction_score if satisfaction_score is not None else self._estimate_satisfaction(user_input, emotion_detected),
                interaction_type=context.get("type", "general")
            )
            
            self.interaction_history.append(interaction)
            
            # حفظ في قاعدة البيانات
            await self._save_interaction(interaction)
            
            # تحديث الشخصية بناءً على التفاعل
            await self._update_personality_from_interaction(interaction)
            
        except Exception as e:
            self.logger.error(f"خطأ في تسجيل التفاعل: {e}")

    def _estimate_satisfaction(self, user_input: str, emotion: str) -> float:
        """تقدير مستوى الرضا"""
        
        # تقدير بسيط بناءً على المشاعر
        emotion_scores = {
            "happy": 0.9,
            "satisfied": 0.8,
            "neutral": 0.6,
            "confused": 0.4,
            "frustrated": 0.2,
            "angry": 0.1
        }
        
        return emotion_scores.get(emotion, 0.5)

    async def _save_interaction(self, interaction: UserInteraction):
        """حفظ التفاعل في قاعدة البيانات"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO interactions 
            (timestamp, user_input, assistant_response, user_feedback, 
             emotion_detected, context, satisfaction_score, interaction_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            interaction.timestamp.isoformat(),
            interaction.user_input,
            interaction.assistant_response,
            interaction.user_feedback,
            interaction.emotion_detected,
            json.dumps(interaction.context),
            interaction.satisfaction_score,
            interaction.interaction_type
        ))
        
        conn.commit()
        conn.close()

    async def _update_personality_from_interaction(self, interaction: UserInteraction):
        """تحديث الشخصية بناءً على التفاعل"""
        
        try:
            adaptation_rate = self.personality_profile.adaptation_rate
            satisfaction = interaction.satisfaction_score
            
            # تحديث السمات بناءً على الرضا
            if satisfaction > 0.7:
                # تعزيز السمات الإيجابية
                if interaction.emotion_detected == "happy":
                    self.personality_profile.traits[PersonalityTrait.HUMOR] += adaptation_rate * 0.1
                    self.personality_profile.traits[PersonalityTrait.ENTHUSIASM] += adaptation_rate * 0.1
                
            elif satisfaction < 0.4:
                # تعديل السمات للتحسين
                if interaction.emotion_detected == "frustrated":
                    self.personality_profile.traits[PersonalityTrait.PATIENCE] += adaptation_rate * 0.2
                    self.personality_profile.traits[PersonalityTrait.EMPATHY] += adaptation_rate * 0.1
                
                elif interaction.emotion_detected == "confused":
                    self.personality_profile.traits[PersonalityTrait.PRECISION] += adaptation_rate * 0.2
            
            # تحديث أسلوب التواصل
            await self._update_communication_style(interaction)
            
            # حفظ التغييرات
            await self._save_personality_profile()
            
        except Exception as e:
            self.logger.error(f"خطأ في تحديث الشخصية: {e}")

    async def _update_communication_style(self, interaction: UserInteraction):
        """تحديث أسلوب التواصل"""
        
        user_input = interaction.user_input.lower()
        
        # تحليل أسلوب المستخدم
        if any(word in user_input for word in ["please", "thank you", "sir", "madam"]):
            # المستخدم يفضل الأسلوب المهذب
            if self.personality_profile.communication_style != CommunicationStyle.PROFESSIONAL:
                self.personality_profile.communication_style = CommunicationStyle.PROFESSIONAL
        
        elif any(word in user_input for word in ["hey", "yo", "sup", "what's up"]):
            # المستخدم يفضل الأسلوب العادي
            if self.personality_profile.communication_style != CommunicationStyle.CASUAL:
                self.personality_profile.communication_style = CommunicationStyle.CASUAL
        
        elif interaction.satisfaction_score > 0.8 and "friend" in user_input:
            # المستخدم يقدر الأسلوب الودود
            self.personality_profile.communication_style = CommunicationStyle.FRIENDLY

    async def _save_personality_profile(self):
        """حفظ ملف الشخصية"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        traits_json = json.dumps({trait.value: value for trait, value in self.personality_profile.traits.items()})
        
        cursor.execute("""
            INSERT OR REPLACE INTO personality_profile 
            (id, traits, communication_style, preferred_topics, 
             response_patterns, adaptation_rate, last_updated)
            VALUES (1, ?, ?, ?, ?, ?, ?)
        """, (
            traits_json,
            self.personality_profile.communication_style.value,
            json.dumps(self.personality_profile.preferred_topics),
            json.dumps(self.personality_profile.response_patterns),
            self.personality_profile.adaptation_rate,
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()

    async def generate_adaptive_response(
        self,
        user_input: str,
        context: Dict[str, Any],
        base_response: str
    ) -> str:
        """توليد استجابة تكيفية"""
        
        try:
            # تحليل المزاج المطلوب
            required_mood = self._analyze_required_mood(user_input, context)
            
            # اختيار قالب الاستجابة
            style_templates = self.response_templates.get(
                self.personality_profile.communication_style,
                self.response_templates[CommunicationStyle.FRIENDLY]
            )
            
            # تعديل الاستجابة بناءً على الشخصية
            adapted_response = await self._adapt_response_style(base_response, required_mood, style_templates)
            
            # إضافة العناصر الشخصية
            personalized_response = await self._add_personality_elements(adapted_response, context)
            
            return personalized_response
            
        except Exception as e:
            self.logger.error(f"خطأ في توليد الاستجابة التكيفية: {e}")
            return base_response

    def _analyze_required_mood(self, user_input: str, context: Dict[str, Any]) -> str:
        """تحليل المزاج المطلوب للاستجابة"""
        
        user_input_lower = user_input.lower()
        
        if any(word in user_input_lower for word in ["help", "problem", "issue", "مساعدة", "مشكلة"]):
            return "supportive"
        elif any(word in user_input_lower for word in ["great", "awesome", "excellent", "رائع", "ممتاز"]):
            return "enthusiastic"
        elif any(word in user_input_lower for word in ["explain", "how", "what", "اشرح", "كيف", "ماذا"]):
            return "informative"
        else:
            return "neutral"

    async def _adapt_response_style(
        self,
        base_response: str,
        required_mood: str,
        style_templates: Dict[str, List[str]]
    ) -> str:
        """تكييف أسلوب الاستجابة"""
        
        adapted = base_response
        
        # إضافة عناصر الشخصية
        traits = self.personality_profile.traits
        
        if traits[PersonalityTrait.ENTHUSIASM] > 0.7 and required_mood == "enthusiastic":
            adapted = f"{np.random.choice(style_templates.get('enthusiasm', ['رائع!']))} {adapted}"
        
        if traits[PersonalityTrait.EMPATHY] > 0.7 and required_mood == "supportive":
            adapted = f"أفهم ما تمر به. {adapted}"
        
        if traits[PersonalityTrait.HUMOR] > 0.6 and required_mood == "neutral":
            adapted = f"{adapted} 😊"
        
        if traits[PersonalityTrait.PRECISION] > 0.8 and required_mood == "informative":
            adapted = f"بدقة: {adapted}"
        
        return adapted

    async def _add_personality_elements(self, response: str, context: Dict[str, Any]) -> str:
        """إضافة العناصر الشخصية"""
        
        # إضافة التخصيص بناءً على المواضيع المفضلة
        if "topic" in context and context["topic"] in self.personality_profile.preferred_topics:
            response = f"{response}\n\nمن الجميل أن نتحدث عن {context['topic']} مرة أخرى!"
        
        # إضافة الذكريات
        if len(self.interaction_history) > 0:
            last_interaction = self.interaction_history[-1]
            if (datetime.now() - last_interaction.timestamp).days == 0:
                response = f"{response}\n\nكما ذكرنا سابقاً اليوم..."
        
        return response

    async def get_personality_analysis(self) -> Dict[str, Any]:
        """الحصول على تحليل الشخصية"""
        
        try:
            # حساب الإحصائيات
            total_interactions = len(self.interaction_history)
            avg_satisfaction = np.mean([i.satisfaction_score for i in self.interaction_history if i.satisfaction_score is not None]) if self.interaction_history else 0
            
            # تحليل التطور
            trait_evolution = await self._analyze_trait_evolution()
            
            # أداء الشخصية
            performance_metrics = await self._calculate_performance_metrics()
            
            return {
                "current_personality": {
                    "traits": {trait.value: value for trait, value in self.personality_profile.traits.items()},
                    "communication_style": self.personality_profile.communication_style.value,
                    "adaptation_rate": self.personality_profile.adaptation_rate
                },
                "statistics": {
                    "total_interactions": total_interactions,
                    "average_satisfaction": round(avg_satisfaction, 2),
                    "preferred_topics": self.personality_profile.preferred_topics[:5]
                },
                "evolution": trait_evolution,
                "performance": performance_metrics,
                "last_updated": self.personality_profile.last_updated.isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"خطأ في تحليل الشخصية: {e}")
            return {"error": str(e)}

    async def _analyze_trait_evolution(self) -> Dict[str, Any]:
        """تحليل تطور السمات"""
        
        # هذا مبسط - في التطبيق الحقيقي نحتاج لتتبع التغييرات عبر الزمن
        evolution = {}
        
        for trait, value in self.personality_profile.traits.items():
            if value > 0.7:
                evolution[trait.value] = "مرتفع ومتطور"
            elif value > 0.4:
                evolution[trait.value] = "متوسط ومتوازن"
            else:
                evolution[trait.value] = "منخفض ويحتاج تطوير"
        
        return evolution

    async def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """حساب مقاييس الأداء"""
        
        if not self.interaction_history:
            return {"no_data": True}
        
        recent_interactions = self.interaction_history[-50:]  # آخر 50 تفاعل
        
        # حساب الاتجاهات
        satisfaction_trend = self._calculate_trend([i.satisfaction_score for i in recent_interactions if i.satisfaction_score is not None])
        
        # حساب معدل التحسن
        improvement_rate = satisfaction_trend if satisfaction_trend > 0 else 0
        
        return {
            "satisfaction_trend": satisfaction_trend,
            "improvement_rate": improvement_rate,
            "adaptability_score": self.personality_profile.adaptation_rate,
            "consistency_score": self._calculate_consistency()
        }

    def _calculate_trend(self, values: List[float]) -> float:
        """حساب الاتجاه في القيم"""
        
        if len(values) < 2:
            return 0.0
        
        # حساب بسيط للاتجاه
        first_half = np.mean(values[:len(values)//2])
        second_half = np.mean(values[len(values)//2:])
        
        return second_half - first_half

    def _calculate_consistency(self) -> float:
        """حساب اتساق الأداء"""
        
        if len(self.interaction_history) < 10:
            return 0.5
        
        recent_satisfaction = [i.satisfaction_score for i in self.interaction_history[-20:] if i.satisfaction_score is not None]
        
        if not recent_satisfaction:
            return 0.5
        
        # حساب الانحراف المعياري (أقل = أكثر اتساقاً)
        std_dev = np.std(recent_satisfaction)
        consistency = max(0, 1 - std_dev)
        
        return consistency

# إنشاء مثيل عام
personality_engine = AdaptivePersonalityEngine()

async def get_personality_engine() -> AdaptivePersonalityEngine:
    """الحصول على محرك الشخصية"""
    return personality_engine

if __name__ == "__main__":
    async def test_personality_engine():
        """اختبار محرك الشخصية"""
        print("🧠 اختبار محرك تطوير الشخصية التكيفي")
        print("=" * 50)
        
        engine = await get_personality_engine()
        await engine.initialize()
        
        # محاكاة تفاعلات
        test_interactions = [
            {
                "user_input": "مرحبا، كيف حالك؟",
                "assistant_response": "أهلاً! أنا بخير، كيف يمكنني مساعدتك؟",
                "emotion": "happy",
                "context": {"type": "greeting", "topic": "general"},
                "satisfaction": 0.8
            },
            {
                "user_input": "هل يمكنك شرح الذكاء الاصطناعي؟",
                "assistant_response": "بالطبع! الذكاء الاصطناعي هو...",
                "emotion": "curious",
                "context": {"type": "question", "topic": "ai"},
                "satisfaction": 0.9
            },
            {
                "user_input": "لم أفهم جوابك",
                "assistant_response": "آسف، دعني أوضح بشكل أبسط...",
                "emotion": "confused",
                "context": {"type": "clarification", "topic": "ai"},
                "satisfaction": 0.4
            }
        ]
        
        print("\n🔄 محاكاة التفاعلات والتعلم...")
        for interaction in test_interactions:
            await engine.record_interaction(
                user_input=interaction["user_input"],
                assistant_response=interaction["assistant_response"],
                emotion_detected=interaction["emotion"],
                context=interaction["context"],
                satisfaction_score=interaction["satisfaction"]
            )
        
        # توليد استجابة تكيفية
        print("\n🎭 اختبار توليد الاستجابة التكيفية...")
        base_response = "يمكنني مساعدتك في ذلك"
        adaptive_response = await engine.generate_adaptive_response(
            user_input="أحتاج مساعدة في مشروعي",
            context={"type": "help_request", "topic": "project"},
            base_response=base_response
        )
        print(f"📝 الاستجابة التكيفية: {adaptive_response}")
        
        # عرض تحليل الشخصية
        print("\n📊 تحليل الشخصية:")
        analysis = await engine.get_personality_analysis()
        print(f"🎯 أسلوب التواصل: {analysis['current_personality']['communication_style']}")
        print(f"📈 متوسط الرضا: {analysis['statistics']['average_satisfaction']}")
        print(f"🔄 معدل التكيف: {analysis['current_personality']['adaptation_rate']}")
    
    asyncio.run(test_personality_engine())
