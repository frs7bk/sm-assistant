
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
محرك الذكاء الاصطناعي المتطور
يدمج أحدث تقنيات الذكاء الاصطناعي والتعلم العميق
"""

import asyncio
import logging
import json
import time
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import threading
import queue
import sys

# إضافة مسار المشروع
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# المكتبات المتقدمة (مع معالجة الأخطاء)
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import (
        AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
        pipeline, GPT2LMHeadModel, GPT2Tokenizer
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import cv2
    import face_recognition
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from config.advanced_config import get_config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

@dataclass
class AIResponse:
    """استجابة محرك الذكاء الاصطناعي"""
    text: str
    confidence: float
    context: Dict[str, Any]
    emotions: Dict[str, float]
    entities: List[Dict[str, Any]]
    intent: str
    suggestions: List[str]
    metadata: Dict[str, Any]
    processing_time: float = 0.0
    model_used: str = "unknown"

@dataclass
class UserProfile:
    """ملف المستخدم الشخصي"""
    user_id: str
    preferences: Dict[str, Any]
    interaction_history: List[Dict[str, Any]]
    emotional_state: Dict[str, float]
    learning_progress: Dict[str, Any]
    goals: List[str]
    last_updated: datetime
    
    def __post_init__(self):
        if isinstance(self.last_updated, str):
            self.last_updated = datetime.fromisoformat(self.last_updated)

class NeuralMemoryNetwork(nn.Module):
    """شبكة الذاكرة العصبية المتقدمة"""
    
    def __init__(self, input_size: int = 768, hidden_size: int = 512, memory_size: int = 1000):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        
        # طبقات التشفير
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # آلية الانتباه
        self.attention = nn.MultiheadAttention(
            hidden_size, num_heads=8, dropout=0.1, batch_first=True
        )
        
        # ذاكرة طويلة المدى
        self.memory_bank = nn.Parameter(
            torch.randn(memory_size, hidden_size), requires_grad=True
        )
        
        # طبقة الإخراج
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, input_size)
        )
        
        # مؤشر الذاكرة
        self.memory_pointer = 0
    
    def forward(self, x: torch.Tensor, update_memory: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """المرور الأمامي عبر الشبكة"""
        batch_size = x.size(0)
        
        # تشفير الإدخال
        encoded = self.encoder(x)
        
        # الانتباه مع بنك الذاكرة
        memory_expanded = self.memory_bank.unsqueeze(0).expand(batch_size, -1, -1)
        attended, attention_weights = self.attention(
            encoded.unsqueeze(1), memory_expanded, memory_expanded
        )
        attended = attended.squeeze(1)
        
        # دمج المعلومات
        combined = torch.cat([encoded, attended], dim=-1)
        output = self.decoder(combined)
        
        # تحديث الذاكرة
        if update_memory and self.training:
            self._update_memory(encoded.detach())
        
        return output, attention_weights
    
    def _update_memory(self, new_memory: torch.Tensor):
        """تحديث بنك الذاكرة"""
        with torch.no_grad():
            # إضافة ذكريات جديدة بطريقة دائرية
            for memory in new_memory:
                self.memory_bank[self.memory_pointer] = memory
                self.memory_pointer = (self.memory_pointer + 1) % self.memory_size

class AdvancedAIEngine:
    """محرك الذكاء الاصطناعي المتطور"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # حالة المحرك
        self.is_initialized = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # التكوين
        self.config = get_config() if CONFIG_AVAILABLE else None
        
        # النماذج المحملة
        self.models = {}
        self.tokenizers = {}
        
        # الذاكرة العصبية
        self.neural_memory = None
        
        # ملفات المستخدمين
        self.user_profiles: Dict[str, UserProfile] = {}
        
        # إحصائيات الأداء
        self.performance_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "average_response_time": 0.0,
            "cache_hits": 0,
            "model_switches": 0
        }
        
        # مخزن الذاكرة
        self.memory_store = {}
        self.model_dir = Path("data/models")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # قائمة انتظار المعالجة
        self.processing_queue = queue.PriorityQueue()
        self.background_workers = []
    
    async def initialize(self):
        """تهيئة محرك الذكاء الاصطناعي"""
        self.logger.info("🧠 تهيئة محرك الذكاء الاصطناعي المتطور...")
        
        try:
            # تحميل النماذج الأساسية
            await self._load_base_models()
            
            # تهيئة الذاكرة العصبية
            self._initialize_neural_memory()
            
            # تحميل ملفات المستخدمين
            await self._load_user_profiles()
            
            # تشغيل العمال الخلفيين
            self._start_background_workers()
            
            self.is_initialized = True
            self.logger.info("✅ تم تهيئة محرك الذكاء الاصطناعي بنجاح")
            
        except Exception as e:
            self.logger.error(f"❌ فشل تهيئة محرك الذكاء الاصطناعي: {e}")
            # تهيئة في الوضع الأساسي
            self.is_initialized = True
    
    async def _load_base_models(self):
        """تحميل النماذج الأساسية"""
        if not TRANSFORMERS_AVAILABLE:
            self.logger.warning("⚠️ Transformers غير متاح - تشغيل في الوضع الأساسي")
            return
        
        try:
            # نموذج تحليل المشاعر
            self.models['sentiment'] = pipeline(
                'sentiment-analysis',
                model='cardiffnlp/twitter-xlm-roberta-base-sentiment',
                device=0 if torch.cuda.is_available() else -1
            )
            self.logger.info("✅ تم تحميل نموذج تحليل المشاعر")
            
            # نموذج استخراج الكيانات
            self.models['ner'] = pipeline(
                'ner',
                model='CAMeL-Lab/bert-base-arabic-camelbert-mix-ner',
                device=0 if torch.cuda.is_available() else -1,
                aggregation_strategy='simple'
            )
            self.logger.info("✅ تم تحميل نموذج استخراج الكيانات")
            
            # نموذج التضمين
            model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
            self.tokenizers['embedding'] = AutoTokenizer.from_pretrained(model_name)
            self.models['embedding'] = AutoModel.from_pretrained(model_name)
            self.models['embedding'].to(self.device)
            self.logger.info("✅ تم تحميل نموذج التضمين")
            
        except Exception as e:
            self.logger.warning(f"⚠️ خطأ في تحميل بعض النماذج: {e}")
    
    def _initialize_neural_memory(self):
        """تهيئة الذاكرة العصبية"""
        try:
            if TRANSFORMERS_AVAILABLE:
                self.neural_memory = NeuralMemoryNetwork()
                self.neural_memory.to(self.device)
                
                # تحميل حالة محفوظة إن وجدت
                memory_path = self.model_dir / "neural_memory.pth"
                if memory_path.exists():
                    self.neural_memory.load_state_dict(torch.load(memory_path, map_location=self.device))
                    self.logger.info("✅ تم تحميل الذاكرة العصبية المحفوظة")
                else:
                    self.logger.info("🧠 تم إنشاء ذاكرة عصبية جديدة")
            
        except Exception as e:
            self.logger.warning(f"⚠️ خطأ في تهيئة الذاكرة العصبية: {e}")
    
    async def _load_user_profiles(self):
        """تحميل ملفات المستخدمين"""
        try:
            profiles_path = self.model_dir / "user_profiles.json"
            
            if profiles_path.exists():
                with open(profiles_path, 'r', encoding='utf-8') as f:
                    profiles_data = json.load(f)
                
                for user_id, data in profiles_data.items():
                    self.user_profiles[user_id] = UserProfile(
                        user_id=user_id,
                        preferences=data.get('preferences', {}),
                        interaction_history=data.get('interaction_history', []),
                        emotional_state=data.get('emotional_state', {"neutral": 1.0}),
                        learning_progress=data.get('learning_progress', {}),
                        goals=data.get('goals', []),
                        last_updated=datetime.fromisoformat(data.get('last_updated', datetime.now().isoformat()))
                    )
                
                self.logger.info(f"✅ تم تحميل {len(self.user_profiles)} ملف مستخدم")
            
        except Exception as e:
            self.logger.warning(f"⚠️ خطأ في تحميل ملفات المستخدمين: {e}")
    
    def _start_background_workers(self):
        """تشغيل العمال الخلفيين"""
        num_workers = 2
        
        for i in range(num_workers):
            worker = threading.Thread(
                target=self._background_worker,
                name=f"AIWorker-{i}",
                daemon=True
            )
            worker.start()
            self.background_workers.append(worker)
        
        self.logger.info(f"🔄 تم تشغيل {num_workers} عامل خلفي")
    
    def _background_worker(self):
        """العامل الخلفي للمعالجة"""
        while True:
            try:
                priority, task = self.processing_queue.get(timeout=1)
                
                # تنفيذ المهمة
                task_func, args, kwargs = task
                task_func(*args, **kwargs)
                
                self.processing_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"خطأ في العامل الخلفي: {e}")
    
    async def process_natural_language(
        self, 
        text: str, 
        user_id: str = "default",
        context: Optional[Dict[str, Any]] = None
    ) -> AIResponse:
        """معالجة اللغة الطبيعية المتقدمة"""
        
        start_time = time.time()
        
        try:
            self.performance_stats["total_requests"] += 1
            
            # تنظيف النص
            cleaned_text = self._preprocess_text(text)
            
            # الحصول على ملف المستخدم
            user_profile = self._get_or_create_user_profile(user_id)
            
            # تحليل المشاعر
            emotions = await self._analyze_emotions(cleaned_text)
            
            # استخراج الكيانات
            entities = await self._extract_entities(cleaned_text)
            
            # تحديد القصد
            intent = await self._classify_intent(cleaned_text, context)
            
            # توليد الاستجابة
            response_text = await self._generate_response(
                cleaned_text, intent, emotions, entities, user_profile, context
            )
            
            # حساب مستوى الثقة
            confidence = self._calculate_confidence(intent, emotions, entities)
            
            # توليد الاقتراحات
            suggestions = await self._generate_suggestions(intent, context)
            
            # تحديث ملف المستخدم
            self._update_user_profile(user_profile, text, response_text, emotions, intent)
            
            # إنشاء الاستجابة
            response = AIResponse(
                text=response_text,
                confidence=confidence,
                context=context or {},
                emotions=emotions,
                entities=entities,
                intent=intent,
                suggestions=suggestions,
                metadata={
                    "user_id": user_id,
                    "model_used": "advanced_ai_engine",
                    "processing_steps": ["emotion_analysis", "entity_extraction", "intent_classification", "response_generation"]
                },
                processing_time=time.time() - start_time,
                model_used="advanced_ai_engine"
            )
            
            self.performance_stats["successful_requests"] += 1
            self._update_performance_stats(response.processing_time)
            
            return response
            
        except Exception as e:
            self.logger.error(f"خطأ في معالجة اللغة الطبيعية: {e}")
            
            # استجابة احتياطية
            return AIResponse(
                text="عذراً، حدث خطأ في المعالجة. يرجى المحاولة مرة أخرى.",
                confidence=0.0,
                context=context or {},
                emotions={"neutral": 1.0},
                entities=[],
                intent="error",
                suggestions=["إعادة المحاولة", "تبسيط السؤال"],
                metadata={"error": str(e)},
                processing_time=time.time() - start_time,
                model_used="fallback"
            )
    
    def _preprocess_text(self, text: str) -> str:
        """تنظيف وتحضير النص"""
        # إزالة المسافات الزائدة
        text = ' '.join(text.split())
        
        # تطبيع النص العربي (يمكن إضافة المزيد)
        text = text.strip()
        
        return text
    
    async def _analyze_emotions(self, text: str) -> Dict[str, float]:
        """تحليل المشاعر المتقدم"""
        try:
            if 'sentiment' in self.models and TRANSFORMERS_AVAILABLE:
                result = self.models['sentiment'](text)
                
                emotions = {"neutral": 0.5}
                
                if result:
                    label = result[0]['label'].lower()
                    score = float(result[0]['score'])
                    
                    if 'positive' in label or 'pos' in label:
                        emotions = {"happy": score, "neutral": 1-score}
                    elif 'negative' in label or 'neg' in label:
                        emotions = {"sad": score, "neutral": 1-score}
                    else:
                        emotions = {"neutral": score}
                
                return emotions
            else:
                return await self._basic_emotion_analysis(text)
                
        except Exception as e:
            self.logger.warning(f"خطأ في تحليل المشاعر: {e}")
            return {"neutral": 1.0}
    
    async def _basic_emotion_analysis(self, text: str) -> Dict[str, float]:
        """تحليل مشاعر أساسي"""
        positive_words = ["سعيد", "ممتاز", "رائع", "جيد", "مذهل", "حب", "فرح"]
        negative_words = ["حزين", "سيء", "فظيع", "مشكلة", "خطأ", "غضب", "كره"]
        
        text_lower = text.lower()
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        total = positive_count + negative_count
        
        if total == 0:
            return {"neutral": 1.0}
        
        emotions = {
            "happy": positive_count / total if positive_count > negative_count else 0.0,
            "sad": negative_count / total if negative_count > positive_count else 0.0,
            "neutral": 1.0 - max(positive_count, negative_count) / total
        }
        
        return emotions
    
    async def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """استخراج الكيانات المتقدم"""
        try:
            if 'ner' in self.models and TRANSFORMERS_AVAILABLE:
                entities = self.models['ner'](text)
                
                processed_entities = []
                for entity in entities:
                    processed_entities.append({
                        "text": entity['word'],
                        "label": entity['entity_group'],
                        "confidence": float(entity['score']),
                        "start": int(entity['start']),
                        "end": int(entity['end'])
                    })
                
                return processed_entities
            else:
                return await self._basic_entity_extraction(text)
                
        except Exception as e:
            self.logger.warning(f"خطأ في استخراج الكيانات: {e}")
            return []
    
    async def _basic_entity_extraction(self, text: str) -> List[Dict[str, Any]]:
        """استخراج كيانات أساسي"""
        entities = []
        
        # البحث عن أرقام
        import re
        numbers = re.findall(r'\d+', text)
        for num in numbers:
            entities.append({
                "text": num,
                "label": "NUMBER",
                "confidence": 0.8,
                "start": text.find(num),
                "end": text.find(num) + len(num)
            })
        
        return entities
    
    async def _classify_intent(self, text: str, context: Optional[Dict[str, Any]]) -> str:
        """تصنيف القصد المتقدم"""
        text_lower = text.lower()
        
        # قصود أساسية
        if any(word in text_lower for word in ["مرحبا", "أهلا", "سلام", "صباح", "مساء"]):
            return "greeting"
        elif any(word in text_lower for word in ["شكرا", "متشكر", "أشكرك"]):
            return "thanks"
        elif any(word in text_lower for word in ["وداعا", "مع السلامة", "إلى اللقاء"]):
            return "goodbye"
        elif "؟" in text or any(word in text_lower for word in ["كيف", "ماذا", "متى", "أين", "لماذا", "من"]):
            return "question"
        elif any(word in text_lower for word in ["ساعدني", "أريد", "أحتاج", "يمكنك"]):
            return "request"
        elif any(word in text_lower for word in ["لا", "توقف", "كفى", "إيقاف"]):
            return "stop"
        else:
            return "general"
    
    async def _generate_response(
        self,
        text: str,
        intent: str,
        emotions: Dict[str, float],
        entities: List[Dict[str, Any]],
        user_profile: UserProfile,
        context: Optional[Dict[str, Any]]
    ) -> str:
        """توليد الاستجابة المتقدم"""
        
        # استخدام GPT إذا كان متاحاً
        if OPENAI_AVAILABLE and self.config and self.config.ai_models.openai_api_key:
            try:
                return await self._generate_with_gpt(text, intent, emotions, user_profile, context)
            except Exception as e:
                self.logger.warning(f"خطأ في GPT: {e}")
        
        # استجابات محلية متقدمة
        return await self._generate_local_response(text, intent, emotions, entities, user_profile, context)
    
    async def _generate_with_gpt(
        self,
        text: str,
        intent: str,
        emotions: Dict[str, float],
        user_profile: UserProfile,
        context: Optional[Dict[str, Any]]
    ) -> str:
        """توليد الاستجابة باستخدام GPT"""
        try:
            openai.api_key = self.config.ai_models.openai_api_key
            
            # بناء الرسالة المتقدمة
            system_message = self._build_system_message(user_profile, emotions, context)
            
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": text}
                ],
                temperature=0.7,
                max_tokens=1024
            )
            
            return response['choices'][0]['message']['content']
            
        except Exception as e:
            self.logger.error(f"خطأ في GPT: {e}")
            raise
    
    def _build_system_message(
        self,
        user_profile: UserProfile,
        emotions: Dict[str, float],
        context: Optional[Dict[str, Any]]
    ) -> str:
        """بناء رسالة النظام المتقدمة"""
        
        base_message = "أنت مساعد ذكي متطور وودود. تتحدث بالعربية وتفهم السياق جيداً."
        
        # إضافة معلومات المستخدم
        if user_profile.preferences:
            base_message += f" تفضيلات المستخدم: {user_profile.preferences}"
        
        # إضافة الحالة العاطفية
        dominant_emotion = max(emotions, key=emotions.get)
        if dominant_emotion != "neutral":
            base_message += f" المستخدم يبدو {dominant_emotion}."
        
        # إضافة السياق
        if context and "recent_topics" in context:
            base_message += f" المواضيع الأخيرة: {context['recent_topics']}"
        
        return base_message
    
    async def _generate_local_response(
        self,
        text: str,
        intent: str,
        emotions: Dict[str, float],
        entities: List[Dict[str, Any]],
        user_profile: UserProfile,
        context: Optional[Dict[str, Any]]
    ) -> str:
        """توليد استجابة محلية متقدمة"""
        
        responses = {
            "greeting": [
                "مرحباً! كيف يمكنني مساعدتك اليوم؟",
                "أهلاً وسهلاً! ما الذي تريد أن نتحدث عنه؟",
                "السلام عليكم! أنا هنا للمساعدة."
            ],
            "thanks": [
                "عفواً! أسعدني أن أساعدك.",
                "لا شكر على واجب! هل تحتاج أي شيء آخر؟",
                "كل التقدير لك! كيف يمكنني خدمتك أكثر؟"
            ],
            "goodbye": [
                "وداعاً! أتمنى لك يوماً سعيداً.",
                "إلى اللقاء! أراك قريباً.",
                "مع السلامة! عودة ميمونة."
            ],
            "question": [
                "سؤال ممتاز! دعني أفكر في أفضل إجابة...",
                "هذا سؤال مهم. سأحاول مساعدتك بأفضل طريقة ممكنة.",
                "أقدر فضولك! إليك ما أعرفه..."
            ],
            "request": [
                "بالطبع! سأفعل ما بوسعي لمساعدتك.",
                "سأكون سعيداً لمساعدتك في ذلك.",
                "دعني أرى كيف يمكنني تحقيق طلبك."
            ]
        }
        
        # اختيار استجابة أساسية
        intent_responses = responses.get(intent, ["أفهم ما تقوله. كيف يمكنني مساعدتك؟"])
        
        # تخصيص الاستجابة حسب المشاعر
        dominant_emotion = max(emotions, key=emotions.get)
        
        if dominant_emotion == "sad" and emotions[dominant_emotion] > 0.6:
            response = "أشعر أنك قد تمر بوقت صعب. " + intent_responses[0]
        elif dominant_emotion == "happy" and emotions[dominant_emotion] > 0.6:
            response = "يسعدني أن أراك في مزاج جيد! " + intent_responses[0]
        else:
            import random
            response = random.choice(intent_responses)
        
        # إضافة معلومات عن الكيانات المستخرجة
        if entities:
            entity_texts = [entity['text'] for entity in entities]
            response += f" لاحظت أنك ذكرت: {', '.join(entity_texts[:3])}."
        
        return response
    
    def _calculate_confidence(
        self,
        intent: str,
        emotions: Dict[str, float],
        entities: List[Dict[str, Any]]
    ) -> float:
        """حساب مستوى الثقة"""
        
        base_confidence = 0.5
        
        # زيادة الثقة للقصود الواضحة
        clear_intents = ["greeting", "thanks", "goodbye"]
        if intent in clear_intents:
            base_confidence += 0.3
        
        # زيادة الثقة للمشاعر الواضحة
        max_emotion_score = max(emotions.values()) if emotions else 0
        base_confidence += max_emotion_score * 0.2
        
        # زيادة الثقة للكيانات المستخرجة
        if entities:
            avg_entity_confidence = sum(e['confidence'] for e in entities) / len(entities)
            base_confidence += avg_entity_confidence * 0.1
        
        return min(base_confidence, 1.0)
    
    async def _generate_suggestions(
        self,
        intent: str,
        context: Optional[Dict[str, Any]]
    ) -> List[str]:
        """توليد الاقتراحات"""
        
        suggestions_map = {
            "greeting": ["كيف يمكنني مساعدتك؟", "هل تريد أن نتحدث عن شيء معين؟"],
            "thanks": ["هل تحتاج مساعدة أخرى؟", "ما رأيك في موضوع جديد؟"],
            "goodbye": ["نراك قريباً!", "استمتع بباقي يومك!"],
            "question": ["هل تريد المزيد من التفاصيل؟", "هل لديك أسئلة أخرى؟"],
            "request": ["هل هذا ما كنت تبحث عنه؟", "هل تحتاج شيء آخر؟"]
        }
        
        base_suggestions = suggestions_map.get(intent, ["كيف يمكنني مساعدتك أكثر؟"])
        
        # إضافة اقتراحات تعتمد على السياق
        if context and "recent_topics" in context:
            base_suggestions.append("هل تريد مواصلة موضوع سابق؟")
        
        return base_suggestions[:3]  # أقصى 3 اقتراحات
    
    def _get_or_create_user_profile(self, user_id: str) -> UserProfile:
        """الحصول على ملف المستخدم أو إنشاؤه"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserProfile(
                user_id=user_id,
                preferences={},
                interaction_history=[],
                emotional_state={"neutral": 1.0},
                learning_progress={},
                goals=[],
                last_updated=datetime.now()
            )
        
        return self.user_profiles[user_id]
    
    def _update_user_profile(
        self,
        user_profile: UserProfile,
        input_text: str,
        response_text: str,
        emotions: Dict[str, float],
        intent: str
    ):
        """تحديث ملف المستخدم"""
        
        # إضافة التفاعل للتاريخ
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "input": input_text[:100],  # أول 100 حرف
            "response": response_text[:100],
            "emotions": emotions,
            "intent": intent
        }
        
        user_profile.interaction_history.append(interaction)
        
        # الحفاظ على آخر 50 تفاعل فقط
        if len(user_profile.interaction_history) > 50:
            user_profile.interaction_history = user_profile.interaction_history[-50:]
        
        # تحديث الحالة العاطفية
        for emotion, score in emotions.items():
            if emotion in user_profile.emotional_state:
                user_profile.emotional_state[emotion] = (
                    user_profile.emotional_state[emotion] * 0.8 + score * 0.2
                )
            else:
                user_profile.emotional_state[emotion] = score
        
        user_profile.last_updated = datetime.now()
    
    def _update_performance_stats(self, processing_time: float):
        """تحديث إحصائيات الأداء"""
        total = self.performance_stats["total_requests"]
        current_avg = self.performance_stats["average_response_time"]
        
        new_avg = (current_avg * (total - 1) + processing_time) / total
        self.performance_stats["average_response_time"] = new_avg
    
    async def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """تحليل الصور المتقدم"""
        try:
            if not CV2_AVAILABLE:
                return {"error": "مكتبات الرؤية الحاسوبية غير متاحة"}
            
            import cv2
            
            image = cv2.imread(image_path)
            if image is None:
                return {"error": "تعذر تحميل الصورة"}
            
            results = {
                "image_size": image.shape,
                "faces_detected": 0,
                "objects_detected": [],
                "colors_analysis": {},
                "brightness": 0.0
            }
            
            # كشف الوجوه
            try:
                if face_recognition:
                    face_locations = face_recognition.face_locations(image)
                    results["faces_detected"] = len(face_locations)
                else:
                    # استخدام OpenCV للكشف الأساسي
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                    results["faces_detected"] = len(faces)
                    
            except Exception as e:
                self.logger.warning(f"خطأ في كشف الوجوه: {e}")
            
            # تحليل الألوان
            try:
                # متوسط الألوان
                mean_color = np.mean(image, axis=(0, 1))
                results["colors_analysis"] = {
                    "dominant_blue": float(mean_color[0]),
                    "dominant_green": float(mean_color[1]),
                    "dominant_red": float(mean_color[2])
                }
                
                # السطوع
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                results["brightness"] = float(np.mean(gray))
                
            except Exception as e:
                self.logger.warning(f"خطأ في تحليل الألوان: {e}")
            
            return results
            
        except Exception as e:
            return {"error": str(e)}
    
    async def save_memory(self):
        """حفظ الذاكرة والنماذج"""
        try:
            # حفظ الذاكرة العصبية
            if self.neural_memory:
                memory_path = self.model_dir / "neural_memory.pth"
                torch.save(self.neural_memory.state_dict(), memory_path)
            
            # حفظ ملفات المستخدمين
            profiles_data = {}
            for user_id, profile in self.user_profiles.items():
                profiles_data[user_id] = {
                    "preferences": profile.preferences,
                    "interaction_history": profile.interaction_history,
                    "emotional_state": profile.emotional_state,
                    "learning_progress": profile.learning_progress,
                    "goals": profile.goals,
                    "last_updated": profile.last_updated.isoformat()
                }
            
            profiles_path = self.model_dir / "user_profiles.json"
            with open(profiles_path, 'w', encoding='utf-8') as f:
                json.dump(profiles_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info("✅ تم حفظ الذاكرة والملفات الشخصية")
            
        except Exception as e:
            self.logger.error(f"❌ فشل حفظ الذاكرة: {e}")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """الحصول على تقرير الأداء"""
        success_rate = (
            self.performance_stats["successful_requests"] / 
            max(self.performance_stats["total_requests"], 1)
        ) * 100
        
        return {
            "total_requests": self.performance_stats["total_requests"],
            "success_rate": f"{success_rate:.1f}%",
            "average_response_time": f"{self.performance_stats['average_response_time']:.3f}s",
            "models_loaded": len(self.models),
            "users_registered": len(self.user_profiles),
            "memory_initialized": self.neural_memory is not None,
            "device": str(self.device)
        }

# مثيل عام لمحرك الذكاء الاصطناعي
ai_engine = AdvancedAIEngine()

async def get_ai_engine() -> AdvancedAIEngine:
    """الحصول على محرك الذكاء الاصطناعي"""
    if not ai_engine.is_initialized:
        await ai_engine.initialize()
    return ai_engine

if __name__ == "__main__":
    async def main():
        """اختبار محرك الذكاء الاصطناعي"""
        print("🧠 اختبار محرك الذكاء الاصطناعي المتطور")
        print("=" * 50)
        
        engine = await get_ai_engine()
        
        # اختبار المعالجة
        test_inputs = [
            "مرحباً، كيف حالك؟",
            "أشعر بالحزن اليوم",
            "هل يمكنك مساعدتي في حل مشكلة؟",
            "شكراً لك على المساعدة"
        ]
        
        for text in test_inputs:
            print(f"\n📝 الإدخال: {text}")
            response = await engine.process_natural_language(text)
            print(f"🤖 الاستجابة: {response.text}")
            print(f"📊 الثقة: {response.confidence:.1%}")
            print(f"🎭 المشاعر: {response.emotions}")
            print(f"🎯 القصد: {response.intent}")
        
        # عرض تقرير الأداء
        print(f"\n📈 تقرير الأداء:")
        report = engine.get_performance_report()
        for key, value in report.items():
            print(f"   • {key}: {value}")
    
    asyncio.run(main())
