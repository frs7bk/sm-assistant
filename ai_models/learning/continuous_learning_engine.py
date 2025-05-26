
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
محرك التعلم المستمر والذاكرة طويلة المدى
نظام ذكي يتذكر كل شيء ويتعلم من كل تفاعل
"""

import asyncio
import logging
import json
import sqlite3
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import pickle
import hashlib
from dataclasses import dataclass, asdict
import threading
import queue
from collections import defaultdict, deque

@dataclass
class MemoryItem:
    """عنصر في الذاكرة"""
    id: str
    content: str
    context: Dict[str, Any]
    timestamp: datetime
    importance: float
    access_count: int
    last_accessed: datetime
    emotional_weight: float
    category: str
    embeddings: Optional[List[float]] = None
    related_items: List[str] = None

class LongTermMemorySystem:
    """نظام الذاكرة طويلة المدى"""
    
    def __init__(self, db_path: str = "data/memory/long_term.db"):
        self.logger = logging.getLogger(__name__)
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # إعداد قاعدة البيانات
        self._init_database()
        
        # ذاكرة قصيرة المدى (في الذاكرة)
        self.short_term_memory = deque(maxlen=1000)
        
        # فهرس للبحث السريع
        self.search_index = {}
        
        # إحصائيات الذاكرة
        self.memory_stats = {
            "total_items": 0,
            "categories": defaultdict(int),
            "avg_importance": 0.0,
            "most_accessed": None
        }
        
        # تحميل الفهرس
        self._load_search_index()
    
    def _init_database(self):
        """تهيئة قاعدة البيانات"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memory_items (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                context TEXT,
                timestamp REAL,
                importance REAL,
                access_count INTEGER DEFAULT 0,
                last_accessed REAL,
                emotional_weight REAL,
                category TEXT,
                embeddings BLOB,
                related_items TEXT
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp ON memory_items(timestamp);
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_importance ON memory_items(importance);
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_category ON memory_items(category);
        """)
        
        conn.commit()
        conn.close()
    
    def store_memory(self, content: str, context: Dict[str, Any], 
                    importance: float = 0.5, category: str = "general") -> str:
        """تخزين ذكرى جديدة"""
        memory_id = hashlib.md5(f"{content}{datetime.now()}".encode()).hexdigest()
        
        # تحديد الوزن العاطفي
        emotional_weight = self._calculate_emotional_weight(content, context)
        
        memory_item = MemoryItem(
            id=memory_id,
            content=content,
            context=context,
            timestamp=datetime.now(),
            importance=importance,
            access_count=0,
            last_accessed=datetime.now(),
            emotional_weight=emotional_weight,
            category=category
        )
        
        # إضافة للذاكرة قصيرة المدى
        self.short_term_memory.append(memory_item)
        
        # تخزين في قاعدة البيانات
        self._store_to_database(memory_item)
        
        # تحديث الفهرس
        self._update_search_index(memory_item)
        
        self.logger.info(f"تم تخزين ذكرى جديدة: {memory_id}")
        return memory_id
    
    def retrieve_memories(self, query: str, limit: int = 10, 
                         min_importance: float = 0.0) -> List[MemoryItem]:
        """استرجاع الذكريات المرتبطة"""
        # البحث في الذاكرة قصيرة المدى أولاً
        recent_matches = []
        for item in self.short_term_memory:
            if self._is_relevant(query, item):
                recent_matches.append(item)
        
        # البحث في قاعدة البيانات
        database_matches = self._search_database(query, limit, min_importance)
        
        # دمج النتائج وترتيبها
        all_matches = recent_matches + database_matches
        all_matches = sorted(all_matches, 
                           key=lambda x: x.importance * (1 + x.access_count * 0.1),
                           reverse=True)
        
        # تحديث عدد الوصول
        for item in all_matches[:limit]:
            self._update_access_count(item.id)
        
        return all_matches[:limit]
    
    def forget_low_importance(self, threshold: float = 0.1, 
                            older_than_days: int = 30):
        """نسيان الذكريات قليلة الأهمية"""
        cutoff_date = datetime.now() - timedelta(days=older_than_days)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            DELETE FROM memory_items 
            WHERE importance < ? AND timestamp < ? AND access_count < 2
        """, (threshold, cutoff_date.timestamp()))
        
        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()
        
        self.logger.info(f"تم حذف {deleted_count} ذكرى قليلة الأهمية")
        return deleted_count
    
    def _calculate_emotional_weight(self, content: str, context: Dict[str, Any]) -> float:
        """حساب الوزن العاطفي للذكرى"""
        emotional_keywords = {
            "سعيد": 0.8, "حزين": 0.7, "غاضب": 0.9, "خائف": 0.8,
            "متحمس": 0.7, "قلق": 0.6, "فخور": 0.8, "محبط": 0.6,
            "مبهج": 0.7, "صادم": 0.9, "مثير": 0.8, "مؤلم": 0.9
        }
        
        weight = 0.5  # وزن افتراضي
        content_lower = content.lower()
        
        for keyword, value in emotional_keywords.items():
            if keyword in content_lower:
                weight = max(weight, value)
        
        # زيادة الوزن إذا كان السياق يحتوي على مشاعر
        if "emotions" in context:
            emotions = context["emotions"]
            if isinstance(emotions, dict):
                max_emotion = max(emotions.values()) if emotions else 0.5
                weight = max(weight, max_emotion)
        
        return min(weight, 1.0)
    
    def _is_relevant(self, query: str, item: MemoryItem) -> bool:
        """تحديد ما إذا كانت الذكرى مرتبطة بالاستعلام"""
        query_lower = query.lower()
        content_lower = item.content.lower()
        
        # تطابق مباشر
        if query_lower in content_lower:
            return True
        
        # تطابق في السياق
        context_str = str(item.context).lower()
        if query_lower in context_str:
            return True
        
        # تطابق الكلمات المفتاحية
        query_words = set(query_lower.split())
        content_words = set(content_lower.split())
        
        overlap = len(query_words.intersection(content_words))
        return overlap > 0
    
    def _search_database(self, query: str, limit: int, 
                        min_importance: float) -> List[MemoryItem]:
        """البحث في قاعدة البيانات"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # استعلام مع تطابق النص
        cursor.execute("""
            SELECT * FROM memory_items 
            WHERE (content LIKE ? OR context LIKE ?) 
            AND importance >= ?
            ORDER BY importance DESC, access_count DESC
            LIMIT ?
        """, (f"%{query}%", f"%{query}%", min_importance, limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        memories = []
        for row in rows:
            context = json.loads(row[2]) if row[2] else {}
            related_items = json.loads(row[10]) if row[10] else []
            
            memory = MemoryItem(
                id=row[0],
                content=row[1],
                context=context,
                timestamp=datetime.fromtimestamp(row[3]),
                importance=row[4],
                access_count=row[5],
                last_accessed=datetime.fromtimestamp(row[6]),
                emotional_weight=row[7],
                category=row[8],
                related_items=related_items
            )
            memories.append(memory)
        
        return memories
    
    def _store_to_database(self, item: MemoryItem):
        """تخزين العنصر في قاعدة البيانات"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO memory_items 
            (id, content, context, timestamp, importance, access_count, 
             last_accessed, emotional_weight, category, related_items)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            item.id,
            item.content,
            json.dumps(item.context, ensure_ascii=False),
            item.timestamp.timestamp(),
            item.importance,
            item.access_count,
            item.last_accessed.timestamp(),
            item.emotional_weight,
            item.category,
            json.dumps(item.related_items or [], ensure_ascii=False)
        ))
        
        conn.commit()
        conn.close()
    
    def _update_access_count(self, memory_id: str):
        """تحديث عدد مرات الوصول"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE memory_items 
            SET access_count = access_count + 1, last_accessed = ?
            WHERE id = ?
        """, (datetime.now().timestamp(), memory_id))
        
        conn.commit()
        conn.close()
    
    def _load_search_index(self):
        """تحميل فهرس البحث"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM memory_items")
            self.memory_stats["total_items"] = cursor.fetchone()[0]
            
            conn.close()
        except Exception as e:
            self.logger.error(f"خطأ في تحميل الفهرس: {e}")
    
    def _update_search_index(self, item: MemoryItem):
        """تحديث فهرس البحث"""
        self.memory_stats["total_items"] += 1
        self.memory_stats["categories"][item.category] += 1

class ContinuousLearningEngine:
    """محرك التعلم المستمر"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # نظام الذاكرة
        self.memory_system = LongTermMemorySystem()
        
        # نماذج التعلم
        self.learning_models = {
            "preferences": {},  # تفضيلات المستخدم
            "patterns": {},     # أنماط السلوك
            "responses": {},    # فعالية الاستجابات
            "context": {}       # فهم السياق
        }
        
        # إعدادات التعلم
        self.learning_settings = {
            "adaptation_rate": 0.1,
            "memory_threshold": 0.3,
            "pattern_detection": True,
            "feedback_learning": True,
            "auto_improvement": True
        }
        
        # قائمة انتظار التعلم
        self.learning_queue = queue.Queue()
        self.learning_worker = None
        
        # إحصائيات التعلم
        self.learning_stats = {
            "total_interactions": 0,
            "successful_adaptations": 0,
            "pattern_discoveries": 0,
            "preference_updates": 0
        }
        
        self._start_learning_worker()
    
    async def process_interaction(self, user_input: str, context: Dict[str, Any], 
                                response: Dict[str, Any], feedback: Optional[Dict[str, Any]] = None):
        """معالجة تفاعل للتعلم منه"""
        
        # تخزين في الذاكرة
        memory_context = {
            "user_input": user_input,
            "response": response,
            "feedback": feedback,
            "context": context,
            "timestamp": datetime.now().isoformat()
        }
        
        importance = self._calculate_interaction_importance(
            user_input, context, response, feedback
        )
        
        memory_id = self.memory_system.store_memory(
            content=f"تفاعل: {user_input}",
            context=memory_context,
            importance=importance,
            category="interaction"
        )
        
        # إضافة للتعلم
        learning_item = {
            "memory_id": memory_id,
            "user_input": user_input,
            "context": context,
            "response": response,
            "feedback": feedback,
            "importance": importance
        }
        
        self.learning_queue.put(learning_item)
        self.learning_stats["total_interactions"] += 1
        
        self.logger.info(f"تم معالجة تفاعل للتعلم: {memory_id}")
    
    async def adapt_response(self, user_input: str, context: Dict[str, Any], 
                          base_response: Dict[str, Any]) -> Dict[str, Any]:
        """تكييف الاستجابة بناءً على التعلم السابق"""
        
        # البحث عن تفاعلات مشابهة
        similar_memories = self.memory_system.retrieve_memories(
            user_input, limit=5, min_importance=0.3
        )
        
        adapted_response = base_response.copy()
        
        # تطبيق التعلم من الذكريات المشابهة
        for memory in similar_memories:
            memory_context = memory.context
            
            if "feedback" in memory_context and memory_context["feedback"]:
                feedback = memory_context["feedback"]
                
                # تحسين على أساس التغذية الراجعة السابقة
                if feedback.get("satisfaction", 0) > 0.7:
                    # استخدام نفس النمط للاستجابات الناجحة
                    if "tone" in memory_context.get("response", {}):
                        adapted_response["tone"] = memory_context["response"]["tone"]
                
                elif feedback.get("satisfaction", 0) < 0.3:
                    # تجنب أنماط الاستجابات الفاشلة
                    if "avoid_patterns" not in adapted_response:
                        adapted_response["avoid_patterns"] = []
                    
                    failed_response = memory_context.get("response", {})
                    adapted_response["avoid_patterns"].append(failed_response.get("text", ""))
        
        # تطبيق تفضيلات المستخدم المتعلمة
        user_id = context.get("user_id", "default")
        if user_id in self.learning_models["preferences"]:
            preferences = self.learning_models["preferences"][user_id]
            
            # تطبيق تفضيلات اللغة
            if "language_style" in preferences:
                adapted_response["style"] = preferences["language_style"]
            
            # تطبيق تفضيلات الطول
            if "response_length" in preferences:
                adapted_response["preferred_length"] = preferences["response_length"]
        
        adapted_response["adaptation_applied"] = True
        adapted_response["similar_memories_count"] = len(similar_memories)
        
        return adapted_response
    
    def learn_from_feedback(self, interaction_id: str, feedback: Dict[str, Any]):
        """التعلم من التغذية الراجعة"""
        
        # البحث عن التفاعل في الذاكرة
        memories = self.memory_system.retrieve_memories(interaction_id, limit=1)
        
        if memories:
            memory = memories[0]
            memory_context = memory.context
            
            # تحديث السياق مع التغذية الراجعة
            memory_context["feedback"] = feedback
            
            # تحديث الأهمية بناءً على التغذية الراجعة
            satisfaction = feedback.get("satisfaction", 0.5)
            new_importance = min(memory.importance + satisfaction * 0.2, 1.0)
            
            # إعادة تخزين مع الأهمية المحدثة
            self.memory_system.store_memory(
                content=memory.content,
                context=memory_context,
                importance=new_importance,
                category=memory.category
            )
            
            # تحديث نماذج التعلم
            self._update_learning_models(memory_context, feedback)
            
            self.learning_stats["successful_adaptations"] += 1
            self.logger.info(f"تم التعلم من التغذية الراجعة: {interaction_id}")
    
    def _calculate_interaction_importance(self, user_input: str, context: Dict[str, Any], 
                                        response: Dict[str, Any], feedback: Optional[Dict[str, Any]]) -> float:
        """حساب أهمية التفاعل"""
        
        importance = 0.5  # أهمية أساسية
        
        # زيادة الأهمية للأسئلة المعقدة
        if len(user_input.split()) > 10:
            importance += 0.1
        
        # زيادة الأهمية للاستجابات عالية الثقة
        confidence = response.get("confidence", 0.5)
        importance += confidence * 0.2
        
        # زيادة الأهمية عند وجود تغذية راجعة
        if feedback:
            satisfaction = feedback.get("satisfaction", 0.5)
            importance += satisfaction * 0.3
        
        # زيادة الأهمية للسياقات المعقدة
        if len(context) > 3:
            importance += 0.1
        
        return min(importance, 1.0)
    
    def _start_learning_worker(self):
        """بدء عامل التعلم في الخلفية"""
        def learning_worker():
            while True:
                try:
                    item = self.learning_queue.get(timeout=1)
                    if item is None:
                        break
                    
                    self._process_learning_item(item)
                    self.learning_queue.task_done()
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    self.logger.error(f"خطأ في عامل التعلم: {e}")
        
        self.learning_worker = threading.Thread(target=learning_worker, daemon=True)
        self.learning_worker.start()
    
    def _process_learning_item(self, item: Dict[str, Any]):
        """معالجة عنصر تعلم"""
        try:
            user_input = item["user_input"]
            context = item["context"]
            response = item["response"]
            feedback = item.get("feedback")
            
            # كشف الأنماط
            self._detect_patterns(user_input, context, response)
            
            # تحديث التفضيلات
            self._update_user_preferences(context, response, feedback)
            
            # تحسين فهم السياق
            self._improve_context_understanding(context, response)
            
        except Exception as e:
            self.logger.error(f"خطأ في معالجة التعلم: {e}")
    
    def _detect_patterns(self, user_input: str, context: Dict[str, Any], response: Dict[str, Any]):
        """كشف الأنماط في التفاعلات"""
        
        user_id = context.get("user_id", "default")
        
        # تحليل أنماط الوقت
        current_hour = datetime.now().hour
        if user_id not in self.learning_models["patterns"]:
            self.learning_models["patterns"][user_id] = {
                "time_patterns": defaultdict(int),
                "topic_patterns": defaultdict(int),
                "response_patterns": defaultdict(int)
            }
        
        patterns = self.learning_models["patterns"][user_id]
        
        # تسجيل نمط الوقت
        time_category = "morning" if current_hour < 12 else "afternoon" if current_hour < 18 else "evening"
        patterns["time_patterns"][time_category] += 1
        
        # تسجيل نمط الموضوع
        intent = response.get("intent", "general")
        patterns["topic_patterns"][intent] += 1
        
        # تسجيل نمط الاستجابة
        response_type = "short" if len(response.get("text", "")) < 100 else "long"
        patterns["response_patterns"][response_type] += 1
        
        self.learning_stats["pattern_discoveries"] += 1
    
    def _update_user_preferences(self, context: Dict[str, Any], response: Dict[str, Any], 
                               feedback: Optional[Dict[str, Any]]):
        """تحديث تفضيلات المستخدم"""
        
        user_id = context.get("user_id", "default")
        
        if user_id not in self.learning_models["preferences"]:
            self.learning_models["preferences"][user_id] = {
                "language_style": "formal",
                "response_length": "medium",
                "detail_level": "moderate",
                "topics_of_interest": defaultdict(float)
            }
        
        preferences = self.learning_models["preferences"][user_id]
        
        if feedback:
            satisfaction = feedback.get("satisfaction", 0.5)
            
            # تحديث تفضيل طول الاستجابة
            response_length = len(response.get("text", ""))
            if satisfaction > 0.7:
                if response_length < 100:
                    preferences["response_length"] = "short"
                elif response_length > 300:
                    preferences["response_length"] = "long"
                else:
                    preferences["response_length"] = "medium"
            
            # تحديث مستوى التفاصيل
            detail_indicators = ["تفاصيل", "اشرح", "وضح", "كيف"]
            if any(indicator in context.get("user_input", "").lower() for indicator in detail_indicators):
                if satisfaction > 0.7:
                    preferences["detail_level"] = "high"
                elif satisfaction < 0.3:
                    preferences["detail_level"] = "low"
        
        self.learning_stats["preference_updates"] += 1
    
    def _improve_context_understanding(self, context: Dict[str, Any], response: Dict[str, Any]):
        """تحسين فهم السياق"""
        
        # تحليل العلاقات بين السياق والاستجابة
        context_keys = list(context.keys())
        response_intent = response.get("intent", "general")
        
        if response_intent not in self.learning_models["context"]:
            self.learning_models["context"][response_intent] = {
                "important_context_keys": defaultdict(int),
                "success_indicators": defaultdict(float)
            }
        
        context_model = self.learning_models["context"][response_intent]
        
        # تسجيل مفاتيح السياق المهمة
        for key in context_keys:
            context_model["important_context_keys"][key] += 1
        
        # تسجيل مؤشرات النجاح
        confidence = response.get("confidence", 0.5)
        for key in context_keys:
            current_success = context_model["success_indicators"][key]
            # متوسط متحرك للنجاح
            context_model["success_indicators"][key] = (current_success * 0.9) + (confidence * 0.1)
    
    def _update_learning_models(self, memory_context: Dict[str, Any], feedback: Dict[str, Any]):
        """تحديث نماذج التعلم بناءً على التغذية الراجعة"""
        
        satisfaction = feedback.get("satisfaction", 0.5)
        user_id = memory_context.get("context", {}).get("user_id", "default")
        
        # تحديث فعالية الاستجابات
        response = memory_context.get("response", {})
        response_pattern = {
            "intent": response.get("intent"),
            "confidence": response.get("confidence"),
            "length": len(response.get("text", ""))
        }
        
        pattern_key = f"{response_pattern['intent']}_{response_pattern['confidence']:.1f}"
        
        if user_id not in self.learning_models["responses"]:
            self.learning_models["responses"][user_id] = {}
        
        if pattern_key not in self.learning_models["responses"][user_id]:
            self.learning_models["responses"][user_id][pattern_key] = {
                "total_uses": 0,
                "avg_satisfaction": 0.0
            }
        
        pattern_data = self.learning_models["responses"][user_id][pattern_key]
        pattern_data["total_uses"] += 1
        
        # متوسط متحرك للرضا
        current_avg = pattern_data["avg_satisfaction"]
        new_avg = (current_avg * (pattern_data["total_uses"] - 1) + satisfaction) / pattern_data["total_uses"]
        pattern_data["avg_satisfaction"] = new_avg
    
    def get_learning_insights(self) -> Dict[str, Any]:
        """الحصول على رؤى التعلم"""
        
        insights = {
            "memory_stats": self.memory_system.memory_stats,
            "learning_stats": self.learning_stats,
            "total_users": len(self.learning_models["preferences"]),
            "pattern_discoveries": self.learning_stats["pattern_discoveries"],
            "adaptation_success_rate": 0.0
        }
        
        # حساب معدل نجاح التكيف
        total_interactions = self.learning_stats["total_interactions"]
        if total_interactions > 0:
            insights["adaptation_success_rate"] = (
                self.learning_stats["successful_adaptations"] / total_interactions
            )
        
        return insights
    
    async def cleanup_old_data(self, days_threshold: int = 90):
        """تنظيف البيانات القديمة"""
        
        # حذف الذكريات قليلة الأهمية
        deleted_memories = self.memory_system.forget_low_importance(
            threshold=0.2, older_than_days=days_threshold
        )
        
        # تنظيف نماذج التعلم غير المستخدمة
        cleaned_models = 0
        for user_id in list(self.learning_models["preferences"].keys()):
            # إزالة المستخدمين غير النشطين
            if user_id not in self.learning_models["patterns"]:
                del self.learning_models["preferences"][user_id]
                cleaned_models += 1
        
        self.logger.info(f"تم تنظيف {deleted_memories} ذكرى و {cleaned_models} نموذج تعلم")
        
        return {
            "deleted_memories": deleted_memories,
            "cleaned_models": cleaned_models
        }

# إنشاء مثيل عام
continuous_learning_engine = ContinuousLearningEngine()

def get_continuous_learning_engine() -> ContinuousLearningEngine:
    """الحصول على محرك التعلم المستمر"""
    return continuous_learning_engine

if __name__ == "__main__":
    # اختبار النظام
    async def test_learning_system():
        engine = get_continuous_learning_engine()
        
        # محاكاة تفاعل
        await engine.process_interaction(
            user_input="ما هو الطقس اليوم؟",
            context={"user_id": "test_user", "location": "الرياض"},
            response={"text": "الطقس مشمس اليوم", "confidence": 0.8, "intent": "weather"},
            feedback={"satisfaction": 0.9, "helpful": True}
        )
        
        # اختبار التكيف
        adapted = await engine.adapt_response(
            user_input="كيف الطقس؟",
            context={"user_id": "test_user", "location": "الرياض"},
            base_response={"text": "الطقس جميل", "confidence": 0.7}
        )
        
        print("اختبار التعلم المستمر اكتمل بنجاح!")
        print(f"الاستجابة المكيفة: {adapted}")
    
    asyncio.run(test_learning_system())
