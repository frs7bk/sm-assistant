
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
محرك التنبؤ الذكي المتقدم
يتنبأ بالاحتياجات والسلوكيات ويقترح الحلول قبل الطلب
"""

import asyncio
import logging
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import pickle
from dataclasses import dataclass, asdict
import threading
import queue
from collections import defaultdict, deque
import sqlite3

@dataclass
class PredictionPattern:
    """نمط التنبؤ"""
    pattern_id: str
    pattern_type: str  # time, behavior, context, hybrid
    frequency: float
    confidence: float
    conditions: Dict[str, Any]
    expected_action: str
    last_occurrence: datetime
    success_rate: float

class SmartPredictionEngine:
    """محرك التنبؤ الذكي"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # قاعدة بيانات الأنماط
        self.patterns_db_path = Path("data/predictions/patterns.db")
        self.patterns_db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_patterns_database()
        
        # الأنماط المكتشفة
        self.discovered_patterns = {}
        
        # تاريخ السلوكيات
        self.behavior_history = deque(maxlen=10000)
        
        # نماذج التنبؤ
        self.prediction_models = {
            "time_based": {},      # أنماط زمنية
            "behavior_based": {},  # أنماط سلوكية
            "context_based": {},   # أنماط سياقية
            "hybrid": {}          # أنماط مختلطة
        }
        
        # إحصائيات التنبؤ
        self.prediction_stats = {
            "total_predictions": 0,
            "successful_predictions": 0,
            "patterns_discovered": 0,
            "accuracy_rate": 0.0,
            "user_satisfaction": 0.0
        }
        
        # إعدادات التنبؤ
        self.prediction_settings = {
            "min_pattern_frequency": 3,
            "min_confidence_threshold": 0.6,
            "prediction_horizon_hours": 24,
            "max_predictions_per_session": 5,
            "learning_rate": 0.1
        }
        
        # قائمة التنبؤات النشطة
        self.active_predictions = {}
        
        # بدء عامل التنبؤ
        self._start_prediction_worker()
    
    def _init_patterns_database(self):
        """تهيئة قاعدة بيانات الأنماط"""
        conn = sqlite3.connect(self.patterns_db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS behavior_patterns (
                pattern_id TEXT PRIMARY KEY,
                pattern_type TEXT,
                frequency REAL,
                confidence REAL,
                conditions TEXT,
                expected_action TEXT,
                last_occurrence REAL,
                success_rate REAL,
                created_at REAL,
                updated_at REAL
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prediction_history (
                prediction_id TEXT PRIMARY KEY,
                pattern_id TEXT,
                predicted_at REAL,
                actual_action TEXT,
                was_correct BOOLEAN,
                user_feedback REAL,
                context TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    async def analyze_behavior(self, user_action: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """تحليل السلوك واكتشاف الأنماط"""
        try:
            # تسجيل السلوك
            behavior_entry = {
                "timestamp": datetime.now(),
                "action": user_action,
                "context": context,
                "hour": datetime.now().hour,
                "day_of_week": datetime.now().weekday(),
                "day_of_month": datetime.now().day
            }
            
            self.behavior_history.append(behavior_entry)
            
            # اكتشاف الأنماط الجديدة
            await self._discover_patterns()
            
            # تحديث النماذج الموجودة
            await self._update_prediction_models(behavior_entry)
            
            # توليد تنبؤات جديدة
            predictions = await self._generate_predictions(context)
            
            return {
                "behavior_recorded": True,
                "patterns_updated": True,
                "new_predictions": predictions,
                "total_patterns": len(self.discovered_patterns)
            }
        
        except Exception as e:
            self.logger.error(f"خطأ في تحليل السلوك: {e}")
            return {"error": str(e)}
    
    async def _discover_patterns(self):
        """اكتشاف أنماط جديدة"""
        try:
            if len(self.behavior_history) < 10:
                return
            
            # تحليل الأنماط الزمنية
            await self._discover_time_patterns()
            
            # تحليل الأنماط السلوكية
            await self._discover_behavior_patterns()
            
            # تحليل الأنماط السياقية
            await self._discover_context_patterns()
            
            # تحليل الأنماط المختلطة
            await self._discover_hybrid_patterns()
            
        except Exception as e:
            self.logger.error(f"خطأ في اكتشاف الأنماط: {e}")
    
    async def _discover_time_patterns(self):
        """اكتشاف الأنماط الزمنية"""
        # تحليل أنماط الساعة
        hour_patterns = defaultdict(list)
        for entry in self.behavior_history:
            hour_patterns[entry["hour"]].append(entry["action"])
        
        for hour, actions in hour_patterns.items():
            if len(actions) >= self.prediction_settings["min_pattern_frequency"]:
                most_common_action = max(set(actions), key=actions.count)
                frequency = actions.count(most_common_action) / len(actions)
                
                if frequency >= 0.5:  # 50% من الوقت
                    pattern_id = f"time_hour_{hour}_{most_common_action}"
                    pattern = PredictionPattern(
                        pattern_id=pattern_id,
                        pattern_type="time_based",
                        frequency=frequency,
                        confidence=min(frequency * 1.2, 1.0),
                        conditions={"hour": hour},
                        expected_action=most_common_action,
                        last_occurrence=datetime.now(),
                        success_rate=0.0
                    )
                    
                    self.discovered_patterns[pattern_id] = pattern
                    self.prediction_models["time_based"][pattern_id] = pattern
    
    async def _discover_behavior_patterns(self):
        """اكتشاف الأنماط السلوكية (تسلسل الأفعال)"""
        # تحليل تسلسل الأفعال
        sequences = []
        for i in range(len(self.behavior_history) - 2):
            sequence = [
                self.behavior_history[i]["action"],
                self.behavior_history[i + 1]["action"],
                self.behavior_history[i + 2]["action"]
            ]
            sequences.append(tuple(sequence))
        
        # العثور على التسلسلات المتكررة
        sequence_counts = defaultdict(int)
        for seq in sequences:
            sequence_counts[seq] += 1
        
        for sequence, count in sequence_counts.items():
            if count >= self.prediction_settings["min_pattern_frequency"]:
                pattern_id = f"behavior_seq_{'_'.join(sequence)}"
                pattern = PredictionPattern(
                    pattern_id=pattern_id,
                    pattern_type="behavior_based",
                    frequency=count / len(sequences),
                    confidence=min((count / len(sequences)) * 1.5, 1.0),
                    conditions={"previous_actions": list(sequence[:-1])},
                    expected_action=sequence[-1],
                    last_occurrence=datetime.now(),
                    success_rate=0.0
                )
                
                self.discovered_patterns[pattern_id] = pattern
                self.prediction_models["behavior_based"][pattern_id] = pattern
    
    async def _discover_context_patterns(self):
        """اكتشاف الأنماط السياقية"""
        # تحليل السياق والأفعال
        context_patterns = defaultdict(list)
        
        for entry in self.behavior_history:
            context = entry["context"]
            if context:
                # تحليل مفاتيح السياق المهمة
                for key, value in context.items():
                    if isinstance(value, (str, int, float, bool)):
                        context_key = f"{key}:{value}"
                        context_patterns[context_key].append(entry["action"])
        
        for context_key, actions in context_patterns.items():
            if len(actions) >= self.prediction_settings["min_pattern_frequency"]:
                most_common_action = max(set(actions), key=actions.count)
                frequency = actions.count(most_common_action) / len(actions)
                
                if frequency >= 0.4:  # 40% من الوقت
                    pattern_id = f"context_{context_key}_{most_common_action}"
                    key, value = context_key.split(":", 1)
                    
                    pattern = PredictionPattern(
                        pattern_id=pattern_id,
                        pattern_type="context_based",
                        frequency=frequency,
                        confidence=frequency,
                        conditions={key: value},
                        expected_action=most_common_action,
                        last_occurrence=datetime.now(),
                        success_rate=0.0
                    )
                    
                    self.discovered_patterns[pattern_id] = pattern
                    self.prediction_models["context_based"][pattern_id] = pattern
    
    async def _discover_hybrid_patterns(self):
        """اكتشاف الأنماط المختلطة (زمنية + سياقية)"""
        # دمج الأنماط الزمنية والسياقية
        hybrid_patterns = defaultdict(list)
        
        for entry in self.behavior_history:
            hour = entry["hour"]
            day_of_week = entry["day_of_week"]
            context = entry["context"]
            
            # نمط الوقت + نوع النشاط
            if "activity_type" in context:
                hybrid_key = f"time_{hour}_activity_{context['activity_type']}"
                hybrid_patterns[hybrid_key].append(entry["action"])
            
            # نمط يوم الأسبوع + السياق
            if "location" in context:
                hybrid_key = f"day_{day_of_week}_location_{context['location']}"
                hybrid_patterns[hybrid_key].append(entry["action"])
        
        for hybrid_key, actions in hybrid_patterns.items():
            if len(actions) >= self.prediction_settings["min_pattern_frequency"]:
                most_common_action = max(set(actions), key=actions.count)
                frequency = actions.count(most_common_action) / len(actions)
                
                if frequency >= 0.6:  # 60% من الوقت للأنماط المختلطة
                    pattern_id = f"hybrid_{hybrid_key}_{most_common_action}"
                    
                    # تحليل الشروط
                    conditions = {}
                    parts = hybrid_key.split("_")
                    for i in range(0, len(parts), 2):
                        if i + 1 < len(parts):
                            conditions[parts[i]] = parts[i + 1]
                    
                    pattern = PredictionPattern(
                        pattern_id=pattern_id,
                        pattern_type="hybrid",
                        frequency=frequency,
                        confidence=min(frequency * 1.3, 1.0),
                        conditions=conditions,
                        expected_action=most_common_action,
                        last_occurrence=datetime.now(),
                        success_rate=0.0
                    )
                    
                    self.discovered_patterns[pattern_id] = pattern
                    self.prediction_models["hybrid"][pattern_id] = pattern
    
    async def _generate_predictions(self, current_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """توليد التنبؤات بناءً على السياق الحالي"""
        predictions = []
        current_hour = datetime.now().hour
        current_day = datetime.now().weekday()
        
        try:
            # فحص جميع الأنماط المكتشفة
            for pattern_id, pattern in self.discovered_patterns.items():
                if pattern.confidence < self.prediction_settings["min_confidence_threshold"]:
                    continue
                
                # فحص الشروط
                conditions_met = True
                
                for condition_key, condition_value in pattern.conditions.items():
                    if condition_key == "hour" and condition_value != current_hour:
                        conditions_met = False
                        break
                    elif condition_key == "day" and condition_value != current_day:
                        conditions_met = False
                        break
                    elif condition_key in current_context:
                        if str(current_context[condition_key]) != str(condition_value):
                            conditions_met = False
                            break
                    elif condition_key == "previous_actions":
                        # فحص الأفعال السابقة
                        if len(self.behavior_history) >= len(condition_value):
                            recent_actions = [
                                entry["action"] for entry in 
                                list(self.behavior_history)[-len(condition_value):]
                            ]
                            if recent_actions != condition_value:
                                conditions_met = False
                                break
                        else:
                            conditions_met = False
                            break
                
                if conditions_met:
                    prediction = {
                        "prediction_id": f"pred_{datetime.now().timestamp()}_{pattern_id}",
                        "pattern_id": pattern_id,
                        "expected_action": pattern.expected_action,
                        "confidence": pattern.confidence,
                        "pattern_type": pattern.pattern_type,
                        "reasoning": self._generate_reasoning(pattern),
                        "suggested_time": datetime.now() + timedelta(minutes=5),
                        "priority": self._calculate_priority(pattern)
                    }
                    predictions.append(prediction)
            
            # ترتيب التنبؤات حسب الأولوية
            predictions.sort(key=lambda x: x["priority"], reverse=True)
            
            # الحد من عدد التنبؤات
            max_predictions = self.prediction_settings["max_predictions_per_session"]
            predictions = predictions[:max_predictions]
            
            # حفظ التنبؤات النشطة
            for prediction in predictions:
                self.active_predictions[prediction["prediction_id"]] = prediction
            
            self.prediction_stats["total_predictions"] += len(predictions)
            
            return predictions
        
        except Exception as e:
            self.logger.error(f"خطأ في توليد التنبؤات: {e}")
            return []
    
    def _generate_reasoning(self, pattern: PredictionPattern) -> str:
        """توليد تفسير للتنبؤ"""
        reasoning_templates = {
            "time_based": f"بناءً على نشاطك في الساعة {pattern.conditions.get('hour', 'هذه')}، عادة ما تقوم بـ {pattern.expected_action}",
            "behavior_based": f"بناءً على الأنشطة الأخيرة، التالي عادة ما يكون {pattern.expected_action}",
            "context_based": f"في هذا السياق، عادة ما تحتاج إلى {pattern.expected_action}",
            "hybrid": f"بناءً على الوقت والسياق، من المتوقع أن تحتاج لـ {pattern.expected_action}"
        }
        
        base_reasoning = reasoning_templates.get(pattern.pattern_type, f"متوقع: {pattern.expected_action}")
        confidence_text = f" (ثقة: {pattern.confidence:.1%})"
        
        return base_reasoning + confidence_text
    
    def _calculate_priority(self, pattern: PredictionPattern) -> float:
        """حساب أولوية التنبؤ"""
        priority = pattern.confidence * pattern.frequency
        
        # زيادة الأولوية للأنماط الناجحة
        if pattern.success_rate > 0:
            priority *= (1 + pattern.success_rate)
        
        # زيادة الأولوية للأنماط المختلطة
        if pattern.pattern_type == "hybrid":
            priority *= 1.2
        
        return min(priority, 1.0)
    
    async def validate_prediction(self, prediction_id: str, actual_action: str, 
                                user_feedback: float = 0.5) -> Dict[str, Any]:
        """تقييم دقة التنبؤ"""
        try:
            if prediction_id not in self.active_predictions:
                return {"error": "التنبؤ غير موجود"}
            
            prediction = self.active_predictions[prediction_id]
            pattern_id = prediction["pattern_id"]
            expected_action = prediction["expected_action"]
            
            was_correct = (actual_action == expected_action)
            
            # تحديث إحصائيات التنبؤ
            if was_correct:
                self.prediction_stats["successful_predictions"] += 1
            
            # تحديث معدل النجاح للنمط
            pattern = self.discovered_patterns.get(pattern_id)
            if pattern:
                if pattern.success_rate == 0.0:
                    pattern.success_rate = 1.0 if was_correct else 0.0
                else:
                    # متوسط متحرك
                    pattern.success_rate = (pattern.success_rate * 0.8) + (0.2 if was_correct else 0.0)
                
                # تحديث الثقة بناءً على الأداء
                if was_correct:
                    pattern.confidence = min(pattern.confidence * 1.1, 1.0)
                else:
                    pattern.confidence = max(pattern.confidence * 0.9, 0.1)
            
            # حفظ في التاريخ
            await self._save_prediction_validation(prediction_id, actual_action, was_correct, user_feedback)
            
            # حساب معدل الدقة العام
            total_predictions = self.prediction_stats["total_predictions"]
            if total_predictions > 0:
                self.prediction_stats["accuracy_rate"] = (
                    self.prediction_stats["successful_predictions"] / total_predictions
                )
            
            # تحديث رضا المستخدم
            current_satisfaction = self.prediction_stats["user_satisfaction"]
            self.prediction_stats["user_satisfaction"] = (current_satisfaction * 0.9) + (user_feedback * 0.1)
            
            # إزالة من التنبؤات النشطة
            del self.active_predictions[prediction_id]
            
            return {
                "validated": True,
                "was_correct": was_correct,
                "pattern_updated": True,
                "new_accuracy_rate": self.prediction_stats["accuracy_rate"],
                "user_satisfaction": self.prediction_stats["user_satisfaction"]
            }
        
        except Exception as e:
            self.logger.error(f"خطأ في تقييم التنبؤ: {e}")
            return {"error": str(e)}
    
    async def _save_prediction_validation(self, prediction_id: str, actual_action: str, 
                                        was_correct: bool, user_feedback: float):
        """حفظ تقييم التنبؤ في قاعدة البيانات"""
        try:
            conn = sqlite3.connect(self.patterns_db_path)
            cursor = conn.cursor()
            
            prediction = self.active_predictions.get(prediction_id, {})
            
            cursor.execute("""
                INSERT INTO prediction_history 
                (prediction_id, pattern_id, predicted_at, actual_action, was_correct, user_feedback, context)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                prediction_id,
                prediction.get("pattern_id", ""),
                datetime.now().timestamp(),
                actual_action,
                was_correct,
                user_feedback,
                json.dumps(prediction.get("context", {}), ensure_ascii=False)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"خطأ في حفظ تقييم التنبؤ: {e}")
    
    def _start_prediction_worker(self):
        """بدء عامل التنبؤ في الخلفية"""
        def prediction_worker():
            while True:
                try:
                    # تنظيف التنبؤات المنتهية الصلاحية
                    asyncio.run(self._cleanup_expired_predictions())
                    
                    # حفظ الأنماط في قاعدة البيانات
                    asyncio.run(self._save_patterns_to_database())
                    
                    # انتظار 5 دقائق قبل الدورة التالية
                    threading.Event().wait(300)
                    
                except Exception as e:
                    self.logger.error(f"خطأ في عامل التنبؤ: {e}")
        
        worker_thread = threading.Thread(target=prediction_worker, daemon=True)
        worker_thread.start()
    
    async def _cleanup_expired_predictions(self):
        """تنظيف التنبؤات المنتهية الصلاحية"""
        current_time = datetime.now()
        expired_predictions = []
        
        for pred_id, prediction in self.active_predictions.items():
            suggested_time = prediction.get("suggested_time", current_time)
            if isinstance(suggested_time, str):
                suggested_time = datetime.fromisoformat(suggested_time)
            
            # إزالة التنبؤات التي مر عليها أكثر من ساعة
            if (current_time - suggested_time).total_seconds() > 3600:
                expired_predictions.append(pred_id)
        
        for pred_id in expired_predictions:
            del self.active_predictions[pred_id]
    
    async def _save_patterns_to_database(self):
        """حفظ الأنماط في قاعدة البيانات"""
        try:
            conn = sqlite3.connect(self.patterns_db_path)
            cursor = conn.cursor()
            
            for pattern_id, pattern in self.discovered_patterns.items():
                cursor.execute("""
                    INSERT OR REPLACE INTO behavior_patterns 
                    (pattern_id, pattern_type, frequency, confidence, conditions, 
                     expected_action, last_occurrence, success_rate, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    pattern.pattern_id,
                    pattern.pattern_type,
                    pattern.frequency,
                    pattern.confidence,
                    json.dumps(pattern.conditions, ensure_ascii=False),
                    pattern.expected_action,
                    pattern.last_occurrence.timestamp(),
                    pattern.success_rate,
                    datetime.now().timestamp(),
                    datetime.now().timestamp()
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"خطأ في حفظ الأنماط: {e}")
    
    async def get_smart_suggestions(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """الحصول على اقتراحات ذكية للمستخدم"""
        try:
            predictions = await self._generate_predictions(context)
            
            suggestions = []
            for prediction in predictions:
                suggestion = {
                    "type": "predictive_suggestion",
                    "action": prediction["expected_action"],
                    "confidence": prediction["confidence"],
                    "reasoning": prediction["reasoning"],
                    "priority": prediction["priority"],
                    "estimated_time": prediction["suggested_time"].isoformat() if isinstance(prediction["suggested_time"], datetime) else prediction["suggested_time"]
                }
                suggestions.append(suggestion)
            
            return suggestions
        
        except Exception as e:
            self.logger.error(f"خطأ في الحصول على الاقتراحات: {e}")
            return []
    
    def get_prediction_insights(self) -> Dict[str, Any]:
        """الحصول على رؤى التنبؤ"""
        try:
            return {
                "stats": self.prediction_stats,
                "total_patterns": len(self.discovered_patterns),
                "patterns_by_type": {
                    pattern_type: len(patterns) 
                    for pattern_type, patterns in self.prediction_models.items()
                },
                "active_predictions_count": len(self.active_predictions),
                "behavior_history_size": len(self.behavior_history),
                "top_patterns": self._get_top_patterns(),
                "prediction_accuracy_trend": self._calculate_accuracy_trend()
            }
        except Exception as e:
            self.logger.error(f"خطأ في الحصول على رؤى التنبؤ: {e}")
            return {"error": str(e)}
    
    def _get_top_patterns(self) -> List[Dict[str, Any]]:
        """الحصول على أفضل الأنماط"""
        patterns_list = list(self.discovered_patterns.values())
        patterns_list.sort(key=lambda p: p.confidence * p.frequency, reverse=True)
        
        top_patterns = []
        for pattern in patterns_list[:5]:
            top_patterns.append({
                "pattern_id": pattern.pattern_id,
                "type": pattern.pattern_type,
                "confidence": pattern.confidence,
                "frequency": pattern.frequency,
                "success_rate": pattern.success_rate,
                "expected_action": pattern.expected_action
            })
        
        return top_patterns
    
    def _calculate_accuracy_trend(self) -> List[float]:
        """حساب اتجاه دقة التنبؤات"""
        # هذه مبسطة - في التطبيق الحقيقي نحتاج بيانات تاريخية أكثر
        accuracy = self.prediction_stats["accuracy_rate"]
        return [max(0.0, accuracy - 0.1), accuracy, min(1.0, accuracy + 0.05)]

# إنشاء مثيل عام
smart_prediction_engine = SmartPredictionEngine()

def get_smart_prediction_engine() -> SmartPredictionEngine:
    """الحصول على محرك التنبؤ الذكي"""
    return smart_prediction_engine

if __name__ == "__main__":
    # اختبار النظام
    async def test_prediction_system():
        engine = get_smart_prediction_engine()
        
        # محاكاة سلوكيات
        behaviors = [
            {"action": "فتح بريد إلكتروني", "context": {"hour": 9, "activity_type": "work"}},
            {"action": "كتابة كود", "context": {"hour": 10, "activity_type": "work"}},
            {"action": "استراحة", "context": {"hour": 12, "activity_type": "break"}},
            {"action": "فتح بريد إلكتروني", "context": {"hour": 9, "activity_type": "work"}},
            {"action": "كتابة كود", "context": {"hour": 10, "activity_type": "work"}},
        ]
        
        for behavior in behaviors:
            result = await engine.analyze_behavior(behavior["action"], behavior["context"])
            print(f"تحليل السلوك: {result}")
        
        # الحصول على رؤى
        insights = engine.get_prediction_insights()
        print(f"رؤى التنبؤ: {insights}")
    
    asyncio.run(test_prediction_system())
