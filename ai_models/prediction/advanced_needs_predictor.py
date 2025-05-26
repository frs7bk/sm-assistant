
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
محرك التنبؤ المتقدم بالاحتياجات
Advanced Needs Prediction Engine
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import sqlite3
from enum import Enum
import pickle
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class NeedCategory(Enum):
    """فئات الاحتياجات"""
    WORK = "work"
    HEALTH = "health"
    ENTERTAINMENT = "entertainment"
    SOCIAL = "social"
    LEARNING = "learning"
    SHOPPING = "shopping"
    TRAVEL = "travel"
    FINANCE = "finance"
    FOOD = "food"
    MAINTENANCE = "maintenance"

class Priority(Enum):
    """مستويات الأولوية"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5

@dataclass
class PredictedNeed:
    """حاجة متوقعة"""
    need_id: str
    category: NeedCategory
    description: str
    predicted_time: datetime
    confidence: float
    priority: Priority
    context: Dict[str, Any]
    suggested_actions: List[str]
    
@dataclass
class UserPattern:
    """نمط المستخدم"""
    pattern_id: str
    name: str
    frequency: str  # daily, weekly, monthly
    typical_times: List[str]
    conditions: Dict[str, Any]
    associated_needs: List[str]
    accuracy: float

class AdvancedNeedsPredictor:
    """محرك التنبؤ المتقدم بالاحتياجات"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # قاعدة البيانات
        self.db_path = Path("data/needs_predictor.db")
        self.db_path.parent.mkdir(exist_ok=True)
        
        # النماذج التنبؤية
        self.models = {
            "time_predictor": RandomForestRegressor(n_estimators=100),
            "category_classifier": GradientBoostingClassifier(n_estimators=100),
            "priority_predictor": RandomForestRegressor(n_estimators=50)
        }
        
        self.scalers = {
            "features": StandardScaler(),
            "time": StandardScaler()
        }
        
        # البيانات التاريخية
        self.historical_needs: List[Dict[str, Any]] = []
        self.user_patterns: Dict[str, UserPattern] = {}
        
        # متغيرات السياق
        self.current_context = {
            "day_of_week": datetime.now().weekday(),
            "hour": datetime.now().hour,
            "season": self._get_season(),
            "weather_influence": 0.5,  # محاكاة
            "stress_level": 0.3,
            "energy_level": 0.7,
            "social_activity": 0.5
        }
        
        # قواعد التنبؤ الذكية
        self.prediction_rules = {
            "morning_routine": {
                "time_range": (6, 10),
                "typical_needs": ["work", "health", "food"],
                "confidence_boost": 0.2
            },
            "lunch_time": {
                "time_range": (11, 14),
                "typical_needs": ["food", "social"],
                "confidence_boost": 0.3
            },
            "evening_wind_down": {
                "time_range": (18, 22),
                "typical_needs": ["entertainment", "social", "health"],
                "confidence_boost": 0.25
            },
            "weekend_pattern": {
                "days": [5, 6],  # Saturday, Sunday
                "typical_needs": ["entertainment", "social", "maintenance"],
                "confidence_boost": 0.15
            }
        }

    async def initialize(self):
        """تهيئة محرك التنبؤ"""
        
        try:
            self.logger.info("🔮 تهيئة محرك التنبؤ المتقدم بالاحتياجات...")
            
            # إنشاء قاعدة البيانات
            await self._initialize_database()
            
            # تحميل البيانات التاريخية
            await self._load_historical_data()
            
            # تدريب النماذج
            await self._train_prediction_models()
            
            # اكتشاف الأنماط
            await self._discover_user_patterns()
            
            self.logger.info("✅ تم تهيئة محرك التنبؤ")
            
        except Exception as e:
            self.logger.error(f"❌ فشل تهيئة محرك التنبؤ: {e}")

    async def _initialize_database(self):
        """تهيئة قاعدة البيانات"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # جدول الاحتياجات التاريخية
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS historical_needs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                category TEXT NOT NULL,
                description TEXT,
                fulfilled_time TEXT,
                priority INTEGER,
                context TEXT,
                outcome TEXT
            )
        """)
        
        # جدول أنماط المستخدم
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_patterns (
                pattern_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                frequency TEXT,
                typical_times TEXT,
                conditions TEXT,
                associated_needs TEXT,
                accuracy REAL
            )
        """)
        
        # جدول التنبؤات
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                prediction_id TEXT PRIMARY KEY,
                predicted_time TEXT NOT NULL,
                category TEXT NOT NULL,
                description TEXT,
                confidence REAL,
                priority INTEGER,
                status TEXT,
                actual_outcome TEXT
            )
        """)
        
        conn.commit()
        conn.close()

    async def _load_historical_data(self):
        """تحميل البيانات التاريخية"""
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # تحميل الاحتياجات التاريخية
            cursor.execute("SELECT * FROM historical_needs ORDER BY timestamp DESC LIMIT 1000")
            for row in cursor.fetchall():
                need_data = {
                    "id": row[0],
                    "timestamp": datetime.fromisoformat(row[1]),
                    "category": row[2],
                    "description": row[3],
                    "fulfilled_time": datetime.fromisoformat(row[4]) if row[4] else None,
                    "priority": row[5],
                    "context": json.loads(row[6]) if row[6] else {},
                    "outcome": row[7]
                }
                self.historical_needs.append(need_data)
            
            # إضافة بيانات تجريبية إذا كانت قاعدة البيانات فارغة
            if not self.historical_needs:
                await self._generate_sample_data()
            
            conn.close()
            
        except Exception as e:
            self.logger.error(f"خطأ في تحميل البيانات التاريخية: {e}")

    async def _generate_sample_data(self):
        """توليد بيانات تجريبية"""
        
        sample_needs = [
            {"category": "work", "description": "مراجعة الإيميلات", "hour": 9, "priority": 3},
            {"category": "food", "description": "طلب الغداء", "hour": 12, "priority": 4},
            {"category": "health", "description": "شرب الماء", "hour": 14, "priority": 2},
            {"category": "entertainment", "description": "مشاهدة فيديو", "hour": 19, "priority": 1},
            {"category": "social", "description": "التحدث مع الأصدقاء", "hour": 20, "priority": 2}
        ]
        
        base_date = datetime.now() - timedelta(days=30)
        
        for i in range(100):  # 100 حاجة تجريبية
            for need in sample_needs:
                # إضافة تنويع عشوائي
                date = base_date + timedelta(days=i//5, hours=need["hour"] + np.random.randint(-2, 3))
                
                need_data = {
                    "id": len(self.historical_needs) + 1,
                    "timestamp": date,
                    "category": need["category"],
                    "description": need["description"],
                    "fulfilled_time": date + timedelta(minutes=np.random.randint(5, 60)),
                    "priority": need["priority"],
                    "context": {
                        "day_of_week": date.weekday(),
                        "hour": date.hour,
                        "weather": np.random.choice(["sunny", "cloudy", "rainy"]),
                        "stress_level": np.random.uniform(0, 1)
                    },
                    "outcome": np.random.choice(["fulfilled", "postponed", "ignored"])
                }
                
                self.historical_needs.append(need_data)

    async def _train_prediction_models(self):
        """تدريب النماذج التنبؤية"""
        
        if len(self.historical_needs) < 50:
            self.logger.warning("⚠️ بيانات غير كافية لتدريب النماذج")
            return
        
        try:
            # إعداد البيانات للتدريب
            features, targets = await self._prepare_training_data()
            
            if len(features) == 0:
                return
            
            # تقسيم البيانات
            X_train, X_test, y_time_train, y_time_test = train_test_split(
                features, targets["time"], test_size=0.2, random_state=42
            )
            
            _, _, y_cat_train, y_cat_test = train_test_split(
                features, targets["category"], test_size=0.2, random_state=42
            )
            
            _, _, y_priority_train, y_priority_test = train_test_split(
                features, targets["priority"], test_size=0.2, random_state=42
            )
            
            # تطبيع البيانات
            X_train_scaled = self.scalers["features"].fit_transform(X_train)
            X_test_scaled = self.scalers["features"].transform(X_test)
            
            # تدريب النماذج
            self.models["time_predictor"].fit(X_train_scaled, y_time_train)
            self.models["category_classifier"].fit(X_train_scaled, y_cat_train)
            self.models["priority_predictor"].fit(X_train_scaled, y_priority_train)
            
            # تقييم الأداء
            time_score = self.models["time_predictor"].score(X_test_scaled, y_time_test)
            cat_score = self.models["category_classifier"].score(X_test_scaled, y_cat_test)
            priority_score = self.models["priority_predictor"].score(X_test_scaled, y_priority_test)
            
            self.logger.info(f"📊 أداء النماذج - الوقت: {time_score:.2f}, الفئة: {cat_score:.2f}, الأولوية: {priority_score:.2f}")
            
        except Exception as e:
            self.logger.error(f"خطأ في تدريب النماذج: {e}")

    async def _prepare_training_data(self) -> Tuple[List[List[float]], Dict[str, List]]:
        """إعداد البيانات للتدريب"""
        
        features = []
        targets = {"time": [], "category": [], "priority": []}
        
        for need in self.historical_needs:
            if need["fulfilled_time"]:
                # استخراج الميزات
                feature_vector = [
                    need["context"].get("day_of_week", 0),
                    need["context"].get("hour", 12),
                    need["context"].get("stress_level", 0.5),
                    need["priority"],
                    hash(need["category"]) % 100,  # تشفير الفئة
                    len(need["description"]),
                    (need["fulfilled_time"] - need["timestamp"]).total_seconds() / 3600  # الوقت للتنفيذ بالساعات
                ]
                
                features.append(feature_vector)
                
                # الأهداف
                time_to_fulfillment = (need["fulfilled_time"] - need["timestamp"]).total_seconds() / 3600
                targets["time"].append(time_to_fulfillment)
                targets["category"].append(need["category"])
                targets["priority"].append(need["priority"])
        
        return features, targets

    async def _discover_user_patterns(self):
        """اكتشاف أنماط المستخدم"""
        
        try:
            # تحليل الأنماط الزمنية
            await self._analyze_temporal_patterns()
            
            # تحليل أنماط الفئات
            await self._analyze_category_patterns()
            
            # تحليل أنماط السياق
            await self._analyze_context_patterns()
            
        except Exception as e:
            self.logger.error(f"خطأ في اكتشاف الأنماط: {e}")

    async def _analyze_temporal_patterns(self):
        """تحليل الأنماط الزمنية"""
        
        # تجميع البيانات حسب الساعة
        hourly_data = {}
        for need in self.historical_needs:
            hour = need["timestamp"].hour
            category = need["category"]
            
            if hour not in hourly_data:
                hourly_data[hour] = {}
            
            if category not in hourly_data[hour]:
                hourly_data[hour][category] = 0
            
            hourly_data[hour][category] += 1
        
        # اكتشاف الأنماط
        for hour, categories in hourly_data.items():
            dominant_category = max(categories, key=categories.get)
            frequency = categories[dominant_category]
            
            if frequency >= 3:  # حد أدنى للتكرار
                pattern = UserPattern(
                    pattern_id=f"hourly_{hour}_{dominant_category}",
                    name=f"نمط الساعة {hour} - {dominant_category}",
                    frequency="daily",
                    typical_times=[str(hour)],
                    conditions={"hour": hour},
                    associated_needs=[dominant_category],
                    accuracy=min(frequency / 10, 1.0)
                )
                
                self.user_patterns[pattern.pattern_id] = pattern

    async def _analyze_category_patterns(self):
        """تحليل أنماط الفئات"""
        
        # تحليل تسلسل الفئات
        category_sequences = []
        
        sorted_needs = sorted(self.historical_needs, key=lambda x: x["timestamp"])
        
        for i in range(len(sorted_needs) - 2):
            sequence = [
                sorted_needs[i]["category"],
                sorted_needs[i+1]["category"],
                sorted_needs[i+2]["category"]
            ]
            category_sequences.append(sequence)
        
        # البحث عن التسلسلات المتكررة
        sequence_counts = {}
        for seq in category_sequences:
            seq_str = "->".join(seq)
            sequence_counts[seq_str] = sequence_counts.get(seq_str, 0) + 1
        
        # إنشاء أنماط للتسلسلات الشائعة
        for sequence, count in sequence_counts.items():
            if count >= 3:
                pattern = UserPattern(
                    pattern_id=f"sequence_{hash(sequence) % 1000}",
                    name=f"نمط التسلسل: {sequence}",
                    frequency="variable",
                    typical_times=[],
                    conditions={"sequence": sequence.split("->")},
                    associated_needs=sequence.split("->"),
                    accuracy=min(count / 10, 1.0)
                )
                
                self.user_patterns[pattern.pattern_id] = pattern

    async def _analyze_context_patterns(self):
        """تحليل أنماط السياق"""
        
        # تحليل تأثير الطقس
        weather_influence = {}
        for need in self.historical_needs:
            weather = need["context"].get("weather", "unknown")
            category = need["category"]
            
            if weather not in weather_influence:
                weather_influence[weather] = {}
            
            if category not in weather_influence[weather]:
                weather_influence[weather][category] = 0
            
            weather_influence[weather][category] += 1
        
        # إنشاء أنماط الطقس
        for weather, categories in weather_influence.items():
            if weather != "unknown" and len(categories) > 0:
                dominant_category = max(categories, key=categories.get)
                
                pattern = UserPattern(
                    pattern_id=f"weather_{weather}_{dominant_category}",
                    name=f"نمط الطقس {weather} - {dominant_category}",
                    frequency="conditional",
                    typical_times=[],
                    conditions={"weather": weather},
                    associated_needs=[dominant_category],
                    accuracy=0.6
                )
                
                self.user_patterns[pattern.pattern_id] = pattern

    def _get_season(self) -> str:
        """تحديد الموسم الحالي"""
        
        month = datetime.now().month
        if month in [12, 1, 2]:
            return "winter"
        elif month in [3, 4, 5]:
            return "spring"
        elif month in [6, 7, 8]:
            return "summer"
        else:
            return "autumn"

    async def predict_upcoming_needs(
        self,
        time_horizon: int = 24,  # ساعات
        max_predictions: int = 10
    ) -> List[PredictedNeed]:
        """التنبؤ بالاحتياجات القادمة"""
        
        try:
            predictions = []
            current_time = datetime.now()
            
            # التنبؤ بناءً على النماذج المدربة
            ml_predictions = await self._predict_with_ml_models(time_horizon)
            predictions.extend(ml_predictions)
            
            # التنبؤ بناءً على الأنماط المكتشفة
            pattern_predictions = await self._predict_with_patterns(time_horizon)
            predictions.extend(pattern_predictions)
            
            # التنبؤ بناءً على القواعد الذكية
            rule_predictions = await self._predict_with_rules(time_horizon)
            predictions.extend(rule_predictions)
            
            # ترتيب وتصفية التنبؤات
            predictions = sorted(predictions, key=lambda x: (x.predicted_time, -x.confidence))
            
            # إزالة التكرارات وتحديد العدد
            unique_predictions = []
            seen_descriptions = set()
            
            for pred in predictions:
                if pred.description not in seen_descriptions and len(unique_predictions) < max_predictions:
                    unique_predictions.append(pred)
                    seen_descriptions.add(pred.description)
            
            return unique_predictions
            
        except Exception as e:
            self.logger.error(f"خطأ في التنبؤ بالاحتياجات: {e}")
            return []

    async def _predict_with_ml_models(self, time_horizon: int) -> List[PredictedNeed]:
        """التنبؤ باستخدام نماذج التعلم الآلي"""
        
        predictions = []
        
        try:
            # إنشاء نقاط زمنية للتنبؤ
            time_points = []
            for hour_offset in range(1, time_horizon + 1):
                future_time = datetime.now() + timedelta(hours=hour_offset)
                time_points.append(future_time)
            
            for future_time in time_points:
                # إعداد ميزات السياق للوقت المستقبلي
                context_features = [
                    future_time.weekday(),
                    future_time.hour,
                    self.current_context["stress_level"],
                    3,  # أولوية افتراضية
                    50,  # فئة افتراضية مشفرة
                    10,  # طول وصف افتراضي
                    1.0  # وقت للتنفيذ افتراضي
                ]
                
                # تطبيق التطبيع
                features_scaled = self.scalers["features"].transform([context_features])
                
                # التنبؤ
                if hasattr(self.models["time_predictor"], "predict"):
                    time_pred = self.models["time_predictor"].predict(features_scaled)[0]
                    category_pred = self.models["category_classifier"].predict(features_scaled)[0]
                    priority_pred = self.models["priority_predictor"].predict(features_scaled)[0]
                    
                    # حساب الثقة بناءً على الوقت والسياق
                    confidence = self._calculate_ml_confidence(future_time, context_features)
                    
                    if confidence > 0.3:  # حد أدنى للثقة
                        prediction = PredictedNeed(
                            need_id=f"ml_{future_time.isoformat()}_{category_pred}",
                            category=NeedCategory(category_pred) if category_pred in [c.value for c in NeedCategory] else NeedCategory.WORK,
                            description=self._generate_need_description(category_pred, future_time),
                            predicted_time=future_time,
                            confidence=confidence,
                            priority=Priority(max(1, min(5, int(priority_pred)))),
                            context={"source": "ml_model", "hour": future_time.hour},
                            suggested_actions=self._generate_suggested_actions(category_pred)
                        )
                        
                        predictions.append(prediction)
            
        except Exception as e:
            self.logger.error(f"خطأ في التنبؤ بالنماذج: {e}")
        
        return predictions

    async def _predict_with_patterns(self, time_horizon: int) -> List[PredictedNeed]:
        """التنبؤ بناءً على الأنماط المكتشفة"""
        
        predictions = []
        
        for pattern in self.user_patterns.values():
            try:
                # التحقق من شروط النمط
                if await self._pattern_conditions_met(pattern):
                    # تحديد الوقت المتوقع
                    predicted_time = await self._calculate_pattern_time(pattern)
                    
                    if predicted_time and predicted_time <= datetime.now() + timedelta(hours=time_horizon):
                        for need_category in pattern.associated_needs:
                            prediction = PredictedNeed(
                                need_id=f"pattern_{pattern.pattern_id}_{need_category}",
                                category=NeedCategory(need_category) if need_category in [c.value for c in NeedCategory] else NeedCategory.WORK,
                                description=self._generate_need_description(need_category, predicted_time),
                                predicted_time=predicted_time,
                                confidence=pattern.accuracy * 0.8,  # تقليل الثقة قليلاً
                                priority=Priority.MEDIUM,
                                context={"source": "pattern", "pattern_id": pattern.pattern_id},
                                suggested_actions=self._generate_suggested_actions(need_category)
                            )
                            
                            predictions.append(prediction)
            
            except Exception as e:
                self.logger.error(f"خطأ في تطبيق النمط {pattern.pattern_id}: {e}")
        
        return predictions

    async def _predict_with_rules(self, time_horizon: int) -> List[PredictedNeed]:
        """التنبؤ بناءً على القواعد الذكية"""
        
        predictions = []
        
        current_time = datetime.now()
        
        for rule_name, rule_config in self.prediction_rules.items():
            try:
                # التحقق من شروط القاعدة
                if await self._rule_conditions_met(rule_name, rule_config):
                    # حساب الوقت المتوقع
                    predicted_time = await self._calculate_rule_time(rule_config)
                    
                    if predicted_time and predicted_time <= current_time + timedelta(hours=time_horizon):
                        for need_category in rule_config["typical_needs"]:
                            confidence = 0.6 + rule_config.get("confidence_boost", 0)
                            
                            prediction = PredictedNeed(
                                need_id=f"rule_{rule_name}_{need_category}",
                                category=NeedCategory(need_category) if need_category in [c.value for c in NeedCategory] else NeedCategory.WORK,
                                description=self._generate_need_description(need_category, predicted_time),
                                predicted_time=predicted_time,
                                confidence=min(confidence, 1.0),
                                priority=Priority.MEDIUM,
                                context={"source": "rule", "rule_name": rule_name},
                                suggested_actions=self._generate_suggested_actions(need_category)
                            )
                            
                            predictions.append(prediction)
            
            except Exception as e:
                self.logger.error(f"خطأ في تطبيق القاعدة {rule_name}: {e}")
        
        return predictions

    def _calculate_ml_confidence(self, future_time: datetime, features: List[float]) -> float:
        """حساب الثقة للتنبؤات التعلم الآلي"""
        
        base_confidence = 0.5
        
        # زيادة الثقة للأوقات المألوفة
        hour = future_time.hour
        if 9 <= hour <= 17:  # ساعات العمل
            base_confidence += 0.2
        elif 18 <= hour <= 22:  # ساعات المساء
            base_confidence += 0.1
        
        # تقليل الثقة للأوقات البعيدة
        hours_ahead = (future_time - datetime.now()).total_seconds() / 3600
        if hours_ahead > 12:
            base_confidence -= 0.2
        
        return max(0.1, min(1.0, base_confidence))

    async def _pattern_conditions_met(self, pattern: UserPattern) -> bool:
        """فحص شروط النمط"""
        
        conditions = pattern.conditions
        current_time = datetime.now()
        
        # فحص الساعة
        if "hour" in conditions:
            return abs(current_time.hour - conditions["hour"]) <= 1
        
        # فحص الطقس
        if "weather" in conditions:
            # محاكاة - في التطبيق الحقيقي نحصل على الطقس من API
            return True
        
        # فحص التسلسل
        if "sequence" in conditions:
            # فحص آخر الاحتياجات
            recent_needs = [need["category"] for need in self.historical_needs[-3:]]
            expected_sequence = conditions["sequence"][:-1]  # كل شيء عدا الأخير
            return recent_needs == expected_sequence
        
        return True

    async def _calculate_pattern_time(self, pattern: UserPattern) -> Optional[datetime]:
        """حساب الوقت المتوقع للنمط"""
        
        current_time = datetime.now()
        
        if pattern.frequency == "daily" and pattern.typical_times:
            # البحث عن أقرب وقت مناسب
            target_hour = int(pattern.typical_times[0])
            
            # إذا كان الوقت قد مضى اليوم، اختر الغد
            if current_time.hour >= target_hour:
                target_time = current_time.replace(hour=target_hour, minute=0, second=0, microsecond=0) + timedelta(days=1)
            else:
                target_time = current_time.replace(hour=target_hour, minute=0, second=0, microsecond=0)
            
            return target_time
        
        return None

    async def _rule_conditions_met(self, rule_name: str, rule_config: Dict[str, Any]) -> bool:
        """فحص شروط القاعدة"""
        
        current_time = datetime.now()
        
        # فحص نطاق الوقت
        if "time_range" in rule_config:
            start_hour, end_hour = rule_config["time_range"]
            if not (start_hour <= current_time.hour <= end_hour):
                return False
        
        # فحص أيام الأسبوع
        if "days" in rule_config:
            if current_time.weekday() not in rule_config["days"]:
                return False
        
        return True

    async def _calculate_rule_time(self, rule_config: Dict[str, Any]) -> Optional[datetime]:
        """حساب الوقت المتوقع للقاعدة"""
        
        current_time = datetime.now()
        
        if "time_range" in rule_config:
            start_hour, end_hour = rule_config["time_range"]
            
            # اختيار وقت عشوائي في النطاق
            if current_time.hour < start_hour:
                target_hour = start_hour
                target_time = current_time.replace(hour=target_hour, minute=0, second=0, microsecond=0)
            elif current_time.hour > end_hour:
                target_hour = start_hour
                target_time = current_time.replace(hour=target_hour, minute=0, second=0, microsecond=0) + timedelta(days=1)
            else:
                # الوقت الحالي في النطاق
                target_time = current_time + timedelta(minutes=np.random.randint(15, 120))
            
            return target_time
        
        return current_time + timedelta(hours=1)

    def _generate_need_description(self, category: str, predicted_time: datetime) -> str:
        """توليد وصف للحاجة"""
        
        descriptions = {
            "work": [
                "مراجعة الإيميلات",
                "إنجاز المهام المعلقة",
                "حضور اجتماع",
                "تحديث التقارير"
            ],
            "food": [
                "تناول وجبة",
                "شرب الماء",
                "طلب الطعام",
                "تحضير وجبة خفيفة"
            ],
            "health": [
                "أخذ استراحة",
                "ممارسة التمارين",
                "فحص الصحة",
                "الاسترخاء"
            ],
            "entertainment": [
                "مشاهدة فيديو",
                "قراءة كتاب",
                "لعب لعبة",
                "الاستماع للموسيقى"
            ],
            "social": [
                "التواصل مع الأصدقاء",
                "المشاركة في نشاط اجتماعي",
                "زيارة العائلة",
                "إجراء مكالمة"
            ]
        }
        
        category_descriptions = descriptions.get(category, ["نشاط عام"])
        base_description = np.random.choice(category_descriptions)
        
        # إضافة السياق الزمني
        hour = predicted_time.hour
        if 6 <= hour <= 10:
            time_context = "صباحية"
        elif 11 <= hour <= 14:
            time_context = "ظهر"
        elif 15 <= hour <= 18:
            time_context = "بعد الظهر"
        else:
            time_context = "مسائية"
        
        return f"{base_description} ({time_context})"

    def _generate_suggested_actions(self, category: str) -> List[str]:
        """توليد الإجراءات المقترحة"""
        
        actions = {
            "work": [
                "فتح تطبيق البريد الإلكتروني",
                "مراجعة قائمة المهام",
                "إعداد مساحة العمل"
            ],
            "food": [
                "التحقق من المطبخ",
                "البحث عن المطاعم القريبة",
                "إعداد جدول الوجبات"
            ],
            "health": [
                "ضبط تذكير للراحة",
                "فتح تطبيق اللياقة",
                "تحضير مكان للتمارين"
            ],
            "entertainment": [
                "فتح تطبيق الترفيه",
                "البحث عن محتوى جديد",
                "تجهيز مكان مريح"
            ],
            "social": [
                "فتح تطبيق المراسلة",
                "مراجعة جهات الاتصال",
                "التحقق من الأحداث الاجتماعية"
            ]
        }
        
        return actions.get(category, ["التخطيط للنشاط"])

    async def update_context(self, new_context: Dict[str, Any]):
        """تحديث سياق المستخدم"""
        
        self.current_context.update(new_context)
        
        # إعادة تقييم التنبؤات إذا كان هناك تغيير مهم
        significant_changes = ["stress_level", "energy_level", "location"]
        if any(key in new_context for key in significant_changes):
            self.logger.info("🔄 إعادة تقييم التنبؤات بناءً على تغيير السياق")

    async def record_need_fulfillment(
        self,
        need_description: str,
        category: str,
        fulfilled: bool,
        notes: str = ""
    ):
        """تسجيل تنفيذ الحاجة"""
        
        try:
            need_data = {
                "timestamp": datetime.now(),
                "category": category,
                "description": need_description,
                "fulfilled_time": datetime.now() if fulfilled else None,
                "priority": 3,
                "context": self.current_context.copy(),
                "outcome": "fulfilled" if fulfilled else "unfulfilled"
            }
            
            self.historical_needs.append(need_data)
            
            # حفظ في قاعدة البيانات
            await self._save_need_to_db(need_data)
            
            # إعادة تدريب النماذج إذا كان لدينا بيانات كافية
            if len(self.historical_needs) % 50 == 0:
                await self._train_prediction_models()
            
        except Exception as e:
            self.logger.error(f"خطأ في تسجيل تنفيذ الحاجة: {e}")

    async def _save_need_to_db(self, need_data: Dict[str, Any]):
        """حفظ الحاجة في قاعدة البيانات"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO historical_needs 
            (timestamp, category, description, fulfilled_time, priority, context, outcome)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            need_data["timestamp"].isoformat(),
            need_data["category"],
            need_data["description"],
            need_data["fulfilled_time"].isoformat() if need_data["fulfilled_time"] else None,
            need_data["priority"],
            json.dumps(need_data["context"]),
            need_data["outcome"]
        ))
        
        conn.commit()
        conn.close()

    async def get_prediction_analytics(self) -> Dict[str, Any]:
        """الحصول على تحليلات التنبؤ"""
        
        try:
            # إحصائيات عامة
            total_needs = len(self.historical_needs)
            fulfilled_needs = len([n for n in self.historical_needs if n["outcome"] == "fulfilled"])
            fulfillment_rate = fulfilled_needs / total_needs if total_needs > 0 else 0
            
            # تحليل دقة الأنماط
            pattern_accuracy = {}
            for pattern_id, pattern in self.user_patterns.items():
                pattern_accuracy[pattern_id] = {
                    "name": pattern.name,
                    "accuracy": pattern.accuracy,
                    "frequency": pattern.frequency
                }
            
            # تحليل الفئات الأكثر شيوعاً
            category_counts = {}
            for need in self.historical_needs:
                category = need["category"]
                category_counts[category] = category_counts.get(category, 0) + 1
            
            most_common_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            
            # تحليل الأوقات الأكثر نشاطاً
            hourly_activity = {}
            for need in self.historical_needs:
                hour = need["timestamp"].hour
                hourly_activity[hour] = hourly_activity.get(hour, 0) + 1
            
            peak_hours = sorted(hourly_activity.items(), key=lambda x: x[1], reverse=True)[:3]
            
            return {
                "general_statistics": {
                    "total_needs": total_needs,
                    "fulfillment_rate": round(fulfillment_rate * 100, 1),
                    "active_patterns": len(self.user_patterns)
                },
                "pattern_analysis": pattern_accuracy,
                "category_analysis": {
                    "most_common": most_common_categories,
                    "distribution": category_counts
                },
                "temporal_analysis": {
                    "peak_hours": peak_hours,
                    "hourly_distribution": hourly_activity
                },
                "context_influence": {
                    "stress_correlation": self._calculate_stress_correlation(),
                    "day_of_week_preferences": self._calculate_day_preferences()
                },
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"خطأ في تحليلات التنبؤ: {e}")
            return {"error": str(e)}

    def _calculate_stress_correlation(self) -> float:
        """حساب الارتباط بين التوتر والاحتياجات"""
        
        stress_levels = []
        need_counts = []
        
        # تجميع البيانات حسب مستوى التوتر
        stress_groups = {}
        for need in self.historical_needs:
            stress = need["context"].get("stress_level", 0.5)
            stress_bucket = round(stress * 10) / 10  # تقريب لأقرب 0.1
            
            if stress_bucket not in stress_groups:
                stress_groups[stress_bucket] = 0
            stress_groups[stress_bucket] += 1
        
        for stress, count in stress_groups.items():
            stress_levels.append(stress)
            need_counts.append(count)
        
        if len(stress_levels) > 1:
            correlation = np.corrcoef(stress_levels, need_counts)[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
        
        return 0.0

    def _calculate_day_preferences(self) -> Dict[str, int]:
        """حساب تفضيلات أيام الأسبوع"""
        
        day_counts = {}
        day_names = ["الاثنين", "الثلاثاء", "الأربعاء", "الخميس", "الجمعة", "السبت", "الأحد"]
        
        for need in self.historical_needs:
            day = need["timestamp"].weekday()
            day_name = day_names[day]
            day_counts[day_name] = day_counts.get(day_name, 0) + 1
        
        return day_counts

# إنشاء مثيل عام
needs_predictor = AdvancedNeedsPredictor()

async def get_needs_predictor() -> AdvancedNeedsPredictor:
    """الحصول على محرك التنبؤ"""
    return needs_predictor

if __name__ == "__main__":
    async def test_needs_predictor():
        """اختبار محرك التنبؤ"""
        print("🔮 اختبار محرك التنبؤ المتقدم بالاحتياجات")
        print("=" * 50)
        
        predictor = await get_needs_predictor()
        await predictor.initialize()
        
        # اختبار التنبؤ
        print("\n🔍 التنبؤ بالاحتياجات القادمة...")
        predictions = await predictor.predict_upcoming_needs(time_horizon=12, max_predictions=5)
        
        for i, pred in enumerate(predictions, 1):
            print(f"\n{i}. {pred.description}")
            print(f"   📅 الوقت المتوقع: {pred.predicted_time.strftime('%H:%M')}")
            print(f"   🎯 الثقة: {pred.confidence:.1%}")
            print(f"   ⭐ الأولوية: {pred.priority.name}")
            print(f"   💡 إجراءات مقترحة: {', '.join(pred.suggested_actions[:2])}")
        
        # اختبار تسجيل تنفيذ الحاجة
        print("\n📝 تسجيل تنفيذ حاجة...")
        await predictor.record_need_fulfillment(
            need_description="شرب الماء",
            category="health",
            fulfilled=True,
            notes="تم التنفيذ في الوقت المحدد"
        )
        
        # عرض التحليلات
        print("\n📊 تحليلات التنبؤ:")
        analytics = await predictor.get_prediction_analytics()
        stats = analytics["general_statistics"]
        print(f"📈 إجمالي الاحتياجات: {stats['total_needs']}")
        print(f"✅ معدل التنفيذ: {stats['fulfillment_rate']}%")
        print(f"🔄 الأنماط النشطة: {stats['active_patterns']}")
        
        # عرض أكثر الفئات شيوعاً
        print(f"\n🏆 الفئات الأكثر شيوعاً:")
        for category, count in analytics["category_analysis"]["most_common"][:3]:
            print(f"   • {category}: {count} مرة")
    
    asyncio.run(test_needs_predictor())
