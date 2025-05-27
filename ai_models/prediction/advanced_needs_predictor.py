
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
محرك التنبؤ المتقدم للاحتياجات والتوقعات الذكية
Advanced Needs Prediction Engine with Intelligent Forecasting
"""

import asyncio
import logging
import numpy as np
import pandas as pd
import json
import pickle
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from enum import Enum
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.pipeline import Pipeline

# Time Series
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    TIMESERIES_AVAILABLE = True
except ImportError:
    TIMESERIES_AVAILABLE = False
    logging.warning("مكتبات السلاسل الزمنية غير متوفرة")

# Deep Learning
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    logging.warning("PyTorch غير متوفر")

class NeedCategory(Enum):
    """فئات الاحتياجات"""
    IMMEDIATE = "immediate"          # فورية
    ROUTINE = "routine"              # روتينية
    PERSONAL = "personal"            # شخصية
    PROFESSIONAL = "professional"    # مهنية
    ENTERTAINMENT = "entertainment"  # ترفيهية
    SOCIAL = "social"               # اجتماعية
    HEALTH = "health"               # صحية
    LEARNING = "learning"           # تعليمية
    FINANCIAL = "financial"         # مالية
    TRAVEL = "travel"               # سفر

class UrgencyLevel(Enum):
    """مستويات الأولوية"""
    CRITICAL = 5     # حرجة
    HIGH = 4         # عالية
    MEDIUM = 3       # متوسطة
    LOW = 2          # منخفضة
    OPTIONAL = 1     # اختيارية

class PredictionConfidence(Enum):
    """مستويات ثقة التنبؤ"""
    VERY_HIGH = "very_high"    # عالية جداً
    HIGH = "high"              # عالية
    MEDIUM = "medium"          # متوسطة
    LOW = "low"                # منخفضة
    UNCERTAIN = "uncertain"    # غير مؤكدة

@dataclass
class UserNeed:
    """حاجة المستخدم"""
    need_id: str
    category: NeedCategory
    description: str
    urgency: UrgencyLevel
    predicted_time: datetime
    confidence: float
    context: Dict[str, Any] = field(default_factory=dict)
    triggers: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    estimated_duration: int = 30  # بالدقائق
    resources_needed: List[str] = field(default_factory=list)
    satisfaction_score: Optional[float] = None
    fulfillment_history: List[datetime] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PredictionResult:
    """نتيجة التنبؤ"""
    predictions: List[UserNeed]
    prediction_horizon: timedelta
    overall_confidence: float
    trending_categories: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
    context_insights: Dict[str, Any] = field(default_factory=dict)
    prediction_timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class UserContext:
    """سياق المستخدم"""
    user_id: str
    current_location: Optional[str] = None
    time_of_day: str = "morning"
    day_of_week: str = "monday"
    mood_state: str = "neutral"
    energy_level: int = 5  # 1-10
    stress_level: int = 3  # 1-10
    social_context: str = "alone"
    device_type: str = "mobile"
    activity_context: str = "work"
    weather_condition: Optional[str] = None
    calendar_events: List[Dict] = field(default_factory=list)
    recent_activities: List[str] = field(default_factory=list)
    preferences: Dict[str, Any] = field(default_factory=dict)

class NeuralNeedsPredictor(nn.Module):
    """شبكة عصبية للتنبؤ بالاحتياجات"""
    
    def __init__(self, input_size: int, hidden_sizes: List[int], num_categories: int):
        super().__init__()
        
        self.layers = nn.ModuleList()
        prev_size = input_size
        
        # الطبقات المخفية
        for hidden_size in hidden_sizes:
            self.layers.append(nn.Linear(prev_size, hidden_size))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.BatchNorm1d(hidden_size))
            self.layers.append(nn.Dropout(0.3))
            prev_size = hidden_size
        
        # طبقات الإخراج
        self.category_head = nn.Linear(prev_size, num_categories)
        self.urgency_head = nn.Linear(prev_size, 5)  # 5 مستويات أولوية
        self.confidence_head = nn.Linear(prev_size, 1)
        
    def forward(self, x):
        for layer in self.layers:
            if isinstance(layer, nn.BatchNorm1d) and x.size(0) == 1:
                continue  # تخطي BatchNorm للعينة الواحدة
            x = layer(x)
        
        category_logits = torch.softmax(self.category_head(x), dim=1)
        urgency_logits = torch.softmax(self.urgency_head(x), dim=1)
        confidence = torch.sigmoid(self.confidence_head(x))
        
        return category_logits, urgency_logits, confidence

class AdvancedNeedsPredictor:
    """محرك التنبؤ المتقدم للاحتياجات"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # النماذج
        self.ml_models: Dict[str, Any] = {}
        self.neural_model: Optional[NeuralNeedsPredictor] = None
        self.time_series_models: Dict[str, Any] = {}
        
        # البيانات
        self.user_profiles: Dict[str, Dict[str, Any]] = {}
        self.needs_history: Dict[str, List[UserNeed]] = defaultdict(list)
        self.context_history: Dict[str, List[UserContext]] = defaultdict(list)
        self.prediction_cache: Dict[str, PredictionResult] = {}
        
        # معالجات البيانات
        self.scalers: Dict[str, StandardScaler] = {}
        self.encoders: Dict[str, LabelEncoder] = {}
        
        # الأنماط المكتشفة
        self.behavior_patterns: Dict[str, Dict[str, Any]] = {}
        self.temporal_patterns: Dict[str, List[Tuple]] = {}
        self.contextual_rules: List[Dict[str, Any]] = []
        
        # الإحصائيات
        self.prediction_accuracy: Dict[str, float] = {}
        self.category_frequency: Dict[str, int] = defaultdict(int)
        self.success_rates: Dict[str, float] = {}
        
        # مسارات الحفظ
        self.data_dir = Path("data/needs_prediction")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # تهيئة النماذج
        self._initialize_models()
        
        self.logger.info("تم تهيئة محرك التنبؤ المتقدم للاحتياجات")

    def _initialize_models(self):
        """تهيئة النماذج الأساسية"""
        try:
            # نماذج التعلم الآلي التقليدية
            self.ml_models = {
                'category_classifier': RandomForestRegressor(n_estimators=100, random_state=42),
                'urgency_predictor': GradientBoostingRegressor(n_estimators=50, random_state=42),
                'timing_predictor': Ridge(alpha=1.0),
                'duration_predictor': LinearRegression()
            }
            
            # نموذج التجميع للأنماط
            self.ml_models['pattern_clustering'] = KMeans(n_clusters=10, random_state=42)
            
            # الشبكة العصبية
            if PYTORCH_AVAILABLE:
                self.neural_model = NeuralNeedsPredictor(
                    input_size=50,  # سيتم تعديله ديناميكياً
                    hidden_sizes=[128, 64, 32],
                    num_categories=len(NeedCategory)
                )
                self.neural_optimizer = optim.Adam(self.neural_model.parameters(), lr=0.001)
                self.neural_criterion = nn.CrossEntropyLoss()
            
            # نماذج السلاسل الزمنية
            if TIMESERIES_AVAILABLE:
                self.time_series_models = {
                    'daily_patterns': None,
                    'weekly_patterns': None,
                    'monthly_trends': None
                }
            
        except Exception as e:
            self.logger.error(f"خطأ في تهيئة النماذج: {e}")

    async def predict_user_needs(
        self,
        user_id: str,
        current_context: UserContext,
        prediction_horizon: timedelta = timedelta(hours=24),
        max_predictions: int = 10
    ) -> PredictionResult:
        """التنبؤ باحتياجات المستخدم"""
        try:
            # التحقق من وجود cache
            cache_key = f"{user_id}_{current_context.time_of_day}_{prediction_horizon.total_seconds()}"
            if cache_key in self.prediction_cache:
                cached_result = self.prediction_cache[cache_key]
                # التحقق من صلاحية Cache (أقل من ساعة)
                if (datetime.now() - cached_result.prediction_timestamp).total_seconds() < 3600:
                    return cached_result
            
            # إنشاء ملف المستخدم إذا لم يكن موجوداً
            if user_id not in self.user_profiles:
                await self._create_user_profile(user_id, current_context)
            
            # تحليل الأنماط التاريخية
            historical_patterns = await self._analyze_historical_patterns(user_id, current_context)
            
            # التنبؤ بالفئات المحتملة
            predicted_categories = await self._predict_categories(user_id, current_context, historical_patterns)
            
            # التنبؤ بالتوقيت والأولوية
            predictions = []
            for category_info in predicted_categories[:max_predictions]:
                need = await self._create_predicted_need(
                    user_id, 
                    category_info, 
                    current_context, 
                    prediction_horizon
                )
                predictions.append(need)
            
            # ترتيب التنبؤات حسب الأولوية والثقة
            predictions.sort(key=lambda x: (x.urgency.value, x.confidence), reverse=True)
            
            # حساب الثقة الإجمالية
            overall_confidence = np.mean([p.confidence for p in predictions]) if predictions else 0.0
            
            # تحديد الاتجاهات
            trending_categories = await self._identify_trending_categories(user_id)
            
            # إنشاء التوصيات
            recommended_actions = await self._generate_recommendations(predictions, current_context)
            
            # رؤى السياق
            context_insights = await self._analyze_context_insights(user_id, current_context)
            
            # إنشاء النتيجة
            result = PredictionResult(
                predictions=predictions,
                prediction_horizon=prediction_horizon,
                overall_confidence=overall_confidence,
                trending_categories=trending_categories,
                recommended_actions=recommended_actions,
                context_insights=context_insights
            )
            
            # حفظ في Cache
            self.prediction_cache[cache_key] = result
            
            # تحديث السياق
            self.context_history[user_id].append(current_context)
            if len(self.context_history[user_id]) > 1000:
                self.context_history[user_id] = self.context_history[user_id][-1000:]
            
            return result
            
        except Exception as e:
            self.logger.error(f"خطأ في التنبؤ بالاحتياجات: {e}")
            raise

    async def _create_user_profile(self, user_id: str, initial_context: UserContext):
        """إنشاء ملف تعريف المستخدم"""
        try:
            self.user_profiles[user_id] = {
                'creation_date': datetime.now(),
                'preferences': initial_context.preferences.copy(),
                'typical_patterns': {},
                'favorite_categories': [],
                'activity_frequency': defaultdict(int),
                'context_preferences': defaultdict(list),
                'satisfaction_scores': [],
                'learning_rate': 0.1
            }
            
            self.logger.info(f"تم إنشاء ملف تعريف جديد للمستخدم: {user_id}")
            
        except Exception as e:
            self.logger.error(f"خطأ في إنشاء ملف المستخدم: {e}")

    async def _analyze_historical_patterns(
        self, 
        user_id: str, 
        current_context: UserContext
    ) -> Dict[str, Any]:
        """تحليل الأنماط التاريخية"""
        try:
            patterns = {
                'temporal_patterns': {},
                'contextual_patterns': {},
                'behavioral_patterns': {},
                'preference_patterns': {}
            }
            
            if user_id not in self.needs_history:
                return patterns
            
            user_needs = self.needs_history[user_id]
            user_contexts = self.context_history[user_id]
            
            # تحليل الأنماط الزمنية
            time_patterns = defaultdict(list)
            for need in user_needs:
                hour = need.predicted_time.hour
                day_of_week = need.predicted_time.weekday()
                time_patterns[f"hour_{hour}"].append(need.category.value)
                time_patterns[f"day_{day_of_week}"].append(need.category.value)
            
            patterns['temporal_patterns'] = {
                time_key: max(set(categories), key=categories.count) if categories else None
                for time_key, categories in time_patterns.items()
            }
            
            # تحليل الأنماط السياقية
            context_patterns = defaultdict(list)
            for i, context in enumerate(user_contexts):
                if i < len(user_needs):
                    need = user_needs[i]
                    context_patterns[context.activity_context].append(need.category.value)
                    context_patterns[context.mood_state].append(need.category.value)
                    context_patterns[f"energy_{context.energy_level//2}"].append(need.category.value)
            
            patterns['contextual_patterns'] = {
                ctx_key: max(set(categories), key=categories.count) if categories else None
                for ctx_key, categories in context_patterns.items()
            }
            
            # تحليل الأنماط السلوكية
            category_sequences = []
            for i in range(len(user_needs) - 2):
                seq = [user_needs[i].category.value, 
                       user_needs[i+1].category.value, 
                       user_needs[i+2].category.value]
                category_sequences.append(tuple(seq))
            
            if category_sequences:
                most_common_seq = max(set(category_sequences), key=category_sequences.count)
                patterns['behavioral_patterns']['common_sequence'] = most_common_seq
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"خطأ في تحليل الأنماط التاريخية: {e}")
            return {}

    async def _predict_categories(
        self, 
        user_id: str, 
        current_context: UserContext, 
        patterns: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """التنبؤ بالفئات المحتملة"""
        try:
            category_scores = defaultdict(float)
            
            # النقاط الأساسية لكل فئة
            base_scores = {
                NeedCategory.IMMEDIATE.value: 0.3,
                NeedCategory.ROUTINE.value: 0.4,
                NeedCategory.PERSONAL.value: 0.2,
                NeedCategory.PROFESSIONAL.value: 0.3,
                NeedCategory.ENTERTAINMENT.value: 0.25,
                NeedCategory.SOCIAL.value: 0.2,
                NeedCategory.HEALTH.value: 0.15,
                NeedCategory.LEARNING.value: 0.1,
                NeedCategory.FINANCIAL.value: 0.05,
                NeedCategory.TRAVEL.value: 0.02
            }
            
            # تطبيق النقاط الأساسية
            for category, score in base_scores.items():
                category_scores[category] = score
            
            # تعديل بناءً على الوقت
            current_hour = datetime.now().hour
            if 6 <= current_hour <= 9:  # الصباح
                category_scores[NeedCategory.ROUTINE.value] += 0.3
                category_scores[NeedCategory.PROFESSIONAL.value] += 0.2
            elif 12 <= current_hour <= 14:  # الظهر
                category_scores[NeedCategory.IMMEDIATE.value] += 0.2
                category_scores[NeedCategory.SOCIAL.value] += 0.15
            elif 18 <= current_hour <= 22:  # المساء
                category_scores[NeedCategory.ENTERTAINMENT.value] += 0.3
                category_scores[NeedCategory.PERSONAL.value] += 0.2
            
            # تعديل بناءً على يوم الأسبوع
            day_of_week = datetime.now().weekday()
            if day_of_week < 5:  # أيام العمل
                category_scores[NeedCategory.PROFESSIONAL.value] += 0.2
            else:  # عطلة نهاية الأسبوع
                category_scores[NeedCategory.ENTERTAINMENT.value] += 0.25
                category_scores[NeedCategory.SOCIAL.value] += 0.2
            
            # تعديل بناءً على السياق
            if current_context.activity_context == "work":
                category_scores[NeedCategory.PROFESSIONAL.value] += 0.3
            elif current_context.activity_context == "leisure":
                category_scores[NeedCategory.ENTERTAINMENT.value] += 0.3
            
            # تعديل بناءً على الحالة المزاجية
            if current_context.mood_state == "stressed":
                category_scores[NeedCategory.HEALTH.value] += 0.2
                category_scores[NeedCategory.ENTERTAINMENT.value] += 0.15
            elif current_context.mood_state == "happy":
                category_scores[NeedCategory.SOCIAL.value] += 0.2
            
            # تعديل بناءً على مستوى الطاقة
            if current_context.energy_level <= 3:
                category_scores[NeedCategory.ROUTINE.value] += 0.2
            elif current_context.energy_level >= 8:
                category_scores[NeedCategory.PROFESSIONAL.value] += 0.15
                category_scores[NeedCategory.LEARNING.value] += 0.1
            
            # تطبيق الأنماط التاريخية
            temporal_patterns = patterns.get('temporal_patterns', {})
            hour_pattern = temporal_patterns.get(f"hour_{current_hour}")
            if hour_pattern:
                category_scores[hour_pattern] += 0.2
            
            contextual_patterns = patterns.get('contextual_patterns', {})
            activity_pattern = contextual_patterns.get(current_context.activity_context)
            if activity_pattern:
                category_scores[activity_pattern] += 0.15
            
            # تطبيق تردد الفئات السابقة
            if user_id in self.user_profiles:
                profile = self.user_profiles[user_id]
                for category, freq in profile.get('activity_frequency', {}).items():
                    if category in category_scores:
                        category_scores[category] += min(freq * 0.01, 0.2)
            
            # تحويل إلى قائمة مرتبة
            sorted_categories = sorted(
                category_scores.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            # إنشاء قائمة النتائج
            results = []
            for category, score in sorted_categories:
                confidence = min(score, 1.0)
                if confidence > 0.1:  # تصفية النتائج الضعيفة
                    results.append({
                        'category': category,
                        'score': score,
                        'confidence': confidence
                    })
            
            return results
            
        except Exception as e:
            self.logger.error(f"خطأ في التنبؤ بالفئات: {e}")
            return []

    async def _create_predicted_need(
        self, 
        user_id: str, 
        category_info: Dict[str, Any], 
        current_context: UserContext, 
        prediction_horizon: timedelta
    ) -> UserNeed:
        """إنشاء حاجة متوقعة"""
        try:
            category = NeedCategory(category_info['category'])
            
            # تحديد الوقت المتوقع
            base_time = datetime.now()
            if category in [NeedCategory.IMMEDIATE, NeedCategory.ROUTINE]:
                predicted_time = base_time + timedelta(minutes=np.random.randint(5, 60))
            elif category in [NeedCategory.PROFESSIONAL, NeedCategory.LEARNING]:
                predicted_time = base_time + timedelta(hours=np.random.randint(1, 8))
            else:
                predicted_time = base_time + timedelta(hours=np.random.randint(2, 24))
            
            # تحديد مستوى الأولوية
            urgency = await self._determine_urgency(category, current_context, category_info['score'])
            
            # إنشاء الوصف
            description = await self._generate_need_description(category, current_context)
            
            # تحديد المدة المتوقعة
            duration = await self._estimate_duration(category, current_context)
            
            # تحديد المشغلات
            triggers = await self._identify_triggers(category, current_context)
            
            # تحديد الموارد المطلوبة
            resources = await self._identify_required_resources(category)
            
            # إنشاء ID فريد
            need_id = f"{user_id}_{category.value}_{int(predicted_time.timestamp())}"
            
            return UserNeed(
                need_id=need_id,
                category=category,
                description=description,
                urgency=urgency,
                predicted_time=predicted_time,
                confidence=category_info['confidence'],
                context=asdict(current_context),
                triggers=triggers,
                estimated_duration=duration,
                resources_needed=resources,
                metadata={
                    'prediction_method': 'ml_hybrid',
                    'context_score': category_info['score'],
                    'user_profile_influence': 0.3
                }
            )
            
        except Exception as e:
            self.logger.error(f"خطأ في إنشاء الحاجة المتوقعة: {e}")
            raise

    async def _determine_urgency(
        self, 
        category: NeedCategory, 
        context: UserContext, 
        score: float
    ) -> UrgencyLevel:
        """تحديد مستوى الأولوية"""
        try:
            base_urgency = {
                NeedCategory.IMMEDIATE: UrgencyLevel.HIGH,
                NeedCategory.ROUTINE: UrgencyLevel.MEDIUM,
                NeedCategory.PERSONAL: UrgencyLevel.MEDIUM,
                NeedCategory.PROFESSIONAL: UrgencyLevel.HIGH,
                NeedCategory.ENTERTAINMENT: UrgencyLevel.LOW,
                NeedCategory.SOCIAL: UrgencyLevel.MEDIUM,
                NeedCategory.HEALTH: UrgencyLevel.HIGH,
                NeedCategory.LEARNING: UrgencyLevel.LOW,
                NeedCategory.FINANCIAL: UrgencyLevel.MEDIUM,
                NeedCategory.TRAVEL: UrgencyLevel.OPTIONAL
            }
            
            urgency = base_urgency.get(category, UrgencyLevel.MEDIUM)
            
            # تعديل بناءً على السياق
            if context.stress_level >= 7:
                if urgency.value < UrgencyLevel.HIGH.value:
                    urgency = UrgencyLevel.HIGH
            
            if context.energy_level <= 3:
                if urgency.value > UrgencyLevel.MEDIUM.value:
                    urgency = UrgencyLevel.MEDIUM
            
            # تعديل بناءً على النقاط
            if score >= 0.8:
                if urgency.value < UrgencyLevel.HIGH.value:
                    urgency = UrgencyLevel.HIGH
            elif score <= 0.3:
                if urgency.value > UrgencyLevel.LOW.value:
                    urgency = UrgencyLevel.LOW
            
            return urgency
            
        except Exception as e:
            self.logger.error(f"خطأ في تحديد مستوى الأولوية: {e}")
            return UrgencyLevel.MEDIUM

    async def _generate_need_description(
        self, 
        category: NeedCategory, 
        context: UserContext
    ) -> str:
        """إنشاء وصف للحاجة"""
        try:
            descriptions = {
                NeedCategory.IMMEDIATE: [
                    "التحقق من الرسائل العاجلة",
                    "إنجاز مهمة مطلوبة بسرعة",
                    "الرد على استفسار مهم",
                    "حل مشكلة تقنية عاجلة"
                ],
                NeedCategory.ROUTINE: [
                    "مراجعة جدول اليوم",
                    "تفقد التحديثات الروتينية",
                    "تنظيم المهام اليومية",
                    "مراجعة التقدم في المشاريع"
                ],
                NeedCategory.PERSONAL: [
                    "قضاء وقت شخصي",
                    "التأمل والاسترخاء",
                    "ممارسة هواية مفضلة",
                    "العناية بالذات"
                ],
                NeedCategory.PROFESSIONAL: [
                    "إكمال مهمة عمل مهمة",
                    "حضور اجتماع أو مؤتمر",
                    "مراجعة تقارير العمل",
                    "التواصل مع زملاء العمل"
                ],
                NeedCategory.ENTERTAINMENT: [
                    "مشاهدة فيلم أو برنامج",
                    "الاستماع إلى الموسيقى",
                    "لعب لعبة مسلية",
                    "قراءة كتاب ممتع"
                ],
                NeedCategory.SOCIAL: [
                    "التواصل مع الأصدقاء",
                    "المشاركة في نشاط اجتماعي",
                    "زيارة العائلة",
                    "حضور مناسبة اجتماعية"
                ],
                NeedCategory.HEALTH: [
                    "ممارسة التمارين الرياضية",
                    "تناول وجبة صحية",
                    "أخذ استراحة للراحة",
                    "فحص الحالة الصحية"
                ],
                NeedCategory.LEARNING: [
                    "دراسة موضوع جديد",
                    "تطوير مهارة معينة",
                    "حضور دورة تدريبية",
                    "قراءة محتوى تعليمي"
                ],
                NeedCategory.FINANCIAL: [
                    "مراجعة الميزانية",
                    "إجراء معاملة مالية",
                    "التخطيط للادخار",
                    "متابعة الاستثمارات"
                ],
                NeedCategory.TRAVEL: [
                    "التخطيط لرحلة",
                    "حجز تذاكر سفر",
                    "البحث عن وجهات سياحية",
                    "تحضير أمتعة السفر"
                ]
            }
            
            category_descriptions = descriptions.get(category, ["نشاط متنوع"])
            base_description = np.random.choice(category_descriptions)
            
            # تخصيص الوصف بناءً على السياق
            if context.activity_context == "work":
                if "عمل" not in base_description:
                    base_description = f"{base_description} (في سياق العمل)"
            elif context.activity_context == "home":
                if "منزل" not in base_description:
                    base_description = f"{base_description} (في المنزل)"
            
            return base_description
            
        except Exception as e:
            self.logger.error(f"خطأ في إنشاء وصف الحاجة: {e}")
            return "نشاط مطلوب"

    async def _estimate_duration(
        self, 
        category: NeedCategory, 
        context: UserContext
    ) -> int:
        """تقدير مدة النشاط بالدقائق"""
        try:
            base_durations = {
                NeedCategory.IMMEDIATE: (5, 20),
                NeedCategory.ROUTINE: (10, 30),
                NeedCategory.PERSONAL: (20, 60),
                NeedCategory.PROFESSIONAL: (30, 120),
                NeedCategory.ENTERTAINMENT: (30, 180),
                NeedCategory.SOCIAL: (30, 120),
                NeedCategory.HEALTH: (20, 90),
                NeedCategory.LEARNING: (45, 180),
                NeedCategory.FINANCIAL: (15, 45),
                NeedCategory.TRAVEL: (60, 300)
            }
            
            min_duration, max_duration = base_durations.get(category, (15, 60))
            
            # تعديل بناءً على مستوى الطاقة
            if context.energy_level <= 3:
                max_duration = min(max_duration, 60)
            elif context.energy_level >= 8:
                min_duration = max(min_duration, 30)
            
            return np.random.randint(min_duration, max_duration + 1)
            
        except Exception as e:
            self.logger.error(f"خطأ في تقدير المدة: {e}")
            return 30

    async def _identify_triggers(
        self, 
        category: NeedCategory, 
        context: UserContext
    ) -> List[str]:
        """تحديد مشغلات الحاجة"""
        try:
            trigger_mapping = {
                NeedCategory.IMMEDIATE: [
                    "إشعار عاجل",
                    "طلب فوري",
                    "مشكلة تحتاج حل سريع"
                ],
                NeedCategory.ROUTINE: [
                    "وقت محدد في اليوم",
                    "تذكير روتيني",
                    "عادة يومية"
                ],
                NeedCategory.PERSONAL: [
                    "الحاجة للاسترخاء",
                    "وقت فراغ متاح",
                    "مستوى إجهاد مرتفع"
                ],
                NeedCategory.PROFESSIONAL: [
                    "مواعيد نهائية للعمل",
                    "اجتماعات مجدولة",
                    "مهام عمل معلقة"
                ],
                NeedCategory.ENTERTAINMENT: [
                    "الحاجة للترفيه",
                    "مستوى طاقة مرتفع",
                    "وقت فراغ"
                ],
                NeedCategory.SOCIAL: [
                    "دعوة اجتماعية",
                    "رسالة من صديق",
                    "مناسبة اجتماعية"
                ],
                NeedCategory.HEALTH: [
                    "مستوى إجهاد مرتفع",
                    "تذكير صحي",
                    "روتين صحي"
                ],
                NeedCategory.LEARNING: [
                    "فضول معرفي",
                    "تطوير مهني",
                    "محتوى تعليمي جديد"
                ],
                NeedCategory.FINANCIAL: [
                    "تذكير مالي",
                    "موعد دفع",
                    "فرصة استثمار"
                ],
                NeedCategory.TRAVEL: [
                    "عرض سفر",
                    "إجازة قادمة",
                    "رغبة في التغيير"
                ]
            }
            
            triggers = trigger_mapping.get(category, ["محفز عام"])
            
            # إضافة محفزات سياقية
            if context.stress_level >= 7:
                triggers.append("مستوى إجهاد مرتفع")
            
            if context.energy_level <= 3:
                triggers.append("مستوى طاقة منخفض")
            
            return triggers
            
        except Exception as e:
            self.logger.error(f"خطأ في تحديد المشغلات: {e}")
            return ["محفز عام"]

    async def _identify_required_resources(self, category: NeedCategory) -> List[str]:
        """تحديد الموارد المطلوبة"""
        try:
            resource_mapping = {
                NeedCategory.IMMEDIATE: ["الهاتف", "الإنترنت"],
                NeedCategory.ROUTINE: ["التطبيقات الأساسية", "التقويم"],
                NeedCategory.PERSONAL: ["وقت خاص", "مكان هادئ"],
                NeedCategory.PROFESSIONAL: ["أدوات العمل", "الإنترنت", "الوثائق"],
                NeedCategory.ENTERTAINMENT: ["الأجهزة الترفيهية", "الإنترنت"],
                NeedCategory.SOCIAL: ["وسائل التواصل", "وقت للقاء"],
                NeedCategory.HEALTH: ["مساحة للتمرين", "معدات رياضية"],
                NeedCategory.LEARNING: ["مواد تعليمية", "وقت للدراسة"],
                NeedCategory.FINANCIAL: ["تطبيقات مالية", "بيانات مالية"],
                NeedCategory.TRAVEL: ["تطبيقات الحجز", "ميزانية سفر"]
            }
            
            return resource_mapping.get(category, ["موارد عامة"])
            
        except Exception as e:
            self.logger.error(f"خطأ في تحديد الموارد: {e}")
            return ["موارد عامة"]

    async def _identify_trending_categories(self, user_id: str) -> List[str]:
        """تحديد الفئات الرائجة"""
        try:
            if user_id not in self.needs_history:
                return []
            
            # تحليل الاتجاهات خلال آخر أسبوع
            recent_date = datetime.now() - timedelta(days=7)
            recent_needs = [
                need for need in self.needs_history[user_id]
                if need.predicted_time >= recent_date
            ]
            
            if not recent_needs:
                return []
            
            # حساب تكرار الفئات
            category_counts = defaultdict(int)
            for need in recent_needs:
                category_counts[need.category.value] += 1
            
            # ترتيب حسب التكرار
            most_common_categories = sorted(
                category_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5]
            
            return [category for category, count in most_common_categories]
            
        except Exception as e:
            self.logger.error(f"خطأ في تحديد الاتجاهات: {e}")
            return []

    async def _generate_recommendations(
        self, 
        predictions: List[UserNeed], 
        context: UserContext
    ) -> List[str]:
        """إنشاء التوصيات"""
        try:
            recommendations = []
            
            if not predictions:
                return ["لا توجد توصيات حالياً"]
            
            # توصيات بناءً على أعلى الأولويات
            high_priority_needs = [
                need for need in predictions 
                if need.urgency.value >= UrgencyLevel.HIGH.value
            ]
            
            if high_priority_needs:
                recommendations.append(
                    f"يوصى بالتركيز على {len(high_priority_needs)} مهام عالية الأولوية"
                )
            
            # توصيات بناءً على الفئات الشائعة
            category_counts = defaultdict(int)
            for prediction in predictions:
                category_counts[prediction.category.value] += 1
            
            if category_counts:
                most_common = max(category_counts.items(), key=lambda x: x[1])
                recommendations.append(
                    f"الفئة الأكثر توقعاً: {most_common[0]} ({most_common[1]} مهام)"
                )
            
            # توصيات بناءً على السياق
            if context.stress_level >= 7:
                recommendations.append("يُنصح بأخذ استراحة للتقليل من الإجهاد")
            
            if context.energy_level <= 3:
                recommendations.append("يُفضل التركيز على المهام البسيطة فقط")
            
            # توصيات زمنية
            now = datetime.now()
            immediate_needs = [
                need for need in predictions
                if (need.predicted_time - now).total_seconds() < 3600
            ]
            
            if immediate_needs:
                recommendations.append(
                    f"هناك {len(immediate_needs)} مهام متوقعة خلال الساعة القادمة"
                )
            
            return recommendations[:5]  # أقصى 5 توصيات
            
        except Exception as e:
            self.logger.error(f"خطأ في إنشاء التوصيات: {e}")
            return ["خطأ في إنشاء التوصيات"]

    async def _analyze_context_insights(
        self, 
        user_id: str, 
        context: UserContext
    ) -> Dict[str, Any]:
        """تحليل رؤى السياق"""
        try:
            insights = {
                'current_state_analysis': {},
                'productivity_indicators': {},
                'behavioral_insights': {},
                'recommendations': []
            }
            
            # تحليل الحالة الحالية
            insights['current_state_analysis'] = {
                'energy_status': self._categorize_energy_level(context.energy_level),
                'stress_status': self._categorize_stress_level(context.stress_level),
                'optimal_activity_time': self._determine_optimal_time(context),
                'context_favorability': self._assess_context_favorability(context)
            }
            
            # مؤشرات الإنتاجية
            insights['productivity_indicators'] = {
                'predicted_productivity': self._predict_productivity(context),
                'best_activities': self._suggest_best_activities(context),
                'energy_optimization': self._suggest_energy_optimization(context)
            }
            
            # رؤى سلوكية
            if user_id in self.context_history:
                insights['behavioral_insights'] = await self._analyze_behavioral_patterns(user_id, context)
            
            return insights
            
        except Exception as e:
            self.logger.error(f"خطأ في تحليل رؤى السياق: {e}")
            return {}

    def _categorize_energy_level(self, energy_level: int) -> str:
        """تصنيف مستوى الطاقة"""
        if energy_level <= 2:
            return "منخفض جداً"
        elif energy_level <= 4:
            return "منخفض"
        elif energy_level <= 6:
            return "متوسط"
        elif energy_level <= 8:
            return "عالي"
        else:
            return "عالي جداً"

    def _categorize_stress_level(self, stress_level: int) -> str:
        """تصنيف مستوى الإجهاد"""
        if stress_level <= 2:
            return "مسترخي"
        elif stress_level <= 4:
            return "طبيعي"
        elif stress_level <= 6:
            return "متوسط"
        elif stress_level <= 8:
            return "مرتفع"
        else:
            return "مرتفع جداً"

    def _determine_optimal_time(self, context: UserContext) -> str:
        """تحديد الوقت الأمثل للأنشطة"""
        if context.energy_level >= 7 and context.stress_level <= 4:
            return "وقت ممتاز للمهام المعقدة"
        elif context.energy_level >= 5 and context.stress_level <= 6:
            return "وقت جيد للمهام المتوسطة"
        elif context.energy_level >= 3:
            return "وقت مناسب للمهام البسيطة"
        else:
            return "وقت للراحة والاسترخاء"

    def _assess_context_favorability(self, context: UserContext) -> float:
        """تقييم ملاءمة السياق"""
        score = 0.5  # نقطة البداية
        
        # تعديل بناءً على الطاقة
        score += (context.energy_level - 5) * 0.05
        
        # تعديل بناءً على الإجهاد (إجهاد أقل = أفضل)
        score += (5 - context.stress_level) * 0.05
        
        # تعديل بناءً على النشاط
        if context.activity_context in ["work", "learning"]:
            score += 0.1
        elif context.activity_context == "leisure":
            score += 0.05
        
        return max(0.0, min(1.0, score))

    def _predict_productivity(self, context: UserContext) -> str:
        """التنبؤ بالإنتاجية"""
        productivity_score = context.energy_level * 0.4 + (10 - context.stress_level) * 0.3
        
        if productivity_score >= 7:
            return "إنتاجية عالية متوقعة"
        elif productivity_score >= 5:
            return "إنتاجية متوسطة متوقعة"
        else:
            return "إنتاجية منخفضة متوقعة"

    def _suggest_best_activities(self, context: UserContext) -> List[str]:
        """اقتراح أفضل الأنشطة"""
        activities = []
        
        if context.energy_level >= 7:
            activities.extend([
                "مهام تتطلب تركيز عالي",
                "مشاريع إبداعية",
                "تعلم مهارات جديدة"
            ])
        elif context.energy_level >= 4:
            activities.extend([
                "مهام روتينية",
                "مراجعة وتنظيم",
                "تواصل اجتماعي"
            ])
        else:
            activities.extend([
                "أنشطة استرخاء",
                "قراءة خفيفة",
                "استراحة قصيرة"
            ])
        
        if context.stress_level <= 3:
            activities.append("أنشطة تحدي جديدة")
        
        return activities

    def _suggest_energy_optimization(self, context: UserContext) -> List[str]:
        """اقتراحات لتحسين الطاقة"""
        suggestions = []
        
        if context.energy_level <= 3:
            suggestions.extend([
                "أخذ استراحة قصيرة",
                "شرب الماء",
                "تناول وجبة خفيفة صحية",
                "التعرض للضوء الطبيعي"
            ])
        elif context.energy_level >= 8:
            suggestions.extend([
                "استثمار الطاقة في مهام مهمة",
                "ممارسة نشاط بدني",
                "تجربة شيء جديد"
            ])
        
        if context.stress_level >= 7:
            suggestions.extend([
                "ممارسة تمارين التنفس",
                "أخذ استراحة طويلة",
                "تقليل المهام غير الضرورية"
            ])
        
        return suggestions

    async def _analyze_behavioral_patterns(
        self, 
        user_id: str, 
        current_context: UserContext
    ) -> Dict[str, Any]:
        """تحليل الأنماط السلوكية"""
        try:
            insights = {}
            user_contexts = self.context_history[user_id]
            
            if len(user_contexts) < 5:
                return {"message": "بيانات غير كافية للتحليل"}
            
            # تحليل أنماط الطاقة
            energy_levels = [ctx.energy_level for ctx in user_contexts[-20:]]
            if energy_levels:
                insights['energy_trend'] = {
                    'average': np.mean(energy_levels),
                    'trend': 'increasing' if energy_levels[-1] > np.mean(energy_levels[:-1]) else 'decreasing',
                    'volatility': np.std(energy_levels)
                }
            
            # تحليل أنماط الإجهاد
            stress_levels = [ctx.stress_level for ctx in user_contexts[-20:]]
            if stress_levels:
                insights['stress_trend'] = {
                    'average': np.mean(stress_levels),
                    'trend': 'increasing' if stress_levels[-1] > np.mean(stress_levels[:-1]) else 'decreasing',
                    'peak_times': self._find_peak_stress_times(user_contexts)
                }
            
            # تحليل أنماط النشاط
            activity_patterns = defaultdict(int)
            for ctx in user_contexts[-50:]:
                activity_patterns[ctx.activity_context] += 1
            
            insights['activity_preferences'] = dict(activity_patterns)
            
            return insights
            
        except Exception as e:
            self.logger.error(f"خطأ في تحليل الأنماط السلوكية: {e}")
            return {}

    def _find_peak_stress_times(self, contexts: List[UserContext]) -> List[str]:
        """العثور على أوقات الذروة للإجهاد"""
        try:
            high_stress_contexts = [ctx for ctx in contexts if ctx.stress_level >= 7]
            
            if not high_stress_contexts:
                return []
            
            # تجميع حسب الوقت
            time_stress = defaultdict(list)
            for ctx in high_stress_contexts:
                time_key = f"{ctx.time_of_day}"
                time_stress[time_key].append(ctx.stress_level)
            
            # العثور على أوقات الذروة
            peak_times = []
            for time_key, stress_levels in time_stress.items():
                if len(stress_levels) >= 3:  # على الأقل 3 مرات
                    peak_times.append(time_key)
            
            return peak_times
            
        except Exception as e:
            self.logger.error(f"خطأ في العثور على أوقات الذروة: {e}")
            return []

    async def record_need_fulfillment(
        self,
        user_id: str,
        need_id: str,
        fulfilled: bool,
        satisfaction_score: Optional[float] = None,
        notes: str = ""
    ):
        """تسجيل تنفيذ الحاجة"""
        try:
            # البحث عن الحاجة في التاريخ
            if user_id in self.needs_history:
                for need in self.needs_history[user_id]:
                    if need.need_id == need_id:
                        if fulfilled:
                            need.fulfillment_history.append(datetime.now())
                        
                        if satisfaction_score is not None:
                            need.satisfaction_score = satisfaction_score
                        
                        if notes:
                            need.metadata['fulfillment_notes'] = notes
                        
                        break
            
            # تحديث إحصائيات الدقة
            await self._update_prediction_accuracy(user_id, need_id, fulfilled)
            
            # تحديث ملف المستخدم
            if user_id in self.user_profiles:
                profile = self.user_profiles[user_id]
                if satisfaction_score is not None:
                    profile['satisfaction_scores'].append(satisfaction_score)
                    # الاحتفاظ بآخر 100 نقطة فقط
                    if len(profile['satisfaction_scores']) > 100:
                        profile['satisfaction_scores'] = profile['satisfaction_scores'][-100:]
            
            self.logger.info(f"تم تسجيل تنفيذ الحاجة: {need_id}")
            
        except Exception as e:
            self.logger.error(f"خطأ في تسجيل تنفيذ الحاجة: {e}")

    async def _update_prediction_accuracy(self, user_id: str, need_id: str, fulfilled: bool):
        """تحديث دقة التنبؤ"""
        try:
            accuracy_key = f"{user_id}_accuracy"
            
            if accuracy_key not in self.prediction_accuracy:
                self.prediction_accuracy[accuracy_key] = 0.5  # قيمة ابتدائية
            
            # تحديث بسيط بناءً على النتيجة
            current_accuracy = self.prediction_accuracy[accuracy_key]
            learning_rate = 0.1
            
            if fulfilled:
                new_accuracy = current_accuracy + learning_rate * (1.0 - current_accuracy)
            else:
                new_accuracy = current_accuracy - learning_rate * current_accuracy
            
            self.prediction_accuracy[accuracy_key] = max(0.0, min(1.0, new_accuracy))
            
        except Exception as e:
            self.logger.error(f"خطأ في تحديث دقة التنبؤ: {e}")

    async def get_user_insights(self, user_id: str) -> Dict[str, Any]:
        """الحصول على رؤى المستخدم"""
        try:
            insights = {
                'profile_summary': {},
                'prediction_performance': {},
                'behavioral_analysis': {},
                'recommendations': []
            }
            
            if user_id not in self.user_profiles:
                return {'error': 'ملف المستخدم غير موجود'}
            
            profile = self.user_profiles[user_id]
            
            # ملخص الملف الشخصي
            insights['profile_summary'] = {
                'creation_date': profile['creation_date'].isoformat(),
                'total_interactions': len(self.needs_history.get(user_id, [])),
                'average_satisfaction': np.mean(profile['satisfaction_scores']) if profile['satisfaction_scores'] else 0,
                'favorite_categories': profile.get('favorite_categories', [])
            }
            
            # أداء التنبؤ
            accuracy_key = f"{user_id}_accuracy"
            insights['prediction_performance'] = {
                'accuracy': self.prediction_accuracy.get(accuracy_key, 0),
                'total_predictions': len(self.needs_history.get(user_id, [])),
                'successful_predictions': len([
                    need for need in self.needs_history.get(user_id, [])
                    if need.fulfillment_history
                ])
            }
            
            # التحليل السلوكي
            if user_id in self.needs_history:
                needs = self.needs_history[user_id]
                category_counts = defaultdict(int)
                for need in needs:
                    category_counts[need.category.value] += 1
                
                insights['behavioral_analysis'] = {
                    'most_common_categories': dict(category_counts),
                    'activity_patterns': profile.get('typical_patterns', {}),
                    'context_preferences': dict(profile.get('context_preferences', {}))
                }
            
            return insights
            
        except Exception as e:
            self.logger.error(f"خطأ في الحصول على رؤى المستخدم: {e}")
            return {'error': str(e)}

    async def save_state(self):
        """حفظ حالة المحرك"""
        try:
            state_data = {
                'user_profiles': {
                    user_id: {
                        **profile,
                        'creation_date': profile['creation_date'].isoformat()
                    }
                    for user_id, profile in self.user_profiles.items()
                },
                'prediction_accuracy': self.prediction_accuracy,
                'category_frequency': dict(self.category_frequency),
                'success_rates': self.success_rates,
                'behavior_patterns': self.behavior_patterns,
                'contextual_rules': self.contextual_rules
            }
            
            state_file = self.data_dir / "predictor_state.json"
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(state_data, f, ensure_ascii=False, indent=2)
            
            # حفظ تاريخ الاحتياجات
            needs_data = {}
            for user_id, needs in self.needs_history.items():
                needs_data[user_id] = [asdict(need) for need in needs]
            
            needs_file = self.data_dir / "needs_history.json"
            with open(needs_file, 'w', encoding='utf-8') as f:
                json.dump(needs_data, f, ensure_ascii=False, indent=2, default=str)
            
            # حفظ النماذج
            for model_name, model in self.ml_models.items():
                model_file = self.data_dir / f"{model_name}.pkl"
                with open(model_file, 'wb') as f:
                    pickle.dump(model, f)
            
            self.logger.info("تم حفظ حالة محرك التنبؤ")
            
        except Exception as e:
            self.logger.error(f"خطأ في حفظ الحالة: {e}")

    async def load_state(self):
        """تحميل حالة المحرك"""
        try:
            state_file = self.data_dir / "predictor_state.json"
            if state_file.exists():
                with open(state_file, 'r', encoding='utf-8') as f:
                    state_data = json.load(f)
                
                # تحميل ملفات المستخدمين
                for user_id, profile in state_data.get('user_profiles', {}).items():
                    profile['creation_date'] = datetime.fromisoformat(profile['creation_date'])
                    self.user_profiles[user_id] = profile
                
                self.prediction_accuracy = state_data.get('prediction_accuracy', {})
                self.category_frequency = defaultdict(int, state_data.get('category_frequency', {}))
                self.success_rates = state_data.get('success_rates', {})
                self.behavior_patterns = state_data.get('behavior_patterns', {})
                self.contextual_rules = state_data.get('contextual_rules', [])
            
            # تحميل تاريخ الاحتياجات
            needs_file = self.data_dir / "needs_history.json"
            if needs_file.exists():
                with open(needs_file, 'r', encoding='utf-8') as f:
                    needs_data = json.load(f)
                
                for user_id, needs in needs_data.items():
                    self.needs_history[user_id] = [UserNeed(**need) for need in needs]
            
            # تحميل النماذج
            for model_name in self.ml_models.keys():
                model_file = self.data_dir / f"{model_name}.pkl"
                if model_file.exists():
                    try:
                        with open(model_file, 'rb') as f:
                            self.ml_models[model_name] = pickle.load(f)
                    except Exception as e:
                        self.logger.warning(f"فشل تحميل النموذج {model_name}: {e}")
            
            self.logger.info("تم تحميل حالة محرك التنبؤ")
            
        except Exception as e:
            self.logger.error(f"خطأ في تحميل الحالة: {e}")

# إنشاء مثيل عام
needs_predictor = AdvancedNeedsPredictor()

async def get_needs_predictor() -> AdvancedNeedsPredictor:
    """الحصول على محرك التنبؤ بالاحتياجات"""
    return needs_predictor

if __name__ == "__main__":
    async def test_needs_predictor():
        """اختبار محرك التنبؤ بالاحتياجات"""
        print("🔮 اختبار محرك التنبؤ المتقدم للاحتياجات")
        print("=" * 60)
        
        predictor = await get_needs_predictor()
        
        # إنشاء سياق تجريبي
        test_context = UserContext(
            user_id="test_user_001",
            current_location="office",
            time_of_day="morning",
            day_of_week="monday",
            mood_state="focused",
            energy_level=7,
            stress_level=4,
            social_context="alone",
            device_type="laptop",
            activity_context="work",
            recent_activities=["checking_email", "planning_day"],
            preferences={"work_style": "focused", "break_frequency": "high"}
        )
        
        print("👤 سياق المستخدم التجريبي:")
        print(f"  • الموقع: {test_context.current_location}")
        print(f"  • وقت اليوم: {test_context.time_of_day}")
        print(f"  • مستوى الطاقة: {test_context.energy_level}/10")
        print(f"  • مستوى الإجهاد: {test_context.stress_level}/10")
        print(f"  • السياق: {test_context.activity_context}")
        
        # التنبؤ بالاحتياجات
        print("\n🔮 التنبؤ بالاحتياجات...")
        prediction_result = await predictor.predict_user_needs(
            user_id="test_user_001",
            current_context=test_context,
            prediction_horizon=timedelta(hours=8),
            max_predictions=5
        )
        
        print(f"\n📊 نتائج التنبؤ:")
        print(f"  • عدد التنبؤات: {len(prediction_result.predictions)}")
        print(f"  • الثقة الإجمالية: {prediction_result.overall_confidence:.2f}")
        print(f"  • أفق التنبؤ: {prediction_result.prediction_horizon}")
        
        print(f"\n🎯 التنبؤات المفصلة:")
        for i, prediction in enumerate(prediction_result.predictions, 1):
            print(f"  {i}. {prediction.description}")
            print(f"     • الفئة: {prediction.category.value}")
            print(f"     • الأولوية: {prediction.urgency.value}")
            print(f"     • الثقة: {prediction.confidence:.2f}")
            print(f"     • التوقيت: {prediction.predicted_time.strftime('%H:%M')}")
            print(f"     • المدة المتوقعة: {prediction.estimated_duration} دقيقة")
            print()
        
        print(f"📈 الفئات الرائجة:")
        for category in prediction_result.trending_categories:
            print(f"  • {category}")
        
        print(f"\n💡 التوصيات:")
        for recommendation in prediction_result.recommended_actions:
            print(f"  • {recommendation}")
        
        print(f"\n🧠 رؤى السياق:")
        insights = prediction_result.context_insights
        if 'current_state_analysis' in insights:
            analysis = insights['current_state_analysis']
            print(f"  • حالة الطاقة: {analysis.get('energy_status', 'غير محدد')}")
            print(f"  • حالة الإجهاد: {analysis.get('stress_status', 'غير محدد')}")
            print(f"  • التوقيت الأمثل: {analysis.get('optimal_activity_time', 'غير محدد')}")
        
        # محاكاة تسجيل تنفيذ الاحتياجات
        print(f"\n✅ محاكاة تسجيل تنفيذ الاحتياجات...")
        for prediction in prediction_result.predictions[:2]:
            await predictor.record_need_fulfillment(
                user_id="test_user_001",
                need_id=prediction.need_id,
                fulfilled=True,
                satisfaction_score=np.random.uniform(0.6, 0.9),
                notes="تم التنفيذ بنجاح"
            )
        
        # الحصول على رؤى المستخدم
        print(f"\n📊 رؤى المستخدم:")
        user_insights = await predictor.get_user_insights("test_user_001")
        if 'profile_summary' in user_insights:
            summary = user_insights['profile_summary']
            print(f"  • إجمالي التفاعلات: {summary.get('total_interactions', 0)}")
            print(f"  • متوسط الرضا: {summary.get('average_satisfaction', 0):.2f}")
        
        if 'prediction_performance' in user_insights:
            performance = user_insights['prediction_performance']
            print(f"  • دقة التنبؤ: {performance.get('accuracy', 0):.2f}")
            print(f"  • التنبؤات الناجحة: {performance.get('successful_predictions', 0)}")
        
        # حفظ الحالة
        await predictor.save_state()
        print(f"\n💾 تم حفظ حالة المحرك")
        
        print("\n✨ انتهى الاختبار بنجاح!")

    # تشغيل الاختبار
    asyncio.run(test_needs_predictor())
