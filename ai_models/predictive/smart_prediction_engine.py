
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
محرك التنبؤ الذكي المتقدم للأنماط والتحليل التنبؤي
Advanced Smart Prediction Engine for Pattern Analysis
"""

import asyncio
import logging
import numpy as np
import pandas as pd
import json
import pickle
import warnings
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from enum import Enum
from collections import defaultdict, deque
import hashlib
import threading
import queue
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, IsolationForest
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Time Series Analysis
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.statespace.sarimax import SARIMAX
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
    logging.warning("PyTorch غير متوفر للتعلم العميق")

# Advanced Analytics
try:
    import xgboost as xgb
    import lightgbm as lgb
    BOOSTING_AVAILABLE = True
except ImportError:
    BOOSTING_AVAILABLE = False
    logging.warning("مكتبات التعزيز المتقدمة غير متوفرة")

class PredictionType(Enum):
    """أنواع التنبؤ"""
    TIME_SERIES = "time_series"          # سلاسل زمنية
    CLASSIFICATION = "classification"    # تصنيف
    REGRESSION = "regression"           # انحدار
    CLUSTERING = "clustering"           # تجميع
    ANOMALY_DETECTION = "anomaly"       # كشف الشذوذ
    PATTERN_RECOGNITION = "pattern"     # تمييز الأنماط
    TREND_ANALYSIS = "trend"            # تحليل الاتجاهات
    FORECASTING = "forecasting"         # توقعات مستقبلية

class ModelComplexity(Enum):
    """مستويات تعقيد النموذج"""
    SIMPLE = "simple"        # بسيط
    MODERATE = "moderate"    # متوسط
    COMPLEX = "complex"      # معقد
    DEEP = "deep"           # عميق

class DataPattern(Enum):
    """أنماط البيانات"""
    LINEAR = "linear"                # خطي
    NON_LINEAR = "non_linear"        # غير خطي
    SEASONAL = "seasonal"            # موسمي
    CYCLIC = "cyclic"               # دوري
    TRENDING = "trending"           # اتجاهي
    RANDOM = "random"               # عشوائي
    MIXED = "mixed"                 # مختلط

@dataclass
class PredictionRequest:
    """طلب التنبؤ"""
    request_id: str
    prediction_type: PredictionType
    data: Dict[str, Any]
    target_variable: Optional[str] = None
    prediction_horizon: int = 10
    confidence_level: float = 0.95
    model_complexity: ModelComplexity = ModelComplexity.MODERATE
    features: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class PredictionResult:
    """نتيجة التنبؤ"""
    request_id: str
    predictions: List[float]
    confidence_intervals: Optional[List[Tuple[float, float]]] = None
    feature_importance: Optional[Dict[str, float]] = None
    model_performance: Dict[str, float] = field(default_factory=dict)
    detected_patterns: List[str] = field(default_factory=list)
    anomalies: List[int] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    processing_time: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class PatternAnalysis:
    """تحليل الأنماط"""
    pattern_type: DataPattern
    strength: float  # 0-1
    frequency: Optional[float] = None
    phase: Optional[float] = None
    trend_direction: Optional[str] = None
    seasonal_components: Optional[Dict[str, float]] = None
    description: str = ""

class AdvancedNeuralPredictor(nn.Module):
    """شبكة عصبية متقدمة للتنبؤ"""
    
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int, dropout_rate: float = 0.2):
        super().__init__()
        
        self.layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # بناء الطبقات
        prev_size = input_size
        for hidden_size in hidden_sizes:
            self.layers.append(nn.Linear(prev_size, hidden_size))
            self.dropout_layers.append(nn.Dropout(dropout_rate))
            self.batch_norms.append(nn.BatchNorm1d(hidden_size))
            prev_size = hidden_size
        
        # طبقة الإخراج
        self.output_layer = nn.Linear(prev_size, output_size)
        
        # طبقات الانتباه (Attention)
        self.attention = nn.MultiheadAttention(embed_dim=prev_size, num_heads=4, batch_first=True)
        
    def forward(self, x):
        # إضافة بُعد للتسلسل إذا لزم الأمر
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # (batch_size, 1, features)
        
        # تطبيق طبقات الانتباه
        attn_output, _ = self.attention(x, x, x)
        x = attn_output.squeeze(1)
        
        # تطبيق الطبقات المخفية
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if x.size(0) > 1:  # تطبيق BatchNorm فقط إذا كان هناك أكثر من عينة
                x = self.batch_norms[i](x)
            x = torch.relu(x)
            x = self.dropout_layers[i](x)
        
        x = self.output_layer(x)
        return x

class SmartPredictionEngine:
    """محرك التنبؤ الذكي المتقدم"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # النماذج المختلفة
        self.models: Dict[str, Any] = {}
        self.model_performance: Dict[str, Dict[str, float]] = {}
        self.pattern_analyzers: Dict[str, Any] = {}
        
        # معالجات البيانات
        self.scalers: Dict[str, StandardScaler] = {}
        self.encoders: Dict[str, LabelEncoder] = {}
        self.feature_selectors: Dict[str, Any] = {}
        
        # ذاكرة التخزين المؤقت
        self.prediction_cache: Dict[str, PredictionResult] = {}
        self.pattern_cache: Dict[str, List[PatternAnalysis]] = {}
        
        # إحصائيات ومقاييس
        self.request_history: List[PredictionRequest] = []
        self.performance_metrics: Dict[str, float] = {}
        self.model_usage_stats: Dict[str, int] = defaultdict(int)
        
        # خيوط المعالجة
        self.prediction_queue = queue.Queue()
        self.processing_thread = None
        self.is_running = False
        
        # مسارات الحفظ
        self.models_dir = Path("data/prediction_models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # تهيئة النماذج
        self._initialize_models()
        
        self.logger.info("تم تهيئة محرك التنبؤ الذكي المتقدم")

    def _initialize_models(self):
        """تهيئة النماذج الأساسية"""
        try:
            # النماذج التقليدية
            self.models['linear_regression'] = LinearRegression()
            self.models['ridge_regression'] = Ridge(alpha=1.0)
            self.models['random_forest'] = RandomForestRegressor(n_estimators=100, random_state=42)
            self.models['gradient_boosting'] = GradientBoostingRegressor(n_estimators=100, random_state=42)
            
            # نماذج التجميع
            self.models['kmeans'] = KMeans(n_clusters=5, random_state=42)
            self.models['dbscan'] = DBSCAN(eps=0.5, min_samples=5)
            
            # كشف الشذوذ
            self.models['isolation_forest'] = IsolationForest(contamination=0.1, random_state=42)
            
            # نماذج متقدمة إذا كانت متوفرة
            if BOOSTING_AVAILABLE:
                self.models['xgboost'] = xgb.XGBRegressor(random_state=42)
                self.models['lightgbm'] = lgb.LGBMRegressor(random_state=42)
            
            # محللات الأنماط
            self.pattern_analyzers['trend'] = self._analyze_trend
            self.pattern_analyzers['seasonality'] = self._analyze_seasonality
            self.pattern_analyzers['cyclical'] = self._analyze_cyclical_patterns
            
        except Exception as e:
            self.logger.error(f"خطأ في تهيئة النماذج: {e}")

    async def start_prediction_engine(self):
        """بدء محرك التنبؤ"""
        try:
            self.is_running = True
            self.processing_thread = threading.Thread(target=self._prediction_processing_loop)
            self.processing_thread.start()
            self.logger.info("تم بدء محرك التنبؤ")
            
        except Exception as e:
            self.logger.error(f"خطأ في بدء محرك التنبؤ: {e}")
            raise

    async def stop_prediction_engine(self):
        """إيقاف محرك التنبؤ"""
        try:
            self.is_running = False
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=5)
            
            await self.save_models()
            self.logger.info("تم إيقاف محرك التنبؤ")
            
        except Exception as e:
            self.logger.error(f"خطأ في إيقاف محرك التنبؤ: {e}")

    def _prediction_processing_loop(self):
        """حلقة معالجة طلبات التنبؤ"""
        while self.is_running:
            try:
                try:
                    request = self.prediction_queue.get(timeout=1)
                    asyncio.create_task(self._process_prediction_request(request))
                except queue.Empty:
                    continue
                    
            except Exception as e:
                self.logger.error(f"خطأ في حلقة معالجة التنبؤ: {e}")

    async def predict(self, request: PredictionRequest) -> PredictionResult:
        """تنفيذ طلب التنبؤ"""
        try:
            start_time = datetime.now()
            
            # التحقق من وجود نتيجة مخزنة مؤقتاً
            cache_key = self._generate_cache_key(request)
            if cache_key in self.prediction_cache:
                cached_result = self.prediction_cache[cache_key]
                if (datetime.now() - cached_result.created_at).total_seconds() < 3600:  # ساعة واحدة
                    return cached_result
            
            # تحضير البيانات
            processed_data = await self._prepare_data(request)
            
            # تحليل الأنماط
            patterns = await self._analyze_patterns(processed_data, request)
            
            # اختيار النموذج المناسب
            best_model = await self._select_best_model(processed_data, request, patterns)
            
            # تنفيذ التنبؤ
            predictions = await self._execute_prediction(best_model, processed_data, request)
            
            # حساب فترات الثقة
            confidence_intervals = await self._calculate_confidence_intervals(
                predictions, processed_data, request
            )
            
            # كشف الشذوذ
            anomalies = await self._detect_anomalies(processed_data, predictions)
            
            # تحليل أهمية الميزات
            feature_importance = await self._analyze_feature_importance(best_model, processed_data)
            
            # إنشاء التوصيات
            recommendations = await self._generate_recommendations(
                predictions, patterns, anomalies, request
            )
            
            # حساب مقاييس الأداء
            performance = await self._evaluate_model_performance(best_model, processed_data, request)
            
            # إنشاء النتيجة
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = PredictionResult(
                request_id=request.request_id,
                predictions=predictions.tolist() if isinstance(predictions, np.ndarray) else predictions,
                confidence_intervals=confidence_intervals,
                feature_importance=feature_importance,
                model_performance=performance,
                detected_patterns=[p.pattern_type.value for p in patterns],
                anomalies=anomalies,
                recommendations=recommendations,
                processing_time=processing_time
            )
            
            # حفظ في الذاكرة المؤقتة
            self.prediction_cache[cache_key] = result
            
            # تحديث الإحصائيات
            self.request_history.append(request)
            self.model_usage_stats[best_model] += 1
            
            return result
            
        except Exception as e:
            self.logger.error(f"خطأ في تنفيذ التنبؤ: {e}")
            raise

    async def _prepare_data(self, request: PredictionRequest) -> Dict[str, Any]:
        """تحضير البيانات للتنبؤ"""
        try:
            data = request.data
            
            # تحويل البيانات إلى DataFrame
            if isinstance(data, dict):
                if 'dataframe' in data:
                    df = pd.DataFrame(data['dataframe'])
                elif 'series' in data:
                    df = pd.DataFrame({'value': data['series']})
                else:
                    df = pd.DataFrame([data])
            else:
                df = pd.DataFrame(data)
            
            # معالجة القيم المفقودة
            df = df.fillna(df.mean(numeric_only=True))
            
            # ترميز المتغيرات الفئوية
            categorical_columns = df.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                    df[col] = self.encoders[col].fit_transform(df[col].astype(str))
                else:
                    df[col] = self.encoders[col].transform(df[col].astype(str))
            
            # تطبيع البيانات الرقمية
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 0:
                scaler_key = f"scaler_{request.prediction_type.value}"
                if scaler_key not in self.scalers:
                    self.scalers[scaler_key] = StandardScaler()
                    df[numeric_columns] = self.scalers[scaler_key].fit_transform(df[numeric_columns])
                else:
                    df[numeric_columns] = self.scalers[scaler_key].transform(df[numeric_columns])
            
            # استخراج الميزات الزمنية إذا كان هناك عمود تاريخ
            if 'timestamp' in df.columns or 'date' in df.columns:
                time_col = 'timestamp' if 'timestamp' in df.columns else 'date'
                df[time_col] = pd.to_datetime(df[time_col])
                df['hour'] = df[time_col].dt.hour
                df['day_of_week'] = df[time_col].dt.dayofweek
                df['month'] = df[time_col].dt.month
                df['quarter'] = df[time_col].dt.quarter
            
            return {
                'dataframe': df,
                'target_variable': request.target_variable,
                'features': request.features or list(df.columns),
                'original_data': data
            }
            
        except Exception as e:
            self.logger.error(f"خطأ في تحضير البيانات: {e}")
            raise

    async def _analyze_patterns(self, data: Dict[str, Any], request: PredictionRequest) -> List[PatternAnalysis]:
        """تحليل الأنماط في البيانات"""
        try:
            df = data['dataframe']
            patterns = []
            
            # تحليل الاتجاه العام
            if request.target_variable and request.target_variable in df.columns:
                target_series = df[request.target_variable]
                
                # تحليل الاتجاه
                trend_analysis = await self._analyze_trend(target_series)
                if trend_analysis:
                    patterns.append(trend_analysis)
                
                # تحليل الموسمية
                if len(target_series) >= 24:  # بيانات كافية للتحليل الموسمي
                    seasonal_analysis = await self._analyze_seasonality(target_series)
                    if seasonal_analysis:
                        patterns.append(seasonal_analysis)
                
                # تحليل الدورية
                cyclical_analysis = await self._analyze_cyclical_patterns(target_series)
                if cyclical_analysis:
                    patterns.append(cyclical_analysis)
            
            # تحليل العلاقات بين المتغيرات
            correlation_patterns = await self._analyze_correlations(df)
            patterns.extend(correlation_patterns)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"خطأ في تحليل الأنماط: {e}")
            return []

    async def _analyze_trend(self, series: pd.Series) -> Optional[PatternAnalysis]:
        """تحليل الاتجاه في السلسلة الزمنية"""
        try:
            # حساب الاتجاه باستخدام الانحدار الخطي
            x = np.arange(len(series))
            coeffs = np.polyfit(x, series, 1)
            slope = coeffs[0]
            
            # تحديد قوة الاتجاه
            correlation = np.corrcoef(x, series)[0, 1]
            strength = abs(correlation)
            
            if strength > 0.5:  # اتجاه قوي
                direction = "صاعد" if slope > 0 else "هابط"
                
                return PatternAnalysis(
                    pattern_type=DataPattern.TRENDING,
                    strength=strength,
                    trend_direction=direction,
                    description=f"اتجاه {direction} بقوة {strength:.2f}"
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"خطأ في تحليل الاتجاه: {e}")
            return None

    async def _analyze_seasonality(self, series: pd.Series) -> Optional[PatternAnalysis]:
        """تحليل الموسمية في السلسلة الزمنية"""
        try:
            if not TIMESERIES_AVAILABLE or len(series) < 24:
                return None
            
            # تحليل الموسمية
            decomposition = seasonal_decompose(series, model='additive', period=12)
            seasonal_component = decomposition.seasonal
            
            # حساب قوة الموسمية
            seasonal_strength = np.var(seasonal_component) / np.var(series)
            
            if seasonal_strength > 0.1:  # موسمية واضحة
                return PatternAnalysis(
                    pattern_type=DataPattern.SEASONAL,
                    strength=seasonal_strength,
                    seasonal_components={
                        'amplitude': np.max(seasonal_component) - np.min(seasonal_component),
                        'period': 12
                    },
                    description=f"نمط موسمي بقوة {seasonal_strength:.2f}"
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"خطأ في تحليل الموسمية: {e}")
            return None

    async def _analyze_cyclical_patterns(self, series: pd.Series) -> Optional[PatternAnalysis]:
        """تحليل الأنماط الدورية"""
        try:
            # تحليل FFT للعثور على الترددات المهيمنة
            fft = np.fft.fft(series.values)
            freqs = np.fft.fftfreq(len(series))
            
            # العثور على أقوى تردد
            dominant_freq_idx = np.argmax(np.abs(fft[1:len(fft)//2])) + 1
            dominant_freq = freqs[dominant_freq_idx]
            
            if abs(dominant_freq) > 0.01:  # تردد واضح
                period = 1 / abs(dominant_freq)
                strength = np.abs(fft[dominant_freq_idx]) / np.sum(np.abs(fft))
                
                if strength > 0.1:
                    return PatternAnalysis(
                        pattern_type=DataPattern.CYCLIC,
                        strength=strength,
                        frequency=abs(dominant_freq),
                        description=f"نمط دوري بفترة {period:.1f} وقوة {strength:.2f}"
                    )
            
            return None
            
        except Exception as e:
            self.logger.error(f"خطأ في تحليل الأنماط الدورية: {e}")
            return None

    async def _analyze_correlations(self, df: pd.DataFrame) -> List[PatternAnalysis]:
        """تحليل العلاقات بين المتغيرات"""
        try:
            patterns = []
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_columns) > 1:
                correlation_matrix = df[numeric_columns].corr()
                
                # البحث عن ارتباطات قوية
                for i in range(len(correlation_matrix.columns)):
                    for j in range(i+1, len(correlation_matrix.columns)):
                        correlation = correlation_matrix.iloc[i, j]
                        
                        if abs(correlation) > 0.7:  # ارتباط قوي
                            col1 = correlation_matrix.columns[i]
                            col2 = correlation_matrix.columns[j]
                            
                            pattern = PatternAnalysis(
                                pattern_type=DataPattern.LINEAR if correlation > 0 else DataPattern.NON_LINEAR,
                                strength=abs(correlation),
                                description=f"ارتباط {'إيجابي' if correlation > 0 else 'سلبي'} قوي بين {col1} و {col2}"
                            )
                            patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"خطأ في تحليل الارتباطات: {e}")
            return []

    async def _select_best_model(
        self, 
        data: Dict[str, Any], 
        request: PredictionRequest, 
        patterns: List[PatternAnalysis]
    ) -> str:
        """اختيار أفضل نموذج للتنبؤ"""
        try:
            df = data['dataframe']
            prediction_type = request.prediction_type
            complexity = request.model_complexity
            
            # اختيار النموذج بناءً على نوع التنبؤ
            if prediction_type == PredictionType.TIME_SERIES:
                # للسلاسل الزمنية
                if any(p.pattern_type == DataPattern.SEASONAL for p in patterns):
                    return 'arima' if TIMESERIES_AVAILABLE else 'gradient_boosting'
                elif any(p.pattern_type == DataPattern.TRENDING for p in patterns):
                    return 'linear_regression'
                else:
                    return 'random_forest'
                    
            elif prediction_type == PredictionType.CLASSIFICATION:
                if complexity == ModelComplexity.SIMPLE:
                    return 'linear_regression'  # للتصنيف البسيط
                elif complexity == ModelComplexity.COMPLEX:
                    return 'xgboost' if BOOSTING_AVAILABLE else 'gradient_boosting'
                else:
                    return 'random_forest'
                    
            elif prediction_type == PredictionType.REGRESSION:
                if any(p.pattern_type == DataPattern.LINEAR for p in patterns):
                    return 'linear_regression'
                elif complexity == ModelComplexity.COMPLEX:
                    return 'xgboost' if BOOSTING_AVAILABLE else 'gradient_boosting'
                else:
                    return 'random_forest'
                    
            elif prediction_type == PredictionType.CLUSTERING:
                return 'kmeans'
                
            elif prediction_type == PredictionType.ANOMALY_DETECTION:
                return 'isolation_forest'
            
            # النموذج الافتراضي
            return 'random_forest'
            
        except Exception as e:
            self.logger.error(f"خطأ في اختيار النموذج: {e}")
            return 'random_forest'

    async def _execute_prediction(
        self, 
        model_name: str, 
        data: Dict[str, Any], 
        request: PredictionRequest
    ) -> np.ndarray:
        """تنفيذ التنبؤ بالنموذج المحدد"""
        try:
            df = data['dataframe']
            target_variable = data['target_variable']
            
            if model_name == 'arima' and TIMESERIES_AVAILABLE:
                return await self._predict_with_arima(df, target_variable, request)
            
            # تحضير البيانات للنماذج العادية
            if target_variable and target_variable in df.columns:
                X = df.drop(columns=[target_variable])
                y = df[target_variable]
            else:
                X = df
                y = None
            
            model = self.models[model_name]
            
            # تدريب النموذج إذا لم يكن مدرباً
            if hasattr(model, 'fit') and y is not None:
                model.fit(X, y)
            
            # التنبؤ
            if request.prediction_type == PredictionType.CLUSTERING:
                predictions = model.fit_predict(X)
            elif hasattr(model, 'predict'):
                if request.prediction_horizon > len(X):
                    # للتنبؤ المستقبلي، نستخدم آخر القيم
                    last_values = X.tail(1)
                    predictions = []
                    for _ in range(request.prediction_horizon):
                        pred = model.predict(last_values)[0]
                        predictions.append(pred)
                        # تحديث القيم للتنبؤ التالي (بسيط)
                        last_values.iloc[0, -1] = pred
                    predictions = np.array(predictions)
                else:
                    predictions = model.predict(X[:request.prediction_horizon])
            else:
                predictions = np.zeros(request.prediction_horizon)
            
            return np.array(predictions)
            
        except Exception as e:
            self.logger.error(f"خطأ في تنفيذ التنبؤ: {e}")
            return np.zeros(request.prediction_horizon)

    async def _predict_with_arima(
        self, 
        df: pd.DataFrame, 
        target_variable: str, 
        request: PredictionRequest
    ) -> np.ndarray:
        """التنبؤ باستخدام نموذج ARIMA"""
        try:
            series = df[target_variable]
            
            # تدريب نموذج ARIMA
            model = ARIMA(series, order=(1, 1, 1))
            fitted_model = model.fit()
            
            # التنبؤ
            forecast = fitted_model.forecast(steps=request.prediction_horizon)
            
            return np.array(forecast)
            
        except Exception as e:
            self.logger.error(f"خطأ في التنبؤ بـ ARIMA: {e}")
            return np.zeros(request.prediction_horizon)

    async def _calculate_confidence_intervals(
        self, 
        predictions: np.ndarray, 
        data: Dict[str, Any], 
        request: PredictionRequest
    ) -> List[Tuple[float, float]]:
        """حساب فترات الثقة للتنبؤات"""
        try:
            # حساب الانحراف المعياري من البيانات التاريخية
            df = data['dataframe']
            target_variable = data['target_variable']
            
            if target_variable and target_variable in df.columns:
                std = df[target_variable].std()
            else:
                std = np.std(predictions)
            
            # حساب فترات الثقة
            confidence_level = request.confidence_level
            z_score = 1.96 if confidence_level == 0.95 else 2.58  # للثقة 95% أو 99%
            
            margin = z_score * std
            
            intervals = []
            for pred in predictions:
                lower = pred - margin
                upper = pred + margin
                intervals.append((lower, upper))
            
            return intervals
            
        except Exception as e:
            self.logger.error(f"خطأ في حساب فترات الثقة: {e}")
            return [(p, p) for p in predictions]

    async def _detect_anomalies(self, data: Dict[str, Any], predictions: np.ndarray) -> List[int]:
        """كشف الشذوذ في التنبؤات"""
        try:
            # استخدام IQR لكشف الشذوذ
            Q1 = np.percentile(predictions, 25)
            Q3 = np.percentile(predictions, 75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            anomalies = []
            for i, pred in enumerate(predictions):
                if pred < lower_bound or pred > upper_bound:
                    anomalies.append(i)
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"خطأ في كشف الشذوذ: {e}")
            return []

    async def _analyze_feature_importance(
        self, 
        model_name: str, 
        data: Dict[str, Any]
    ) -> Optional[Dict[str, float]]:
        """تحليل أهمية الميزات"""
        try:
            model = self.models[model_name]
            features = data['features']
            
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                return dict(zip(features, importances))
            elif hasattr(model, 'coef_'):
                coefficients = np.abs(model.coef_)
                return dict(zip(features, coefficients))
            
            return None
            
        except Exception as e:
            self.logger.error(f"خطأ في تحليل أهمية الميزات: {e}")
            return None

    async def _generate_recommendations(
        self, 
        predictions: np.ndarray, 
        patterns: List[PatternAnalysis], 
        anomalies: List[int], 
        request: PredictionRequest
    ) -> List[str]:
        """إنشاء التوصيات بناءً على التنبؤات"""
        try:
            recommendations = []
            
            # تحليل الاتجاه العام
            if len(predictions) > 1:
                trend = np.mean(np.diff(predictions))
                if trend > 0:
                    recommendations.append("الاتجاه العام صاعد - قد تكون هناك فرصة للنمو")
                elif trend < 0:
                    recommendations.append("الاتجاه العام هابط - يُنصح بالحذر والمراقبة")
            
            # توصيات بناءً على الأنماط المكتشفة
            for pattern in patterns:
                if pattern.pattern_type == DataPattern.SEASONAL:
                    recommendations.append("تم اكتشاف نمط موسمي - خطط للتغيرات الموسمية")
                elif pattern.pattern_type == DataPattern.CYCLIC:
                    recommendations.append("نمط دوري مكتشف - توقع تكرار هذا النمط")
                elif pattern.pattern_type == DataPattern.TRENDING:
                    if pattern.trend_direction == "صاعد":
                        recommendations.append("اتجاه نمو قوي - استثمر في هذا المجال")
                    else:
                        recommendations.append("اتجاه هبوط - قد تحتاج لاستراتيجية تصحيحية")
            
            # توصيات للشذوذ
            if anomalies:
                recommendations.append(f"تم اكتشاف {len(anomalies)} قيمة شاذة - تحقق من البيانات")
            
            # توصيات عامة
            if np.std(predictions) > np.mean(predictions) * 0.3:
                recommendations.append("تقلبات عالية متوقعة - خطط لإدارة المخاطر")
            
            return recommendations[:5]  # أقصى 5 توصيات
            
        except Exception as e:
            self.logger.error(f"خطأ في إنشاء التوصيات: {e}")
            return ["تعذر إنشاء توصيات"]

    async def _evaluate_model_performance(
        self, 
        model_name: str, 
        data: Dict[str, Any], 
        request: PredictionRequest
    ) -> Dict[str, float]:
        """تقييم أداء النموذج"""
        try:
            performance = {}
            df = data['dataframe']
            target_variable = data['target_variable']
            
            if target_variable and target_variable in df.columns:
                model = self.models[model_name]
                X = df.drop(columns=[target_variable])
                y = df[target_variable]
                
                # تقسيم البيانات للاختبار
                if len(X) > 10:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42
                    )
                    
                    # تدريب وتقييم
                    if hasattr(model, 'fit'):
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        
                        # حساب المقاييس
                        performance['mse'] = float(mean_squared_error(y_test, y_pred))
                        performance['rmse'] = float(np.sqrt(performance['mse']))
                        performance['r2'] = float(r2_score(y_test, y_pred))
                        
                        # دقة إضافية للتصنيف
                        if request.prediction_type == PredictionType.CLASSIFICATION:
                            y_pred_class = np.round(y_pred)
                            performance['accuracy'] = float(accuracy_score(y_test, y_pred_class))
            
            performance['model_name'] = model_name
            performance['evaluation_time'] = datetime.now().isoformat()
            
            return performance
            
        except Exception as e:
            self.logger.error(f"خطأ في تقييم الأداء: {e}")
            return {'error': str(e)}

    def _generate_cache_key(self, request: PredictionRequest) -> str:
        """إنشاء مفتاح للذاكرة المؤقتة"""
        try:
            # إنشاء hash من البيانات المهمة
            key_data = {
                'prediction_type': request.prediction_type.value,
                'target_variable': request.target_variable,
                'prediction_horizon': request.prediction_horizon,
                'model_complexity': request.model_complexity.value,
                'features': sorted(request.features) if request.features else []
            }
            
            key_string = json.dumps(key_data, sort_keys=True)
            return hashlib.md5(key_string.encode()).hexdigest()
            
        except Exception as e:
            self.logger.error(f"خطأ في إنشاء مفتاح الذاكرة المؤقتة: {e}")
            return f"key_{datetime.now().timestamp()}"

    async def get_prediction_statistics(self) -> Dict[str, Any]:
        """الحصول على إحصائيات التنبؤ"""
        try:
            stats = {
                'total_requests': len(self.request_history),
                'model_usage': dict(self.model_usage_stats),
                'cache_size': len(self.prediction_cache),
                'performance_metrics': self.performance_metrics,
                'prediction_types_distribution': {},
                'average_processing_time': 0.0,
                'engine_status': 'running' if self.is_running else 'stopped'
            }
            
            # توزيع أنواع التنبؤ
            type_counts = defaultdict(int)
            total_time = 0
            
            for request in self.request_history:
                type_counts[request.prediction_type.value] += 1
            
            stats['prediction_types_distribution'] = dict(type_counts)
            
            # متوسط وقت المعالجة من النتائج المخزنة
            if self.prediction_cache:
                total_time = sum(result.processing_time for result in self.prediction_cache.values())
                stats['average_processing_time'] = total_time / len(self.prediction_cache)
            
            return stats
            
        except Exception as e:
            self.logger.error(f"خطأ في الحصول على الإحصائيات: {e}")
            return {'error': str(e)}

    async def save_models(self):
        """حفظ النماذج والحالة"""
        try:
            # حفظ النماذج
            models_file = self.models_dir / "models.pkl"
            with open(models_file, 'wb') as f:
                pickle.dump(self.models, f)
            
            # حفظ معالجات البيانات
            scalers_file = self.models_dir / "scalers.pkl"
            with open(scalers_file, 'wb') as f:
                pickle.dump(self.scalers, f)
            
            encoders_file = self.models_dir / "encoders.pkl"
            with open(encoders_file, 'wb') as f:
                pickle.dump(self.encoders, f)
            
            # حفظ الإحصائيات
            stats_file = self.models_dir / "statistics.json"
            stats = await self.get_prediction_statistics()
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=2, default=str)
            
            self.logger.info("تم حفظ نماذج التنبؤ والحالة")
            
        except Exception as e:
            self.logger.error(f"خطأ في حفظ النماذج: {e}")

    async def load_models(self):
        """تحميل النماذج والحالة"""
        try:
            # تحميل النماذج
            models_file = self.models_dir / "models.pkl"
            if models_file.exists():
                with open(models_file, 'rb') as f:
                    self.models.update(pickle.load(f))
            
            # تحميل معالجات البيانات
            scalers_file = self.models_dir / "scalers.pkl"
            if scalers_file.exists():
                with open(scalers_file, 'rb') as f:
                    self.scalers.update(pickle.load(f))
            
            encoders_file = self.models_dir / "encoders.pkl"
            if encoders_file.exists():
                with open(encoders_file, 'rb') as f:
                    self.encoders.update(pickle.load(f))
            
            self.logger.info("تم تحميل نماذج التنبؤ والحالة")
            
        except Exception as e:
            self.logger.error(f"خطأ في تحميل النماذج: {e}")

# إنشاء مثيل عام
smart_prediction_engine = SmartPredictionEngine()

async def get_prediction_engine() -> SmartPredictionEngine:
    """الحصول على محرك التنبؤ الذكي"""
    return smart_prediction_engine

if __name__ == "__main__":
    async def test_prediction_engine():
        """اختبار محرك التنبؤ الذكي"""
        print("🔮 اختبار محرك التنبؤ الذكي المتقدم")
        print("=" * 60)
        
        engine = await get_prediction_engine()
        await engine.start_prediction_engine()
        
        # بيانات تجريبية للاختبار
        sample_data = {
            'dataframe': {
                'value': [10, 12, 11, 13, 15, 14, 16, 18, 17, 19, 21, 20],
                'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                'feature2': [0.5, 0.7, 0.6, 0.8, 1.0, 0.9, 1.1, 1.3, 1.2, 1.4, 1.6, 1.5]
            }
        }
        
        # طلب تنبؤ
        request = PredictionRequest(
            request_id="test_001",
            prediction_type=PredictionType.TIME_SERIES,
            data=sample_data,
            target_variable='value',
            prediction_horizon=5,
            confidence_level=0.95,
            model_complexity=ModelComplexity.MODERATE
        )
        
        print("📊 بيانات الاختبار:")
        print(f"  • نوع التنبؤ: {request.prediction_type.value}")
        print(f"  • المتغير المستهدف: {request.target_variable}")
        print(f"  • أفق التنبؤ: {request.prediction_horizon}")
        print(f"  • مستوى الثقة: {request.confidence_level}")
        
        # تنفيذ التنبؤ
        print("\n🔍 تنفيذ التنبؤ...")
        result = await engine.predict(request)
        
        print(f"\n📈 نتائج التنبؤ:")
        print(f"  • التنبؤات: {[f'{p:.2f}' for p in result.predictions]}")
        print(f"  • وقت المعالجة: {result.processing_time:.3f} ثانية")
        print(f"  • الأنماط المكتشفة: {result.detected_patterns}")
        
        if result.confidence_intervals:
            print(f"  • فترات الثقة:")
            for i, (lower, upper) in enumerate(result.confidence_intervals):
                print(f"    - التنبؤ {i+1}: [{lower:.2f}, {upper:.2f}]")
        
        if result.feature_importance:
            print(f"  • أهمية الميزات:")
            for feature, importance in result.feature_importance.items():
                print(f"    - {feature}: {importance:.3f}")
        
        if result.anomalies:
            print(f"  • الشذوذ المكتشف: {result.anomalies}")
        
        print(f"\n💡 التوصيات:")
        for recommendation in result.recommendations:
            print(f"  • {recommendation}")
        
        print(f"\n📊 مقاييس الأداء:")
        for metric, value in result.model_performance.items():
            print(f"  • {metric}: {value}")
        
        # اختبار أنواع تنبؤ مختلفة
        print(f"\n🎯 اختبار أنواع التنبؤ المختلفة:")
        
        # تنبؤ بالتصنيف
        classification_request = PredictionRequest(
            request_id="test_002",
            prediction_type=PredictionType.CLASSIFICATION,
            data=sample_data,
            target_variable='value',
            prediction_horizon=3
        )
        
        classification_result = await engine.predict(classification_request)
        print(f"  • التصنيف: {[f'{p:.2f}' for p in classification_result.predictions]}")
        
        # كشف الشذوذ
        anomaly_request = PredictionRequest(
            request_id="test_003",
            prediction_type=PredictionType.ANOMALY_DETECTION,
            data=sample_data,
            prediction_horizon=len(sample_data['dataframe']['value'])
        )
        
        anomaly_result = await engine.predict(anomaly_request)
        print(f"  • كشف الشذوذ: {anomaly_result.anomalies}")
        
        # الحصول على الإحصائيات
        print(f"\n📊 إحصائيات المحرك:")
        stats = await engine.get_prediction_statistics()
        print(f"  • إجمالي الطلبات: {stats['total_requests']}")
        print(f"  • استخدام النماذج: {stats['model_usage']}")
        print(f"  • حجم الذاكرة المؤقتة: {stats['cache_size']}")
        print(f"  • متوسط وقت المعالجة: {stats['average_processing_time']:.3f} ثانية")
        
        # حفظ النماذج
        await engine.save_models()
        print(f"\n💾 تم حفظ النماذج والحالة")
        
        # إيقاف المحرك
        await engine.stop_prediction_engine()
        print(f"\n⏹️ تم إيقاف محرك التنبؤ")
        
        print("\n✨ انتهى الاختبار بنجاح!")

    # تشغيل الاختبار
    asyncio.run(test_prediction_engine())
