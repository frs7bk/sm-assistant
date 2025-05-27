#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
محرك التنبؤ الذكي المتقدم
Smart Prediction Engine with Advanced Machine Learning
"""

import asyncio
import logging
import numpy as np
import pandas as pd
import pickle
import json
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict, field
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split, TimeSeriesSplit
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Deep Learning
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

# Time Series
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

@dataclass
class PredictionRequest:
    """طلب التنبؤ"""
    request_id: str
    user_id: str
    prediction_type: str
    input_data: Dict[str, Any]
    time_horizon: timedelta
    confidence_level: float = 0.95
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class PredictionResult:
    """نتيجة التنبؤ"""
    prediction_id: str
    request_id: str
    predicted_value: Any
    confidence_score: float
    prediction_interval: Tuple[float, float]
    feature_importance: Dict[str, float]
    model_used: str
    accuracy_metrics: Dict[str, float]
    explanation: str
    recommendations: List[str]
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class PredictionModel:
    """نموذج التنبؤ"""
    model_id: str
    model_type: str
    model_object: Any
    performance_metrics: Dict[str, float]
    feature_names: List[str]
    last_trained: datetime
    training_data_size: int
    is_active: bool = True

class LSTMPredictor(nn.Module):
    """شبكة LSTM للتنبؤ"""

    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, output_size: int = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)

        return out

class SmartPredictionEngine:
    """محرك التنبؤ الذكي المتقدم"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # حالة المحرك
        self.is_initialized = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # النماذج
        self.models: Dict[str, PredictionModel] = {}
        self.model_selector = None

        # معالجات البيانات
        self.scalers: Dict[str, StandardScaler] = {}
        self.encoders: Dict[str, LabelEncoder] = {}

        # تاريخ التنبؤات
        self.prediction_history = deque(maxlen=10000)
        self.model_performance_history = defaultdict(list)

        # مخزن البيانات
        self.training_data: Dict[str, pd.DataFrame] = {}
        self.feature_store = {}

        # إحصائيات الأداء
        self.performance_stats = {
            "total_predictions": 0,
            "successful_predictions": 0,
            "average_accuracy": 0.0,
            "models_trained": 0,
            "avg_processing_time": 0.0
        }

        # مسارات الحفظ
        self.models_dir = Path("data/prediction_models")
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # إعدادات التنبؤ
        self.prediction_config = {
            "min_training_samples": 50,
            "retrain_threshold": 0.1,  # تراجع في الدقة
            "max_models_per_type": 5,
            "ensemble_threshold": 3,  # عدد النماذج للتصويت
            "confidence_threshold": 0.7
        }

    async def initialize(self):
        """تهيئة محرك التنبؤ"""
        self.logger.info("🔮 تهيئة محرك التنبؤ الذكي المتقدم...")

        try:
            # تحميل النماذج المحفوظة
            await self._load_saved_models()

            # تهيئة محدد النماذج
            await self._initialize_model_selector()

            # تحميل البيانات التدريبية
            await self._load_training_data()

            self.is_initialized = True
            self.logger.info("✅ تم تهيئة محرك التنبؤ بنجاح")

        except Exception as e:
            self.logger.error(f"❌ فشل تهيئة محرك التنبؤ: {e}")
            # تهيئة أساسية
            self.is_initialized = True

    async def _load_saved_models(self):
        """تحميل النماذج المحفوظة"""
        try:
            model_files = list(self.models_dir.glob("*.pkl"))

            for model_file in model_files:
                try:
                    with open(model_file, 'rb') as f:
                        model_data = pickle.load(f)

                    self.models[model_data['model_id']] = PredictionModel(**model_data)
                    self.logger.info(f"✅ تم تحميل النموذج: {model_data['model_id']}")

                except Exception as e:
                    self.logger.warning(f"⚠️ فشل تحميل النموذج {model_file}: {e}")

            if self.models:
                self.logger.info(f"📊 تم تحميل {len(self.models)} نموذج")

        except Exception as e:
            self.logger.error(f"خطأ في تحميل النماذج: {e}")

    async def _initialize_model_selector(self):
        """تهيئة محدد النماذج"""
        try:
            if SKLEARN_AVAILABLE:
                # نموذج لاختيار أفضل نموذج تنبؤ
                self.model_selector = RandomForestRegressor(
                    n_estimators=50,
                    random_state=42
                )

                self.logger.info("✅ تم تهيئة محدد النماذج")

        except Exception as e:
            self.logger.warning(f"⚠️ خطأ في تهيئة محدد النماذج: {e}")

    async def _load_training_data(self):
        """تحميل البيانات التدريبية"""
        try:
            data_dir = Path("data/training")
            if data_dir.exists():
                for data_file in data_dir.glob("*.csv"):
                    try:
                        df = pd.read_csv(data_file)
                        dataset_name = data_file.stem
                        self.training_data[dataset_name] = df

                        self.logger.info(f"✅ تم تحميل بيانات: {dataset_name}")

                    except Exception as e:
                        self.logger.warning(f"⚠️ فشل تحميل {data_file}: {e}")

        except Exception as e:
            self.logger.error(f"خطأ في تحميل البيانات التدريبية: {e}")

    async def predict(self, request: PredictionRequest) -> PredictionResult:
        """تنفيذ التنبؤ"""
        start_time = datetime.now()

        try:
            self.performance_stats["total_predictions"] += 1

            # اختيار أفضل نموذج
            best_model = await self._select_best_model(request)

            if not best_model:
                # إنشاء نموذج جديد
                best_model = await self._create_new_model(request)

            # تحضير البيانات
            processed_data = await self._prepare_prediction_data(request, best_model)

            # تنفيذ التنبؤ
            prediction_value = await self._execute_prediction(best_model, processed_data)

            # حساب الثقة والفترة
            confidence, interval = await self._calculate_confidence(
                best_model, processed_data, prediction_value
            )

            # توليد التفسير والتوصيات
            explanation = await self._generate_explanation(best_model, processed_data, prediction_value)
            recommendations = await self._generate_recommendations(request, prediction_value)

            # إنشاء النتيجة
            result = PredictionResult(
                prediction_id=f"pred_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{request.user_id}",
                request_id=request.request_id,
                predicted_value=prediction_value,
                confidence_score=confidence,
                prediction_interval=interval,
                feature_importance=await self._get_feature_importance(best_model),
                model_used=best_model.model_id,
                accuracy_metrics=best_model.performance_metrics,
                explanation=explanation,
                recommendations=recommendations
            )

            # حفظ في التاريخ
            self.prediction_history.append(result)

            # تحديث الإحصائيات
            processing_time = (datetime.now() - start_time).total_seconds()
            self.performance_stats["successful_predictions"] += 1
            self._update_performance_stats(processing_time)

            self.logger.info(f"✅ تم التنبؤ بنجاح: {result.prediction_id}")

            return result

        except Exception as e:
            self.logger.error(f"❌ فشل التنبؤ: {e}")

            # نتيجة احتياطية
            return PredictionResult(
                prediction_id=f"pred_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                request_id=request.request_id,
                predicted_value=None,
                confidence_score=0.0,
                prediction_interval=(0.0, 0.0),
                feature_importance={},
                model_used="error",
                accuracy_metrics={"error": str(e)},
                explanation="حدث خطأ في التنبؤ",
                recommendations=["يرجى المحاولة مرة أخرى"]
            )

    async def _select_best_model(self, request: PredictionRequest) -> Optional[PredictionModel]:
        """اختيار أفضل نموذج للتنبؤ"""
        try:
            # تصفية النماذج المناسبة
            suitable_models = [
                model for model in self.models.values()
                if model.is_active and model.model_type == request.prediction_type
            ]

            if not suitable_models:
                return None

            # ترتيب النماذج حسب الأداء
            suitable_models.sort(
                key=lambda m: m.performance_metrics.get('accuracy', 0.0),
                reverse=True
            )

            return suitable_models[0]

        except Exception as e:
            self.logger.error(f"خطأ في اختيار النموذج: {e}")
            return None

    async def _create_new_model(self, request: PredictionRequest) -> PredictionModel:
        """إنشاء نموذج جديد"""
        try:
            model_id = f"{request.prediction_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # اختيار نوع النموذج بناءً على طبيعة البيانات
            if request.prediction_type == "time_series":
                model_object = await self._create_time_series_model()
            elif request.prediction_type == "regression":
                model_object = await self._create_regression_model()
            elif request.prediction_type == "classification":
                model_object = await self._create_classification_model()
            else:
                model_object = await self._create_general_model()

            # إنشاء نموذج التنبؤ
            prediction_model = PredictionModel(
                model_id=model_id,
                model_type=request.prediction_type,
                model_object=model_object,
                performance_metrics={},
                feature_names=[],
                last_trained=datetime.now(),
                training_data_size=0
            )

            # تدريب النموذج إذا توفرت بيانات
            await self._train_model(prediction_model, request)

            # حفظ النموذج
            self.models[model_id] = prediction_model
            await self._save_model(prediction_model)

            self.performance_stats["models_trained"] += 1

            return prediction_model

        except Exception as e:
            self.logger.error(f"خطأ في إنشاء النموذج الجديد: {e}")
            raise

    async def _create_time_series_model(self):
        """إنشاء نموذج السلاسل الزمنية"""
        if PYTORCH_AVAILABLE:
            return LSTMPredictor(input_size=10, hidden_size=64, num_layers=2)
        elif SKLEARN_AVAILABLE:
            return RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            return None

    async def _create_regression_model(self):
        """إنشاء نموذج الانحدار"""
        if SKLEARN_AVAILABLE:
            return GradientBoostingRegressor(n_estimators=100, random_state=42)
        else:
            return None

    async def _create_classification_model(self):
        """إنشاء نموذج التصنيف"""
        if SKLEARN_AVAILABLE:
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            return None

    async def _create_general_model(self):
        """إنشاء نموذج عام"""
        if SKLEARN_AVAILABLE:
            return RandomForestRegressor(n_estimators=50, random_state=42)
        else:
            return None

    async def _train_model(self, model: PredictionModel, request: PredictionRequest):
        """تدريب النموذج"""
        try:
            # البحث عن بيانات تدريبية مناسبة
            training_data = await self._get_training_data(request.prediction_type)

            if training_data is None or len(training_data) < self.prediction_config["min_training_samples"]:
                self.logger.warning("لا توجد بيانات تدريبية كافية")
                return

            # تحضير البيانات
            X, y = await self._prepare_training_data(training_data, request)

            if X is None or y is None:
                return

            # تقسيم البيانات
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # تدريب النموذج
            if PYTORCH_AVAILABLE and isinstance(model.model_object, nn.Module):
                await self._train_neural_model(model, X_train, y_train, X_test, y_test)
            elif SKLEARN_AVAILABLE:
                await self._train_sklearn_model(model, X_train, y_train, X_test, y_test)

            # تحديث معلومات النموذج
            model.feature_names = list(X.columns) if isinstance(X, pd.DataFrame) else [f"feature_{i}" for i in range(X.shape[1])]
            model.training_data_size = len(training_data)
            model.last_trained = datetime.now()

            self.logger.info(f"✅ تم تدريب النموذج: {model.model_id}")

        except Exception as e:
            self.logger.error(f"خطأ في تدريب النموذج: {e}")

    async def _train_neural_model(self, model: PredictionModel, X_train, y_train, X_test, y_test):
        """تدريب النموذج العصبي"""
        try:
            model.model_object.to(self.device)

            # تحويل البيانات
            X_train_tensor = torch.FloatTensor(X_train.values if isinstance(X_train, pd.DataFrame) else X_train)
            y_train_tensor = torch.FloatTensor(y_train.values if isinstance(y_train, pd.Series) else y_train)
            X_test_tensor = torch.FloatTensor(X_test.values if isinstance(X_test, pd.DataFrame) else X_test)
            y_test_tensor = torch.FloatTensor(y_test.values if isinstance(y_test, pd.Series) else y_test)

            # إعداد التدريب
            optimizer = optim.Adam(model.model_object.parameters(), lr=0.001)
            criterion = nn.MSELoss()

            # التدريب
            model.model_object.train()
            for epoch in range(100):
                optimizer.zero_grad()
                outputs = model.model_object(X_train_tensor.unsqueeze(1))
                loss = criterion(outputs.squeeze(), y_train_tensor)
                loss.backward()
                optimizer.step()

            # التقييم
            model.model_object.eval()
            with torch.no_grad():
                test_outputs = model.model_object(X_test_tensor.unsqueeze(1))
                test_predictions = test_outputs.squeeze().cpu().numpy()

                mse = mean_squared_error(y_test, test_predictions)
                r2 = r2_score(y_test, test_predictions)

                model.performance_metrics = {
                    "mse": float(mse),
                    "r2": float(r2),
                    "accuracy": float(max(0, r2))  # استخدام R² كمقياس دقة
                }

        except Exception as e:
            self.logger.error(f"خطأ في تدريب النموذج العصبي: {e}")

    async def _train_sklearn_model(self, model: PredictionModel, X_train, y_train, X_test, y_test):
        """تدريب نموذج sklearn"""
        try:
            # تدريب النموذج
            model.model_object.fit(X_train, y_train)

            # التنبؤ على بيانات الاختبار
            y_pred = model.model_object.predict(X_test)

            # حساب المقاييس
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            model.performance_metrics = {
                "mse": float(mse),
                "mae": float(mae),
                "r2": float(r2),
                "accuracy": float(max(0, r2))
            }

        except Exception as e:
            self.logger.error(f"خطأ في تدريب نموذج sklearn: {e}")

    async def _get_training_data(self, prediction_type: str) -> Optional[pd.DataFrame]:
        """الحصول على بيانات التدريب"""
        try:
            # البحث في البيانات المحفوظة
            for dataset_name, data in self.training_data.items():
                if prediction_type in dataset_name.lower():
                    return data

            # إنشاء بيانات تجريبية إذا لم توجد
            return await self._generate_synthetic_data(prediction_type)

        except Exception as e:
            self.logger.error(f"خطأ في الحصول على بيانات التدريب: {e}")
            return None

    async def _generate_synthetic_data(self, prediction_type: str) -> pd.DataFrame:
        """توليد بيانات تجريبية"""
        try:
            np.random.seed(42)
            n_samples = 1000

            if prediction_type == "time_series":
                # بيانات سلسلة زمنية
                dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='D')
                trend = np.linspace(100, 200, n_samples)
                seasonal = 10 * np.sin(2 * np.pi * np.arange(n_samples) / 365.25)
                noise = np.random.normal(0, 5, n_samples)
                values = trend + seasonal + noise

                return pd.DataFrame({
                    'date': dates,
                    'value': values,
                    'target': values + np.random.normal(0, 2, n_samples)
                })

            else:
                # بيانات عامة
                data = {
                    'feature_1': np.random.normal(0, 1, n_samples),
                    'feature_2': np.random.normal(5, 2, n_samples),
                    'feature_3': np.random.uniform(0, 10, n_samples),
                    'feature_4': np.random.exponential(2, n_samples)
                }

                # إنشاء المتغير التابع
                df = pd.DataFrame(data)
                df['target'] = (
                    2 * df['feature_1'] + 
                    0.5 * df['feature_2'] - 
                    0.3 * df['feature_3'] + 
                    0.1 * df['feature_4'] + 
                    np.random.normal(0, 0.5, n_samples)
                )

                return df

        except Exception as e:
            self.logger.error(f"خطأ في توليد البيانات التجريبية: {e}")
            return pd.DataFrame()

    async def _prepare_training_data(self, data: pd.DataFrame, request: PredictionRequest) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
        """تحضير بيانات التدريب"""
        try:
            if data.empty:
                return None, None

            # تحديد المتغير التابع
            target_column = 'target'
            if target_column not in data.columns:
                # اختيار آخر عمود كمتغير تابع
                target_column = data.columns[-1]

            # فصل الميزات والمتغير التابع
            X = data.drop(columns=[target_column])
            y = data[target_column]

            # معالجة البيانات المفقودة
            X = X.fillna(X.mean() if X.select_dtypes(include=[np.number]).shape[1] > 0 else X.mode().iloc[0])
            y = y.fillna(y.mean())

            # تشفير البيانات النصية
            for column in X.select_dtypes(include=['object']).columns:
                if column not in self.encoders:
                    self.encoders[column] = LabelEncoder()

                X[column] = self.encoders[column].fit_transform(X[column].astype(str))

            # تطبيع البيانات
            scaler_key = f"{request.prediction_type}_scaler"
            if scaler_key not in self.scalers:
                self.scalers[scaler_key] = StandardScaler()

            X_scaled = pd.DataFrame(
                self.scalers[scaler_key].fit_transform(X),
                columns=X.columns,
                index=X.index
            )

            return X_scaled, y

        except Exception as e:
            self.logger.error(f"خطأ في تحضير بيانات التدريب: {e}")
            return None, None

    async def _prepare_prediction_data(self, request: PredictionRequest, model: PredictionModel) -> Optional[np.ndarray]:
        """تحضير بيانات التنبؤ"""
        try:
            # تحويل بيانات الإدخال إلى DataFrame
            input_df = pd.DataFrame([request.input_data])

            # التأكد من وجود جميع الميزات المطلوبة
            for feature in model.feature_names:
                if feature not in input_df.columns:
                    input_df[feature] = 0  # قيمة افتراضية

            # ترتيب الأعمدة
            input_df = input_df[model.feature_names]

            # تطبيق نفس المعالجة المستخدمة في التدريب
            scaler_key = f"{request.prediction_type}_scaler"
            if scaler_key in self.scalers:
                input_scaled = self.scalers[scaler_key].transform(input_df)
                return input_scaled

            return input_df.values

        except Exception as e:
            self.logger.error(f"خطأ في تحضير بيانات التنبؤ: {e}")
            return None

    async def _execute_prediction(self, model: PredictionModel, data: np.ndarray) -> Any:
        """تنفيذ التنبؤ"""
        try:
            if PYTORCH_AVAILABLE and isinstance(model.model_object, nn.Module):
                # تنبؤ النموذج العصبي
                model.model_object.eval()
                with torch.no_grad():
                    input_tensor = torch.FloatTensor(data)
                    output = model.model_object(input_tensor.unsqueeze(1))
                    return float(output.squeeze().cpu().numpy())

            elif SKLEARN_AVAILABLE and hasattr(model.model_object, 'predict'):
                # تنبؤ نموذج sklearn
                prediction = model.model_object.predict(data)
                return float(prediction[0]) if isinstance(prediction, np.ndarray) else float(prediction)

            else:
                # نموذج افتراضي
                return float(np.mean(data))

        except Exception as e:
            self.logger.error(f"خطأ في تنفيذ التنبؤ: {e}")
            return 0.0

    async def _calculate_confidence(self, model: PredictionModel, data: np.ndarray, prediction: Any) -> Tuple[float, Tuple[float, float]]:
        """حساب الثقة وفترة التنبؤ"""
        try:
            # حساب الثقة بناءً على أداء النموذج
            base_confidence = model.performance_metrics.get('accuracy', 0.5)

            # تعديل الثقة بناءً على البيانات
            data_quality = 1.0 - (np.std(data) / (np.mean(np.abs(data)) + 1e-8))
            data_quality = max(0.0, min(1.0, data_quality))

            confidence = (base_confidence + data_quality) / 2

            # حساب فترة الثقة
            error_margin = model.performance_metrics.get('mse', 1.0) ** 0.5
            interval = (
                float(prediction - 1.96 * error_margin),
                float(prediction + 1.96 * error_margin)
            )

            return float(confidence), interval

        except Exception as e:
            self.logger.error(f"خطأ في حساب الثقة: {e}")
            return 0.5, (0.0, 0.0)

    async def _get_feature_importance(self, model: PredictionModel) -> Dict[str, float]:
        """الحصول على أهمية الميزات"""
        try:
            importance = {}

            if hasattr(model.model_object, 'feature_importances_'):
                # نماذج sklearn مع feature_importances_
                importances = model.model_object.feature_importances_
                for i, importance_value in enumerate(importances):
                    feature_name = model.feature_names[i] if i < len(model.feature_names) else f"feature_{i}"
                    importance[feature_name] = float(importance_value)

            elif hasattr(model.model_object, 'coef_'):
                # نماذج خطية
                coefficients = model.model_object.coef_
                for i, coef in enumerate(coefficients):
                    feature_name = model.feature_names[i] if i < len(model.feature_names) else f"feature_{i}"
                    importance[feature_name] = float(abs(coef))

            else:
                # أهمية افتراضية متساوية
                for feature_name in model.feature_names:
                    importance[feature_name] = 1.0 / len(model.feature_names)

            return importance

        except Exception as e:
            self.logger.error(f"خطأ في حساب أهمية الميزات: {e}")
            return {}

    async def _generate_explanation(self, model: PredictionModel, data: np.ndarray, prediction: Any) -> str:
        """توليد تفسير للتنبؤ"""
        try:
            feature_importance = await self._get_feature_importance(model)

            if not feature_importance:
                return f"التنبؤ: {prediction:.2f} باستخدام نموذج {model.model_type}"

            # العثور على أهم الميزات
            top_features = sorted(
                feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]

            explanation = f"التنبؤ: {prediction:.2f}\n"
            explanation += f"النموذج المستخدم: {model.model_type}\n"
            explanation += f"دقة النموذج: {model.performance_metrics.get('accuracy', 0):.1%}\n"
            explanation += "أهم العوامل المؤثرة:\n"

            for feature, importance in top_features:
                explanation += f"• {feature}: {importance:.1%}\n"

            return explanation

        except Exception as e:
            self.logger.error(f"خطأ في توليد التفسير: {e}")
            return f"التنبؤ: {prediction}"

    async def _generate_recommendations(self, request: PredictionRequest, prediction: Any) -> List[str]:
        """توليد التوصيات"""
        try:
            recommendations = []

            if request.prediction_type == "time_series":
                if isinstance(prediction, (int, float)):
                    if prediction > 0:
                        recommendations.append("الاتجاه العام إيجابي")
                        recommendations.append("استعد للنمو المتوقع")
                    else:
                        recommendations.append("الاتجاه العام سلبي")
                        recommendations.append("اتخذ إجراءات وقائية")

            elif request.prediction_type == "regression":
                recommendations.append("راقب القيم الفعلية مقابل المتوقعة")
                recommendations.append("حدث النموذج بانتظام")

            # توصيات عامة
            recommendations.extend([
                "استخدم هذا التنبؤ كدليل وليس قرار نهائي",
                "راجع البيانات المدخلة للتأكد من صحتها",
                "قارن مع تنبؤات من مصادر أخرى"
            ])

            return recommendations[:5]  # أقصى 5 توصيات

        except Exception as e:
            self.logger.error(f"خطأ في توليد التوصيات: {e}")
            return ["استخدم النتائج بحذر"]

    async def _save_model(self, model: PredictionModel):
        """حفظ النموذج"""
        try:
            model_data = {
                'model_id': model.model_id,
                'model_type': model.model_type,
                'model_object': model.model_object,
                'performance_metrics': model.performance_metrics,
                'feature_names': model.feature_names,
                'last_trained': model.last_trained,
                'training_data_size': model.training_data_size,
                'is_active': model.is_active
            }

            model_file = self.models_dir / f"{model.model_id}.pkl"
            with open(model_file, 'wb') as f:
                pickle.dump(model_data, f)

            self.logger.info(f"✅ تم حفظ النموذج: {model.model_id}")

        except Exception as e:
            self.logger.error(f"خطأ في حفظ النموذج: {e}")

    def _update_performance_stats(self, processing_time: float):
        """تحديث إحصائيات الأداء"""
        try:
            total = self.performance_stats["total_predictions"]
            current_avg = self.performance_stats["avg_processing_time"]

            new_avg = (current_avg * (total - 1) + processing_time) / total
            self.performance_stats["avg_processing_time"] = new_avg

            # حساب متوسط الدقة
            if self.models:
                total_accuracy = sum(
                    model.performance_metrics.get('accuracy', 0.0)
                    for model in self.models.values()
                )
                self.performance_stats["average_accuracy"] = total_accuracy / len(self.models)

        except Exception as e:
            self.logger.error(f"خطأ في تحديث الإحصائيات: {e}")

    async def get_prediction_statistics(self) -> Dict[str, Any]:
        """الحصول على إحصائيات التنبؤ"""
        try:
            stats = {
                "performance": self.performance_stats,
                "models": {
                    "total_models": len(self.models),
                    "active_models": sum(1 for m in self.models.values() if m.is_active),
                    "model_types": list(set(m.model_type for m in self.models.values())),
                    "best_model": max(
                        self.models.values(),
                        key=lambda m: m.performance_metrics.get('accuracy', 0),
                        default=None
                    ).model_id if self.models else None
                },
                "predictions": {
                    "total_predictions": len(self.prediction_history),
                    "recent_predictions": len([
                        p for p in self.prediction_history
                        if (datetime.now() - p.timestamp).days < 7
                    ]),
                    "average_confidence": np.mean([
                        p.confidence_score for p in self.prediction_history
                    ]) if self.prediction_history else 0.0
                },
                "data": {
                    "training_datasets": len(self.training_data),
                    "feature_encoders": len(self.encoders),
                    "data_scalers": len(self.scalers)
                }
            }

            return stats

        except Exception as e:
            self.logger.error(f"خطأ في الحصول على الإحصائيات: {e}")
            return {"error": str(e)}

# إنشاء مثيل عام
smart_prediction_engine = SmartPredictionEngine()

async def get_smart_prediction_engine() -> SmartPredictionEngine:
    """الحصول على محرك التنبؤ الذكي"""
    if not smart_prediction_engine.is_initialized:
        await smart_prediction_engine.initialize()
    return smart_prediction_engine

if __name__ == "__main__":
    async def test_prediction_engine():
        """اختبار محرك التنبؤ"""
        print("🔮 اختبار محرك التنبؤ الذكي المتقدم")
        print("=" * 50)

        engine = await get_smart_prediction_engine()

        # إنشاء طلب تنبؤ تجريبي
        request = PredictionRequest(
            request_id="test_request_001",
            user_id="test_user",
            prediction_type="regression",
            input_data={
                "feature_1": 1.5,
                "feature_2": 2.3,
                "feature_3": 0.8,
                "feature_4": 1.2
            },
            time_horizon=timedelta(days=7),
            confidence_level=0.95
        )

        print(f"📝 طلب التنبؤ:")
        print(f"  • النوع: {request.prediction_type}")
        print(f"  • البيانات: {request.input_data}")
        print(f"  • الأفق الزمني: {request.time_horizon.days} أيام")

        # تنفيذ التنبؤ
        result = await engine.predict(request)

        print(f"\n🎯 نتيجة التنبؤ:")
        print(f"  • القيمة المتوقعة: {result.predicted_value}")
        print(f"  • درجة الثقة: {result.confidence_score:.1%}")
        print(f"  • فترة التنبؤ: {result.prediction_interval}")
        print(f"  • النموذج المستخدم: {result.model_used}")

        if result.feature_importance:
            print(f"\n🔍 أهمية الميزات:")
            for feature, importance in result.feature_importance.items():
                print(f"  • {feature}: {importance:.1%}")

        print(f"\n📝 التفسير:")
        print(f"  {result.explanation}")

        if result.recommendations:
            print(f"\n💡 التوصيات:")
            for rec in result.recommendations:
                print(f"  • {rec}")

        # إحصائيات النظام
        stats = await engine.get_prediction_statistics()
        print(f"\n📊 إحصائيات النظام:")
        print(f"  • إجمالي التنبؤات: {stats['performance']['total_predictions']}")
        print(f"  • التنبؤات الناجحة: {stats['performance']['successful_predictions']}")
        print(f"  • متوسط الدقة: {stats['performance']['average_accuracy']:.1%}")
        print(f"  • عدد النماذج: {stats['models']['total_models']}")

        print("\n✨ انتهى الاختبار بنجاح!")

    # تشغيل الاختبار
    asyncio.run(test_prediction_engine())