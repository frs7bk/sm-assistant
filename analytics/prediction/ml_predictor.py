
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
محرك التنبؤ بالتعلم الآلي المتقدم والذكي
Advanced Machine Learning Prediction Engine
"""

import asyncio
import logging
import numpy as np
import pandas as pd
import pickle
import json
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Scientific Computing
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    RandomForestClassifier, GradientBoostingClassifier,
    VotingRegressor, VotingClassifier
)
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet,
    LogisticRegression, SGDRegressor, SGDClassifier
)
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA, FastICA, LatentDirichletAllocation
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    LabelEncoder, OneHotEncoder, PolynomialFeatures
)
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV,
    RandomizedSearchCV, TimeSeriesSplit
)
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.pipeline import Pipeline

# Advanced Libraries
try:
    import xgboost as xgb
    import lightgbm as lgb
    from catboost import CatBoostRegressor, CatBoostClassifier
    ADVANCED_MODELS_AVAILABLE = True
except ImportError:
    ADVANCED_MODELS_AVAILABLE = False
    logging.warning("المكتبات المتقدمة غير متوفرة (XGBoost, LightGBM, CatBoost)")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    logging.warning("PyTorch غير متوفر")

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    TIMESERIES_AVAILABLE = True
except ImportError:
    TIMESERIES_AVAILABLE = False
    logging.warning("مكتبات السلاسل الزمنية غير متوفرة")

class ModelType(Enum):
    """أنواع النماذج"""
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    CLUSTERING = "clustering"
    TIMESERIES = "timeseries"
    DEEP_LEARNING = "deep_learning"
    ENSEMBLE = "ensemble"

class PredictionTask(Enum):
    """مهام التنبؤ"""
    USER_BEHAVIOR = "user_behavior"
    SYSTEM_PERFORMANCE = "system_performance"
    CONTENT_RECOMMENDATION = "content_recommendation"
    TASK_COMPLETION_TIME = "task_completion_time"
    ERROR_PREDICTION = "error_prediction"
    RESOURCE_USAGE = "resource_usage"
    USER_SATISFACTION = "user_satisfaction"
    CUSTOM = "custom"

@dataclass
class ModelPerformance:
    """أداء النموذج"""
    model_name: str
    model_type: ModelType
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    r2_score: float = 0.0
    mse: float = float('inf')
    mae: float = float('inf')
    training_time: float = 0.0
    prediction_time: float = 0.0
    memory_usage: float = 0.0
    feature_importance: Dict[str, float] = None
    cross_val_scores: List[float] = None
    confusion_matrix: List[List[int]] = None
    roc_auc: float = 0.0
    last_updated: datetime = None

    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.now()
        if self.feature_importance is None:
            self.feature_importance = {}
        if self.cross_val_scores is None:
            self.cross_val_scores = []

@dataclass
class PredictionResult:
    """نتيجة التنبؤ"""
    prediction: Union[float, int, str, List]
    confidence: float
    probability_distribution: Dict[str, float] = None
    feature_contributions: Dict[str, float] = None
    model_used: str = ""
    prediction_time: float = 0.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.probability_distribution is None:
            self.probability_distribution = {}
        if self.feature_contributions is None:
            self.feature_contributions = {}
        if self.metadata is None:
            self.metadata = {}

class NeuralNetworkModel(nn.Module):
    """نموذج شبكة عصبية متقدم"""
    
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int, dropout_rate: float = 0.2):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # طبقة الإدخال
        prev_size = input_size
        
        # الطبقات المخفية
        for hidden_size in hidden_sizes:
            self.layers.append(nn.Linear(prev_size, hidden_size))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.BatchNorm1d(hidden_size))
            self.layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        # طبقة الإخراج
        self.layers.append(nn.Linear(prev_size, output_size))
        
    def forward(self, x):
        for layer in self.layers:
            if isinstance(layer, nn.BatchNorm1d) and x.size(0) == 1:
                # تخطي BatchNorm للعينة الواحدة
                continue
            x = layer(x)
        return x

class AdvancedMLPredictor:
    """محرك التنبؤ بالتعلم الآلي المتقدم"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}
        self.encoders: Dict[str, Any] = {}
        self.model_performance: Dict[str, ModelPerformance] = {}
        self.feature_columns: Dict[str, List[str]] = {}
        self.prediction_history: List[Dict] = []
        self.model_registry: Dict[PredictionTask, List[str]] = {}
        
        # مسارات الحفظ
        self.models_dir = Path("data/models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.performance_file = self.models_dir / "model_performance.json"
        self.history_file = self.models_dir / "prediction_history.json"
        
        # تحديد النماذج المتاحة
        self._initialize_model_registry()
        
        # تحميل الأداء المحفوظ
        self._load_performance_data()
        
        self.logger.info("تم تهيئة محرك التنبؤ بالتعلم الآلي المتقدم")

    def _initialize_model_registry(self):
        """تهيئة سجل النماذج"""
        base_models = {
            ModelType.REGRESSION: [
                'linear_regression', 'ridge', 'lasso', 'elastic_net',
                'random_forest', 'gradient_boosting', 'svr',
                'decision_tree', 'knn', 'mlp'
            ],
            ModelType.CLASSIFICATION: [
                'logistic_regression', 'random_forest', 'gradient_boosting',
                'svm', 'decision_tree', 'knn', 'naive_bayes', 'mlp'
            ],
            ModelType.CLUSTERING: [
                'kmeans', 'dbscan', 'agglomerative'
            ],
            ModelType.TIMESERIES: [
                'arima', 'exponential_smoothing', 'seasonal_decompose'
            ] if TIMESERIES_AVAILABLE else [],
            ModelType.DEEP_LEARNING: [
                'neural_network', 'autoencoder'
            ] if PYTORCH_AVAILABLE else [],
            ModelType.ENSEMBLE: [
                'voting_regressor', 'voting_classifier', 'stacking'
            ]
        }
        
        if ADVANCED_MODELS_AVAILABLE:
            base_models[ModelType.REGRESSION].extend(['xgboost', 'lightgbm', 'catboost'])
            base_models[ModelType.CLASSIFICATION].extend(['xgboost', 'lightgbm', 'catboost'])
        
        # ربط مهام التنبؤ بالنماذج
        self.model_registry = {
            PredictionTask.USER_BEHAVIOR: base_models[ModelType.CLASSIFICATION] + base_models[ModelType.CLUSTERING],
            PredictionTask.SYSTEM_PERFORMANCE: base_models[ModelType.REGRESSION] + base_models[ModelType.TIMESERIES],
            PredictionTask.CONTENT_RECOMMENDATION: base_models[ModelType.CLASSIFICATION] + base_models[ModelType.CLUSTERING],
            PredictionTask.TASK_COMPLETION_TIME: base_models[ModelType.REGRESSION],
            PredictionTask.ERROR_PREDICTION: base_models[ModelType.CLASSIFICATION],
            PredictionTask.RESOURCE_USAGE: base_models[ModelType.REGRESSION] + base_models[ModelType.TIMESERIES],
            PredictionTask.USER_SATISFACTION: base_models[ModelType.REGRESSION] + base_models[ModelType.CLASSIFICATION],
            PredictionTask.CUSTOM: list(set().union(*base_models.values()))
        }

    def _create_model(self, model_name: str, model_type: ModelType, **kwargs):
        """إنشاء نموذج"""
        try:
            if model_name == 'linear_regression':
                return LinearRegression(**kwargs)
            elif model_name == 'ridge':
                return Ridge(alpha=kwargs.get('alpha', 1.0))
            elif model_name == 'lasso':
                return Lasso(alpha=kwargs.get('alpha', 1.0))
            elif model_name == 'elastic_net':
                return ElasticNet(alpha=kwargs.get('alpha', 1.0))
            elif model_name == 'random_forest':
                if model_type == ModelType.REGRESSION:
                    return RandomForestRegressor(
                        n_estimators=kwargs.get('n_estimators', 100),
                        random_state=42
                    )
                else:
                    return RandomForestClassifier(
                        n_estimators=kwargs.get('n_estimators', 100),
                        random_state=42
                    )
            elif model_name == 'gradient_boosting':
                if model_type == ModelType.REGRESSION:
                    return GradientBoostingRegressor(
                        n_estimators=kwargs.get('n_estimators', 100),
                        random_state=42
                    )
                else:
                    return GradientBoostingClassifier(
                        n_estimators=kwargs.get('n_estimators', 100),
                        random_state=42
                    )
            elif model_name == 'svm' or model_name == 'svr':
                if model_type == ModelType.REGRESSION:
                    return SVR(kernel=kwargs.get('kernel', 'rbf'))
                else:
                    return SVC(kernel=kwargs.get('kernel', 'rbf'), probability=True)
            elif model_name == 'decision_tree':
                if model_type == ModelType.REGRESSION:
                    return DecisionTreeRegressor(random_state=42)
                else:
                    return DecisionTreeClassifier(random_state=42)
            elif model_name == 'knn':
                n_neighbors = kwargs.get('n_neighbors', 5)
                if model_type == ModelType.REGRESSION:
                    return KNeighborsRegressor(n_neighbors=n_neighbors)
                else:
                    return KNeighborsClassifier(n_neighbors=n_neighbors)
            elif model_name == 'naive_bayes':
                return GaussianNB()
            elif model_name == 'mlp':
                hidden_layer_sizes = kwargs.get('hidden_layer_sizes', (100,))
                if model_type == ModelType.REGRESSION:
                    return MLPRegressor(
                        hidden_layer_sizes=hidden_layer_sizes,
                        random_state=42
                    )
                else:
                    return MLPClassifier(
                        hidden_layer_sizes=hidden_layer_sizes,
                        random_state=42
                    )
            elif model_name == 'logistic_regression':
                return LogisticRegression(random_state=42)
            elif model_name == 'kmeans':
                return KMeans(n_clusters=kwargs.get('n_clusters', 3), random_state=42)
            elif model_name == 'dbscan':
                return DBSCAN(eps=kwargs.get('eps', 0.5))
            elif model_name == 'agglomerative':
                return AgglomerativeClustering(n_clusters=kwargs.get('n_clusters', 3))
            
            # النماذج المتقدمة
            elif ADVANCED_MODELS_AVAILABLE:
                if model_name == 'xgboost':
                    if model_type == ModelType.REGRESSION:
                        return xgb.XGBRegressor(random_state=42)
                    else:
                        return xgb.XGBClassifier(random_state=42)
                elif model_name == 'lightgbm':
                    if model_type == ModelType.REGRESSION:
                        return lgb.LGBMRegressor(random_state=42, verbose=-1)
                    else:
                        return lgb.LGBMClassifier(random_state=42, verbose=-1)
                elif model_name == 'catboost':
                    if model_type == ModelType.REGRESSION:
                        return CatBoostRegressor(random_state=42, verbose=False)
                    else:
                        return CatBoostClassifier(random_state=42, verbose=False)
            
            # نماذج السلاسل الزمنية
            elif TIMESERIES_AVAILABLE and model_name in ['arima', 'exponential_smoothing']:
                return model_name  # سيتم التعامل معها بشكل خاص
            
            else:
                self.logger.warning(f"نموذج غير معروف: {model_name}")
                return None
                
        except Exception as e:
            self.logger.error(f"خطأ في إنشاء النموذج {model_name}: {e}")
            return None

    async def train_model(
        self,
        data: pd.DataFrame,
        target_column: str,
        model_name: str,
        model_type: ModelType,
        task: PredictionTask = PredictionTask.CUSTOM,
        test_size: float = 0.2,
        cross_validation: bool = True,
        hyperparameter_tuning: bool = False,
        **model_kwargs
    ) -> ModelPerformance:
        """تدريب النموذج"""
        start_time = datetime.now()
        
        try:
            # التحضير
            if target_column not in data.columns:
                raise ValueError(f"العمود المستهدف '{target_column}' غير موجود في البيانات")
            
            # فصل الميزات والهدف
            X = data.drop(columns=[target_column])
            y = data[target_column]
            
            # معالجة البيانات
            X_processed, y_processed = await self._preprocess_data(X, y, model_type)
            
            # تقسيم البيانات
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y_processed, test_size=test_size, random_state=42
            )
            
            # إنشاء النموذج
            model = self._create_model(model_name, model_type, **model_kwargs)
            if model is None:
                raise ValueError(f"فشل في إنشاء النموذج: {model_name}")
            
            # تحسين المعاملات الفائقة
            if hyperparameter_tuning:
                model = await self._tune_hyperparameters(model, X_train, y_train, model_type)
            
            # التدريب
            if model_name in ['arima', 'exponential_smoothing'] and TIMESERIES_AVAILABLE:
                model = await self._train_timeseries_model(model_name, y_train)
            else:
                model.fit(X_train, y_train)
            
            # التقييم
            performance = await self._evaluate_model(
                model, X_test, y_test, model_name, model_type, cross_validation, X_train, y_train
            )
            
            # حساب وقت التدريب
            training_time = (datetime.now() - start_time).total_seconds()
            performance.training_time = training_time
            
            # حفظ النموذج
            model_key = f"{task.value}_{model_name}_{target_column}"
            self.models[model_key] = model
            self.model_performance[model_key] = performance
            self.feature_columns[model_key] = list(X.columns)
            
            # حفظ البيانات
            await self._save_model(model_key, model)
            await self._save_performance_data()
            
            self.logger.info(f"تم تدريب النموذج بنجاح: {model_key}")
            return performance
            
        except Exception as e:
            self.logger.error(f"خطأ في تدريب النموذج: {e}")
            raise

    async def _preprocess_data(self, X: pd.DataFrame, y: pd.Series, model_type: ModelType) -> Tuple[np.ndarray, np.ndarray]:
        """معالجة البيانات"""
        try:
            # نسخ البيانات
            X_processed = X.copy()
            y_processed = y.copy()
            
            # التعامل مع القيم المفقودة
            X_processed = X_processed.fillna(X_processed.mean() if X_processed.select_dtypes(include=[np.number]).shape[1] > 0 else X_processed.mode().iloc[0])
            
            # ترميز المتغيرات الفئوية
            categorical_columns = X_processed.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                    X_processed[col] = self.encoders[col].fit_transform(X_processed[col].astype(str))
                else:
                    X_processed[col] = self.encoders[col].transform(X_processed[col].astype(str))
            
            # تطبيع البيانات
            if model_type in [ModelType.REGRESSION, ModelType.CLASSIFICATION, ModelType.CLUSTERING]:
                scaler_key = f"{model_type.value}_scaler"
                if scaler_key not in self.scalers:
                    self.scalers[scaler_key] = StandardScaler()
                    X_processed = self.scalers[scaler_key].fit_transform(X_processed)
                else:
                    X_processed = self.scalers[scaler_key].transform(X_processed)
            
            # ترميز الهدف للتصنيف
            if model_type == ModelType.CLASSIFICATION and y_processed.dtype == 'object':
                target_encoder_key = f"{model_type.value}_target_encoder"
                if target_encoder_key not in self.encoders:
                    self.encoders[target_encoder_key] = LabelEncoder()
                    y_processed = self.encoders[target_encoder_key].fit_transform(y_processed.astype(str))
                else:
                    y_processed = self.encoders[target_encoder_key].transform(y_processed.astype(str))
            
            return X_processed, y_processed.values
            
        except Exception as e:
            self.logger.error(f"خطأ في معالجة البيانات: {e}")
            raise

    async def _tune_hyperparameters(self, model, X_train, y_train, model_type: ModelType):
        """تحسين المعاملات الفائقة"""
        try:
            param_grids = {
                'RandomForestRegressor': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10]
                },
                'RandomForestClassifier': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10]
                },
                'SVR': {
                    'C': [0.1, 1, 10],
                    'kernel': ['rbf', 'linear']
                },
                'SVC': {
                    'C': [0.1, 1, 10],
                    'kernel': ['rbf', 'linear']
                }
            }
            
            model_name = model.__class__.__name__
            if model_name in param_grids:
                grid_search = GridSearchCV(
                    model, param_grids[model_name], 
                    cv=5, scoring='r2' if model_type == ModelType.REGRESSION else 'accuracy'
                )
                grid_search.fit(X_train, y_train)
                return grid_search.best_estimator_
            
            return model
            
        except Exception as e:
            self.logger.error(f"خطأ في تحسين المعاملات: {e}")
            return model

    async def _train_timeseries_model(self, model_name: str, y_train):
        """تدريب نماذج السلاسل الزمنية"""
        try:
            if model_name == 'arima':
                model = ARIMA(y_train, order=(1, 1, 1))
                return model.fit()
            elif model_name == 'exponential_smoothing':
                model = ExponentialSmoothing(y_train, trend='add', seasonal='add', seasonal_periods=12)
                return model.fit()
            
            return None
            
        except Exception as e:
            self.logger.error(f"خطأ في تدريب نموذج السلسلة الزمنية: {e}")
            return None

    async def _evaluate_model(
        self, 
        model, 
        X_test, 
        y_test, 
        model_name: str, 
        model_type: ModelType, 
        cross_validation: bool,
        X_train=None, 
        y_train=None
    ) -> ModelPerformance:
        """تقييم النموذج"""
        try:
            performance = ModelPerformance(
                model_name=model_name,
                model_type=model_type
            )
            
            # التنبؤ
            if hasattr(model, 'predict'):
                y_pred = model.predict(X_test)
            else:
                # للنماذج الخاصة مثل ARIMA
                y_pred = model.forecast(len(y_test))
            
            # حساب المقاييس
            if model_type == ModelType.REGRESSION:
                performance.r2_score = r2_score(y_test, y_pred)
                performance.mse = mean_squared_error(y_test, y_pred)
                performance.mae = mean_absolute_error(y_test, y_pred)
            
            elif model_type == ModelType.CLASSIFICATION:
                performance.accuracy = accuracy_score(y_test, y_pred)
                performance.precision = precision_score(y_test, y_pred, average='weighted')
                performance.recall = recall_score(y_test, y_pred, average='weighted')
                performance.f1_score = f1_score(y_test, y_pred, average='weighted')
                
                # مصفوفة الخلط
                cm = confusion_matrix(y_test, y_pred)
                performance.confusion_matrix = cm.tolist()
                
                # ROC AUC للتصنيف الثنائي
                if len(np.unique(y_test)) == 2 and hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_test)[:, 1]
                    performance.roc_auc = roc_auc_score(y_test, y_proba)
            
            # التحقق المتقاطع
            if cross_validation and X_train is not None and y_train is not None:
                try:
                    if model_type == ModelType.REGRESSION:
                        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
                    else:
                        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
                    
                    performance.cross_val_scores = cv_scores.tolist()
                except Exception as cv_error:
                    self.logger.warning(f"فشل التحقق المتقاطع: {cv_error}")
            
            # أهمية الميزات
            if hasattr(model, 'feature_importances_'):
                feature_names = [f"feature_{i}" for i in range(len(model.feature_importances_))]
                performance.feature_importance = dict(zip(feature_names, model.feature_importances_.tolist()))
            elif hasattr(model, 'coef_'):
                coef = model.coef_
                if len(coef.shape) > 1:
                    coef = coef[0]  # للتصنيف متعدد الفئات
                feature_names = [f"feature_{i}" for i in range(len(coef))]
                performance.feature_importance = dict(zip(feature_names, coef.tolist()))
            
            return performance
            
        except Exception as e:
            self.logger.error(f"خطأ في تقييم النموذج: {e}")
            return ModelPerformance(model_name=model_name, model_type=model_type)

    async def predict(
        self,
        data: Union[pd.DataFrame, dict, list],
        task: PredictionTask = PredictionTask.CUSTOM,
        model_name: Optional[str] = None,
        target_column: Optional[str] = None,
        return_probabilities: bool = False
    ) -> PredictionResult:
        """التنبؤ"""
        start_time = datetime.now()
        
        try:
            # تحديد النموذج
            if model_name and target_column:
                model_key = f"{task.value}_{model_name}_{target_column}"
            else:
                # البحث عن أفضل نموذج متاح
                model_key = await self._find_best_model(task)
            
            if model_key not in self.models:
                raise ValueError(f"النموذج غير موجود: {model_key}")
            
            model = self.models[model_key]
            
            # تحضير البيانات
            if isinstance(data, dict):
                data = pd.DataFrame([data])
            elif isinstance(data, list):
                data = pd.DataFrame(data)
            
            # معالجة البيانات
            X_processed = await self._preprocess_prediction_data(data, model_key)
            
            # التنبؤ
            prediction = model.predict(X_processed)
            
            # حساب الثقة
            confidence = await self._calculate_confidence(model, X_processed, model_key)
            
            # الاحتماليات
            probabilities = {}
            if return_probabilities and hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_processed)
                if hasattr(model, 'classes_'):
                    probabilities = dict(zip(model.classes_, proba[0]))
            
            # مساهمة الميزات
            feature_contributions = await self._calculate_feature_contributions(
                model, X_processed, model_key
            )
            
            # إنشاء النتيجة
            result = PredictionResult(
                prediction=prediction[0] if len(prediction) == 1 else prediction.tolist(),
                confidence=confidence,
                probability_distribution=probabilities,
                feature_contributions=feature_contributions,
                model_used=model_key,
                prediction_time=(datetime.now() - start_time).total_seconds(),
                metadata={
                    'task': task.value,
                    'data_shape': data.shape,
                    'timestamp': datetime.now().isoformat()
                }
            )
            
            # حفظ في التاريخ
            self.prediction_history.append({
                'timestamp': datetime.now().isoformat(),
                'task': task.value,
                'model_used': model_key,
                'prediction': result.prediction,
                'confidence': result.confidence
            })
            
            # حفظ التاريخ بشكل دوري
            if len(self.prediction_history) % 100 == 0:
                await self._save_prediction_history()
            
            return result
            
        except Exception as e:
            self.logger.error(f"خطأ في التنبؤ: {e}")
            raise

    async def _find_best_model(self, task: PredictionTask) -> str:
        """البحث عن أفضل نموذج للمهمة"""
        try:
            # البحث عن النماذج المدربة للمهمة
            task_models = [key for key in self.models.keys() if key.startswith(task.value)]
            
            if not task_models:
                raise ValueError(f"لا توجد نماذج مدربة للمهمة: {task.value}")
            
            # اختيار أفضل نموذج بناءً على الأداء
            best_model = None
            best_score = -float('inf')
            
            for model_key in task_models:
                if model_key in self.model_performance:
                    performance = self.model_performance[model_key]
                    # استخدام R2 للانحدار أو الدقة للتصنيف
                    score = performance.r2_score if performance.model_type == ModelType.REGRESSION else performance.accuracy
                    
                    if score > best_score:
                        best_score = score
                        best_model = model_key
            
            if best_model is None:
                best_model = task_models[0]  # استخدام أول نموذج متاح
            
            return best_model
            
        except Exception as e:
            self.logger.error(f"خطأ في البحث عن أفضل نموذج: {e}")
            raise

    async def _preprocess_prediction_data(self, data: pd.DataFrame, model_key: str) -> np.ndarray:
        """معالجة بيانات التنبؤ"""
        try:
            # الحصول على أعمدة الميزات المطلوبة
            if model_key in self.feature_columns:
                required_columns = self.feature_columns[model_key]
                
                # التأكد من وجود جميع الأعمدة المطلوبة
                missing_columns = set(required_columns) - set(data.columns)
                if missing_columns:
                    # ملء الأعمدة المفقودة بالقيم الافتراضية
                    for col in missing_columns:
                        data[col] = 0
                
                # ترتيب الأعمدة
                data = data[required_columns]
            
            # تطبيق نفس معالجة التدريب
            data_processed = data.copy()
            
            # التعامل مع القيم المفقودة
            data_processed = data_processed.fillna(data_processed.mean() if data_processed.select_dtypes(include=[np.number]).shape[1] > 0 else data_processed.mode().iloc[0])
            
            # ترميز المتغيرات الفئوية
            categorical_columns = data_processed.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                if col in self.encoders:
                    try:
                        data_processed[col] = self.encoders[col].transform(data_processed[col].astype(str))
                    except ValueError:
                        # قيمة جديدة غير موجودة في التدريب
                        data_processed[col] = 0
            
            # تطبيع البيانات
            model_type = self.model_performance[model_key].model_type
            if model_type in [ModelType.REGRESSION, ModelType.CLASSIFICATION, ModelType.CLUSTERING]:
                scaler_key = f"{model_type.value}_scaler"
                if scaler_key in self.scalers:
                    data_processed = self.scalers[scaler_key].transform(data_processed)
            
            return data_processed
            
        except Exception as e:
            self.logger.error(f"خطأ في معالجة بيانات التنبؤ: {e}")
            raise

    async def _calculate_confidence(self, model, X_processed, model_key: str) -> float:
        """حساب ثقة التنبؤ"""
        try:
            # للنماذج التي تدعم الاحتماليات
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_processed)
                confidence = np.max(proba[0])
            elif hasattr(model, 'decision_function'):
                # للنماذج مثل SVM
                decision = model.decision_function(X_processed)
                confidence = abs(decision[0]) / 10  # تطبيع تقريبي
                confidence = min(confidence, 1.0)
            else:
                # استخدام أداء النموذج كبديل
                if model_key in self.model_performance:
                    performance = self.model_performance[model_key]
                    if performance.model_type == ModelType.REGRESSION:
                        confidence = performance.r2_score
                    else:
                        confidence = performance.accuracy
                else:
                    confidence = 0.5  # قيمة افتراضية
            
            return max(0.0, min(1.0, confidence))
            
        except Exception as e:
            self.logger.error(f"خطأ في حساب الثقة: {e}")
            return 0.5

    async def _calculate_feature_contributions(self, model, X_processed, model_key: str) -> Dict[str, float]:
        """حساب مساهمة الميزات"""
        try:
            contributions = {}
            
            # للنماذج التي تدعم أهمية الميزات
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feature_names = self.feature_columns.get(model_key, [f"feature_{i}" for i in range(len(importances))])
                
                for i, importance in enumerate(importances):
                    if i < len(feature_names):
                        contributions[feature_names[i]] = float(importance)
            
            elif hasattr(model, 'coef_'):
                coef = model.coef_
                if len(coef.shape) > 1:
                    coef = coef[0]
                
                feature_names = self.feature_columns.get(model_key, [f"feature_{i}" for i in range(len(coef))])
                
                for i, coeff in enumerate(coef):
                    if i < len(feature_names):
                        contributions[feature_names[i]] = float(abs(coeff))
            
            return contributions
            
        except Exception as e:
            self.logger.error(f"خطأ في حساب مساهمة الميزات: {e}")
            return {}

    async def auto_ml_pipeline(
        self,
        data: pd.DataFrame,
        target_column: str,
        task: PredictionTask = PredictionTask.CUSTOM,
        max_models: int = 5,
        time_limit: int = 300
    ) -> List[ModelPerformance]:
        """خط أنابيب التعلم الآلي التلقائي"""
        start_time = datetime.now()
        results = []
        
        try:
            # تحديد نوع المهمة
            if task == PredictionTask.CUSTOM:
                # تحديد تلقائي بناءً على البيانات
                if data[target_column].dtype in ['object', 'category']:
                    model_type = ModelType.CLASSIFICATION
                elif len(data[target_column].unique()) < 10:
                    model_type = ModelType.CLASSIFICATION
                else:
                    model_type = ModelType.REGRESSION
            else:
                # تحديد بناءً على نوع المهمة
                if task in [PredictionTask.USER_BEHAVIOR, PredictionTask.ERROR_PREDICTION]:
                    model_type = ModelType.CLASSIFICATION
                else:
                    model_type = ModelType.REGRESSION
            
            # الحصول على النماذج المناسبة
            available_models = self.model_registry.get(task, [])
            if not available_models:
                available_models = self.model_registry[PredictionTask.CUSTOM]
            
            # تحديد عدد النماذج المراد تجربتها
            models_to_try = available_models[:max_models]
            
            self.logger.info(f"بدء خط الأنابيب التلقائي: {len(models_to_try)} نماذج")
            
            # تدريب النماذج
            for model_name in models_to_try:
                # فحص الوقت المحدد
                if (datetime.now() - start_time).total_seconds() > time_limit:
                    self.logger.warning("تم الوصول للحد الأقصى للوقت")
                    break
                
                try:
                    self.logger.info(f"تدريب النموذج: {model_name}")
                    
                    performance = await self.train_model(
                        data=data,
                        target_column=target_column,
                        model_name=model_name,
                        model_type=model_type,
                        task=task,
                        cross_validation=True,
                        hyperparameter_tuning=True
                    )
                    
                    results.append(performance)
                    
                except Exception as model_error:
                    self.logger.error(f"فشل تدريب النموذج {model_name}: {model_error}")
                    continue
            
            # ترتيب النتائج
            if model_type == ModelType.REGRESSION:
                results.sort(key=lambda x: x.r2_score, reverse=True)
            else:
                results.sort(key=lambda x: x.accuracy, reverse=True)
            
            self.logger.info(f"انتهى خط الأنابيب التلقائي: {len(results)} نماذج مدربة")
            return results
            
        except Exception as e:
            self.logger.error(f"خطأ في خط الأنابيب التلقائي: {e}")
            return results

    async def get_model_recommendations(self, task: PredictionTask = None) -> List[str]:
        """الحصول على توصيات النماذج"""
        try:
            recommendations = []
            
            if not self.model_performance:
                recommendations.append("لا توجد نماذج مدربة حالياً - ابدأ بتدريب النماذج أولاً للحصول على توصيات")
                return recommendations
            
            # تحليل الأداء
            avg_r2 = np.mean([p.r2_score for p in self.model_performance.values() if p.model_type == ModelType.REGRESSION])
            avg_accuracy = np.mean([p.accuracy for p in self.model_performance.values() if p.model_type == ModelType.CLASSIFICATION])
            avg_training_time = np.mean([p.training_time for p in self.model_performance.values()])
            
            # توصيات الأداء
            if avg_r2 < 0.7:
                recommendations.append("أداء نماذج الانحدار منخفض - فكر في إضافة ميزات جديدة أو معالجة البيانات")
            
            if avg_accuracy < 0.8:
                recommendations.append("أداء نماذج التصنيف منخفض - جرب نماذج أكثر تعقيداً أو تحسين البيانات")
            
            if avg_training_time > 60:
                recommendations.append("أوقات التدريب طويلة - فكر في تقليل حجم البيانات أو استخدام نماذج أبسط")
            
            # توصيات النماذج
            best_models = sorted(
                self.model_performance.items(),
                key=lambda x: x[1].r2_score if x[1].model_type == ModelType.REGRESSION else x[1].accuracy,
                reverse=True
            )[:3]
            
            if best_models:
                recommendations.append(f"أفضل النماذج الحالية: {', '.join([model[0] for model in best_models])}")
            
            # توصيات التحسين
            if ADVANCED_MODELS_AVAILABLE:
                recommendations.append("جرب النماذج المتقدمة: XGBoost, LightGBM, CatBoost للحصول على أداء أفضل")
            
            if PYTORCH_AVAILABLE:
                recommendations.append("فكر في استخدام الشبكات العصبية للمشاكل المعقدة")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"خطأ في الحصول على التوصيات: {e}")
            return ["خطأ في تحليل النماذج"]

    async def _save_model(self, model_key: str, model):
        """حفظ النموذج"""
        try:
            model_path = self.models_dir / f"{model_key}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        except Exception as e:
            self.logger.error(f"خطأ في حفظ النموذج: {e}")

    async def _load_model(self, model_key: str):
        """تحميل النموذج"""
        try:
            model_path = self.models_dir / f"{model_key}.pkl"
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            self.logger.error(f"خطأ في تحميل النموذج: {e}")
        return None

    async def _save_performance_data(self):
        """حفظ بيانات الأداء"""
        try:
            performance_data = {}
            for key, performance in self.model_performance.items():
                performance_data[key] = asdict(performance)
            
            with open(self.performance_file, 'w', encoding='utf-8') as f:
                json.dump(performance_data, f, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"خطأ في حفظ بيانات الأداء: {e}")

    async def _load_performance_data(self):
        """تحميل بيانات الأداء"""
        try:
            if self.performance_file.exists():
                with open(self.performance_file, 'r', encoding='utf-8') as f:
                    performance_data = json.load(f)
                
                for key, data in performance_data.items():
                    # تحويل التاريخ
                    if 'last_updated' in data and isinstance(data['last_updated'], str):
                        data['last_updated'] = datetime.fromisoformat(data['last_updated'])
                    
                    self.model_performance[key] = ModelPerformance(**data)
        except Exception as e:
            self.logger.error(f"خطأ في تحميل بيانات الأداء: {e}")

    async def _save_prediction_history(self):
        """حفظ تاريخ التنبؤات"""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.prediction_history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"خطأ في حفظ تاريخ التنبؤات: {e}")

    def get_model_status(self) -> Dict[str, Any]:
        """الحصول على حالة النماذج"""
        return {
            "total_models": len(self.models),
            "model_types": list(set([p.model_type.value for p in self.model_performance.values()])),
            "best_performing_models": [
                {
                    "name": name,
                    "type": perf.model_type.value,
                    "score": perf.r2_score if perf.model_type == ModelType.REGRESSION else perf.accuracy
                }
                for name, perf in sorted(
                    self.model_performance.items(),
                    key=lambda x: x[1].r2_score if x[1].model_type == ModelType.REGRESSION else x[1].accuracy,
                    reverse=True
                )[:5]
            ],
            "total_predictions": len(self.prediction_history),
            "advanced_features_available": {
                "xgboost_lightgbm_catboost": ADVANCED_MODELS_AVAILABLE,
                "pytorch": PYTORCH_AVAILABLE,
                "timeseries": TIMESERIES_AVAILABLE
            }
        }

# إنشاء مثيل عام
ml_predictor = AdvancedMLPredictor()

async def get_ml_predictor() -> AdvancedMLPredictor:
    """الحصول على محرك التنبؤ"""
    return ml_predictor

if __name__ == "__main__":
    async def test_ml_predictor():
        """اختبار محرك التنبؤ"""
        print("🤖 اختبار محرك التنبؤ بالتعلم الآلي المتقدم")
        print("=" * 60)
        
        predictor = await get_ml_predictor()
        
        # إنشاء بيانات تجريبية
        np.random.seed(42)
        data = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randint(0, 5, 100),
            'target': np.random.randn(100)
        })
        
        print("📊 بيانات الاختبار:")
        print(data.head())
        
        # تدريب نموذج
        print("\n🏋️ تدريب نموذج...")
        performance = await predictor.train_model(
            data=data,
            target_column='target',
            model_name='random_forest',
            model_type=ModelType.REGRESSION,
            task=PredictionTask.CUSTOM
        )
        
        print(f"✅ أداء النموذج: R² = {performance.r2_score:.3f}")
        
        # تجربة التنبؤ
        print("\n🎯 اختبار التنبؤ...")
        test_data = {'feature1': 0.5, 'feature2': -0.3, 'feature3': 2}
        
        result = await predictor.predict(
            data=test_data,
            task=PredictionTask.CUSTOM,
            model_name='random_forest',
            target_column='target'
        )
        
        print(f"📈 التنبؤ: {result.prediction:.3f}")
        print(f"🎯 مستوى الثقة: {result.confidence:.3f}")
        
        # خط الأنابيب التلقائي
        print("\n⚡ خط الأنابيب التلقائي...")
        results = await predictor.auto_ml_pipeline(
            data=data,
            target_column='target',
            max_models=3,
            time_limit=60
        )
        
        print(f"🏆 أفضل النماذج:")
        for i, result in enumerate(results[:3]):
            print(f"  {i+1}. {result.model_name}: R² = {result.r2_score:.3f}")
        
        # التوصيات
        print("\n💡 التوصيات:")
        recommendations = await predictor.get_model_recommendations()
        for rec in recommendations:
            print(f"  • {rec}")
        
        # حالة النماذج
        print("\n📊 حالة النماذج:")
        status = predictor.get_model_status()
        print(f"  • إجمالي النماذج: {status['total_models']}")
        print(f"  • أنواع النماذج: {', '.join(status['model_types'])}")
        print(f"  • إجمالي التنبؤات: {status['total_predictions']}")
        
        print("\n✨ انتهى الاختبار بنجاح!")

    # تشغيل الاختبار
    asyncio.run(test_ml_predictor())
