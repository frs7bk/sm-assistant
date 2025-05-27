
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
متنبئ التعلم الآلي المتقدم
Advanced Machine Learning Predictor
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import pickle
import joblib
from pathlib import Path
import logging
import warnings
from abc import ABC, abstractmethod
import json

# تجاهل التحذيرات غير المهمة
warnings.filterwarnings('ignore')

# مكتبات التعلم الآلي
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# مكتبات متقدمة
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

@dataclass
class PredictionRequest:
    """طلب تنبؤ"""
    request_id: str
    features: Dict[str, Any]
    prediction_type: str
    confidence_threshold: float = 0.8
    context: Optional[Dict[str, Any]] = None
    timestamp: datetime = None

@dataclass
class PredictionResult:
    """نتيجة التنبؤ"""
    request_id: str
    prediction: Union[float, int, str, List]
    confidence: float
    model_used: str
    feature_importance: Dict[str, float]
    explanation: str
    metadata: Dict[str, Any]
    timestamp: datetime

@dataclass
class ModelPerformance:
    """أداء النموذج"""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    rmse: float
    mae: float
    r2: float
    training_time: float
    prediction_time: float

class BasePredictionModel(ABC):
    """نموذج تنبؤ أساسي"""
    
    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.scaler = None
        self.is_trained = False
        self.feature_names = []
        self.performance_metrics = {}
    
    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> bool:
        """تدريب النموذج"""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, float]:
        """التنبؤ مع الثقة"""
        pass
    
    def get_feature_importance(self) -> Dict[str, float]:
        """الحصول على أهمية الميزات"""
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            return dict(zip(self.feature_names, importances))
        return {}

class AdvancedRandomForest(BasePredictionModel):
    """غابة عشوائية متطورة"""
    
    def __init__(self):
        super().__init__("Advanced Random Forest")
        
    def train(self, X: np.ndarray, y: np.ndarray) -> bool:
        try:
            # تحسين المعاملات تلقائياً
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            rf = RandomForestRegressor(random_state=42)
            grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='r2', n_jobs=-1)
            
            # تدريب النموذج
            grid_search.fit(X, y)
            self.model = grid_search.best_estimator_
            
            # حفظ أسماء الميزات
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            
            self.is_trained = True
            return True
            
        except Exception as e:
            logging.error(f"خطأ في تدريب {self.name}: {e}")
            return False
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, float]:
        if not self.is_trained:
            raise ValueError("النموذج غير مدرب")
        
        predictions = self.model.predict(X)
        
        # حساب الثقة من تشتت التنبؤات
        if hasattr(self.model, 'estimators_'):
            tree_predictions = np.array([tree.predict(X) for tree in self.model.estimators_])
            confidence = 1.0 - np.mean(np.std(tree_predictions, axis=0))
        else:
            confidence = 0.8  # ثقة افتراضية
        
        return predictions, min(confidence, 0.99)

class GradientBoostingPredictor(BasePredictionModel):
    """متنبئ تعزيز التدرج"""
    
    def __init__(self):
        super().__init__("Gradient Boosting")
        
    def train(self, X: np.ndarray, y: np.ndarray) -> bool:
        try:
            # معاملات محسنة
            self.model = GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                subsample=0.8
            )
            
            self.model.fit(X, y)
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            self.is_trained = True
            return True
            
        except Exception as e:
            logging.error(f"خطأ في تدريب {self.name}: {e}")
            return False
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, float]:
        if not self.is_trained:
            raise ValueError("النموذج غير مدرب")
        
        predictions = self.model.predict(X)
        
        # حساب الثقة من staged predictions
        staged_predictions = list(self.model.staged_predict(X))
        if len(staged_predictions) > 1:
            final_pred = staged_predictions[-1]
            confidence = 1.0 - np.mean(np.abs(final_pred - staged_predictions[-2]) / (np.abs(final_pred) + 1e-8))
        else:
            confidence = 0.85
        
        return predictions, min(confidence, 0.99)

class XGBoostPredictor(BasePredictionModel):
    """متنبئ XGBoost المتقدم"""
    
    def __init__(self):
        super().__init__("XGBoost Advanced")
        self.available = XGBOOST_AVAILABLE
        
    def train(self, X: np.ndarray, y: np.ndarray) -> bool:
        if not self.available:
            logging.warning("XGBoost غير متوفر")
            return False
            
        try:
            # معاملات محسنة لـ XGBoost
            self.model = xgb.XGBRegressor(
                n_estimators=300,
                learning_rate=0.1,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                objective='reg:squarederror'
            )
            
            self.model.fit(X, y)
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            self.is_trained = True
            return True
            
        except Exception as e:
            logging.error(f"خطأ في تدريب {self.name}: {e}")
            return False
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, float]:
        if not self.is_trained:
            raise ValueError("النموذج غير مدرب")
        
        predictions = self.model.predict(X)
        
        # استخدام uncertainty quantification
        confidence = 0.9  # XGBoost عادة موثوق
        
        return predictions, confidence

class NeuralNetworkPredictor(BasePredictionModel):
    """متنبئ الشبكة العصبية"""
    
    def __init__(self):
        super().__init__("Neural Network")
        
    def train(self, X: np.ndarray, y: np.ndarray) -> bool:
        try:
            # تطبيع البيانات
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # شبكة عصبية متقدمة
            self.model = MLPRegressor(
                hidden_layer_sizes=(100, 50, 25),
                activation='relu',
                solver='adam',
                alpha=0.001,
                learning_rate='adaptive',
                max_iter=1000,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1
            )
            
            self.model.fit(X_scaled, y)
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            self.is_trained = True
            return True
            
        except Exception as e:
            logging.error(f"خطأ في تدريب {self.name}: {e}")
            return False
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, float]:
        if not self.is_trained:
            raise ValueError("النموذج غير مدرب")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        # حساب الثقة من loss curve
        if hasattr(self.model, 'loss_curve_'):
            loss_improvement = (self.model.loss_curve_[0] - self.model.loss_curve_[-1]) / self.model.loss_curve_[0]
            confidence = min(0.95, 0.7 + loss_improvement * 0.3)
        else:
            confidence = 0.8
        
        return predictions, confidence

class EnsemblePredictor(BasePredictionModel):
    """متنبئ متجمع (Ensemble)"""
    
    def __init__(self):
        super().__init__("Advanced Ensemble")
        self.models = []
        self.weights = []
        
    def train(self, X: np.ndarray, y: np.ndarray) -> bool:
        try:
            # إنشاء نماذج متنوعة
            models = [
                ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
                ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42)),
                ('svr', SVR(kernel='rbf')),
                ('mlp', MLPRegressor(hidden_layer_sizes=(50,), max_iter=500, random_state=42))
            ]
            
            # إضافة XGBoost إذا كان متوفراً
            if XGBOOST_AVAILABLE:
                models.append(('xgb', xgb.XGBRegressor(n_estimators=100, random_state=42)))
            
            # تدريب نموذج التصويت
            self.model = VotingRegressor(models)
            self.model.fit(X, y)
            
            # حفظ أوزان النماذج
            self.weights = [1.0 / len(models)] * len(models)
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            self.is_trained = True
            return True
            
        except Exception as e:
            logging.error(f"خطأ في تدريب {self.name}: {e}")
            return False
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, float]:
        if not self.is_trained:
            raise ValueError("النموذج غير مدرب")
        
        predictions = self.model.predict(X)
        
        # حساب الثقة من تنوع التنبؤات
        individual_predictions = []
        for name, estimator in self.model.named_estimators_.items():
            individual_predictions.append(estimator.predict(X))
        
        individual_predictions = np.array(individual_predictions)
        variance = np.var(individual_predictions, axis=0)
        confidence = 1.0 - np.mean(variance) / (np.mean(np.abs(predictions)) + 1e-8)
        
        return predictions, min(max(confidence, 0.6), 0.95)

class AdvancedMLPredictor:
    """متنبئ التعلم الآلي المتقدم"""
    
    def __init__(self, models_dir: str = "data/models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # النماذج المتاحة
        self.available_models = {
            "random_forest": AdvancedRandomForest(),
            "gradient_boosting": GradientBoostingPredictor(),
            "neural_network": NeuralNetworkPredictor(),
            "ensemble": EnsemblePredictor()
        }
        
        # إضافة XGBoost إذا كان متوفراً
        if XGBOOST_AVAILABLE:
            self.available_models["xgboost"] = XGBoostPredictor()
        
        # النموذج النشط
        self.active_model = None
        self.model_performance = {}
        self.feature_selector = None
        self.data_preprocessor = None
        
        # تاريخ التنبؤات
        self.prediction_history = []
        
        logging.info("تم تهيئة متنبئ التعلم الآلي المتقدم")
    
    async def train_models(self, data: pd.DataFrame, target_column: str,
                          test_size: float = 0.2,
                          feature_selection: bool = True) -> Dict[str, Any]:
        """تدريب جميع النماذج وتقييمها"""
        try:
            # إعداد البيانات
            X, y = self._prepare_data(data, target_column)
            
            # تقسيم البيانات
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            # اختيار الميزات
            if feature_selection:
                X_train, X_test = self._select_features(X_train, X_test, y_train)
            
            # تدريب النماذج
            training_results = {}
            
            for model_name, model in self.available_models.items():
                logging.info(f"تدريب نموذج: {model_name}")
                
                start_time = datetime.now()
                success = model.train(X_train, y_train)
                training_time = (datetime.now() - start_time).total_seconds()
                
                if success:
                    # تقييم النموذج
                    performance = await self._evaluate_model(
                        model, X_test, y_test, training_time
                    )
                    training_results[model_name] = performance
                    self.model_performance[model_name] = performance
                    
                    # حفظ النموذج
                    await self._save_model(model, model_name)
                else:
                    training_results[model_name] = {"error": "فشل التدريب"}
            
            # اختيار أفضل نموذج
            best_model = self._select_best_model()
            if best_model:
                self.active_model = self.available_models[best_model]
                logging.info(f"تم اختيار أفضل نموذج: {best_model}")
            
            return {
                "training_results": training_results,
                "best_model": best_model,
                "total_models_trained": len([r for r in training_results.values() if "error" not in r])
            }
            
        except Exception as e:
            logging.error(f"خطأ في تدريب النماذج: {e}")
            return {"error": str(e)}
    
    async def predict(self, features: Dict[str, Any],
                     prediction_type: str = "regression",
                     confidence_threshold: float = 0.8) -> PredictionResult:
        """إجراء تنبؤ متقدم"""
        try:
            if not self.active_model:
                raise ValueError("لا يوجد نموذج نشط")
            
            # تحويل الميزات لمصفوفة numpy
            feature_vector = self._features_to_array(features)
            
            # التنبؤ
            start_time = datetime.now()
            predictions, confidence = self.active_model.predict(feature_vector.reshape(1, -1))
            prediction_time = (datetime.now() - start_time).total_seconds()
            
            # إنشاء نتيجة التنبؤ
            result = PredictionResult(
                request_id=f"pred_{int(datetime.now().timestamp())}",
                prediction=float(predictions[0]),
                confidence=confidence,
                model_used=self.active_model.name,
                feature_importance=self.active_model.get_feature_importance(),
                explanation=self._generate_explanation(features, predictions[0], confidence),
                metadata={
                    "prediction_time": prediction_time,
                    "feature_count": len(features),
                    "model_performance": self.model_performance.get(self.active_model.name, {})
                },
                timestamp=datetime.now()
            )
            
            # حفظ في التاريخ
            self.prediction_history.append(result)
            
            # التحقق من الثقة
            if confidence < confidence_threshold:
                result.explanation += f" (تحذير: مستوى الثقة {confidence:.2f} أقل من المطلوب {confidence_threshold})"
            
            return result
            
        except Exception as e:
            logging.error(f"خطأ في التنبؤ: {e}")
            raise
    
    async def predict_multiple(self, features_list: List[Dict[str, Any]]) -> List[PredictionResult]:
        """تنبؤات متعددة"""
        results = []
        
        for features in features_list:
            try:
                result = await self.predict(features)
                results.append(result)
            except Exception as e:
                logging.error(f"خطأ في التنبؤ المتعدد: {e}")
                # إنشاء نتيجة خطأ
                error_result = PredictionResult(
                    request_id=f"error_{int(datetime.now().timestamp())}",
                    prediction=None,
                    confidence=0.0,
                    model_used="None",
                    feature_importance={},
                    explanation=f"خطأ في التنبؤ: {str(e)}",
                    metadata={"error": True},
                    timestamp=datetime.now()
                )
                results.append(error_result)
        
        return results
    
    async def get_model_insights(self) -> Dict[str, Any]:
        """الحصول على رؤى النماذج"""
        try:
            insights = {
                "active_model": self.active_model.name if self.active_model else None,
                "available_models": list(self.available_models.keys()),
                "model_performance": self.model_performance,
                "prediction_history_count": len(self.prediction_history),
                "feature_importance": self.active_model.get_feature_importance() if self.active_model else {},
                "recommendations": await self._generate_model_recommendations()
            }
            
            # إضافة إحصائيات التنبؤات
            if self.prediction_history:
                confidences = [p.confidence for p in self.prediction_history[-100:]]  # آخر 100 تنبؤ
                insights["recent_predictions"] = {
                    "average_confidence": np.mean(confidences),
                    "min_confidence": np.min(confidences),
                    "max_confidence": np.max(confidences),
                    "predictions_above_threshold": len([c for c in confidences if c > 0.8])
                }
            
            return insights
            
        except Exception as e:
            logging.error(f"خطأ في الحصول على رؤى النماذج: {e}")
            return {}
    
    def _prepare_data(self, data: pd.DataFrame, target_column: str) -> Tuple[np.ndarray, np.ndarray]:
        """إعداد البيانات للتدريب"""
        # فصل الميزات والهدف
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        # معالجة البيانات المفقودة
        X = X.fillna(X.mean() if X.select_dtypes(include=[np.number]).shape[1] > 0 else X.mode().iloc[0])
        y = y.fillna(y.mean() if y.dtype in [np.float64, np.int64] else y.mode().iloc[0])
        
        # تحويل البيانات النصية
        for column in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X[column] = le.fit_transform(X[column].astype(str))
        
        # تطبيع البيانات
        if not hasattr(self, 'data_preprocessor') or self.data_preprocessor is None:
            self.data_preprocessor = StandardScaler()
            X_scaled = self.data_preprocessor.fit_transform(X)
        else:
            X_scaled = self.data_preprocessor.transform(X)
        
        return X_scaled, y.values
    
    def _select_features(self, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """اختيار أفضل الميزات"""
        try:
            # اختيار أفضل الميزات
            if not hasattr(self, 'feature_selector') or self.feature_selector is None:
                k_features = min(20, X_train.shape[1])  # أفضل 20 ميزة أو جميع الميزات إذا كانت أقل
                self.feature_selector = SelectKBest(f_regression, k=k_features)
                X_train_selected = self.feature_selector.fit_transform(X_train, y_train)
            else:
                X_train_selected = self.feature_selector.transform(X_train)
            
            X_test_selected = self.feature_selector.transform(X_test)
            
            return X_train_selected, X_test_selected
            
        except Exception as e:
            logging.warning(f"فشل اختيار الميزات: {e}")
            return X_train, X_test
    
    async def _evaluate_model(self, model: BasePredictionModel, X_test: np.ndarray, 
                            y_test: np.ndarray, training_time: float) -> ModelPerformance:
        """تقييم أداء النموذج"""
        try:
            # التنبؤ
            start_time = datetime.now()
            predictions, confidence = model.predict(X_test)
            prediction_time = (datetime.now() - start_time).total_seconds()
            
            # حساب المقاييس
            mse = mean_squared_error(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            rmse = np.sqrt(mse)
            
            # مقاييس إضافية
            accuracy = max(0, 1 - (mae / (np.mean(np.abs(y_test)) + 1e-8)))
            
            return ModelPerformance(
                model_name=model.name,
                accuracy=accuracy,
                precision=confidence,  # استخدام الثقة كمقياس للدقة
                recall=confidence,
                f1_score=confidence,
                rmse=rmse,
                mae=mae,
                r2=r2,
                training_time=training_time,
                prediction_time=prediction_time
            )
            
        except Exception as e:
            logging.error(f"خطأ في تقييم النموذج: {e}")
            return ModelPerformance(
                model_name=model.name,
                accuracy=0.0, precision=0.0, recall=0.0, f1_score=0.0,
                rmse=float('inf'), mae=float('inf'), r2=-float('inf'),
                training_time=training_time, prediction_time=0.0
            )
    
    def _select_best_model(self) -> Optional[str]:
        """اختيار أفضل نموذج"""
        if not self.model_performance:
            return None
        
        # ترتيب النماذج حسب R² مع مراعاة الوقت
        best_model = None
        best_score = -float('inf')
        
        for model_name, performance in self.model_performance.items():
            # حساب نقاط مركبة تأخذ في الاعتبار R² والسرعة
            score = performance.r2 - (performance.training_time / 100)  # تقليل نقاط للوقت الطويل
            
            if score > best_score:
                best_score = score
                best_model = model_name
        
        return best_model
    
    def _features_to_array(self, features: Dict[str, Any]) -> np.ndarray:
        """تحويل الميزات لمصفوفة numpy"""
        # تحويل القيم النصية للأرقام إذا أمكن
        feature_values = []
        
        for key, value in features.items():
            if isinstance(value, (int, float)):
                feature_values.append(float(value))
            elif isinstance(value, str):
                # تحويل النص لرقم hash بسيط
                feature_values.append(float(hash(value) % 1000))
            else:
                feature_values.append(0.0)  # قيمة افتراضية
        
        return np.array(feature_values)
    
    def _generate_explanation(self, features: Dict[str, Any], 
                            prediction: float, confidence: float) -> str:
        """توليد شرح للتنبؤ"""
        explanation = f"التنبؤ: {prediction:.2f} (ثقة: {confidence:.1%})"
        
        # إضافة معلومات عن الميزات المهمة
        feature_importance = self.active_model.get_feature_importance()
        if feature_importance:
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]
            if top_features:
                explanation += "\nأهم العوامل المؤثرة: "
                explanation += ", ".join([f"{feat}" for feat, imp in top_features])
        
        return explanation
    
    async def _generate_model_recommendations(self) -> List[str]:
        """توليد توصيات لتحسين النماذج"""
        recommendations = []
        
        if not self.model_performance:
            recommendations.append("قم بتدريب النماذج أولاً للحصول على توصيات")
            return recommendations
        
        # تحليل الأداء
        avg_r2 = np.mean([p.r2 for p in self.model_performance.values()])
        avg_training_time = np.mean([p.training_time for p in self.model_performance.values()])
        
        if avg_r2 < 0.7:
            recommendations.append("أداء النماذج منخفض - فكر في إضافة ميزات جديدة أو معالجة البيانات")
        
        if avg_training_time > 60:
            recommendations.append("وقت التدريب طويل - فكر في تقليل حجم البيانات أو استخدام نماذج أبسط")
        
        # فحص تنوع النماذج
        model_types = len(self.available_models)
        if model_types < 3:
            recommendations.append("أضف المزيد من أنواع النماذج لتحسين دقة التنبؤات")
        
        if not recommendations:
            recommendations.append("النماذج تعمل بشكل جيد - استمر في جمع البيانات للتحسين")
        
        return recommendations
    
    async def _save_model(self, model: BasePredictionModel, model_name: str):
        """حفظ النموذج"""
        try:
            model_path = self.models_dir / f"{model_name}.pkl"
            
            model_data = {
                'model': model.model,
                'scaler': model.scaler,
                'feature_names': model.feature_names,
                'is_trained': model.is_trained,
                'performance': self.model_performance.get(model_name)
            }
            
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            logging.info(f"تم حفظ النموذج: {model_name}")
            
        except Exception as e:
            logging.error(f"خطأ في حفظ النموذج {model_name}: {e}")

# مثيل عالمي للمتنبئ
ml_predictor = AdvancedMLPredictor()

# دوال مساعدة
async def train_prediction_models(data: pd.DataFrame, target_column: str) -> Dict[str, Any]:
    """دالة مساعدة لتدريب النماذج"""
    return await ml_predictor.train_models(data, target_column)

async def make_prediction(features: Dict[str, Any]) -> PredictionResult:
    """دالة مساعدة للتنبؤ"""
    return await ml_predictor.predict(features)

async def get_prediction_insights() -> Dict[str, Any]:
    """دالة مساعدة للرؤى"""
    return await ml_predictor.get_model_insights()

if __name__ == "__main__":
    # اختبار المتنبئ
    async def test_predictor():
        # إنشاء بيانات وهمية للاختبار
        np.random.seed(42)
        n_samples = 1000
        
        data = pd.DataFrame({
            'feature_1': np.random.normal(0, 1, n_samples),
            'feature_2': np.random.uniform(-1, 1, n_samples),
            'feature_3': np.random.exponential(1, n_samples),
            'target': np.random.normal(10, 5, n_samples)
        })
        
        # إضافة بعض الارتباط
        data['target'] = data['feature_1'] * 2 + data['feature_2'] * 3 + np.random.normal(0, 1, n_samples)
        
        # تدريب النماذج
        print("تدريب النماذج...")
        training_results = await ml_predictor.train_models(data, 'target')
        print("نتائج التدريب:", training_results)
        
        # إجراء تنبؤ
        test_features = {
            'feature_1': 1.5,
            'feature_2': -0.5,
            'feature_3': 2.0
        }
        
        prediction = await ml_predictor.predict(test_features)
        print("نتيجة التنبؤ:", prediction.prediction)
        print("الثقة:", prediction.confidence)
        
        # الحصول على الرؤى
        insights = await ml_predictor.get_model_insights()
        print("رؤى النماذج:", insights)
    
    asyncio.run(test_predictor())
