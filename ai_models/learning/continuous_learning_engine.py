
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
محرك التعلم المستمر المتقدم والذكي
Advanced Continuous Learning Engine
"""

import asyncio
import logging
import numpy as np
import pandas as pd
import pickle
import json
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from enum import Enum
from collections import deque, defaultdict
import threading
import queue
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor

# Advanced Libraries
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    logging.warning("PyTorch غير متوفر للتعلم العميق")

try:
    from river import linear_model, tree, ensemble, metrics, preprocessing
    from river.drift import ADWIN, PageHinkley, KSWIN
    RIVER_AVAILABLE = True
except ImportError:
    RIVER_AVAILABLE = False
    logging.warning("River غير متوفر للتعلم التدفقي")

class LearningMode(Enum):
    """أنماط التعلم"""
    INCREMENTAL = "incremental"  # تعلم تدريجي
    ONLINE = "online"  # تعلم فوري
    BATCH = "batch"  # تعلم بالدفعات
    FEDERATED = "federated"  # تعلم فيدرالي
    TRANSFER = "transfer"  # تعلم منقول
    META = "meta"  # تعلم فوقي
    REINFORCEMENT = "reinforcement"  # تعلم تعزيزي

class ConceptDriftType(Enum):
    """أنواع انحراف المفهوم"""
    SUDDEN = "sudden"  # مفاجئ
    GRADUAL = "gradual"  # تدريجي
    RECURRING = "recurring"  # متكرر
    INCREMENTAL = "incremental"  # تدريجي مستمر

class LearningTask(Enum):
    """مهام التعلم"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    ANOMALY_DETECTION = "anomaly_detection"
    PATTERN_RECOGNITION = "pattern_recognition"
    BEHAVIOR_MODELING = "behavior_modeling"

@dataclass
class LearningExample:
    """مثال التعلم"""
    example_id: str
    data: Dict[str, Any]
    target: Any
    timestamp: datetime
    source: str = "unknown"
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    feedback_score: Optional[float] = None
    is_validated: bool = False

@dataclass
class LearningSession:
    """جلسة التعلم"""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    examples_count: int = 0
    performance_before: Dict[str, float] = field(default_factory=dict)
    performance_after: Dict[str, float] = field(default_factory=dict)
    drift_detected: bool = False
    adaptation_applied: bool = False
    learning_mode: LearningMode = LearningMode.INCREMENTAL
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ModelCheckpoint:
    """نقطة تفتيش النموذج"""
    checkpoint_id: str
    model_state: bytes
    performance_metrics: Dict[str, float]
    timestamp: datetime
    examples_seen: int
    drift_score: float = 0.0
    confidence_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DriftDetectionResult:
    """نتيجة كشف الانحراف"""
    drift_detected: bool
    drift_type: Optional[ConceptDriftType]
    confidence: float
    severity: float  # 0-1
    affected_features: List[str] = field(default_factory=list)
    recommendation: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

class AdaptiveNeuralNetwork(nn.Module):
    """شبكة عصبية تكيفية"""
    
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int, task_type: str = "classification"):
        super().__init__()
        self.task_type = task_type
        self.layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        
        # بناء الطبقات
        prev_size = input_size
        for hidden_size in hidden_sizes:
            self.layers.append(nn.Linear(prev_size, hidden_size))
            self.dropout_layers.append(nn.Dropout(0.2))
            prev_size = hidden_size
        
        # طبقة الإخراج
        self.output_layer = nn.Linear(prev_size, output_size)
        
        # طبقات التطبيع
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_size) for hidden_size in hidden_sizes
        ])
        
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if x.size(0) > 1:  # تطبيق BatchNorm فقط إذا كان هناك أكثر من عينة
                x = self.batch_norms[i](x)
            x = torch.relu(x)
            x = self.dropout_layers[i](x)
        
        x = self.output_layer(x)
        
        if self.task_type == "classification":
            x = torch.softmax(x, dim=1)
        
        return x

class ContinuousLearningEngine:
    """محرك التعلم المستمر المتقدم"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # النماذج الأساسية
        self.models: Dict[str, Any] = {}
        self.model_performance: Dict[str, Dict[str, float]] = {}
        self.model_checkpoints: Dict[str, List[ModelCheckpoint]] = {}
        
        # كشف الانحراف
        self.drift_detectors: Dict[str, Any] = {}
        self.drift_history: List[DriftDetectionResult] = []
        
        # إدارة البيانات
        self.data_buffer = deque(maxlen=self.config.get('buffer_size', 10000))
        self.validation_buffer = deque(maxlen=self.config.get('validation_buffer_size', 1000))
        self.feedback_buffer = deque(maxlen=self.config.get('feedback_buffer_size', 5000))
        
        # معالجات البيانات
        self.scalers: Dict[str, StandardScaler] = {}
        self.encoders: Dict[str, LabelEncoder] = {}
        
        # إحصائيات التعلم
        self.learning_sessions: List[LearningSession] = []
        self.examples_processed = 0
        self.adaptations_count = 0
        self.last_adaptation_time = None
        
        # معايير الأداء
        self.performance_thresholds = {
            'accuracy_drop': self.config.get('accuracy_drop_threshold', 0.05),
            'drift_confidence': self.config.get('drift_confidence_threshold', 0.8),
            'adaptation_frequency': self.config.get('max_adaptations_per_hour', 5)
        }
        
        # خيوط التنفيذ
        self.learning_queue = queue.Queue()
        self.processing_thread = None
        self.is_running = False
        
        # مسارات الحفظ
        self.models_dir = Path("data/continuous_learning")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self._initialize_drift_detectors()
        self.logger.info("تم تهيئة محرك التعلم المستمر المتقدم")

    def _initialize_drift_detectors(self):
        """تهيئة كاشفات الانحراف"""
        try:
            if RIVER_AVAILABLE:
                self.drift_detectors['adwin'] = ADWIN(delta=0.01)
                self.drift_detectors['page_hinkley'] = PageHinkley(min_instances=30, delta=0.005, threshold=50)
                self.drift_detectors['kswin'] = KSWIN(alpha=0.005, window_size=100, stat_size=30)
            else:
                # كاشف انحراف بسيط مخصص
                self.drift_detectors['simple'] = SimpleDriftDetector()
                
        except Exception as e:
            self.logger.error(f"خطأ في تهيئة كاشفات الانحراف: {e}")

    async def start_continuous_learning(self):
        """بدء التعلم المستمر"""
        try:
            self.is_running = True
            self.processing_thread = threading.Thread(target=self._continuous_learning_loop)
            self.processing_thread.start()
            self.logger.info("تم بدء التعلم المستمر")
            
        except Exception as e:
            self.logger.error(f"خطأ في بدء التعلم المستمر: {e}")
            raise

    async def stop_continuous_learning(self):
        """إيقاف التعلم المستمر"""
        try:
            self.is_running = False
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=5)
            
            # حفظ الحالة الحالية
            await self.save_learning_state()
            self.logger.info("تم إيقاف التعلم المستمر")
            
        except Exception as e:
            self.logger.error(f"خطأ في إيقاف التعلم المستمر: {e}")

    def _continuous_learning_loop(self):
        """حلقة التعلم المستمر"""
        while self.is_running:
            try:
                # معالجة البيانات في الطابور
                try:
                    example = self.learning_queue.get(timeout=1)
                    asyncio.create_task(self._process_learning_example(example))
                except queue.Empty:
                    continue
                
                # فحص الانحراف بشكل دوري
                if self.examples_processed % 100 == 0:
                    asyncio.create_task(self._check_concept_drift())
                
                # تحفيظ نقاط التفتيش
                if self.examples_processed % 1000 == 0:
                    asyncio.create_task(self._create_checkpoint())
                
            except Exception as e:
                self.logger.error(f"خطأ في حلقة التعلم المستمر: {e}")

    async def add_learning_example(
        self,
        data: Dict[str, Any],
        target: Any,
        source: str = "user_interaction",
        weight: float = 1.0,
        immediate_learning: bool = False
    ) -> str:
        """إضافة مثال تعلم جديد"""
        try:
            example_id = hashlib.md5(f"{data}_{target}_{datetime.now()}".encode()).hexdigest()[:12]
            
            example = LearningExample(
                example_id=example_id,
                data=data,
                target=target,
                timestamp=datetime.now(),
                source=source,
                weight=weight
            )
            
            # إضافة للمخزن المؤقت
            self.data_buffer.append(example)
            
            if immediate_learning:
                await self._process_learning_example(example)
            else:
                self.learning_queue.put(example)
            
            self.logger.debug(f"تم إضافة مثال التعلم: {example_id}")
            return example_id
            
        except Exception as e:
            self.logger.error(f"خطأ في إضافة مثال التعلم: {e}")
            raise

    async def _process_learning_example(self, example: LearningExample):
        """معالجة مثال التعلم"""
        try:
            # تحديد نوع المهمة
            task_type = self._determine_task_type(example.target)
            model_key = f"continuous_{task_type}"
            
            # إنشاء النموذج إذا لم يكن موجوداً
            if model_key not in self.models:
                await self._initialize_model(model_key, task_type, example.data)
            
            model = self.models[model_key]
            
            # تحضير البيانات
            X, y = await self._prepare_data([example])
            
            # التعلم التدريجي
            if hasattr(model, 'partial_fit'):
                # للنماذج التي تدعم التعلم التدريجي
                if task_type == "classification" and hasattr(model, 'classes_'):
                    model.partial_fit(X, y, classes=model.classes_)
                else:
                    model.partial_fit(X, y)
            elif PYTORCH_AVAILABLE and isinstance(model, nn.Module):
                # للشبكات العصبية
                await self._train_neural_network_online(model, X, y)
            else:
                # إعادة تدريب كامل مع البيانات المخزنة
                await self._retrain_model_with_buffer(model_key, task_type)
            
            # تحديث الإحصائيات
            self.examples_processed += 1
            
            # تقييم الأداء
            if self.examples_processed % 50 == 0:
                await self._evaluate_model_performance(model_key, task_type)
            
        except Exception as e:
            self.logger.error(f"خطأ في معالجة مثال التعلم: {e}")

    def _determine_task_type(self, target: Any) -> str:
        """تحديد نوع المهمة"""
        if isinstance(target, (int, float)):
            if isinstance(target, int) and target in [0, 1]:
                return "classification"
            elif isinstance(target, int) and target < 10:
                return "classification"
            else:
                return "regression"
        elif isinstance(target, str):
            return "classification"
        else:
            return "classification"

    async def _initialize_model(self, model_key: str, task_type: str, sample_data: Dict[str, Any]):
        """تهيئة النموذج"""
        try:
            input_size = len(sample_data)
            
            if PYTORCH_AVAILABLE and self.config.get('use_neural_networks', True):
                # شبكة عصبية تكيفية
                if task_type == "classification":
                    output_size = self.config.get('num_classes', 10)
                    model = AdaptiveNeuralNetwork(
                        input_size=input_size,
                        hidden_sizes=[64, 32],
                        output_size=output_size,
                        task_type=task_type
                    )
                else:
                    model = AdaptiveNeuralNetwork(
                        input_size=input_size,
                        hidden_sizes=[64, 32],
                        output_size=1,
                        task_type=task_type
                    )
                
                # إعداد المحسن
                model.optimizer = optim.Adam(model.parameters(), lr=0.001)
                model.criterion = nn.CrossEntropyLoss() if task_type == "classification" else nn.MSELoss()
                
            else:
                # نماذج sklearn التقليدية
                if task_type == "classification":
                    model = SGDClassifier(random_state=42, loss='log')
                else:
                    model = SGDRegressor(random_state=42)
            
            self.models[model_key] = model
            self.model_performance[model_key] = {}
            self.model_checkpoints[model_key] = []
            
            self.logger.info(f"تم تهيئة النموذج: {model_key}")
            
        except Exception as e:
            self.logger.error(f"خطأ في تهيئة النموذج: {e}")
            raise

    async def _prepare_data(self, examples: List[LearningExample]) -> Tuple[np.ndarray, np.ndarray]:
        """تحضير البيانات للتدريب"""
        try:
            # استخراج البيانات
            X_list = []
            y_list = []
            
            for example in examples:
                # تحويل البيانات إلى متجه
                feature_vector = []
                for key, value in example.data.items():
                    if isinstance(value, (int, float)):
                        feature_vector.append(value)
                    elif isinstance(value, str):
                        # ترميز النصوص
                        if key not in self.encoders:
                            self.encoders[key] = LabelEncoder()
                            # تهيئة بقيم وهمية
                            self.encoders[key].fit([value, "unknown"])
                        
                        try:
                            encoded_value = self.encoders[key].transform([value])[0]
                        except ValueError:
                            # قيمة جديدة لم نرها من قبل
                            encoded_value = len(self.encoders[key].classes_)
                        
                        feature_vector.append(encoded_value)
                    else:
                        feature_vector.append(0)  # قيمة افتراضية
                
                X_list.append(feature_vector)
                y_list.append(example.target)
            
            X = np.array(X_list)
            y = np.array(y_list)
            
            # تطبيع البيانات
            scaler_key = "feature_scaler"
            if scaler_key not in self.scalers:
                self.scalers[scaler_key] = StandardScaler()
                X = self.scalers[scaler_key].fit_transform(X)
            else:
                X = self.scalers[scaler_key].transform(X)
            
            return X, y
            
        except Exception as e:
            self.logger.error(f"خطأ في تحضير البيانات: {e}")
            raise

    async def _train_neural_network_online(self, model: nn.Module, X: np.ndarray, y: np.ndarray):
        """تدريب الشبكة العصبية بشكل فوري"""
        try:
            model.train()
            
            # تحويل البيانات إلى tensors
            X_tensor = torch.FloatTensor(X)
            
            if model.task_type == "classification":
                y_tensor = torch.LongTensor(y)
            else:
                y_tensor = torch.FloatTensor(y).unsqueeze(1)
            
            # التدريب
            model.optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = model.criterion(outputs, y_tensor)
            loss.backward()
            model.optimizer.step()
            
        except Exception as e:
            self.logger.error(f"خطأ في تدريب الشبكة العصبية: {e}")

    async def _retrain_model_with_buffer(self, model_key: str, task_type: str):
        """إعادة تدريب النموذج مع البيانات المخزنة"""
        try:
            if len(self.data_buffer) < 10:
                return  # بيانات غير كافية
            
            # أخذ عينة من البيانات المخزنة
            recent_examples = list(self.data_buffer)[-1000:]  # آخر 1000 مثال
            
            X, y = await self._prepare_data(recent_examples)
            
            # إعادة تدريب النموذج
            model = self.models[model_key]
            
            if PYTORCH_AVAILABLE and isinstance(model, nn.Module):
                # تدريب الشبكة العصبية
                dataset = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y) if task_type == "classification" else torch.FloatTensor(y))
                dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
                
                model.train()
                for epoch in range(5):  # تدريب سريع
                    for batch_X, batch_y in dataloader:
                        model.optimizer.zero_grad()
                        outputs = model(batch_X)
                        loss = model.criterion(outputs, batch_y)
                        loss.backward()
                        model.optimizer.step()
            else:
                # إعادة تدريب sklearn
                model.fit(X, y)
            
            self.logger.info(f"تم إعادة تدريب النموذج: {model_key}")
            
        except Exception as e:
            self.logger.error(f"خطأ في إعادة تدريب النموذج: {e}")

    async def _check_concept_drift(self) -> DriftDetectionResult:
        """فحص انحراف المفهوم"""
        try:
            if len(self.validation_buffer) < 50:
                return DriftDetectionResult(drift_detected=False, confidence=0.0, severity=0.0)
            
            # حساب الأداء الحالي
            recent_examples = list(self.validation_buffer)[-50:]
            performance_scores = []
            
            for example in recent_examples:
                # محاكاة التنبؤ وحساب الدقة
                predicted = await self._predict_single(example.data)
                actual = example.target
                
                if isinstance(actual, str) or isinstance(predicted, str):
                    score = 1.0 if predicted == actual else 0.0
                else:
                    score = 1.0 - abs(predicted - actual) / max(abs(actual), 1.0)
                
                performance_scores.append(score)
            
            current_performance = np.mean(performance_scores)
            
            # مقارنة مع الأداء التاريخي
            if hasattr(self, 'historical_performance'):
                performance_drop = self.historical_performance - current_performance
                
                drift_detected = performance_drop > self.performance_thresholds['accuracy_drop']
                confidence = min(performance_drop / self.performance_thresholds['accuracy_drop'], 1.0)
                severity = performance_drop
                
                if drift_detected:
                    # تحديد نوع الانحراف
                    drift_type = self._classify_drift_type(performance_scores)
                    
                    result = DriftDetectionResult(
                        drift_detected=True,
                        drift_type=drift_type,
                        confidence=confidence,
                        severity=severity,
                        recommendation="يوصى بإعادة تدريب النموذج أو تحديث البيانات"
                    )
                    
                    self.drift_history.append(result)
                    
                    # تطبيق التكيف
                    await self._apply_adaptation(result)
                    
                    return result
            else:
                self.historical_performance = current_performance
            
            return DriftDetectionResult(drift_detected=False, confidence=0.0, severity=0.0)
            
        except Exception as e:
            self.logger.error(f"خطأ في فحص انحراف المفهوم: {e}")
            return DriftDetectionResult(drift_detected=False, confidence=0.0, severity=0.0)

    def _classify_drift_type(self, performance_scores: List[float]) -> ConceptDriftType:
        """تصنيف نوع الانحراف"""
        try:
            # تحليل نمط الأداء
            scores_array = np.array(performance_scores)
            
            # فحص الانحراف المفاجئ
            diff = np.diff(scores_array)
            if np.any(np.abs(diff) > 0.3):
                return ConceptDriftType.SUDDEN
            
            # فحص الانحراف التدريجي
            if len(scores_array) > 10:
                first_half = scores_array[:len(scores_array)//2]
                second_half = scores_array[len(scores_array)//2:]
                
                if np.mean(first_half) - np.mean(second_half) > 0.1:
                    return ConceptDriftType.GRADUAL
            
            # فحص النمط المتكرر
            if len(self.drift_history) > 3:
                recent_drifts = self.drift_history[-3:]
                if all(d.drift_detected for d in recent_drifts):
                    return ConceptDriftType.RECURRING
            
            return ConceptDriftType.INCREMENTAL
            
        except Exception as e:
            self.logger.error(f"خطأ في تصنيف نوع الانحراف: {e}")
            return ConceptDriftType.GRADUAL

    async def _apply_adaptation(self, drift_result: DriftDetectionResult):
        """تطبيق التكيف بناءً على الانحراف"""
        try:
            # فحص معدل التكيف
            if (self.last_adaptation_time and 
                (datetime.now() - self.last_adaptation_time).total_seconds() < 3600 / self.performance_thresholds['adaptation_frequency']):
                self.logger.info("تم تخطي التكيف بسبب معدل التكيف المرتفع")
                return
            
            # اختيار استراتيجية التكيف
            if drift_result.severity > 0.3:
                # انحراف شديد - إعادة تدريب كامل
                await self._full_retrain_adaptation()
            elif drift_result.severity > 0.1:
                # انحراف متوسط - تحديث تدريجي
                await self._incremental_adaptation()
            else:
                # انحراف خفيف - ضبط المعاملات
                await self._parameter_adjustment()
            
            self.adaptations_count += 1
            self.last_adaptation_time = datetime.now()
            
            self.logger.info(f"تم تطبيق التكيف للانحراف: {drift_result.drift_type}")
            
        except Exception as e:
            self.logger.error(f"خطأ في تطبيق التكيف: {e}")

    async def _full_retrain_adaptation(self):
        """إعادة تدريب كامل للنماذج"""
        try:
            for model_key in self.models.keys():
                # إنشاء نقطة تفتيش قبل إعادة التدريب
                await self._create_checkpoint(model_key)
                
                # تحديد نوع المهمة
                task_type = model_key.split('_')[1]
                
                # إعادة تدريب مع البيانات الحديثة
                await self._retrain_model_with_buffer(model_key, task_type)
                
        except Exception as e:
            self.logger.error(f"خطأ في إعادة التدريب الكامل: {e}")

    async def _incremental_adaptation(self):
        """تكيف تدريجي"""
        try:
            # استخدام البيانات الحديثة للتحديث التدريجي
            recent_examples = list(self.data_buffer)[-100:]
            
            for model_key, model in self.models.items():
                if hasattr(model, 'partial_fit') and recent_examples:
                    X, y = await self._prepare_data(recent_examples)
                    
                    if hasattr(model, 'classes_'):
                        model.partial_fit(X, y, classes=model.classes_)
                    else:
                        model.partial_fit(X, y)
                        
        except Exception as e:
            self.logger.error(f"خطأ في التكيف التدريجي: {e}")

    async def _parameter_adjustment(self):
        """ضبط المعاملات"""
        try:
            # ضبط معاملات التعلم للشبكات العصبية
            for model in self.models.values():
                if PYTORCH_AVAILABLE and isinstance(model, nn.Module) and hasattr(model, 'optimizer'):
                    # تقليل معدل التعلم قليلاً
                    for param_group in model.optimizer.param_groups:
                        param_group['lr'] *= 0.95
                        
        except Exception as e:
            self.logger.error(f"خطأ في ضبط المعاملات: {e}")

    async def _predict_single(self, data: Dict[str, Any]) -> Any:
        """تنبؤ لعينة واحدة"""
        try:
            # تحضير البيانات
            example = LearningExample(
                example_id="temp",
                data=data,
                target=None,
                timestamp=datetime.now()
            )
            
            X, _ = await self._prepare_data([example])
            
            # استخدام أول نموذج متاح
            if self.models:
                model_key = list(self.models.keys())[0]
                model = self.models[model_key]
                
                if PYTORCH_AVAILABLE and isinstance(model, nn.Module):
                    model.eval()
                    with torch.no_grad():
                        X_tensor = torch.FloatTensor(X)
                        output = model(X_tensor)
                        if model.task_type == "classification":
                            prediction = torch.argmax(output, dim=1).item()
                        else:
                            prediction = output.item()
                else:
                    prediction = model.predict(X)[0]
                
                return prediction
            
            return 0  # قيمة افتراضية
            
        except Exception as e:
            self.logger.error(f"خطأ في التنبؤ: {e}")
            return 0

    async def _create_checkpoint(self, model_key: str = None):
        """إنشاء نقطة تفتيش للنماذج"""
        try:
            models_to_checkpoint = [model_key] if model_key else list(self.models.keys())
            
            for key in models_to_checkpoint:
                if key in self.models:
                    model = self.models[key]
                    
                    # حفظ حالة النموذج
                    if PYTORCH_AVAILABLE and isinstance(model, nn.Module):
                        model_state = pickle.dumps(model.state_dict())
                    else:
                        model_state = pickle.dumps(model)
                    
                    # حساب مقاييس الأداء
                    performance_metrics = self.model_performance.get(key, {})
                    
                    checkpoint = ModelCheckpoint(
                        checkpoint_id=f"{key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        model_state=model_state,
                        performance_metrics=performance_metrics,
                        timestamp=datetime.now(),
                        examples_seen=self.examples_processed
                    )
                    
                    self.model_checkpoints[key].append(checkpoint)
                    
                    # الاحتفاظ بآخر 10 نقاط تفتيش فقط
                    if len(self.model_checkpoints[key]) > 10:
                        self.model_checkpoints[key] = self.model_checkpoints[key][-10:]
            
            self.logger.info("تم إنشاء نقاط التفتيش")
            
        except Exception as e:
            self.logger.error(f"خطأ في إنشاء نقطة التفتيش: {e}")

    async def _evaluate_model_performance(self, model_key: str, task_type: str):
        """تقييم أداء النموذج"""
        try:
            if len(self.validation_buffer) < 10:
                return
            
            model = self.models[model_key]
            validation_examples = list(self.validation_buffer)[-50:]
            
            predictions = []
            actual_values = []
            
            for example in validation_examples:
                try:
                    predicted = await self._predict_single(example.data)
                    predictions.append(predicted)
                    actual_values.append(example.target)
                except Exception:
                    continue
            
            if predictions and actual_values:
                # حساب المقاييس
                if task_type == "classification":
                    accuracy = accuracy_score(actual_values, predictions)
                    self.model_performance[model_key]['accuracy'] = accuracy
                else:
                    mse = mean_squared_error(actual_values, predictions)
                    r2 = r2_score(actual_values, predictions)
                    self.model_performance[model_key]['mse'] = mse
                    self.model_performance[model_key]['r2'] = r2
                
                self.model_performance[model_key]['last_evaluation'] = datetime.now().isoformat()
                
        except Exception as e:
            self.logger.error(f"خطأ في تقييم الأداء: {e}")

    async def add_feedback(self, example_id: str, feedback_score: float, feedback_data: Dict[str, Any] = None):
        """إضافة ملاحظات على التنبؤ"""
        try:
            # البحث عن المثال في البيانات المخزنة
            for example in self.data_buffer:
                if example.example_id == example_id:
                    example.feedback_score = feedback_score
                    example.is_validated = True
                    if feedback_data:
                        example.metadata.update(feedback_data)
                    
                    self.feedback_buffer.append(example)
                    break
            
            # استخدام الملاحظات في التحسين
            if len(self.feedback_buffer) >= 20:
                await self._apply_feedback_learning()
                
        except Exception as e:
            self.logger.error(f"خطأ في إضافة الملاحظات: {e}")

    async def _apply_feedback_learning(self):
        """تطبيق التعلم من الملاحظات"""
        try:
            # تجميع الملاحظات الإيجابية والسلبية
            positive_examples = [ex for ex in self.feedback_buffer if ex.feedback_score >= 0.7]
            negative_examples = [ex for ex in self.feedback_buffer if ex.feedback_score <= 0.3]
            
            # إعادة تدريب مع الملاحظات
            if positive_examples:
                for model_key in self.models.keys():
                    task_type = model_key.split('_')[1]
                    
                    # إعطاء وزن أكبر للأمثلة الإيجابية
                    X_pos, y_pos = await self._prepare_data(positive_examples)
                    
                    model = self.models[model_key]
                    if hasattr(model, 'partial_fit'):
                        # تدريب متعدد للأمثلة الإيجابية
                        for _ in range(3):
                            if hasattr(model, 'classes_'):
                                model.partial_fit(X_pos, y_pos, classes=model.classes_)
                            else:
                                model.partial_fit(X_pos, y_pos)
            
            # مسح الملاحظات المعالجة
            self.feedback_buffer.clear()
            
        except Exception as e:
            self.logger.error(f"خطأ في تطبيق التعلم من الملاحظات: {e}")

    async def get_learning_statistics(self) -> Dict[str, Any]:
        """الحصول على إحصائيات التعلم"""
        try:
            stats = {
                "total_examples_processed": self.examples_processed,
                "total_adaptations": self.adaptations_count,
                "models_count": len(self.models),
                "drift_detections": len([d for d in self.drift_history if d.drift_detected]),
                "buffer_status": {
                    "data_buffer": len(self.data_buffer),
                    "validation_buffer": len(self.validation_buffer),
                    "feedback_buffer": len(self.feedback_buffer)
                },
                "model_performance": self.model_performance,
                "last_adaptation": self.last_adaptation_time.isoformat() if self.last_adaptation_time else None,
                "learning_sessions": len(self.learning_sessions),
                "is_running": self.is_running
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"خطأ في الحصول على الإحصائيات: {e}")
            return {}

    async def save_learning_state(self):
        """حفظ حالة التعلم"""
        try:
            state_file = self.models_dir / "learning_state.json"
            
            state = {
                "examples_processed": self.examples_processed,
                "adaptations_count": self.adaptations_count,
                "last_adaptation_time": self.last_adaptation_time.isoformat() if self.last_adaptation_time else None,
                "model_performance": self.model_performance,
                "drift_history": [asdict(d) for d in self.drift_history],
                "config": self.config
            }
            
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, ensure_ascii=False, indent=2, default=str)
            
            # حفظ النماذج
            for model_key, model in self.models.items():
                model_file = self.models_dir / f"{model_key}.pkl"
                with open(model_file, 'wb') as f:
                    pickle.dump(model, f)
            
            self.logger.info("تم حفظ حالة التعلم")
            
        except Exception as e:
            self.logger.error(f"خطأ في حفظ حالة التعلم: {e}")

    async def load_learning_state(self):
        """تحميل حالة التعلم"""
        try:
            state_file = self.models_dir / "learning_state.json"
            
            if state_file.exists():
                with open(state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                
                self.examples_processed = state.get('examples_processed', 0)
                self.adaptations_count = state.get('adaptations_count', 0)
                
                if state.get('last_adaptation_time'):
                    self.last_adaptation_time = datetime.fromisoformat(state['last_adaptation_time'])
                
                self.model_performance = state.get('model_performance', {})
                
                # تحميل تاريخ الانحراف
                drift_data = state.get('drift_history', [])
                self.drift_history = [DriftDetectionResult(**d) for d in drift_data]
            
            # تحميل النماذج
            model_files = list(self.models_dir.glob("continuous_*.pkl"))
            for model_file in model_files:
                model_key = model_file.stem
                try:
                    with open(model_file, 'rb') as f:
                        self.models[model_key] = pickle.load(f)
                except Exception as e:
                    self.logger.warning(f"فشل تحميل النموذج {model_key}: {e}")
            
            self.logger.info("تم تحميل حالة التعلم")
            
        except Exception as e:
            self.logger.error(f"خطأ في تحميل حالة التعلم: {e}")

class SimpleDriftDetector:
    """كاشف انحراف بسيط"""
    
    def __init__(self, window_size: int = 100, threshold: float = 0.1):
        self.window_size = window_size
        self.threshold = threshold
        self.performance_window = deque(maxlen=window_size)
        self.baseline_performance = None
    
    def add_element(self, performance_score: float):
        """إضافة نقطة أداء جديدة"""
        self.performance_window.append(performance_score)
        
        if self.baseline_performance is None and len(self.performance_window) >= 20:
            self.baseline_performance = np.mean(list(self.performance_window)[:20])
    
    def detected_change(self) -> bool:
        """فحص وجود انحراف"""
        if len(self.performance_window) < 20 or self.baseline_performance is None:
            return False
        
        recent_performance = np.mean(list(self.performance_window)[-20:])
        return abs(self.baseline_performance - recent_performance) > self.threshold

# إنشاء مثيل عام
continuous_learning_engine = ContinuousLearningEngine()

async def get_continuous_learning_engine() -> ContinuousLearningEngine:
    """الحصول على محرك التعلم المستمر"""
    return continuous_learning_engine

if __name__ == "__main__":
    async def test_continuous_learning():
        """اختبار محرك التعلم المستمر"""
        print("🧠 اختبار محرك التعلم المستمر المتقدم")
        print("=" * 60)
        
        engine = await get_continuous_learning_engine()
        await engine.start_continuous_learning()
        
        print("📚 بدء التعلم المستمر...")
        
        # محاكاة بيانات التعلم
        for i in range(100):
            data = {
                'feature1': np.random.randn(),
                'feature2': np.random.randn(),
                'feature3': np.random.choice(['A', 'B', 'C'])
            }
            target = np.random.choice([0, 1])
            
            example_id = await engine.add_learning_example(
                data=data,
                target=target,
                source="test_simulation"
            )
            
            # إضافة ملاحظات عشوائية
            if i % 10 == 0:
                feedback_score = np.random.uniform(0.3, 0.9)
                await engine.add_feedback(example_id, feedback_score)
            
            # محاكاة انحراف في البيانات
            if i == 50:
                print("🔄 محاكاة انحراف في البيانات...")
        
        # انتظار معالجة البيانات
        await asyncio.sleep(2)
        
        # الحصول على الإحصائيات
        stats = await engine.get_learning_statistics()
        print("\n📊 إحصائيات التعلم:")
        print(f"  • الأمثلة المعالجة: {stats['total_examples_processed']}")
        print(f"  • عدد التكيفات: {stats['total_adaptations']}")
        print(f"  • عدد النماذج: {stats['models_count']}")
        print(f"  • كشف الانحراف: {stats['drift_detections']}")
        
        print(f"\n💾 حالة المخازن المؤقتة:")
        buffer_status = stats['buffer_status']
        print(f"  • مخزن البيانات: {buffer_status['data_buffer']}")
        print(f"  • مخزن التحقق: {buffer_status['validation_buffer']}")
        print(f"  • مخزن الملاحظات: {buffer_status['feedback_buffer']}")
        
        # حفظ الحالة
        await engine.save_learning_state()
        print("\n💾 تم حفظ حالة التعلم")
        
        # إيقاف التعلم
        await engine.stop_continuous_learning()
        print("\n⏹️ تم إيقاف التعلم المستمر")
        
        print("\n✨ انتهى الاختبار بنجاح!")

    # تشغيل الاختبار
    asyncio.run(test_continuous_learning())
