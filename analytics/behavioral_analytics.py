
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
محرك تحليل السلوك المتقدم
Advanced Behavioral Analytics Engine
"""

import asyncio
import logging
import numpy as np
import pandas as pd
import sqlite3
import json
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict, field
from collections import defaultdict, deque
import threading
import queue
import hashlib
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
try:
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.ensemble import IsolationForest
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Deep Learning
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

# Visualization
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.io as pio
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

@dataclass
class BehaviorEvent:
    """حدث سلوكي"""
    event_id: str
    user_id: str
    event_type: str
    event_data: Dict[str, Any]
    timestamp: datetime
    session_id: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BehaviorPattern:
    """نمط سلوكي"""
    pattern_id: str
    user_id: str
    pattern_type: str
    pattern_description: str
    frequency: float
    confidence: float
    start_time: datetime
    end_time: Optional[datetime] = None
    triggers: List[str] = field(default_factory=list)
    outcomes: List[str] = field(default_factory=list)
    context_factors: Dict[str, Any] = field(default_factory=dict)
    statistical_significance: float = 0.0

@dataclass
class UserBehaviorProfile:
    """ملف تعريف السلوك الشخصي"""
    user_id: str
    active_patterns: List[BehaviorPattern]
    behavior_trends: Dict[str, float]
    preferences: Dict[str, Any]
    interaction_style: str
    activity_periods: List[Tuple[int, int]]  # (start_hour, end_hour)
    engagement_score: float
    consistency_score: float
    adaptability_score: float
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class BehaviorInsight:
    """رؤية سلوكية"""
    insight_id: str
    user_id: str
    insight_type: str
    description: str
    confidence: float
    impact_score: float
    recommendations: List[str]
    supporting_data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)

class BehaviorEmbeddingNetwork(nn.Module):
    """شبكة تمثيل السلوك العصبية"""
    
    def __init__(self, input_dim: int = 128, embedding_dim: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        
        # طبقات التشفير
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, embedding_dim),
            nn.Tanh()
        )
        
        # طبقات فك التشفير
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )
        
        # طبقة التصنيف
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),  # عدد أنواع السلوك
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        # تشفير
        embedded = self.encoder(x)
        
        # فك التشفير
        reconstructed = self.decoder(embedded)
        
        # تصنيف
        classification = self.classifier(embedded)
        
        return embedded, reconstructed, classification

class AdvancedBehavioralAnalytics:
    """محرك تحليل السلوك المتقدم"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # التكوين الأساسي
        self.is_initialized = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # قاعدة البيانات
        self.db_path = Path("data/behavioral_analytics.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # النماذج والمعالجات
        self.embedding_network = None
        self.clustering_model = None
        self.anomaly_detector = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        
        # مخازن البيانات
        self.behavior_events = deque(maxlen=50000)
        self.user_profiles: Dict[str, UserBehaviorProfile] = {}
        self.active_patterns: Dict[str, List[BehaviorPattern]] = defaultdict(list)
        self.behavior_insights: Dict[str, List[BehaviorInsight]] = defaultdict(list)
        
        # إحصائيات النظام
        self.analytics_stats = {
            "total_events_processed": 0,
            "patterns_discovered": 0,
            "insights_generated": 0,
            "users_analyzed": 0,
            "model_accuracy": 0.0,
            "processing_time_avg": 0.0
        }
        
        # معالجة الخلفية
        self.processing_queue = queue.Queue()
        self.background_workers = []
        self.is_processing = False
        
        # إعدادات التحليل
        self.analysis_config = {
            "min_pattern_frequency": 3,
            "pattern_confidence_threshold": 0.7,
            "anomaly_threshold": 0.1,
            "clustering_eps": 0.5,
            "clustering_min_samples": 5,
            "insight_confidence_threshold": 0.8
        }
    
    async def initialize(self):
        """تهيئة محرك التحليل السلوكي"""
        self.logger.info("🧠 تهيئة محرك تحليل السلوك المتقدم...")
        
        try:
            # إنشاء قاعدة البيانات
            await self._initialize_database()
            
            # تحميل النماذج
            await self._load_models()
            
            # تشغيل العمال الخلفيين
            self._start_background_workers()
            
            # تحميل البيانات التاريخية
            await self._load_historical_data()
            
            self.is_initialized = True
            self.logger.info("✅ تم تهيئة محرك تحليل السلوك بنجاح")
            
        except Exception as e:
            self.logger.error(f"❌ فشل تهيئة محرك تحليل السلوك: {e}")
            # تهيئة أساسية
            self.is_initialized = True
    
    async def _initialize_database(self):
        """تهيئة قاعدة البيانات"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # جدول الأحداث السلوكية
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS behavior_events (
                    event_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    event_data TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    session_id TEXT,
                    context TEXT,
                    metadata TEXT,
                    INDEX(user_id),
                    INDEX(timestamp),
                    INDEX(event_type)
                )
            ''')
            
            # جدول الأنماط السلوكية
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS behavior_patterns (
                    pattern_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    pattern_type TEXT NOT NULL,
                    pattern_description TEXT,
                    frequency REAL,
                    confidence REAL,
                    start_time DATETIME,
                    end_time DATETIME,
                    triggers TEXT,
                    outcomes TEXT,
                    context_factors TEXT,
                    statistical_significance REAL,
                    INDEX(user_id),
                    INDEX(pattern_type)
                )
            ''')
            
            # جدول ملفات المستخدمين
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_behavior_profiles (
                    user_id TEXT PRIMARY KEY,
                    active_patterns TEXT,
                    behavior_trends TEXT,
                    preferences TEXT,
                    interaction_style TEXT,
                    activity_periods TEXT,
                    engagement_score REAL,
                    consistency_score REAL,
                    adaptability_score REAL,
                    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # جدول الرؤى السلوكية
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS behavior_insights (
                    insight_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    insight_type TEXT NOT NULL,
                    description TEXT,
                    confidence REAL,
                    impact_score REAL,
                    recommendations TEXT,
                    supporting_data TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    INDEX(user_id),
                    INDEX(insight_type)
                )
            ''')
            
            conn.commit()
            conn.close()
            self.logger.info("✅ تم تهيئة قاعدة البيانات السلوكية")
            
        except Exception as e:
            self.logger.error(f"❌ فشل تهيئة قاعدة البيانات: {e}")
    
    async def _load_models(self):
        """تحميل النماذج"""
        try:
            if PYTORCH_AVAILABLE:
                # تحميل شبكة التمثيل السلوكي
                self.embedding_network = BehaviorEmbeddingNetwork()
                self.embedding_network.to(self.device)
                
                # تحميل نموذج محفوظ إن وجد
                model_path = Path("data/models/behavior_embedding.pth")
                if model_path.exists():
                    self.embedding_network.load_state_dict(
                        torch.load(model_path, map_location=self.device)
                    )
                    self.logger.info("✅ تم تحميل نموذج التمثيل السلوكي المحفوظ")
                else:
                    self.logger.info("🧠 تم إنشاء نموذج تمثيل سلوكي جديد")
            
            if SKLEARN_AVAILABLE:
                # نموذج التجميع
                self.clustering_model = DBSCAN(
                    eps=self.analysis_config["clustering_eps"],
                    min_samples=self.analysis_config["clustering_min_samples"]
                )
                
                # نموذج كشف الشاذات
                self.anomaly_detector = IsolationForest(
                    contamination=self.analysis_config["anomaly_threshold"],
                    random_state=42
                )
                
                self.logger.info("✅ تم تهيئة نماذج التعلم الآلي")
            
        except Exception as e:
            self.logger.warning(f"⚠️ خطأ في تحميل النماذج: {e}")
    
    def _start_background_workers(self):
        """تشغيل العمال الخلفيين"""
        self.is_processing = True
        num_workers = 3
        
        for i in range(num_workers):
            worker = threading.Thread(
                target=self._background_processor,
                name=f"BehaviorWorker-{i}",
                daemon=True
            )
            worker.start()
            self.background_workers.append(worker)
        
        self.logger.info(f"🔄 تم تشغيل {num_workers} عامل تحليل سلوكي")
    
    def _background_processor(self):
        """معالج الخلفية"""
        while self.is_processing:
            try:
                task = self.processing_queue.get(timeout=1)
                asyncio.create_task(self._process_task(task))
                self.processing_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"خطأ في معالج الخلفية: {e}")
    
    async def _process_task(self, task: Dict[str, Any]):
        """معالجة مهمة"""
        try:
            task_type = task.get("type")
            
            if task_type == "analyze_patterns":
                await self._analyze_behavior_patterns(task.get("user_id"))
            elif task_type == "generate_insights":
                await self._generate_behavior_insights(task.get("user_id"))
            elif task_type == "update_profile":
                await self._update_user_profile(task.get("user_id"))
            elif task_type == "detect_anomalies":
                await self._detect_behavior_anomalies(task.get("user_id"))
                
        except Exception as e:
            self.logger.error(f"خطأ في معالجة المهمة: {e}")
    
    async def track_behavior_event(self, event: BehaviorEvent) -> str:
        """تتبع حدث سلوكي"""
        try:
            # إضافة للمخزن المحلي
            self.behavior_events.append(event)
            
            # حفظ في قاعدة البيانات
            await self._save_behavior_event(event)
            
            # إضافة مهام التحليل للطابور
            self.processing_queue.put({
                "type": "analyze_patterns",
                "user_id": event.user_id
            })
            
            # تحديث الإحصائيات
            self.analytics_stats["total_events_processed"] += 1
            
            self.logger.debug(f"تم تتبع حدث سلوكي: {event.event_id}")
            return event.event_id
            
        except Exception as e:
            self.logger.error(f"خطأ في تتبع الحدث السلوكي: {e}")
            raise
    
    async def _save_behavior_event(self, event: BehaviorEvent):
        """حفظ حدث سلوكي في قاعدة البيانات"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO behavior_events
                (event_id, user_id, event_type, event_data, timestamp, 
                 session_id, context, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                event.event_id,
                event.user_id,
                event.event_type,
                json.dumps(event.event_data, ensure_ascii=False),
                event.timestamp.isoformat(),
                event.session_id,
                json.dumps(event.context, ensure_ascii=False),
                json.dumps(event.metadata, ensure_ascii=False)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"خطأ في حفظ الحدث السلوكي: {e}")
    
    async def _analyze_behavior_patterns(self, user_id: str):
        """تحليل الأنماط السلوكية للمستخدم"""
        try:
            # جمع أحداث المستخدم
            user_events = [
                event for event in self.behavior_events 
                if event.user_id == user_id
            ]
            
            if len(user_events) < self.analysis_config["min_pattern_frequency"]:
                return
            
            # تحليل الأنماط الزمنية
            temporal_patterns = await self._detect_temporal_patterns(user_events)
            
            # تحليل أنماط التسلسل
            sequence_patterns = await self._detect_sequence_patterns(user_events)
            
            # تحليل أنماط السياق
            context_patterns = await self._detect_context_patterns(user_events)
            
            # دمج الأنماط
            all_patterns = temporal_patterns + sequence_patterns + context_patterns
            
            # تقييم وتصفية الأنماط
            validated_patterns = [
                pattern for pattern in all_patterns
                if pattern.confidence >= self.analysis_config["pattern_confidence_threshold"]
            ]
            
            # حفظ الأنماط الجديدة
            for pattern in validated_patterns:
                await self._save_behavior_pattern(pattern)
                self.active_patterns[user_id].append(pattern)
            
            # تحديث إحصائيات
            self.analytics_stats["patterns_discovered"] += len(validated_patterns)
            
        except Exception as e:
            self.logger.error(f"خطأ في تحليل الأنماط السلوكية: {e}")
    
    async def _detect_temporal_patterns(self, events: List[BehaviorEvent]) -> List[BehaviorPattern]:
        """كشف الأنماط الزمنية"""
        patterns = []
        
        try:
            if not events:
                return patterns
            
            # تحليل أوقات النشاط
            activity_hours = defaultdict(int)
            for event in events:
                hour = event.timestamp.hour
                activity_hours[hour] += 1
            
            # العثور على ساعات الذروة
            total_events = len(events)
            peak_hours = []
            
            for hour, count in activity_hours.items():
                frequency = count / total_events
                if frequency > 0.1:  # أكثر من 10% من النشاط
                    peak_hours.append(hour)
            
            if peak_hours:
                pattern_id = hashlib.md5(f"temporal_{events[0].user_id}_{'-'.join(map(str, peak_hours))}".encode()).hexdigest()[:12]
                
                pattern = BehaviorPattern(
                    pattern_id=pattern_id,
                    user_id=events[0].user_id,
                    pattern_type="temporal_activity",
                    pattern_description=f"نشاط مكثف خلال الساعات: {', '.join(map(str, peak_hours))}",
                    frequency=len(peak_hours) / 24,
                    confidence=0.8,
                    start_time=min(event.timestamp for event in events),
                    triggers=["time_based"],
                    context_factors={"peak_hours": peak_hours}
                )
                patterns.append(pattern)
            
            # تحليل أنماط أيام الأسبوع
            weekday_activity = defaultdict(int)
            for event in events:
                weekday = event.timestamp.weekday()
                weekday_activity[weekday] += 1
            
            active_weekdays = []
            for weekday, count in weekday_activity.items():
                frequency = count / total_events
                if frequency > 0.15:  # أكثر من 15% من النشاط
                    active_weekdays.append(weekday)
            
            if active_weekdays:
                pattern_id = hashlib.md5(f"weekly_{events[0].user_id}_{'-'.join(map(str, active_weekdays))}".encode()).hexdigest()[:12]
                
                weekday_names = ["الاثنين", "الثلاثاء", "الأربعاء", "الخميس", "الجمعة", "السبت", "الأحد"]
                active_day_names = [weekday_names[day] for day in active_weekdays]
                
                pattern = BehaviorPattern(
                    pattern_id=pattern_id,
                    user_id=events[0].user_id,
                    pattern_type="weekly_activity",
                    pattern_description=f"نشاط منتظم في أيام: {', '.join(active_day_names)}",
                    frequency=len(active_weekdays) / 7,
                    confidence=0.75,
                    start_time=min(event.timestamp for event in events),
                    triggers=["weekly_pattern"],
                    context_factors={"active_weekdays": active_weekdays}
                )
                patterns.append(pattern)
            
        except Exception as e:
            self.logger.error(f"خطأ في كشف الأنماط الزمنية: {e}")
        
        return patterns
    
    async def _detect_sequence_patterns(self, events: List[BehaviorEvent]) -> List[BehaviorPattern]:
        """كشف أنماط التسلسل"""
        patterns = []
        
        try:
            if len(events) < 3:
                return patterns
            
            # ترتيب الأحداث حسب الوقت
            sorted_events = sorted(events, key=lambda e: e.timestamp)
            
            # البحث عن تسلسلات متكررة
            sequence_counts = defaultdict(int)
            
            for i in range(len(sorted_events) - 2):
                sequence = tuple(
                    event.event_type for event in sorted_events[i:i+3]
                )
                sequence_counts[sequence] += 1
            
            # العثور على التسلسلات المهمة
            total_sequences = len(sorted_events) - 2
            
            for sequence, count in sequence_counts.items():
                frequency = count / total_sequences
                
                if frequency >= 0.1 and count >= 3:  # تكرر 3 مرات على الأقل
                    pattern_id = hashlib.md5(f"sequence_{events[0].user_id}_{'-'.join(sequence)}".encode()).hexdigest()[:12]
                    
                    pattern = BehaviorPattern(
                        pattern_id=pattern_id,
                        user_id=events[0].user_id,
                        pattern_type="sequence_pattern",
                        pattern_description=f"تسلسل متكرر: {' → '.join(sequence)}",
                        frequency=frequency,
                        confidence=min(frequency * 2, 0.9),
                        start_time=min(event.timestamp for event in events),
                        triggers=list(sequence[:-1]),
                        outcomes=[sequence[-1]],
                        context_factors={"sequence": sequence, "occurrences": count}
                    )
                    patterns.append(pattern)
            
        except Exception as e:
            self.logger.error(f"خطأ في كشف أنماط التسلسل: {e}")
        
        return patterns
    
    async def _detect_context_patterns(self, events: List[BehaviorEvent]) -> List[BehaviorPattern]:
        """كشف أنماط السياق"""
        patterns = []
        
        try:
            if not events:
                return patterns
            
            # تجميع الأحداث حسب السياق
            context_groups = defaultdict(list)
            
            for event in events:
                for key, value in event.context.items():
                    context_key = f"{key}:{value}"
                    context_groups[context_key].append(event)
            
            # تحليل كل مجموعة سياق
            for context_key, context_events in context_groups.items():
                if len(context_events) < 3:
                    continue
                
                frequency = len(context_events) / len(events)
                
                if frequency >= 0.15:  # أكثر من 15% من الأحداث
                    pattern_id = hashlib.md5(f"context_{events[0].user_id}_{context_key}".encode()).hexdigest()[:12]
                    
                    # تحليل أنواع الأحداث في هذا السياق
                    event_types = [event.event_type for event in context_events]
                    common_event_type = max(set(event_types), key=event_types.count)
                    
                    pattern = BehaviorPattern(
                        pattern_id=pattern_id,
                        user_id=events[0].user_id,
                        pattern_type="context_pattern",
                        pattern_description=f"نمط سلوكي في سياق {context_key}: غالباً {common_event_type}",
                        frequency=frequency,
                        confidence=0.7,
                        start_time=min(event.timestamp for event in context_events),
                        triggers=[context_key],
                        outcomes=[common_event_type],
                        context_factors={"context": context_key, "dominant_behavior": common_event_type}
                    )
                    patterns.append(pattern)
            
        except Exception as e:
            self.logger.error(f"خطأ في كشف أنماط السياق: {e}")
        
        return patterns
    
    async def _save_behavior_pattern(self, pattern: BehaviorPattern):
        """حفظ نمط سلوكي"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO behavior_patterns
                (pattern_id, user_id, pattern_type, pattern_description,
                 frequency, confidence, start_time, end_time, triggers,
                 outcomes, context_factors, statistical_significance)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                pattern.pattern_id,
                pattern.user_id,
                pattern.pattern_type,
                pattern.pattern_description,
                pattern.frequency,
                pattern.confidence,
                pattern.start_time.isoformat(),
                pattern.end_time.isoformat() if pattern.end_time else None,
                json.dumps(pattern.triggers, ensure_ascii=False),
                json.dumps(pattern.outcomes, ensure_ascii=False),
                json.dumps(pattern.context_factors, ensure_ascii=False),
                pattern.statistical_significance
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"خطأ في حفظ النمط السلوكي: {e}")
    
    async def _generate_behavior_insights(self, user_id: str):
        """توليد رؤى سلوكية"""
        try:
            user_patterns = self.active_patterns.get(user_id, [])
            user_events = [
                event for event in self.behavior_events 
                if event.user_id == user_id
            ]
            
            if not user_patterns or not user_events:
                return
            
            insights = []
            
            # رؤى حول الأنماط المكتشفة
            pattern_insights = await self._analyze_pattern_insights(user_patterns, user_events)
            insights.extend(pattern_insights)
            
            # رؤى حول التغييرات السلوكية
            change_insights = await self._analyze_behavior_changes(user_events)
            insights.extend(change_insights)
            
            # رؤى حول الأداء والفعالية
            performance_insights = await self._analyze_performance_patterns(user_events)
            insights.extend(performance_insights)
            
            # حفظ الرؤى
            for insight in insights:
                if insight.confidence >= self.analysis_config["insight_confidence_threshold"]:
                    await self._save_behavior_insight(insight)
                    self.behavior_insights[user_id].append(insight)
            
            # تحديث الإحصائيات
            self.analytics_stats["insights_generated"] += len(insights)
            
        except Exception as e:
            self.logger.error(f"خطأ في توليد الرؤى السلوكية: {e}")
    
    async def _analyze_pattern_insights(self, patterns: List[BehaviorPattern], events: List[BehaviorEvent]) -> List[BehaviorInsight]:
        """تحليل رؤى الأنماط"""
        insights = []
        
        try:
            # تحليل قوة الأنماط
            strong_patterns = [p for p in patterns if p.confidence > 0.8]
            
            if strong_patterns:
                insight = BehaviorInsight(
                    insight_id=hashlib.md5(f"strong_patterns_{patterns[0].user_id}_{datetime.now()}".encode()).hexdigest()[:12],
                    user_id=patterns[0].user_id,
                    insight_type="pattern_strength",
                    description=f"تم اكتشاف {len(strong_patterns)} نمط سلوكي قوي",
                    confidence=0.9,
                    impact_score=0.8,
                    recommendations=[
                        "استفد من هذه الأنماط لتحسين الإنتاجية",
                        "قم بتطوير روتين يعتمد على هذه الأنماط"
                    ],
                    supporting_data={"strong_patterns": [p.pattern_description for p in strong_patterns]}
                )
                insights.append(insight)
            
            # تحليل الأنماط الزمنية
            temporal_patterns = [p for p in patterns if p.pattern_type == "temporal_activity"]
            
            if temporal_patterns:
                insight = BehaviorInsight(
                    insight_id=hashlib.md5(f"temporal_{patterns[0].user_id}_{datetime.now()}".encode()).hexdigest()[:12],
                    user_id=patterns[0].user_id,
                    insight_type="temporal_optimization",
                    description="لديك أنماط زمنية واضحة في النشاط",
                    confidence=0.85,
                    impact_score=0.7,
                    recommendations=[
                        "جدول المهام المهمة في أوقات النشاط الذروة",
                        "تجنب المهام المعقدة في أوقات النشاط المنخفض"
                    ],
                    supporting_data={"temporal_patterns": [p.context_factors for p in temporal_patterns]}
                )
                insights.append(insight)
            
        except Exception as e:
            self.logger.error(f"خطأ في تحليل رؤى الأنماط: {e}")
        
        return insights
    
    async def _analyze_behavior_changes(self, events: List[BehaviorEvent]) -> List[BehaviorInsight]:
        """تحليل التغييرات السلوكية"""
        insights = []
        
        try:
            if len(events) < 20:
                return insights
            
            # تقسيم الأحداث إلى فترات
            sorted_events = sorted(events, key=lambda e: e.timestamp)
            mid_point = len(sorted_events) // 2
            
            early_events = sorted_events[:mid_point]
            recent_events = sorted_events[mid_point:]
            
            # مقارنة أنواع الأحداث
            early_types = set(event.event_type for event in early_events)
            recent_types = set(event.event_type for event in recent_events)
            
            new_behaviors = recent_types - early_types
            stopped_behaviors = early_types - recent_types
            
            if new_behaviors:
                insight = BehaviorInsight(
                    insight_id=hashlib.md5(f"new_behaviors_{events[0].user_id}_{datetime.now()}".encode()).hexdigest()[:12],
                    user_id=events[0].user_id,
                    insight_type="behavior_evolution",
                    description=f"تطوير سلوكيات جديدة: {', '.join(new_behaviors)}",
                    confidence=0.8,
                    impact_score=0.6,
                    recommendations=[
                        "راقب فعالية السلوكيات الجديدة",
                        "ادمج السلوكيات المفيدة في روتينك اليومي"
                    ],
                    supporting_data={"new_behaviors": list(new_behaviors)}
                )
                insights.append(insight)
            
            if stopped_behaviors:
                insight = BehaviorInsight(
                    insight_id=hashlib.md5(f"stopped_behaviors_{events[0].user_id}_{datetime.now()}".encode()).hexdigest()[:12],
                    user_id=events[0].user_id,
                    insight_type="behavior_reduction",
                    description=f"تقليل السلوكيات: {', '.join(stopped_behaviors)}",
                    confidence=0.75,
                    impact_score=0.5,
                    recommendations=[
                        "راجع أسباب تقليل هذه السلوكيات",
                        "تأكد من أن التغيير إيجابي"
                    ],
                    supporting_data={"stopped_behaviors": list(stopped_behaviors)}
                )
                insights.append(insight)
            
        except Exception as e:
            self.logger.error(f"خطأ في تحليل التغييرات السلوكية: {e}")
        
        return insights
    
    async def _analyze_performance_patterns(self, events: List[BehaviorEvent]) -> List[BehaviorInsight]:
        """تحليل أنماط الأداء"""
        insights = []
        
        try:
            # تحليل معدل النشاط
            if len(events) >= 10:
                # حساب متوسط الأحداث يومياً
                start_date = min(event.timestamp for event in events).date()
                end_date = max(event.timestamp for event in events).date()
                days_span = (end_date - start_date).days + 1
                
                daily_average = len(events) / days_span
                
                if daily_average > 10:
                    performance_level = "عالي"
                    impact = 0.8
                elif daily_average > 5:
                    performance_level = "متوسط"
                    impact = 0.6
                else:
                    performance_level = "منخفض"
                    impact = 0.4
                
                insight = BehaviorInsight(
                    insight_id=hashlib.md5(f"activity_level_{events[0].user_id}_{datetime.now()}".encode()).hexdigest()[:12],
                    user_id=events[0].user_id,
                    insight_type="activity_level",
                    description=f"مستوى النشاط: {performance_level} ({daily_average:.1f} حدث/يوم)",
                    confidence=0.85,
                    impact_score=impact,
                    recommendations=self._get_activity_recommendations(performance_level),
                    supporting_data={"daily_average": daily_average, "total_days": days_span}
                )
                insights.append(insight)
            
        except Exception as e:
            self.logger.error(f"خطأ في تحليل أنماط الأداء: {e}")
        
        return insights
    
    def _get_activity_recommendations(self, performance_level: str) -> List[str]:
        """الحصول على توصيات النشاط"""
        recommendations = {
            "عالي": [
                "حافظ على هذا المستوى من النشاط",
                "تأكد من أخذ فترات راحة كافية",
                "استغل هذه الطاقة في المهام المهمة"
            ],
            "متوسط": [
                "حاول زيادة مستوى النشاط تدريجياً",
                "ركز على جودة الأنشطة",
                "حدد أهدافاً واضحة لزيادة التفاعل"
            ],
            "منخفض": [
                "ابدأ بزيادة النشاط تدريجياً",
                "ابحث عن محفزات للتفاعل أكثر",
                "حدد أوقاتاً ثابتة للأنشطة"
            ]
        }
        return recommendations.get(performance_level, ["راقب مستوى نشاطك بانتظام"])
    
    async def _save_behavior_insight(self, insight: BehaviorInsight):
        """حفظ رؤية سلوكية"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO behavior_insights
                (insight_id, user_id, insight_type, description,
                 confidence, impact_score, recommendations, supporting_data, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                insight.insight_id,
                insight.user_id,
                insight.insight_type,
                insight.description,
                insight.confidence,
                insight.impact_score,
                json.dumps(insight.recommendations, ensure_ascii=False),
                json.dumps(insight.supporting_data, ensure_ascii=False),
                insight.timestamp.isoformat()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"خطأ في حفظ الرؤية السلوكية: {e}")
    
    async def get_user_behavior_analysis(self, user_id: str) -> Dict[str, Any]:
        """الحصول على تحليل سلوك المستخدم"""
        try:
            analysis = {
                "user_id": user_id,
                "patterns": [],
                "insights": [],
                "profile": None,
                "statistics": {},
                "recommendations": []
            }
            
            # الأنماط النشطة
            user_patterns = self.active_patterns.get(user_id, [])
            analysis["patterns"] = [asdict(pattern) for pattern in user_patterns]
            
            # الرؤى السلوكية
            user_insights = self.behavior_insights.get(user_id, [])
            analysis["insights"] = [asdict(insight) for insight in user_insights]
            
            # الملف الشخصي
            if user_id in self.user_profiles:
                analysis["profile"] = asdict(self.user_profiles[user_id])
            
            # إحصائيات المستخدم
            user_events = [
                event for event in self.behavior_events 
                if event.user_id == user_id
            ]
            
            analysis["statistics"] = {
                "total_events": len(user_events),
                "patterns_count": len(user_patterns),
                "insights_count": len(user_insights),
                "last_activity": max(event.timestamp for event in user_events).isoformat() if user_events else None,
                "activity_span_days": (
                    max(event.timestamp for event in user_events) - 
                    min(event.timestamp for event in user_events)
                ).days if len(user_events) > 1 else 0
            }
            
            # توصيات مجمعة
            all_recommendations = []
            for insight in user_insights:
                all_recommendations.extend(insight.recommendations)
            
            # إزالة التكرار والاحتفاظ بأهم التوصيات
            unique_recommendations = list(set(all_recommendations))
            analysis["recommendations"] = unique_recommendations[:5]  # أفضل 5 توصيات
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"خطأ في الحصول على تحليل السلوك: {e}")
            return {"error": str(e)}
    
    async def create_behavior_visualization(self, user_id: str) -> Dict[str, Any]:
        """إنشاء تصورات للسلوك"""
        visualizations = {}
        
        try:
            if not PLOTLY_AVAILABLE:
                return {"error": "مكتبة التصور غير متاحة"}
            
            user_events = [
                event for event in self.behavior_events 
                if event.user_id == user_id
            ]
            
            if not user_events:
                return {"error": "لا توجد بيانات كافية"}
            
            # مخطط النشاط الزمني
            activity_viz = await self._create_activity_timeline(user_events)
            if activity_viz:
                visualizations["activity_timeline"] = activity_viz
            
            # مخطط أنواع الأحداث
            event_types_viz = await self._create_event_types_chart(user_events)
            if event_types_viz:
                visualizations["event_types"] = event_types_viz
            
            # مخطط الأنماط
            patterns_viz = await self._create_patterns_chart(user_id)
            if patterns_viz:
                visualizations["patterns"] = patterns_viz
            
        except Exception as e:
            self.logger.error(f"خطأ في إنشاء التصورات: {e}")
            visualizations = {"error": str(e)}
        
        return visualizations
    
    async def _create_activity_timeline(self, events: List[BehaviorEvent]) -> Optional[str]:
        """إنشاء مخطط زمني للنشاط"""
        try:
            # تجميع الأحداث حسب التاريخ
            daily_counts = defaultdict(int)
            for event in events:
                date = event.timestamp.date()
                daily_counts[date] += 1
            
            dates = list(daily_counts.keys())
            counts = list(daily_counts.values())
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates,
                y=counts,
                mode='lines+markers',
                name='النشاط اليومي',
                line=dict(color='#2E86AB', width=3),
                marker=dict(size=8)
            ))
            
            fig.update_layout(
                title='تطور النشاط عبر الزمن',
                xaxis_title='التاريخ',
                yaxis_title='عدد الأحداث',
                template='plotly_white',
                font=dict(family="Arial", size=12)
            )
            
            return fig.to_json()
            
        except Exception as e:
            self.logger.error(f"خطأ في إنشاء مخطط النشاط الزمني: {e}")
            return None
    
    async def _create_event_types_chart(self, events: List[BehaviorEvent]) -> Optional[str]:
        """إنشاء مخطط أنواع الأحداث"""
        try:
            # تجميع أنواع الأحداث
            event_type_counts = defaultdict(int)
            for event in events:
                event_type_counts[event.event_type] += 1
            
            labels = list(event_type_counts.keys())
            values = list(event_type_counts.values())
            
            fig = go.Figure(data=go.Pie(
                labels=labels,
                values=values,
                hole=0.4,
                textinfo='label+percent',
                textposition='outside'
            ))
            
            fig.update_layout(
                title='توزيع أنواع الأحداث',
                template='plotly_white',
                font=dict(family="Arial", size=12)
            )
            
            return fig.to_json()
            
        except Exception as e:
            self.logger.error(f"خطأ في إنشاء مخطط أنواع الأحداث: {e}")
            return None
    
    async def _create_patterns_chart(self, user_id: str) -> Optional[str]:
        """إنشاء مخطط الأنماط"""
        try:
            patterns = self.active_patterns.get(user_id, [])
            
            if not patterns:
                return None
            
            pattern_names = [pattern.pattern_type for pattern in patterns]
            pattern_confidence = [pattern.confidence for pattern in patterns]
            pattern_frequency = [pattern.frequency for pattern in patterns]
            
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('ثقة الأنماط', 'تكرار الأنماط'),
                specs=[[{"type": "bar"}, {"type": "bar"}]]
            )
            
            fig.add_trace(
                go.Bar(x=pattern_names, y=pattern_confidence, name='الثقة'),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(x=pattern_names, y=pattern_frequency, name='التكرار'),
                row=1, col=2
            )
            
            fig.update_layout(
                title='تحليل الأنماط السلوكية',
                template='plotly_white',
                font=dict(family="Arial", size=12)
            )
            
            return fig.to_json()
            
        except Exception as e:
            self.logger.error(f"خطأ في إنشاء مخطط الأنماط: {e}")
            return None
    
    async def _load_historical_data(self):
        """تحميل البيانات التاريخية"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # تحميل الأحداث الأخيرة
            cursor.execute('''
                SELECT * FROM behavior_events 
                WHERE timestamp > datetime('now', '-30 days')
                ORDER BY timestamp DESC
                LIMIT 1000
            ''')
            
            events_data = cursor.fetchall()
            
            for row in events_data:
                event = BehaviorEvent(
                    event_id=row[0],
                    user_id=row[1],
                    event_type=row[2],
                    event_data=json.loads(row[3]) if row[3] else {},
                    timestamp=datetime.fromisoformat(row[4]),
                    session_id=row[5] or "",
                    context=json.loads(row[6]) if row[6] else {},
                    metadata=json.loads(row[7]) if row[7] else {}
                )
                self.behavior_events.append(event)
            
            # تحميل الأنماط النشطة
            cursor.execute('''
                SELECT * FROM behavior_patterns
                WHERE end_time IS NULL OR end_time > datetime('now', '-7 days')
            ''')
            
            patterns_data = cursor.fetchall()
            
            for row in patterns_data:
                pattern = BehaviorPattern(
                    pattern_id=row[0],
                    user_id=row[1],
                    pattern_type=row[2],
                    pattern_description=row[3],
                    frequency=row[4],
                    confidence=row[5],
                    start_time=datetime.fromisoformat(row[6]),
                    end_time=datetime.fromisoformat(row[7]) if row[7] else None,
                    triggers=json.loads(row[8]) if row[8] else [],
                    outcomes=json.loads(row[9]) if row[9] else [],
                    context_factors=json.loads(row[10]) if row[10] else {},
                    statistical_significance=row[11] or 0.0
                )
                self.active_patterns[pattern.user_id].append(pattern)
            
            conn.close()
            
            self.analytics_stats["users_analyzed"] = len(set(event.user_id for event in self.behavior_events))
            
            self.logger.info(f"✅ تم تحميل {len(events_data)} حدث و {len(patterns_data)} نمط")
            
        except Exception as e:
            self.logger.warning(f"⚠️ خطأ في تحميل البيانات التاريخية: {e}")
    
    async def get_system_analytics(self) -> Dict[str, Any]:
        """الحصول على تحليلات النظام"""
        try:
            return {
                "system_stats": self.analytics_stats,
                "active_users": len(set(event.user_id for event in self.behavior_events)),
                "total_patterns": sum(len(patterns) for patterns in self.active_patterns.values()),
                "total_insights": sum(len(insights) for insights in self.behavior_insights.values()),
                "processing_queue_size": self.processing_queue.qsize(),
                "models_loaded": {
                    "embedding_network": self.embedding_network is not None,
                    "clustering_model": self.clustering_model is not None,
                    "anomaly_detector": self.anomaly_detector is not None
                },
                "database_status": "connected" if self.db_path.exists() else "disconnected"
            }
            
        except Exception as e:
            self.logger.error(f"خطأ في الحصول على تحليلات النظام: {e}")
            return {"error": str(e)}
    
    async def cleanup(self):
        """تنظيف الموارد"""
        try:
            self.is_processing = False
            
            # انتظار انتهاء العمال
            for worker in self.background_workers:
                if worker.is_alive():
                    worker.join(timeout=2)
            
            self.logger.info("✅ تم تنظيف موارد محرك التحليل السلوكي")
            
        except Exception as e:
            self.logger.error(f"خطأ في تنظيف الموارد: {e}")

# إنشاء مثيل عام
behavioral_analytics = AdvancedBehavioralAnalytics()

async def get_behavioral_analytics() -> AdvancedBehavioralAnalytics:
    """الحصول على محرك التحليل السلوكي"""
    if not behavioral_analytics.is_initialized:
        await behavioral_analytics.initialize()
    return behavioral_analytics

if __name__ == "__main__":
    async def test_behavioral_analytics():
        """اختبار محرك التحليل السلوكي"""
        print("🧠 اختبار محرك تحليل السلوك المتقدم")
        print("=" * 50)
        
        analytics = await get_behavioral_analytics()
        
        # إنشاء أحداث تجريبية
        test_events = []
        base_time = datetime.now()
        
        for i in range(20):
            event = BehaviorEvent(
                event_id=f"test_event_{i}",
                user_id="test_user",
                event_type=["login", "search", "create", "share", "logout"][i % 5],
                event_data={"action": f"action_{i}", "value": i},
                timestamp=base_time + timedelta(hours=i),
                session_id=f"session_{i//5}",
                context={"location": "home" if i % 2 == 0 else "office"}
            )
            test_events.append(event)
        
        print(f"📝 إضافة {len(test_events)} حدث تجريبي...")
        
        # تتبع الأحداث
        for event in test_events:
            await analytics.track_behavior_event(event)
        
        # انتظار المعالجة
        await asyncio.sleep(3)
        
        # الحصول على التحليل
        analysis = await analytics.get_user_behavior_analysis("test_user")
        
        print(f"\n📊 نتائج التحليل:")
        print(f"  • إجمالي الأحداث: {analysis['statistics']['total_events']}")
        print(f"  • الأنماط المكتشفة: {analysis['statistics']['patterns_count']}")
        print(f"  • الرؤى المولدة: {analysis['statistics']['insights_count']}")
        
        if analysis['patterns']:
            print(f"\n🔍 الأنماط المكتشفة:")
            for pattern in analysis['patterns']:
                print(f"  • {pattern['pattern_description']} (ثقة: {pattern['confidence']:.1%})")
        
        if analysis['insights']:
            print(f"\n💡 الرؤى السلوكية:")
            for insight in analysis['insights']:
                print(f"  • {insight['description']} (ثقة: {insight['confidence']:.1%})")
        
        if analysis['recommendations']:
            print(f"\n🎯 التوصيات:")
            for recommendation in analysis['recommendations']:
                print(f"  • {recommendation}")
        
        # تحليلات النظام
        system_analytics = await analytics.get_system_analytics()
        print(f"\n🖥️ إحصائيات النظام:")
        print(f"  • المستخدمون النشطون: {system_analytics['active_users']}")
        print(f"  • إجمالي الأنماط: {system_analytics['total_patterns']}")
        print(f"  • إجمالي الرؤى: {system_analytics['total_insights']}")
        
        await analytics.cleanup()
        print("\n✨ انتهى الاختبار بنجاح!")

    # تشغيل الاختبار
    asyncio.run(test_behavioral_analytics())
