
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
مراقب الصحة الذكي المتقدم
مراقبة شاملة للصحة والعافية مع التحليل والتوصيات الذكية
"""

import asyncio
import logging
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import sqlite3
from dataclasses import dataclass, asdict
import threading
import queue
from collections import defaultdict, deque
import cv2
import requests

@dataclass
class HealthMetric:
    """مقياس صحي"""
    metric_id: str
    user_id: str
    metric_type: str  # heart_rate, blood_pressure, weight, steps, sleep, etc.
    value: float
    unit: str
    timestamp: datetime
    source: str  # manual, device, camera, calculated
    confidence: float
    additional_data: Dict[str, Any]

@dataclass
class HealthRecommendation:
    """توصية صحية"""
    recommendation_id: str
    user_id: str
    category: str  # nutrition, exercise, sleep, medical
    priority: str  # low, medium, high, urgent
    title: str
    description: str
    action_items: List[str]
    created_at: datetime
    expires_at: Optional[datetime]
    is_active: bool

class SmartHealthMonitor:
    """مراقب الصحة الذكي"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # قاعدة البيانات الصحية
        self.health_db_path = Path("data/health/health_data.db")
        self.health_db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_health_database()
        
        # البيانات الصحية في الذاكرة
        self.health_metrics = deque(maxlen=10000)
        self.user_profiles = {}
        self.health_recommendations = {}
        
        # نماذج التحليل الصحي
        self.health_analyzers = {
            "heart_rate": self._analyze_heart_rate,
            "blood_pressure": self._analyze_blood_pressure,
            "weight": self._analyze_weight,
            "sleep": self._analyze_sleep,
            "activity": self._analyze_activity,
            "nutrition": self._analyze_nutrition
        }
        
        # المعايير الصحية الطبيعية
        self.normal_ranges = {
            "heart_rate": {"min": 60, "max": 100, "unit": "bpm"},
            "systolic_bp": {"min": 90, "max": 120, "unit": "mmHg"},
            "diastolic_bp": {"min": 60, "max": 80, "unit": "mmHg"},
            "bmi": {"min": 18.5, "max": 24.9, "unit": "kg/m²"},
            "sleep_hours": {"min": 7, "max": 9, "unit": "hours"},
            "daily_steps": {"min": 8000, "max": 12000, "unit": "steps"}
        }
        
        # إحصائيات الصحة
        self.health_stats = {
            "total_measurements": 0,
            "active_users": 0,
            "critical_alerts": 0,
            "recommendations_generated": 0,
            "health_score_average": 0.0
        }
        
        # إعدادات المراقبة
        self.monitoring_settings = {
            "auto_analysis_enabled": True,
            "emergency_alert_threshold": 0.8,
            "recommendation_frequency_hours": 24,
            "data_retention_days": 365,
            "privacy_mode": True
        }
        
        # بدء مراقب الصحة
        self._start_health_monitor()
    
    def _init_health_database(self):
        """تهيئة قاعدة البيانات الصحية"""
        conn = sqlite3.connect(self.health_db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS health_metrics (
                metric_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                metric_type TEXT NOT NULL,
                value REAL,
                unit TEXT,
                timestamp REAL,
                source TEXT,
                confidence REAL,
                additional_data TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_health_profiles (
                user_id TEXT PRIMARY KEY,
                age INTEGER,
                gender TEXT,
                height REAL,
                weight REAL,
                medical_conditions TEXT,
                medications TEXT,
                allergies TEXT,
                emergency_contact TEXT,
                created_at REAL,
                updated_at REAL
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS health_recommendations (
                recommendation_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                category TEXT,
                priority TEXT,
                title TEXT,
                description TEXT,
                action_items TEXT,
                created_at REAL,
                expires_at REAL,
                is_active BOOLEAN DEFAULT 1
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS health_alerts (
                alert_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                alert_type TEXT,
                severity TEXT,
                message TEXT,
                triggered_at REAL,
                resolved_at REAL,
                is_resolved BOOLEAN DEFAULT 0
            )
        """)
        
        conn.commit()
        conn.close()
    
    async def record_health_metric(self, user_id: str, metric_type: str, value: float,
                                 unit: str, source: str = "manual", 
                                 additional_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """تسجيل مقياس صحي"""
        try:
            metric_id = f"health_{user_id}_{metric_type}_{datetime.now().timestamp()}"
            
            if additional_data is None:
                additional_data = {}
            
            metric = HealthMetric(
                metric_id=metric_id,
                user_id=user_id,
                metric_type=metric_type,
                value=value,
                unit=unit,
                timestamp=datetime.now(),
                source=source,
                confidence=1.0 if source == "device" else 0.8,
                additional_data=additional_data
            )
            
            # حفظ في الذاكرة وقاعدة البيانات
            self.health_metrics.append(metric)
            await self._save_metric_to_database(metric)
            
            # تحليل فوري
            analysis_result = await self._analyze_metric(metric)
            
            # فحص التنبيهات
            alerts = await self._check_health_alerts(user_id, metric)
            
            # تحديث الإحصائيات
            self.health_stats["total_measurements"] += 1
            
            result = {
                "success": True,
                "message": "تم تسجيل المقياس الصحي بنجاح",
                "metric_id": metric_id,
                "analysis": analysis_result,
                "alerts": alerts,
                "health_score": await self._calculate_user_health_score(user_id)
            }
            
            # توليد توصيات إذا لزم الأمر
            if analysis_result.get("requires_attention", False):
                recommendations = await self._generate_health_recommendations(user_id, metric)
                result["recommendations"] = recommendations
            
            return result
        
        except Exception as e:
            self.logger.error(f"خطأ في تسجيل المقياس الصحي: {e}")
            return {
                "success": False,
                "message": f"خطأ في تسجيل المقياس الصحي: {str(e)}"
            }
    
    async def monitor_heart_rate_from_camera(self, user_id: str, video_frames: List[np.ndarray]) -> Dict[str, Any]:
        """مراقبة معدل ضربات القلب من الكاميرا"""
        try:
            if not video_frames:
                return {
                    "success": False,
                    "message": "لا توجد إطارات فيديو للتحليل"
                }
            
            # استخراج الوجه من الإطارات
            face_regions = []
            for frame in video_frames:
                # تحويل إلى RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # كشف الوجه
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                faces = face_cascade.detectMultiScale(rgb_frame, 1.1, 4)
                
                if len(faces) > 0:
                    # أخذ أول وجه
                    x, y, w, h = faces[0]
                    face_region = rgb_frame[y:y+h, x:x+w]
                    face_regions.append(face_region)
            
            if len(face_regions) < 10:
                return {
                    "success": False,
                    "message": "عدد غير كافِ من إطارات الوجه للتحليل"
                }
            
            # تحليل تغيرات اللون في منطقة الوجه (PPG)
            heart_rate = await self._calculate_heart_rate_from_ppg(face_regions)
            
            if heart_rate > 0:
                # تسجيل النتيجة
                result = await self.record_health_metric(
                    user_id=user_id,
                    metric_type="heart_rate",
                    value=heart_rate,
                    unit="bpm",
                    source="camera",
                    additional_data={
                        "measurement_method": "ppg_camera",
                        "frames_analyzed": len(face_regions),
                        "measurement_duration": len(video_frames) / 30  # افتراض 30 fps
                    }
                )
                
                return {
                    "success": True,
                    "heart_rate": heart_rate,
                    "confidence": 0.7,  # دقة متوسطة للقياس من الكاميرا
                    "method": "PPG من الكاميرا",
                    "detailed_result": result
                }
            else:
                return {
                    "success": False,
                    "message": "فشل في حساب معدل ضربات القلب من الكاميرا"
                }
        
        except Exception as e:
            self.logger.error(f"خطأ في مراقبة معدل ضربات القلب من الكاميرا: {e}")
            return {
                "success": False,
                "message": f"خطأ في المراقبة: {str(e)}"
            }
    
    async def _calculate_heart_rate_from_ppg(self, face_regions: List[np.ndarray]) -> float:
        """حساب معدل ضربات القلب من إشارة PPG"""
        try:
            # استخراج قناة الأخضر من كل إطار (الأكثر حساسية للدم)
            green_values = []
            
            for face_region in face_regions:
                if face_region.size > 0:
                    # أخذ المتوسط من قناة الأخضر
                    green_channel = face_region[:, :, 1]  # قناة G في RGB
                    green_mean = np.mean(green_channel)
                    green_values.append(green_mean)
            
            if len(green_values) < 10:
                return 0.0
            
            # تطبيق مرشح للتخلص من الضوضاء
            green_signal = np.array(green_values)
            
            # إزالة الاتجاه العام
            green_signal = green_signal - np.mean(green_signal)
            
            # تطبيق FFT لاستخراج التردد المهيمن
            fft = np.fft.fft(green_signal)
            freqs = np.fft.fftfreq(len(green_signal), 1/30)  # افتراض 30 fps
            
            # البحث عن التردد في نطاق معدل ضربات القلب الطبيعي (0.8-3.5 Hz)
            valid_freq_mask = (freqs >= 0.8) & (freqs <= 3.5)
            valid_fft = np.abs(fft[valid_freq_mask])
            valid_freqs = freqs[valid_freq_mask]
            
            if len(valid_fft) > 0:
                # العثور على التردد ذو الأمبليتودة الأعلى
                peak_freq_idx = np.argmax(valid_fft)
                peak_freq = valid_freqs[peak_freq_idx]
                
                # تحويل من Hz إلى BPM
                heart_rate = peak_freq * 60
                
                # التحقق من المعقولية
                if 50 <= heart_rate <= 200:
                    return round(heart_rate, 1)
            
            return 0.0
        
        except Exception as e:
            self.logger.error(f"خطأ في حساب معدل ضربات القلب من PPG: {e}")
            return 0.0
    
    async def analyze_sleep_pattern(self, user_id: str, sleep_data: Dict[str, Any]) -> Dict[str, Any]:
        """تحليل نمط النوم"""
        try:
            # استخراج بيانات النوم
            sleep_start = datetime.fromisoformat(sleep_data.get("sleep_start", ""))
            sleep_end = datetime.fromisoformat(sleep_data.get("sleep_end", ""))
            sleep_quality = sleep_data.get("quality_score", 0.5)
            
            # حساب مدة النوم
            sleep_duration = (sleep_end - sleep_start).total_seconds() / 3600  # بالساعات
            
            # تسجيل المقياس
            await self.record_health_metric(
                user_id=user_id,
                metric_type="sleep",
                value=sleep_duration,
                unit="hours",
                source="sleep_tracker",
                additional_data={
                    "sleep_start": sleep_start.isoformat(),
                    "sleep_end": sleep_end.isoformat(),
                    "quality_score": sleep_quality,
                    "sleep_stages": sleep_data.get("stages", {}),
                    "interruptions": sleep_data.get("interruptions", 0)
                }
            )
            
            # تحليل جودة النوم
            analysis = await self._analyze_sleep_quality(user_id, sleep_duration, sleep_quality)
            
            return {
                "success": True,
                "sleep_duration": sleep_duration,
                "sleep_quality": sleep_quality,
                "analysis": analysis,
                "recommendations": await self._generate_sleep_recommendations(user_id, analysis)
            }
        
        except Exception as e:
            self.logger.error(f"خطأ في تحليل نمط النوم: {e}")
            return {
                "success": False,
                "message": f"خطأ في تحليل النوم: {str(e)}"
            }
    
    async def track_nutrition(self, user_id: str, nutrition_data: Dict[str, Any]) -> Dict[str, Any]:
        """تتبع التغذية"""
        try:
            # استخراج البيانات الغذائية
            calories = nutrition_data.get("calories", 0)
            protein = nutrition_data.get("protein", 0)
            carbs = nutrition_data.get("carbohydrates", 0)
            fat = nutrition_data.get("fat", 0)
            water = nutrition_data.get("water_ml", 0)
            
            # تسجيل المقاييس
            metrics_recorded = []
            
            for metric_type, value, unit in [
                ("calories", calories, "kcal"),
                ("protein", protein, "g"),
                ("carbohydrates", carbs, "g"),
                ("fat", fat, "g"),
                ("water_intake", water, "ml")
            ]:
                if value > 0:
                    result = await self.record_health_metric(
                        user_id=user_id,
                        metric_type=metric_type,
                        value=value,
                        unit=unit,
                        source="nutrition_tracker",
                        additional_data=nutrition_data
                    )
                    metrics_recorded.append(result)
            
            # تحليل التغذية
            nutrition_analysis = await self._analyze_daily_nutrition(user_id)
            
            return {
                "success": True,
                "metrics_recorded": len(metrics_recorded),
                "nutrition_analysis": nutrition_analysis,
                "recommendations": await self._generate_nutrition_recommendations(user_id, nutrition_analysis)
            }
        
        except Exception as e:
            self.logger.error(f"خطأ في تتبع التغذية: {e}")
            return {
                "success": False,
                "message": f"خطأ في تتبع التغذية: {str(e)}"
            }
    
    async def _analyze_metric(self, metric: HealthMetric) -> Dict[str, Any]:
        """تحليل مقياس صحي"""
        try:
            analyzer = self.health_analyzers.get(metric.metric_type)
            if analyzer:
                return await analyzer(metric)
            
            # تحليل عام
            return await self._general_metric_analysis(metric)
        
        except Exception as e:
            self.logger.error(f"خطأ في تحليل المقياس: {e}")
            return {"error": str(e)}
    
    async def _analyze_heart_rate(self, metric: HealthMetric) -> Dict[str, Any]:
        """تحليل معدل ضربات القلب"""
        hr = metric.value
        normal_range = self.normal_ranges["heart_rate"]
        
        status = "طبيعي"
        requires_attention = False
        recommendations = []
        
        if hr < normal_range["min"]:
            status = "أقل من الطبيعي (بطء القلب)"
            requires_attention = True
            recommendations.append("استشارة طبيب إذا كان مصحوباً بأعراض أخرى")
        elif hr > normal_range["max"]:
            status = "أعلى من الطبيعي (تسارع القلب)"
            requires_attention = True
            recommendations.append("تجنب الكافيين وممارسة تقنيات الاسترخاء")
            if hr > 120:
                recommendations.append("استشارة طبيب عاجلة")
        
        return {
            "status": status,
            "value": hr,
            "normal_range": f"{normal_range['min']}-{normal_range['max']} {normal_range['unit']}",
            "requires_attention": requires_attention,
            "recommendations": recommendations,
            "severity": "high" if hr > 120 or hr < 50 else "medium" if requires_attention else "low"
        }
    
    async def _analyze_blood_pressure(self, metric: HealthMetric) -> Dict[str, Any]:
        """تحليل ضغط الدم"""
        # افتراض أن القيمة تحتوي على الضغط الانقباضي والانبساطي
        additional_data = metric.additional_data
        systolic = additional_data.get("systolic", metric.value)
        diastolic = additional_data.get("diastolic", 0)
        
        status = "طبيعي"
        requires_attention = False
        recommendations = []
        
        if systolic < 90 or diastolic < 60:
            status = "منخفض"
            requires_attention = True
            recommendations.append("زيادة تناول السوائل والملح باعتدال")
        elif systolic > 140 or diastolic > 90:
            status = "مرتفع"
            requires_attention = True
            recommendations.append("تقليل الملح وممارسة الرياضة والاسترخاء")
            if systolic > 160 or diastolic > 100:
                recommendations.append("استشارة طبيب عاجلة")
        elif systolic > 120 or diastolic > 80:
            status = "قبل ارتفاع الضغط"
            recommendations.append("مراقبة النظام الغذائي وممارسة الرياضة")
        
        return {
            "status": status,
            "systolic": systolic,
            "diastolic": diastolic,
            "requires_attention": requires_attention,
            "recommendations": recommendations,
            "severity": "high" if systolic > 160 or diastolic > 100 else "medium" if requires_attention else "low"
        }
    
    async def _analyze_weight(self, metric: HealthMetric) -> Dict[str, Any]:
        """تحليل الوزن"""
        weight = metric.value
        user_profile = await self._get_user_health_profile(metric.user_id)
        
        if not user_profile or not user_profile.get("height"):
            return {
                "status": "يتطلب بيانات الطول لحساب BMI",
                "weight": weight,
                "requires_attention": False
            }
        
        height_m = user_profile["height"] / 100  # تحويل من سم إلى متر
        bmi = weight / (height_m ** 2)
        
        status = "طبيعي"
        requires_attention = False
        recommendations = []
        
        if bmi < 18.5:
            status = "نقص في الوزن"
            requires_attention = True
            recommendations.append("زيادة السعرات الحرارية والبروتين")
        elif bmi > 30:
            status = "سمنة"
            requires_attention = True
            recommendations.append("تقليل السعرات الحرارية وزيادة النشاط البدني")
        elif bmi > 25:
            status = "زيادة في الوزن"
            recommendations.append("اتباع نظام غذائي متوازن وممارسة الرياضة")
        
        return {
            "status": status,
            "weight": weight,
            "bmi": round(bmi, 1),
            "bmi_category": status,
            "requires_attention": requires_attention,
            "recommendations": recommendations,
            "severity": "medium" if requires_attention else "low"
        }
    
    async def _analyze_sleep(self, metric: HealthMetric) -> Dict[str, Any]:
        """تحليل النوم"""
        sleep_hours = metric.value
        normal_range = self.normal_ranges["sleep_hours"]
        
        status = "كافِ"
        requires_attention = False
        recommendations = []
        
        if sleep_hours < normal_range["min"]:
            status = "قليل"
            requires_attention = True
            recommendations.append("محاولة النوم مبكراً وتجنب الشاشات قبل النوم")
        elif sleep_hours > normal_range["max"]:
            status = "كثير"
            recommendations.append("تحديد مواعيد ثابتة للنوم والاستيقاظ")
        
        return {
            "status": status,
            "sleep_duration": sleep_hours,
            "normal_range": f"{normal_range['min']}-{normal_range['max']} {normal_range['unit']}",
            "requires_attention": requires_attention,
            "recommendations": recommendations,
            "severity": "medium" if requires_attention else "low"
        }
    
    async def _analyze_activity(self, metric: HealthMetric) -> Dict[str, Any]:
        """تحليل النشاط البدني"""
        steps = metric.value
        normal_range = self.normal_ranges["daily_steps"]
        
        status = "نشط"
        requires_attention = False
        recommendations = []
        
        if steps < normal_range["min"]:
            status = "قليل النشاط"
            requires_attention = True
            recommendations.append("زيادة المشي والحركة اليومية")
        elif steps > normal_range["max"]:
            status = "نشط جداً"
            recommendations.append("الحفاظ على هذا المستوى الرائع من النشاط")
        
        return {
            "status": status,
            "steps": steps,
            "target_range": f"{normal_range['min']}-{normal_range['max']} {normal_range['unit']}",
            "requires_attention": requires_attention,
            "recommendations": recommendations,
            "severity": "medium" if requires_attention else "low"
        }
    
    async def _analyze_nutrition(self, metric: HealthMetric) -> Dict[str, Any]:
        """تحليل التغذية"""
        # تحليل عام للمقاييس الغذائية
        return {
            "status": "تم تسجيله",
            "value": metric.value,
            "unit": metric.unit,
            "requires_attention": False,
            "recommendations": ["الحفاظ على نظام غذائي متوازن"],
            "severity": "low"
        }
    
    async def _general_metric_analysis(self, metric: HealthMetric) -> Dict[str, Any]:
        """تحليل عام للمقاييس"""
        return {
            "status": "تم تسجيله",
            "metric_type": metric.metric_type,
            "value": metric.value,
            "unit": metric.unit,
            "requires_attention": False,
            "recommendations": [],
            "severity": "low"
        }
    
    async def _check_health_alerts(self, user_id: str, metric: HealthMetric) -> List[Dict[str, Any]]:
        """فحص التنبيهات الصحية"""
        alerts = []
        
        try:
            # تنبيهات معدل ضربات القلب
            if metric.metric_type == "heart_rate":
                if metric.value > 120:
                    alerts.append({
                        "type": "high_heart_rate",
                        "severity": "high",
                        "message": f"معدل ضربات القلب مرتفع: {metric.value} bpm"
                    })
                elif metric.value < 50:
                    alerts.append({
                        "type": "low_heart_rate",
                        "severity": "medium",
                        "message": f"معدل ضربات القلب منخفض: {metric.value} bpm"
                    })
            
            # تنبيهات ضغط الدم
            elif metric.metric_type == "blood_pressure":
                systolic = metric.additional_data.get("systolic", metric.value)
                diastolic = metric.additional_data.get("diastolic", 0)
                
                if systolic > 160 or diastolic > 100:
                    alerts.append({
                        "type": "high_blood_pressure",
                        "severity": "urgent",
                        "message": f"ضغط الدم مرتفع جداً: {systolic}/{diastolic}"
                    })
            
            # حفظ التنبيهات في قاعدة البيانات
            for alert in alerts:
                await self._save_health_alert(user_id, alert)
            
            return alerts
        
        except Exception as e:
            self.logger.error(f"خطأ في فحص التنبيهات الصحية: {e}")
            return []
    
    async def _calculate_user_health_score(self, user_id: str) -> float:
        """حساب نقاط الصحة العامة للمستخدم"""
        try:
            # الحصول على آخر القياسات
            recent_metrics = await self._get_recent_user_metrics(user_id, days=7)
            
            if not recent_metrics:
                return 0.0
            
            scores = []
            
            # تقييم معدل ضربات القلب
            hr_metrics = [m for m in recent_metrics if m["metric_type"] == "heart_rate"]
            if hr_metrics:
                avg_hr = np.mean([m["value"] for m in hr_metrics])
                hr_score = self._score_heart_rate(avg_hr)
                scores.append(hr_score)
            
            # تقييم النوم
            sleep_metrics = [m for m in recent_metrics if m["metric_type"] == "sleep"]
            if sleep_metrics:
                avg_sleep = np.mean([m["value"] for m in sleep_metrics])
                sleep_score = self._score_sleep(avg_sleep)
                scores.append(sleep_score)
            
            # تقييم النشاط
            activity_metrics = [m for m in recent_metrics if m["metric_type"] == "steps"]
            if activity_metrics:
                avg_steps = np.mean([m["value"] for m in activity_metrics])
                activity_score = self._score_activity(avg_steps)
                scores.append(activity_score)
            
            # حساب المتوسط
            if scores:
                return round(np.mean(scores), 1)
            else:
                return 0.0
        
        except Exception as e:
            self.logger.error(f"خطأ في حساب نقاط الصحة: {e}")
            return 0.0
    
    def _score_heart_rate(self, hr: float) -> float:
        """تقييم معدل ضربات القلب"""
        if 60 <= hr <= 100:
            return 10.0
        elif 50 <= hr < 60 or 100 < hr <= 110:
            return 7.0
        elif 40 <= hr < 50 or 110 < hr <= 120:
            return 5.0
        else:
            return 2.0
    
    def _score_sleep(self, sleep_hours: float) -> float:
        """تقييم النوم"""
        if 7 <= sleep_hours <= 9:
            return 10.0
        elif 6 <= sleep_hours < 7 or 9 < sleep_hours <= 10:
            return 7.0
        elif 5 <= sleep_hours < 6 or 10 < sleep_hours <= 11:
            return 5.0
        else:
            return 2.0
    
    def _score_activity(self, steps: float) -> float:
        """تقييم النشاط البدني"""
        if steps >= 10000:
            return 10.0
        elif steps >= 8000:
            return 8.0
        elif steps >= 5000:
            return 6.0
        elif steps >= 3000:
            return 4.0
        else:
            return 2.0
    
    async def _generate_health_recommendations(self, user_id: str, metric: HealthMetric) -> List[Dict[str, Any]]:
        """توليد التوصيات الصحية"""
        recommendations = []
        
        try:
            analysis = await self._analyze_metric(metric)
            
            if analysis.get("requires_attention", False):
                for recommendation_text in analysis.get("recommendations", []):
                    recommendation = {
                        "recommendation_id": f"rec_{datetime.now().timestamp()}",
                        "user_id": user_id,
                        "category": metric.metric_type,
                        "priority": analysis.get("severity", "low"),
                        "title": f"توصية لـ {metric.metric_type}",
                        "description": recommendation_text,
                        "action_items": [recommendation_text],
                        "created_at": datetime.now(),
                        "expires_at": datetime.now() + timedelta(days=7),
                        "is_active": True
                    }
                    
                    recommendations.append(recommendation)
                    await self._save_recommendation_to_database(recommendation)
            
            return recommendations
        
        except Exception as e:
            self.logger.error(f"خطأ في توليد التوصيات: {e}")
            return []
    
    async def get_health_dashboard(self, user_id: str) -> Dict[str, Any]:
        """الحصول على لوحة معلومات الصحة"""
        try:
            # الحصول على آخر القياسات
            recent_metrics = await self._get_recent_user_metrics(user_id, days=30)
            
            # حساب نقاط الصحة
            health_score = await self._calculate_user_health_score(user_id)
            
            # الحصول على التوصيات النشطة
            active_recommendations = await self._get_active_recommendations(user_id)
            
            # تجميع البيانات حسب النوع
            metrics_by_type = defaultdict(list)
            for metric in recent_metrics:
                metrics_by_type[metric["metric_type"]].append(metric)
            
            # إحصائيات سريعة
            quick_stats = {}
            for metric_type, metrics in metrics_by_type.items():
                if metrics:
                    values = [m["value"] for m in metrics]
                    quick_stats[metric_type] = {
                        "latest": values[-1],
                        "average": round(np.mean(values), 1),
                        "trend": "stable"  # يمكن تحسينه لحساب الاتجاه الفعلي
                    }
            
            return {
                "user_id": user_id,
                "health_score": health_score,
                "quick_stats": quick_stats,
                "recent_metrics_count": len(recent_metrics),
                "active_recommendations": active_recommendations,
                "last_updated": datetime.now().isoformat(),
                "metrics_summary": {
                    metric_type: len(metrics) 
                    for metric_type, metrics in metrics_by_type.items()
                }
            }
        
        except Exception as e:
            self.logger.error(f"خطأ في إنشاء لوحة معلومات الصحة: {e}")
            return {"error": str(e)}
    
    async def _save_metric_to_database(self, metric: HealthMetric):
        """حفظ المقياس في قاعدة البيانات"""
        try:
            conn = sqlite3.connect(self.health_db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO health_metrics
                (metric_id, user_id, metric_type, value, unit, timestamp, source, confidence, additional_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metric.metric_id,
                metric.user_id,
                metric.metric_type,
                metric.value,
                metric.unit,
                metric.timestamp.timestamp(),
                metric.source,
                metric.confidence,
                json.dumps(metric.additional_data, ensure_ascii=False)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"خطأ في حفظ المقياس: {e}")
    
    async def _save_health_alert(self, user_id: str, alert: Dict[str, Any]):
        """حفظ تنبيه صحي"""
        try:
            alert_id = f"alert_{datetime.now().timestamp()}_{alert['type']}"
            
            conn = sqlite3.connect(self.health_db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO health_alerts
                (alert_id, user_id, alert_type, severity, message, triggered_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                alert_id,
                user_id,
                alert["type"],
                alert["severity"],
                alert["message"],
                datetime.now().timestamp()
            ))
            
            conn.commit()
            conn.close()
            
            self.health_stats["critical_alerts"] += 1
            
        except Exception as e:
            self.logger.error(f"خطأ في حفظ التنبيه: {e}")
    
    async def _save_recommendation_to_database(self, recommendation: Dict[str, Any]):
        """حفظ التوصية في قاعدة البيانات"""
        try:
            conn = sqlite3.connect(self.health_db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO health_recommendations
                (recommendation_id, user_id, category, priority, title, description, 
                 action_items, created_at, expires_at, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                recommendation["recommendation_id"],
                recommendation["user_id"],
                recommendation["category"],
                recommendation["priority"],
                recommendation["title"],
                recommendation["description"],
                json.dumps(recommendation["action_items"], ensure_ascii=False),
                recommendation["created_at"].timestamp(),
                recommendation["expires_at"].timestamp() if recommendation["expires_at"] else None,
                recommendation["is_active"]
            ))
            
            conn.commit()
            conn.close()
            
            self.health_stats["recommendations_generated"] += 1
            
        except Exception as e:
            self.logger.error(f"خطأ في حفظ التوصية: {e}")
    
    def _start_health_monitor(self):
        """بدء مراقب الصحة"""
        def health_monitor():
            while True:
                try:
                    # تنظيف البيانات القديمة
                    asyncio.run(self._cleanup_old_data())
                    
                    # تحديث الإحصائيات
                    asyncio.run(self._update_health_statistics())
                    
                    # فحص التوصيات المنتهية الصلاحية
                    asyncio.run(self._expire_old_recommendations())
                    
                    # انتظار ساعة
                    threading.Event().wait(3600)
                    
                except Exception as e:
                    self.logger.error(f"خطأ في مراقب الصحة: {e}")
        
        monitor_thread = threading.Thread(target=health_monitor, daemon=True)
        monitor_thread.start()
    
    async def _get_recent_user_metrics(self, user_id: str, days: int = 7) -> List[Dict[str, Any]]:
        """الحصول على القياسات الأخيرة للمستخدم"""
        try:
            cutoff_date = (datetime.now() - timedelta(days=days)).timestamp()
            
            conn = sqlite3.connect(self.health_db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM health_metrics 
                WHERE user_id = ? AND timestamp > ?
                ORDER BY timestamp DESC
            """, (user_id, cutoff_date))
            
            rows = cursor.fetchall()
            conn.close()
            
            metrics = []
            for row in rows:
                metrics.append({
                    "metric_id": row[0],
                    "user_id": row[1],
                    "metric_type": row[2],
                    "value": row[3],
                    "unit": row[4],
                    "timestamp": datetime.fromtimestamp(row[5]),
                    "source": row[6],
                    "confidence": row[7],
                    "additional_data": json.loads(row[8]) if row[8] else {}
                })
            
            return metrics
        
        except Exception as e:
            self.logger.error(f"خطأ في الحصول على القياسات الأخيرة: {e}")
            return []
    
    def get_health_statistics(self) -> Dict[str, Any]:
        """الحصول على إحصائيات الصحة"""
        return {
            "global_stats": self.health_stats,
            "total_users": len(set(metric.user_id for metric in self.health_metrics)),
            "monitoring_settings": self.monitoring_settings,
            "normal_ranges": self.normal_ranges,
            "supported_metrics": list(self.health_analyzers.keys())
        }

# إنشاء مثيل عام
smart_health_monitor = SmartHealthMonitor()

def get_smart_health_monitor() -> SmartHealthMonitor:
    """الحصول على مراقب الصحة الذكي"""
    return smart_health_monitor

if __name__ == "__main__":
    # اختبار النظام
    async def test_health_system():
        monitor = get_smart_health_monitor()
        
        print("🏥 اختبار مراقب الصحة الذكي...")
        
        # تسجيل قياسات تجريبية
        hr_result = await monitor.record_health_metric(
            user_id="test_user",
            metric_type="heart_rate",
            value=75,
            unit="bpm",
            source="device"
        )
        print(f"تسجيل معدل ضربات القلب: {hr_result}")
        
        # الحصول على لوحة المعلومات
        dashboard = await monitor.get_health_dashboard("test_user")
        print(f"لوحة معلومات الصحة: {dashboard}")
        
        # الحصول على الإحصائيات
        stats = monitor.get_health_statistics()
        print(f"إحصائيات الصحة: {stats}")
    
    asyncio.run(test_health_system())
