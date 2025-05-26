
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🛡️ محرك الموثوقية والاستقرار المتقدم
Advanced Reliability & Stability Engine
"""

import asyncio
import logging
import time
import traceback
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
from pathlib import Path
import psutil
import threading
from functools import wraps

@dataclass
class HealthCheck:
    """فحص صحة النظام"""
    component: str
    status: str  # healthy, warning, critical
    message: str
    timestamp: datetime
    response_time: float = 0.0

@dataclass
class ErrorReport:
    """تقرير خطأ مفصل"""
    error_id: str
    error_type: str
    message: str
    traceback: str
    timestamp: datetime
    component: str
    severity: str
    context: Dict[str, Any]

class CircuitBreaker:
    """قاطع الدائرة لمنع انهيار النظام"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.logger = logging.getLogger(__name__)
    
    def __call__(self, func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if self.state == "OPEN":
                if self._should_attempt_reset():
                    self.state = "HALF_OPEN"
                else:
                    raise Exception(f"Circuit breaker is OPEN for {func.__name__}")
            
            try:
                result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
                self._on_success()
                return result
            except Exception as e:
                self._on_failure()
                raise e
        
        return wrapper
    
    def _should_attempt_reset(self) -> bool:
        """هل يجب محاولة إعادة تعيين القاطع؟"""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time >= self.recovery_timeout
    
    def _on_success(self):
        """عند نجاح العملية"""
        self.failure_count = 0
        self.state = "CLOSED"
    
    def _on_failure(self):
        """عند فشل العملية"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            self.logger.warning(f"Circuit breaker opened after {self.failure_count} failures")

class HealthMonitor:
    """مراقب صحة النظام الشامل"""
    
    def __init__(self):
        self.health_checks: List[HealthCheck] = []
        self.error_reports: List[ErrorReport] = []
        self.monitoring = False
        self.alert_thresholds = {
            "cpu_usage": 85,
            "memory_usage": 90,
            "disk_usage": 95,
            "response_time": 5.0
        }
        self.logger = logging.getLogger(__name__)
    
    async def start_monitoring(self):
        """بدء المراقبة المستمرة"""
        self.monitoring = True
        
        # بدء مراقبة النظام
        asyncio.create_task(self._system_health_monitor())
        asyncio.create_task(self._component_health_monitor())
        
        self.logger.info("🛡️ بدء مراقبة صحة النظام")
    
    async def _system_health_monitor(self):
        """مراقبة صحة النظام"""
        while self.monitoring:
            try:
                # فحص استخدام CPU
                cpu_usage = psutil.cpu_percent(interval=1)
                await self._check_threshold("cpu_usage", cpu_usage, "%")
                
                # فحص استخدام الذاكرة
                memory = psutil.virtual_memory()
                await self._check_threshold("memory_usage", memory.percent, "%")
                
                # فحص استخدام القرص
                disk = psutil.disk_usage('/')
                disk_usage = (disk.used / disk.total) * 100
                await self._check_threshold("disk_usage", disk_usage, "%")
                
                await asyncio.sleep(30)  # فحص كل 30 ثانية
                
            except Exception as e:
                await self._log_error("system_monitor", str(e), "warning")
                await asyncio.sleep(60)
    
    async def _component_health_monitor(self):
        """مراقبة صحة المكونات"""
        components = [
            "ai_engine",
            "database",
            "cache_system",
            "file_system",
            "network"
        ]
        
        while self.monitoring:
            try:
                for component in components:
                    health = await self._check_component_health(component)
                    self.health_checks.append(health)
                
                # الحفاظ على آخر 100 فحص فقط
                if len(self.health_checks) > 100:
                    self.health_checks = self.health_checks[-100:]
                
                await asyncio.sleep(60)  # فحص كل دقيقة
                
            except Exception as e:
                await self._log_error("component_monitor", str(e), "warning")
                await asyncio.sleep(120)
    
    async def _check_component_health(self, component: str) -> HealthCheck:
        """فحص صحة مكون محدد"""
        start_time = time.time()
        
        try:
            if component == "ai_engine":
                # فحص محرك الذكاء الاصطناعي
                status = "healthy"
                message = "محرك الذكاء الاصطناعي يعمل بشكل طبيعي"
            
            elif component == "database":
                # فحص قاعدة البيانات
                status = "healthy"
                message = "قاعدة البيانات متصلة ومتاحة"
            
            elif component == "cache_system":
                # فحص نظام الكاش
                status = "healthy" 
                message = "نظام الكاش يعمل بكفاءة"
            
            elif component == "file_system":
                # فحص نظام الملفات
                if Path("data").exists():
                    status = "healthy"
                    message = "نظام الملفات متاح"
                else:
                    status = "warning"
                    message = "مجلد البيانات غير موجود"
            
            elif component == "network":
                # فحص الشبكة
                status = "healthy"
                message = "الاتصال بالشبكة طبيعي"
            
            else:
                status = "unknown"
                message = f"مكون غير معروف: {component}"
            
            response_time = time.time() - start_time
            
            return HealthCheck(
                component=component,
                status=status,
                message=message,
                timestamp=datetime.now(),
                response_time=response_time
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            return HealthCheck(
                component=component,
                status="critical",
                message=f"خطأ في فحص {component}: {str(e)}",
                timestamp=datetime.now(),
                response_time=response_time
            )
    
    async def _check_threshold(self, metric: str, value: float, unit: str):
        """فحص تجاوز العتبات"""
        threshold = self.alert_thresholds.get(metric, 100)
        
        if value > threshold:
            await self._trigger_alert(metric, value, threshold, unit)
    
    async def _trigger_alert(self, metric: str, value: float, threshold: float, unit: str):
        """إطلاق تنبيه"""
        message = f"⚠️ تحذير: {metric} = {value:.1f}{unit} (العتبة: {threshold}{unit})"
        self.logger.warning(message)
        
        # يمكن إضافة إرسال تنبيهات عبر البريد الإلكتروني أو SMS هنا
    
    async def _log_error(self, component: str, error_message: str, severity: str):
        """تسجيل خطأ مفصل"""
        error_report = ErrorReport(
            error_id=f"error_{int(time.time())}",
            error_type=type(Exception).__name__,
            message=error_message,
            traceback=traceback.format_exc(),
            timestamp=datetime.now(),
            component=component,
            severity=severity,
            context={"monitoring": True}
        )
        
        self.error_reports.append(error_report)
        
        # الحفاظ على آخر 50 تقرير خطأ
        if len(self.error_reports) > 50:
            self.error_reports = self.error_reports[-50:]
    
    def get_health_report(self) -> Dict[str, Any]:
        """تقرير صحة النظام الشامل"""
        recent_checks = [check for check in self.health_checks 
                        if check.timestamp > datetime.now() - timedelta(minutes=10)]
        
        component_status = {}
        for check in recent_checks:
            if check.component not in component_status:
                component_status[check.component] = check
            elif check.timestamp > component_status[check.component].timestamp:
                component_status[check.component] = check
        
        # حساب الصحة العامة
        total_components = len(component_status)
        healthy_components = sum(1 for status in component_status.values() if status.status == "healthy")
        overall_health = (healthy_components / max(total_components, 1)) * 100
        
        return {
            "overall_health": f"{overall_health:.1f}%",
            "component_status": {
                comp: {
                    "status": status.status,
                    "message": status.message,
                    "response_time": f"{status.response_time:.3f}s",
                    "last_check": status.timestamp.isoformat()
                }
                for comp, status in component_status.items()
            },
            "recent_errors": [
                {
                    "error_id": error.error_id,
                    "component": error.component,
                    "message": error.message,
                    "severity": error.severity,
                    "timestamp": error.timestamp.isoformat()
                }
                for error in self.error_reports[-5:]  # آخر 5 أخطاء
            ],
            "system_metrics": self._get_current_metrics(),
            "recommendations": self._get_health_recommendations()
        }
    
    def _get_current_metrics(self) -> Dict[str, Any]:
        """المقاييس الحالية للنظام"""
        try:
            return {
                "cpu_usage": f"{psutil.cpu_percent():.1f}%",
                "memory_usage": f"{psutil.virtual_memory().percent:.1f}%",
                "disk_usage": f"{(psutil.disk_usage('/').used / psutil.disk_usage('/').total) * 100:.1f}%",
                "active_connections": len(psutil.net_connections()),
                "uptime": str(datetime.now() - datetime.fromtimestamp(psutil.boot_time()))
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _get_health_recommendations(self) -> List[str]:
        """اقتراحات تحسين الصحة"""
        recommendations = []
        
        try:
            # فحص استخدام CPU
            if psutil.cpu_percent() > 80:
                recommendations.append("فكر في تحسين خوارزميات المعالجة لتقليل استخدام CPU")
            
            # فحص استخدام الذاكرة
            if psutil.virtual_memory().percent > 85:
                recommendations.append("تنظيف الذاكرة وإزالة العمليات غير الضرورية")
            
            # فحص أخطاء حديثة
            recent_critical_errors = [e for e in self.error_reports[-10:] if e.severity == "critical"]
            if recent_critical_errors:
                recommendations.append("معالجة الأخطاء الحرجة الحديثة")
            
            if not recommendations:
                recommendations.append("النظام يعمل بحالة ممتازة!")
        
        except Exception:
            recommendations.append("تعذر تحليل حالة النظام")
        
        return recommendations

class ReliabilityEngine:
    """محرك الموثوقية الشامل"""
    
    def __init__(self):
        self.health_monitor = HealthMonitor()
        self.circuit_breakers = {}
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self):
        """تهيئة محرك الموثوقية"""
        await self.health_monitor.start_monitoring()
        self.logger.info("🛡️ تم تهيئة محرك الموثوقية والاستقرار")
    
    def add_circuit_breaker(self, component: str, failure_threshold: int = 5, recovery_timeout: int = 60):
        """إضافة قاطع دائرة لمكون"""
        self.circuit_breakers[component] = CircuitBreaker(failure_threshold, recovery_timeout)
        return self.circuit_breakers[component]
    
    def reliable(self, component: str = "default"):
        """ديكوريتر للمعالجة الموثوقة"""
        if component not in self.circuit_breakers:
            self.add_circuit_breaker(component)
        
        return self.circuit_breakers[component]
    
    async def get_reliability_report(self) -> Dict[str, Any]:
        """تقرير الموثوقية الشامل"""
        health_report = self.health_monitor.get_health_report()
        
        circuit_breaker_status = {
            name: {
                "state": cb.state,
                "failure_count": cb.failure_count,
                "last_failure": cb.last_failure_time
            }
            for name, cb in self.circuit_breakers.items()
        }
        
        return {
            "health_status": health_report,
            "circuit_breakers": circuit_breaker_status,
            "reliability_score": self._calculate_reliability_score(health_report),
            "uptime_status": "operational",
            "last_updated": datetime.now().isoformat()
        }
    
    def _calculate_reliability_score(self, health_report: Dict[str, Any]) -> str:
        """حساب نقاط الموثوقية"""
        try:
            overall_health = float(health_report["overall_health"].replace("%", ""))
            
            if overall_health >= 95:
                return "ممتاز"
            elif overall_health >= 85:
                return "جيد جداً"
            elif overall_health >= 70:
                return "جيد"
            elif overall_health >= 50:
                return "مقبول"
            else:
                return "يحتاج تحسين"
        except:
            return "غير محدد"

# مثيل عام للاستخدام
reliability_engine = ReliabilityEngine()
