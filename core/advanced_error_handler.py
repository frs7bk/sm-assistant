
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🛡️ نظام معالجة الأخطاء المتقدم للمساعد الذكي
Advanced Error Handling System for AI Assistant
"""

import sys
import traceback
import logging
import asyncio
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
from functools import wraps
import threading
from datetime import datetime

class ErrorSeverity(Enum):
    """مستويات خطورة الأخطاء"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """فئات الأخطاء"""
    SYSTEM = "system"
    AI_MODEL = "ai_model"
    API = "api"
    DATABASE = "database"
    NETWORK = "network"
    USER_INPUT = "user_input"
    CONFIGURATION = "configuration"
    PERMISSION = "permission"
    RESOURCE = "resource"
    UNKNOWN = "unknown"

@dataclass
class ErrorContext:
    """سياق معلومات الخطأ"""
    timestamp: float
    error_id: str
    category: ErrorCategory
    severity: ErrorSeverity
    module: str
    function: str
    message: str
    original_exception: str
    traceback_info: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_data: Optional[Dict[str, Any]] = None
    system_state: Optional[Dict[str, Any]] = None
    recovery_attempted: bool = False
    recovery_successful: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """تحويل إلى قاموس للحفظ"""
        data = asdict(self)
        data['category'] = self.category.value
        data['severity'] = self.severity.value
        data['datetime'] = datetime.fromtimestamp(self.timestamp).isoformat()
        return data

class AdvancedErrorHandler:
    """معالج الأخطاء المتقدم"""
    
    def __init__(self):
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # إعدادات المعالج
        self.error_history: List[ErrorContext] = []
        self.max_history_size = 1000
        self.recovery_strategies: Dict[ErrorCategory, List[Callable]] = {}
        self.notification_handlers: List[Callable] = []
        
        # معلومات النظام
        self.system_info = self._get_system_info()
        
        # إعداد معالجات الاسترداد
        self._setup_recovery_strategies()
        
        # مجلد حفظ الأخطاء
        self.error_logs_dir = Path("data/error_logs")
        self.error_logs_dir.mkdir(parents=True, exist_ok=True)
    
    def setup_logging(self):
        """إعداد نظام السجلات المتخصص"""
        # إنشاء logger مخصص للأخطاء
        error_logger = logging.getLogger('error_handler')
        error_logger.setLevel(logging.DEBUG)
        
        # تنسيق مخصص للسجلات
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
        )
        
        # معالج ملف السجلات
        file_handler = logging.FileHandler(
            'data/error_logs/error_handler.log',
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        error_logger.addHandler(file_handler)
        
        # معالج وحدة التحكم للأخطاء الحرجة
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(logging.ERROR)
        console_handler.setFormatter(formatter)
        error_logger.addHandler(console_handler)
    
    def _get_system_info(self) -> Dict[str, Any]:
        """الحصول على معلومات النظام"""
        try:
            import psutil
            import platform
            
            return {
                "platform": platform.platform(),
                "python_version": sys.version,
                "cpu_count": psutil.cpu_count(),
                "memory_total": psutil.virtual_memory().total,
                "disk_usage": psutil.disk_usage('/').free,
                "pid": os.getpid() if 'os' in sys.modules else "unknown"
            }
        except ImportError:
            return {
                "platform": "unknown",
                "python_version": sys.version,
                "limited_info": "psutil not available"
            }
    
    def _setup_recovery_strategies(self):
        """إعداد استراتيجيات الاسترداد"""
        self.recovery_strategies = {
            ErrorCategory.AI_MODEL: [
                self._retry_ai_operation,
                self._fallback_to_basic_model,
                self._use_cached_response
            ],
            ErrorCategory.API: [
                self._retry_api_call,
                self._use_backup_api,
                self._return_cached_data
            ],
            ErrorCategory.DATABASE: [
                self._retry_db_operation,
                self._use_backup_database,
                self._use_local_storage
            ],
            ErrorCategory.NETWORK: [
                self._retry_network_operation,
                self._use_offline_mode
            ],
            ErrorCategory.RESOURCE: [
                self._free_memory,
                self._restart_service,
                self._use_lite_mode
            ]
        }
    
    def capture_error(
        self,
        exception: Exception,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        module: str = "",
        function: str = "",
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        request_data: Optional[Dict[str, Any]] = None,
        attempt_recovery: bool = True
    ) -> ErrorContext:
        """التقاط وتسجيل خطأ"""
        
        # إنشاء معرف فريد للخطأ
        error_id = f"ERR_{int(time.time())}_{len(self.error_history)}"
        
        # الحصول على معلومات التتبع
        tb_info = traceback.format_exc()
        
        # إنشاء سياق الخطأ
        error_context = ErrorContext(
            timestamp=time.time(),
            error_id=error_id,
            category=category,
            severity=severity,
            module=module or self._get_calling_module(),
            function=function or self._get_calling_function(),
            message=str(exception),
            original_exception=exception.__class__.__name__,
            traceback_info=tb_info,
            user_id=user_id,
            session_id=session_id,
            request_data=request_data,
            system_state=self._get_current_system_state()
        )
        
        # تسجيل الخطأ
        self._log_error(error_context)
        
        # إضافة إلى التاريخ
        self._add_to_history(error_context)
        
        # محاولة الاسترداد
        if attempt_recovery:
            self._attempt_recovery(error_context)
        
        # إرسال إشعارات
        self._send_notifications(error_context)
        
        # حفظ تقرير مفصل للأخطاء الحرجة
        if severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            self._save_detailed_report(error_context)
        
        return error_context
    
    def _get_calling_module(self) -> str:
        """الحصول على اسم الوحدة المستدعية"""
        try:
            frame = sys._getframe(3)
            return frame.f_globals.get('__name__', 'unknown')
        except:
            return 'unknown'
    
    def _get_calling_function(self) -> str:
        """الحصول على اسم الدالة المستدعية"""
        try:
            frame = sys._getframe(3)
            return frame.f_code.co_name
        except:
            return 'unknown'
    
    def _get_current_system_state(self) -> Dict[str, Any]:
        """الحصول على حالة النظام الحالية"""
        try:
            import psutil
            return {
                "memory_percent": psutil.virtual_memory().percent,
                "cpu_percent": psutil.cpu_percent(),
                "active_threads": threading.active_count(),
                "open_files": len(psutil.Process().open_files()) if hasattr(psutil.Process(), 'open_files') else 0
            }
        except:
            return {"limited_monitoring": True}
    
    def _log_error(self, error_context: ErrorContext):
        """تسجيل الخطأ في نظام السجلات"""
        log_level_map = {
            ErrorSeverity.LOW: logging.INFO,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL
        }
        
        level = log_level_map.get(error_context.severity, logging.ERROR)
        
        message = (
            f"[{error_context.error_id}] {error_context.category.value.upper()} ERROR "
            f"in {error_context.module}.{error_context.function}: {error_context.message}"
        )
        
        self.logger.log(level, message)
        
        # تسجيل مفصل للأخطاء الحرجة
        if error_context.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(f"CRITICAL ERROR DETAILS:\n{error_context.traceback_info}")
    
    def _add_to_history(self, error_context: ErrorContext):
        """إضافة الخطأ إلى التاريخ"""
        self.error_history.append(error_context)
        
        # إزالة الأخطاء القديمة إذا تجاوزنا الحد الأقصى
        if len(self.error_history) > self.max_history_size:
            self.error_history = self.error_history[-self.max_history_size:]
    
    def _attempt_recovery(self, error_context: ErrorContext):
        """محاولة استرداد من الخطأ"""
        recovery_funcs = self.recovery_strategies.get(error_context.category, [])
        
        error_context.recovery_attempted = len(recovery_funcs) > 0
        
        for recovery_func in recovery_funcs:
            try:
                success = recovery_func(error_context)
                if success:
                    error_context.recovery_successful = True
                    self.logger.info(f"Recovery successful for error {error_context.error_id}")
                    break
            except Exception as recovery_error:
                self.logger.warning(f"Recovery strategy failed: {recovery_error}")
        
        if not error_context.recovery_successful and recovery_funcs:
            self.logger.warning(f"All recovery strategies failed for error {error_context.error_id}")
    
    def _send_notifications(self, error_context: ErrorContext):
        """إرسال إشعارات للمعالجات المسجلة"""
        for handler in self.notification_handlers:
            try:
                handler(error_context)
            except Exception as e:
                self.logger.warning(f"Notification handler failed: {e}")
    
    def _save_detailed_report(self, error_context: ErrorContext):
        """حفظ تقرير مفصل للخطأ"""
        try:
            report_file = self.error_logs_dir / f"error_report_{error_context.error_id}.json"
            
            report_data = error_context.to_dict()
            report_data.update({
                "system_info": self.system_info,
                "recent_errors": [
                    ctx.to_dict() for ctx in self.error_history[-10:]
                    if ctx.timestamp > time.time() - 3600  # آخر ساعة
                ]
            })
            
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Detailed error report saved: {report_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save error report: {e}")
    
    # استراتيجيات الاسترداد
    def _retry_ai_operation(self, error_context: ErrorContext) -> bool:
        """إعادة محاولة عملية الذكاء الاصطناعي"""
        # سيتم تطوير منطق الإعادة هنا
        return False
    
    def _fallback_to_basic_model(self, error_context: ErrorContext) -> bool:
        """التراجع إلى نموذج أساسي"""
        return False
    
    def _use_cached_response(self, error_context: ErrorContext) -> bool:
        """استخدام استجابة محفوظة مؤقتاً"""
        return False
    
    def _retry_api_call(self, error_context: ErrorContext) -> bool:
        """إعادة محاولة استدعاء API"""
        return False
    
    def _use_backup_api(self, error_context: ErrorContext) -> bool:
        """استخدام API احتياطي"""
        return False
    
    def _return_cached_data(self, error_context: ErrorContext) -> bool:
        """إرجاع بيانات محفوظة مؤقتاً"""
        return False
    
    def _retry_db_operation(self, error_context: ErrorContext) -> bool:
        """إعادة محاولة عملية قاعدة البيانات"""
        return False
    
    def _use_backup_database(self, error_context: ErrorContext) -> bool:
        """استخدام قاعدة بيانات احتياطية"""
        return False
    
    def _use_local_storage(self, error_context: ErrorContext) -> bool:
        """استخدام التخزين المحلي"""
        return False
    
    def _retry_network_operation(self, error_context: ErrorContext) -> bool:
        """إعادة محاولة عملية الشبكة"""
        return False
    
    def _use_offline_mode(self, error_context: ErrorContext) -> bool:
        """التبديل إلى الوضع غير المتصل"""
        return False
    
    def _free_memory(self, error_context: ErrorContext) -> bool:
        """تحرير الذاكرة"""
        try:
            import gc
            gc.collect()
            return True
        except:
            return False
    
    def _restart_service(self, error_context: ErrorContext) -> bool:
        """إعادة تشغيل الخدمة"""
        return False
    
    def _use_lite_mode(self, error_context: ErrorContext) -> bool:
        """التبديل إلى الوضع المبسط"""
        return False
    
    # واجهات برمجية للاستخدام
    def add_notification_handler(self, handler: Callable[[ErrorContext], None]):
        """إضافة معالج إشعارات"""
        self.notification_handlers.append(handler)
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """الحصول على إحصائيات الأخطاء"""
        if not self.error_history:
            return {"total_errors": 0}
        
        # تحليل البيانات
        total_errors = len(self.error_history)
        recent_errors = [
            err for err in self.error_history
            if err.timestamp > time.time() - 3600  # آخر ساعة
        ]
        
        # تصنيف حسب الفئة
        category_counts = {}
        for err in self.error_history:
            category_counts[err.category.value] = category_counts.get(err.category.value, 0) + 1
        
        # تصنيف حسب الخطورة
        severity_counts = {}
        for err in self.error_history:
            severity_counts[err.severity.value] = severity_counts.get(err.severity.value, 0) + 1
        
        # معدل الاسترداد
        recovery_attempts = sum(1 for err in self.error_history if err.recovery_attempted)
        successful_recoveries = sum(1 for err in self.error_history if err.recovery_successful)
        recovery_rate = (successful_recoveries / recovery_attempts * 100) if recovery_attempts > 0 else 0
        
        return {
            "total_errors": total_errors,
            "recent_errors_count": len(recent_errors),
            "category_distribution": category_counts,
            "severity_distribution": severity_counts,
            "recovery_rate": f"{recovery_rate:.1f}%",
            "most_common_category": max(category_counts, key=category_counts.get) if category_counts else None,
            "last_error_time": datetime.fromtimestamp(self.error_history[-1].timestamp).isoformat() if self.error_history else None
        }
    
    def get_recent_errors(self, hours: int = 24) -> List[ErrorContext]:
        """الحصول على الأخطاء الحديثة"""
        cutoff_time = time.time() - (hours * 3600)
        return [err for err in self.error_history if err.timestamp > cutoff_time]
    
    def clear_error_history(self):
        """مسح تاريخ الأخطاء"""
        self.error_history.clear()
        self.logger.info("Error history cleared")

# ديكوريتر لمعالجة الأخطاء التلقائية
def handle_errors(
    category: ErrorCategory = ErrorCategory.UNKNOWN,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    attempt_recovery: bool = True,
    reraise: bool = False
):
    """ديكوريتر لمعالجة الأخطاء التلقائية"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                error_handler.capture_error(
                    exception=e,
                    category=category,
                    severity=severity,
                    module=func.__module__,
                    function=func.__name__,
                    attempt_recovery=attempt_recovery
                )
                if reraise:
                    raise
                return None
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_handler.capture_error(
                    exception=e,
                    category=category,
                    severity=severity,
                    module=func.__module__,
                    function=func.__name__,
                    attempt_recovery=attempt_recovery
                )
                if reraise:
                    raise
                return None
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

# مثيل عالمي للمعالج
error_handler = AdvancedErrorHandler()

# دوال مساعدة للاستخدام السريع
def capture_system_error(exception: Exception, **kwargs):
    """التقاط خطأ في النظام"""
    return error_handler.capture_error(exception, ErrorCategory.SYSTEM, **kwargs)

def capture_ai_error(exception: Exception, **kwargs):
    """التقاط خطأ في الذكاء الاصطناعي"""
    return error_handler.capture_error(exception, ErrorCategory.AI_MODEL, **kwargs)

def capture_api_error(exception: Exception, **kwargs):
    """التقاط خطأ في API"""
    return error_handler.capture_error(exception, ErrorCategory.API, **kwargs)

def capture_critical_error(exception: Exception, **kwargs):
    """التقاط خطأ حرج"""
    return error_handler.capture_error(exception, severity=ErrorSeverity.CRITICAL, **kwargs)
