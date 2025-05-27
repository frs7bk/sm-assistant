
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
نظام التكوين المتقدم للمساعد الذكي
Advanced Configuration System
"""

import os
import json
import yaml
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime
import logging
from enum import Enum
import hashlib

from config.settings import settings

class PerformanceLevel(str, Enum):
    """مستويات الأداء"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAXIMUM = "maximum"

class SecurityLevel(str, Enum):
    """مستويات الأمان"""
    BASIC = "basic"
    STANDARD = "standard"
    HIGH = "high"
    MILITARY = "military"

class UITheme(str, Enum):
    """سمات الواجهة"""
    LIGHT = "light"
    DARK = "dark"
    AUTO = "auto"
    CUSTOM = "custom"

@dataclass
class AIModelConfig:
    """تكوين نماذج الذكاء الاصطناعي"""
    primary_model: str = "gpt-4"
    fallback_model: str = "gpt-3.5-turbo"
    max_tokens: int = 4000
    temperature: float = 0.7
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    timeout_seconds: int = 30
    retry_attempts: int = 3
    
    # إعدادات متقدمة
    use_function_calling: bool = True
    enable_streaming: bool = True
    custom_prompts: Dict[str, str] = field(default_factory=dict)
    model_weights: Dict[str, float] = field(default_factory=lambda: {"accuracy": 0.4, "speed": 0.3, "cost": 0.3})

@dataclass
class PerformanceConfig:
    """تكوين الأداء"""
    level: PerformanceLevel = PerformanceLevel.HIGH
    max_memory_gb: float = 8.0
    max_cpu_cores: int = 4
    enable_gpu: bool = True
    parallel_processing: bool = True
    cache_size_mb: int = 1000
    
    # تحسينات متقدمة
    enable_lazy_loading: bool = True
    optimize_memory: bool = True
    use_compression: bool = True
    background_processing: bool = True
    priority_queue: bool = True

@dataclass
class SecurityConfig:
    """تكوين الأمان"""
    level: SecurityLevel = SecurityLevel.HIGH
    enable_encryption: bool = True
    encryption_algorithm: str = "AES-256"
    enable_rate_limiting: bool = True
    max_requests_per_minute: int = 60
    
    # أمان متقدم
    enable_biometric_auth: bool = False
    require_2fa: bool = False
    session_timeout_minutes: int = 60
    audit_logging: bool = True
    data_anonymization: bool = True
    quantum_encryption: bool = False

@dataclass
class UIConfig:
    """تكوين الواجهة"""
    theme: UITheme = UITheme.AUTO
    language: str = "ar"
    font_size: int = 14
    animation_speed: str = "normal"
    
    # تخصيص متقدم
    custom_colors: Dict[str, str] = field(default_factory=dict)
    layout_mode: str = "adaptive"
    accessibility_mode: bool = False
    voice_interface: bool = True
    gesture_control: bool = False

@dataclass
class FeatureFlags:
    """أعلام الميزات"""
    # ميزات أساسية
    enable_learning: bool = True
    enable_prediction: bool = True
    enable_vision: bool = True
    enable_voice: bool = True
    enable_text_analysis: bool = True
    
    # ميزات متقدمة
    enable_ar_vr: bool = False
    enable_iot: bool = False
    enable_quantum_features: bool = False
    enable_blockchain: bool = False
    enable_edge_computing: bool = False
    
    # تطبيقات متخصصة
    enable_financial_advisor: bool = True
    enable_health_monitor: bool = True
    enable_gaming_coach: bool = True
    enable_creative_ai: bool = True
    enable_project_manager: bool = True
    enable_social_intelligence: bool = True
    
    # ميزات تجريبية
    enable_experimental: bool = False
    enable_beta_features: bool = False
    enable_research_mode: bool = False

@dataclass
class IntegrationConfig:
    """تكوين التكاملات"""
    # خدمات سحابية
    aws_integration: bool = False
    azure_integration: bool = False
    gcp_integration: bool = False
    
    # قواعد البيانات
    enable_postgresql: bool = False
    enable_mongodb: bool = False
    enable_redis: bool = True
    enable_elasticsearch: bool = False
    
    # أدوات خارجية
    enable_slack: bool = False
    enable_discord: bool = False
    enable_telegram: bool = False
    enable_whatsapp: bool = False
    
    # APIs خارجية
    weather_api: bool = True
    news_api: bool = True
    translation_api: bool = True
    maps_api: bool = False

@dataclass
class LoggingConfig:
    """تكوين السجلات"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: str = "data/logs/assistant.log"
    max_file_size_mb: int = 100
    backup_count: int = 5
    
    # سجلات متقدمة
    enable_structured_logging: bool = True
    enable_metrics_logging: bool = True
    enable_audit_trail: bool = True
    log_retention_days: int = 30
    enable_real_time_monitoring: bool = True

@dataclass
class AdvancedConfig:
    """التكوين المتقدم الشامل"""
    # معلومات أساسية
    version: str = "3.0.0"
    environment: str = "production"
    deployment_id: str = field(default_factory=lambda: hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8])
    
    # تكوينات فرعية
    ai_models: AIModelConfig = field(default_factory=AIModelConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    features: FeatureFlags = field(default_factory=FeatureFlags)
    integrations: IntegrationConfig = field(default_factory=IntegrationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # إعدادات متقدمة
    auto_update: bool = True
    telemetry_enabled: bool = True
    debug_mode: bool = False
    maintenance_mode: bool = False
    
    # تخصيص المطور
    custom_settings: Dict[str, Any] = field(default_factory=dict)
    plugin_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def __post_init__(self):
        """تهيئة ما بعد الإنشاء"""
        # تطبيق إعدادات من متغيرات البيئة
        self._apply_environment_overrides()
        
        # تحسين الإعدادات بناءً على الأجهزة
        self._optimize_for_hardware()
        
        # التحقق من صحة الإعدادات
        self._validate_config()
    
    def _apply_environment_overrides(self):
        """تطبيق تجاوزات متغيرات البيئة"""
        # مستوى الأداء
        if hasattr(settings, 'PERFORMANCE_LEVEL'):
            try:
                self.performance.level = PerformanceLevel(settings.PERFORMANCE_LEVEL)
            except ValueError:
                logging.warning(f"مستوى أداء غير صحيح: {settings.PERFORMANCE_LEVEL}")
        
        # مستوى الأمان
        if hasattr(settings, 'SECURITY_LEVEL'):
            try:
                self.security.level = SecurityLevel(settings.SECURITY_LEVEL)
            except ValueError:
                logging.warning(f"مستوى أمان غير صحيح: {settings.SECURITY_LEVEL}")
        
        # وضع التطوير
        if hasattr(settings, 'DEBUG_MODE'):
            self.debug_mode = settings.DEBUG_MODE
            
        # ميزات من الإعدادات
        feature_flags = settings.get_feature_flags()
        for flag, value in feature_flags.items():
            if hasattr(self.features, f"enable_{flag}"):
                setattr(self.features, f"enable_{flag}", value)
    
    def _optimize_for_hardware(self):
        """تحسين الإعدادات للأجهزة المتاحة"""
        import psutil
        
        # تحسين الذاكرة
        available_memory = psutil.virtual_memory().total / (1024**3)  # GB
        self.performance.max_memory_gb = min(self.performance.max_memory_gb, available_memory * 0.8)
        
        # تحسين المعالج
        cpu_count = psutil.cpu_count()
        self.performance.max_cpu_cores = min(self.performance.max_cpu_cores, cpu_count)
        
        # تحسين GPU
        try:
            import torch
            self.performance.enable_gpu = torch.cuda.is_available()
        except ImportError:
            self.performance.enable_gpu = False
        
        # تحسين التخزين المؤقت
        if available_memory < 4:
            self.performance.cache_size_mb = 500
        elif available_memory > 16:
            self.performance.cache_size_mb = 2000
    
    def _validate_config(self):
        """التحقق من صحة الإعدادات"""
        errors = []
        
        # التحقق من إعدادات الذكاء الاصطناعي
        if self.ai_models.max_tokens < 100:
            errors.append("max_tokens يجب أن يكون أكبر من 100")
        
        if not 0 <= self.ai_models.temperature <= 2:
            errors.append("temperature يجب أن يكون بين 0 و 2")
        
        # التحقق من إعدادات الأداء
        if self.performance.max_memory_gb < 1:
            errors.append("max_memory_gb يجب أن يكون أكبر من 1")
        
        # التحقق من إعدادات الأمان
        if self.security.max_requests_per_minute < 1:
            errors.append("max_requests_per_minute يجب أن يكون أكبر من 1")
        
        if errors:
            error_msg = "أخطاء في التكوين: " + "; ".join(errors)
            logging.error(error_msg)
            raise ValueError(error_msg)
    
    def to_dict(self) -> Dict[str, Any]:
        """تحويل لقاموس"""
        return asdict(self)
    
    def to_json(self) -> str:
        """تحويل لـ JSON"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
    
    def to_yaml(self) -> str:
        """تحويل لـ YAML"""
        return yaml.dump(self.to_dict(), allow_unicode=True, default_flow_style=False)
    
    def save_to_file(self, file_path: Union[str, Path], format: str = "json"):
        """حفظ في ملف"""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == "json":
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(self.to_json())
        elif format.lower() == "yaml":
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(self.to_yaml())
        else:
            raise ValueError(f"تنسيق غير مدعوم: {format}")
    
    @classmethod
    def load_from_file(cls, file_path: Union[str, Path]) -> 'AdvancedConfig':
        """تحميل من ملف"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"ملف التكوين غير موجود: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.suffix.lower() == '.json':
                data = json.load(f)
            elif file_path.suffix.lower() in ['.yaml', '.yml']:
                data = yaml.safe_load(f)
            else:
                raise ValueError(f"تنسيق ملف غير مدعوم: {file_path.suffix}")
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AdvancedConfig':
        """إنشاء من قاموس"""
        # إنشاء التكوينات الفرعية
        ai_models = AIModelConfig(**data.get('ai_models', {}))
        performance = PerformanceConfig(**data.get('performance', {}))
        security = SecurityConfig(**data.get('security', {}))
        ui = UIConfig(**data.get('ui', {}))
        features = FeatureFlags(**data.get('features', {}))
        integrations = IntegrationConfig(**data.get('integrations', {}))
        logging_config = LoggingConfig(**data.get('logging', {}))
        
        # إزالة التكوينات الفرعية من البيانات الرئيسية
        main_data = {k: v for k, v in data.items() 
                    if k not in ['ai_models', 'performance', 'security', 'ui', 
                               'features', 'integrations', 'logging']}
        
        return cls(
            ai_models=ai_models,
            performance=performance,
            security=security,
            ui=ui,
            features=features,
            integrations=integrations,
            logging=logging_config,
            **main_data
        )
    
    def get_feature_config(self, feature_name: str) -> bool:
        """الحصول على تكوين ميزة معينة"""
        return getattr(self.features, f"enable_{feature_name}", False)
    
    def enable_feature(self, feature_name: str):
        """تفعيل ميزة"""
        setattr(self.features, f"enable_{feature_name}", True)
    
    def disable_feature(self, feature_name: str):
        """إلغاء تفعيل ميزة"""
        setattr(self.features, f"enable_{feature_name}", False)
    
    def get_performance_profile(self) -> Dict[str, Any]:
        """الحصول على ملف الأداء"""
        return {
            "level": self.performance.level.value,
            "memory_limit": f"{self.performance.max_memory_gb}GB",
            "cpu_cores": self.performance.max_cpu_cores,
            "gpu_enabled": self.performance.enable_gpu,
            "parallel_processing": self.performance.parallel_processing,
            "cache_size": f"{self.performance.cache_size_mb}MB"
        }
    
    def get_security_profile(self) -> Dict[str, Any]:
        """الحصول على ملف الأمان"""
        return {
            "level": self.security.level.value,
            "encryption": self.security.enable_encryption,
            "rate_limiting": self.security.enable_rate_limiting,
            "max_requests": self.security.max_requests_per_minute,
            "session_timeout": self.security.session_timeout_minutes,
            "audit_logging": self.security.audit_logging
        }
    
    def update_config(self, updates: Dict[str, Any]):
        """تحديث التكوين"""
        for key, value in updates.items():
            if hasattr(self, key):
                setattr(self, key, value)
            elif '.' in key:
                # دعم التحديث المتداخل مثل "ai_models.temperature"
                parts = key.split('.')
                obj = self
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, parts[-1], value)
    
    def create_profile(self, profile_name: str) -> Dict[str, Any]:
        """إنشاء ملف تعريف مخصص"""
        if profile_name == "development":
            return {
                "debug_mode": True,
                "performance.level": "medium",
                "security.level": "basic",
                "features.enable_experimental": True,
                "logging.level": "DEBUG"
            }
        elif profile_name == "production":
            return {
                "debug_mode": False,
                "performance.level": "high",
                "security.level": "high",
                "features.enable_experimental": False,
                "logging.level": "INFO"
            }
        elif profile_name == "testing":
            return {
                "debug_mode": True,
                "performance.level": "low",
                "security.level": "basic",
                "features.enable_experimental": True,
                "logging.level": "DEBUG"
            }
        else:
            return {}

# إنشاء مثيل التكوين المتقدم
try:
    # محاولة تحميل تكوين مخصص
    config_file = Path("config/advanced_config.json")
    if config_file.exists():
        advanced_config = AdvancedConfig.load_from_file(config_file)
        logging.info("تم تحميل التكوين المتقدم من الملف")
    else:
        advanced_config = AdvancedConfig()
        # حفظ التكوين الافتراضي
        advanced_config.save_to_file(config_file)
        logging.info("تم إنشاء تكوين افتراضي متقدم")

except Exception as e:
    logging.error(f"خطأ في تحميل التكوين المتقدم: {e}")
    advanced_config = AdvancedConfig()

# دوال مساعدة للوصول السريع
def get_ai_config() -> AIModelConfig:
    """الحصول على تكوين الذكاء الاصطناعي"""
    return advanced_config.ai_models

def get_performance_config() -> PerformanceConfig:
    """الحصول على تكوين الأداء"""
    return advanced_config.performance

def get_security_config() -> SecurityConfig:
    """الحصول على تكوين الأمان"""
    return advanced_config.security

def get_ui_config() -> UIConfig:
    """الحصول على تكوين الواجهة"""
    return advanced_config.ui

def get_feature_flags() -> FeatureFlags:
    """الحصول على أعلام الميزات"""
    return advanced_config.features

def is_feature_enabled(feature_name: str) -> bool:
    """فحص ما إذا كانت الميزة مفعلة"""
    return advanced_config.get_feature_config(feature_name)

def update_config(updates: Dict[str, Any]):
    """تحديث التكوين"""
    advanced_config.update_config(updates)

def save_config():
    """حفظ التكوين الحالي"""
    config_file = Path("config/advanced_config.json")
    advanced_config.save_to_file(config_file)

# تصدير المتغيرات المهمة
__all__ = [
    'AdvancedConfig', 'AIModelConfig', 'PerformanceConfig', 'SecurityConfig',
    'UIConfig', 'FeatureFlags', 'IntegrationConfig', 'LoggingConfig',
    'advanced_config', 'get_ai_config', 'get_performance_config',
    'get_security_config', 'get_ui_config', 'get_feature_flags',
    'is_feature_enabled', 'update_config', 'save_config'
]

if __name__ == "__main__":
    # اختبار التكوين
    print("🔧 اختبار نظام التكوين المتقدم")
    print("=" * 50)
    
    # عرض التكوين الحالي
    print("📋 التكوين الحالي:")
    print(f"- إصدار: {advanced_config.version}")
    print(f"- بيئة: {advanced_config.environment}")
    print(f"- معرف النشر: {advanced_config.deployment_id}")
    
    # عرض ملف الأداء
    performance_profile = advanced_config.get_performance_profile()
    print(f"\n⚡ ملف الأداء:")
    for key, value in performance_profile.items():
        print(f"  - {key}: {value}")
    
    # عرض ملف الأمان  
    security_profile = advanced_config.get_security_profile()
    print(f"\n🔒 ملف الأمان:")
    for key, value in security_profile.items():
        print(f"  - {key}: {value}")
    
    # عرض الميزات المفعلة
    enabled_features = []
    for attr in dir(advanced_config.features):
        if attr.startswith('enable_') and getattr(advanced_config.features, attr):
            enabled_features.append(attr[7:])  # إزالة 'enable_'
    
    print(f"\n🎯 الميزات المفعلة ({len(enabled_features)}):")
    for feature in enabled_features[:10]:  # أول 10 ميزات
        print(f"  ✅ {feature}")
    
    if len(enabled_features) > 10:
        print(f"  ... و {len(enabled_features) - 10} ميزة أخرى")
    
    print("\n✅ تم الاختبار بنجاح!")
