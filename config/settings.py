
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
نظام الإعدادات المتقدم للمساعد الذكي
Advanced Settings System for Smart Assistant
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, validator, root_validator
from typing import Literal, Optional, Dict, List, Union, Any
from pathlib import Path
import os
import logging
from enum import Enum
import psutil
import json

class LogLevel(str, Enum):
    """مستويات السجلات المدعومة"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class AIModel(str, Enum):
    """نماذج الذكاء الاصطناعي المدعومة"""
    GPT4 = "gpt-4"
    GPT4_TURBO = "gpt-4-turbo"
    GPT3_5_TURBO = "gpt-3.5-turbo"
    CLAUDE_3 = "claude-3-opus"
    CLAUDE_3_SONNET = "claude-3-sonnet"
    GEMINI_PRO = "gemini-pro"
    GEMINI_ULTRA = "gemini-ultra"

class DatabaseType(str, Enum):
    """أنواع قواعد البيانات المدعومة"""
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    MONGODB = "mongodb"
    REDIS = "redis"

class PerformanceLevel(str, Enum):
    """مستويات الأداء"""
    ECO = "eco"           # توفير الطاقة
    BALANCED = "balanced" # متوازن
    PERFORMANCE = "performance" # أداء عالي
    EXTREME = "extreme"   # أقصى أداء

class SecurityLevel(str, Enum):
    """مستويات الأمان"""
    BASIC = "basic"
    STANDARD = "standard"
    HIGH = "high"
    ENTERPRISE = "enterprise"
    GOVERNMENT = "government"

class Settings(BaseSettings):
    """
    إعدادات المساعد الذكي الشاملة
    Comprehensive Smart Assistant Settings
    """
    
    # === معلومات التطبيق الأساسية ===
    APP_NAME: str = Field(default="المساعد الذكي المتقدم", description="اسم التطبيق")
    APP_VERSION: str = Field(default="3.1.0", description="إصدار التطبيق")
    APP_ENVIRONMENT: Literal["development", "staging", "production"] = Field(default="development")
    APP_DEBUG: bool = Field(default=False, description="وضع التطوير")
    
    # === الإعدادات اللغوية والإقليمية ===
    DEFAULT_LANGUAGE: str = Field(default="ar", regex="^(ar|en|fr|es|de|zh|ja|ru)$")
    SUPPORTED_LANGUAGES: List[str] = Field(default=["ar", "en", "fr", "es"])
    TIMEZONE: str = Field(default="Asia/Riyadh", description="المنطقة الزمنية")
    DATE_FORMAT: str = Field(default="%Y-%m-%d", description="تنسيق التاريخ")
    TIME_FORMAT: str = Field(default="%H:%M:%S", description="تنسيق الوقت")
    
    # === مفاتيح API المطلوبة ===
    OPENAI_API_KEY: str = Field(..., min_length=20, description="مفتاح OpenAI (مطلوب)")
    
    # === مفاتيح API الاختيارية ===
    ANTHROPIC_API_KEY: Optional[str] = Field(default=None, min_length=10)
    GOOGLE_API_KEY: Optional[str] = Field(default=None, min_length=10)
    AZURE_OPENAI_KEY: Optional[str] = Field(default=None, min_length=10)
    HUGGINGFACE_API_KEY: Optional[str] = Field(default=None, min_length=10)
    REPLICATE_API_KEY: Optional[str] = Field(default=None, min_length=10)
    STABILITY_API_KEY: Optional[str] = Field(default=None, min_length=10)
    
    # === إعدادات نماذج الذكاء الاصطناعي ===
    PRIMARY_AI_MODEL: AIModel = Field(default=AIModel.GPT4)
    FALLBACK_AI_MODEL: AIModel = Field(default=AIModel.GPT3_5_TURBO)
    MAX_TOKENS: int = Field(default=4000, ge=100, le=32000)
    TEMPERATURE: float = Field(default=0.7, ge=0.0, le=2.0)
    TOP_P: float = Field(default=0.9, ge=0.0, le=1.0)
    FREQUENCY_PENALTY: float = Field(default=0.0, ge=-2.0, le=2.0)
    PRESENCE_PENALTY: float = Field(default=0.0, ge=-2.0, le=2.0)
    
    # === إعدادات قاعدة البيانات ===
    DATABASE_TYPE: DatabaseType = Field(default=DatabaseType.SQLITE)
    DATABASE_URL: str = Field(default="sqlite:///data/assistant.db")
    DATABASE_POOL_SIZE: int = Field(default=10, ge=1, le=100)
    DATABASE_MAX_OVERFLOW: int = Field(default=20, ge=0, le=100)
    DATABASE_ECHO: bool = Field(default=False, description="عرض استعلامات SQL")
    
    # === إعدادات Redis (التخزين المؤقت) ===
    REDIS_URL: str = Field(default="redis://localhost:6379/0")
    REDIS_PASSWORD: Optional[str] = Field(default=None)
    REDIS_MAX_CONNECTIONS: int = Field(default=20, ge=1, le=100)
    
    # === إعدادات السجلات المتقدمة ===
    LOG_LEVEL: LogLevel = Field(default=LogLevel.INFO)
    LOG_FILE_PATH: Path = Field(default=Path("data/logs/assistant.log"))
    LOG_MAX_SIZE: str = Field(default="100MB", description="الحد الأقصى لحجم ملف السجل")
    LOG_BACKUP_COUNT: int = Field(default=5, ge=1, le=20)
    LOG_FORMAT: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ENABLE_STRUCTURED_LOGGING: bool = Field(default=True)
    ENABLE_LOG_COMPRESSION: bool = Field(default=True)
    
    # === إعدادات الأداء ===
    PERFORMANCE_LEVEL: PerformanceLevel = Field(default=PerformanceLevel.BALANCED)
    MAX_MEMORY_GB: float = Field(default=8.0, ge=1.0, le=64.0)
    MAX_CPU_CORES: int = Field(default=4, ge=1, le=32)
    ENABLE_GPU: bool = Field(default=True)
    ENABLE_PARALLEL_PROCESSING: bool = Field(default=True)
    CACHE_SIZE_MB: int = Field(default=1000, ge=10, le=10000)
    REQUEST_TIMEOUT_SECONDS: int = Field(default=30, ge=5, le=300)
    
    # === إعدادات الأمان ===
    SECURITY_LEVEL: SecurityLevel = Field(default=SecurityLevel.HIGH)
    ENABLE_ENCRYPTION: bool = Field(default=True)
    ENCRYPTION_ALGORITHM: str = Field(default="AES-256")
    SECRET_KEY: Optional[str] = Field(default=None, min_length=32)
    ENABLE_RATE_LIMITING: bool = Field(default=True)
    MAX_REQUESTS_PER_MINUTE: int = Field(default=60, ge=1, le=1000)
    ENABLE_API_KEY_ROTATION: bool = Field(default=False)
    SESSION_TIMEOUT_MINUTES: int = Field(default=60, ge=5, le=1440)
    
    # === إعدادات الشبكة ===
    HOST: str = Field(default="0.0.0.0", description="عنوان الخادم")
    PORT: int = Field(default=5000, ge=1000, le=65535)
    WORKERS: int = Field(default=1, ge=1, le=10)
    ENABLE_CORS: bool = Field(default=True)
    ALLOWED_ORIGINS: List[str] = Field(default=["*"])
    ENABLE_HTTPS: bool = Field(default=False)
    SSL_CERT_PATH: Optional[str] = Field(default=None)
    SSL_KEY_PATH: Optional[str] = Field(default=None)
    
    # === الميزات المتقدمة ===
    ENABLE_LEARNING: bool = Field(default=True, description="تفعيل التعلم المستمر")
    ENABLE_PREDICTION: bool = Field(default=True, description="تفعيل التنبؤات الذكية")
    ENABLE_VISION: bool = Field(default=True, description="تفعيل الرؤية الحاسوبية")
    ENABLE_VOICE: bool = Field(default=True, description="تفعيل المعالجة الصوتية")
    ENABLE_MULTILINGUAL: bool = Field(default=True, description="الدعم متعدد اللغات")
    ENABLE_EMOTIONAL_AI: bool = Field(default=True, description="الذكاء العاطفي")
    
    # === الميزات التجريبية ===
    ENABLE_AR_VR: bool = Field(default=False, description="الواقع المعزز والافتراضي")
    ENABLE_QUANTUM_FEATURES: bool = Field(default=False, description="المعالجة الكمومية")
    ENABLE_BLOCKCHAIN: bool = Field(default=False, description="تقنية البلوك تشين")
    ENABLE_IOT_INTEGRATION: bool = Field(default=False, description="إنترنت الأشياء")
    ENABLE_EDGE_COMPUTING: bool = Field(default=False, description="الحوسبة الطرفية")
    
    # === إعدادات التطبيقات المتخصصة ===
    ENABLE_FINANCIAL_ADVISOR: bool = Field(default=True)
    ENABLE_HEALTH_MONITOR: bool = Field(default=True)
    ENABLE_EDUCATION_TUTOR: bool = Field(default=True)
    ENABLE_CREATIVE_ASSISTANT: bool = Field(default=True)
    ENABLE_BUSINESS_INTELLIGENCE: bool = Field(default=True)
    ENABLE_SCIENTIFIC_RESEARCH: bool = Field(default=False)
    
    # === إعدادات البيانات الضخمة ===
    ENABLE_BIG_DATA_PROCESSING: bool = Field(default=True)
    MAX_DATASET_SIZE_GB: float = Field(default=10.0, ge=0.1, le=1000.0)
    ENABLE_DISTRIBUTED_COMPUTING: bool = Field(default=False)
    SPARK_MASTER_URL: Optional[str] = Field(default="local[*]")
    DASK_SCHEDULER_ADDRESS: Optional[str] = Field(default=None)
    
    # === إعدادات المراقبة والتحليلات ===
    ENABLE_ANALYTICS: bool = Field(default=True)
    ENABLE_PERFORMANCE_MONITORING: bool = Field(default=True)
    ENABLE_ERROR_TRACKING: bool = Field(default=True)
    ENABLE_USER_BEHAVIOR_ANALYSIS: bool = Field(default=True)
    ANALYTICS_RETENTION_DAYS: int = Field(default=90, ge=1, le=365)
    
    # === إعدادات النسخ الاحتياطي ===
    ENABLE_AUTO_BACKUP: bool = Field(default=True)
    BACKUP_INTERVAL_HOURS: int = Field(default=24, ge=1, le=168)
    BACKUP_RETENTION_DAYS: int = Field(default=30, ge=1, le=365)
    BACKUP_STORAGE_PATH: Path = Field(default=Path("data/backups"))
    ENABLE_CLOUD_BACKUP: bool = Field(default=False)
    
    # === إعدادات التخصيص ===
    CUSTOM_BRANDING: Dict[str, str] = Field(default_factory=dict)
    CUSTOM_THEMES: Dict[str, Dict[str, str]] = Field(default_factory=dict)
    PLUGIN_DIRECTORIES: List[str] = Field(default_factory=lambda: ["plugins"])
    EXTENSION_CONFIGS: Dict[str, Any] = Field(default_factory=dict)
    
    # === تكوين Pydantic ===
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding='utf-8',
        case_sensitive=True,
        extra='forbid',
        validate_assignment=True,
        use_enum_values=True
    )
    
    @validator('LOG_FILE_PATH')
    def validate_log_path(cls, v):
        """التحقق من صحة مسار السجلات وإنشاؤه إذا لم يكن موجوداً"""
        log_dir = v.parent
        log_dir.mkdir(parents=True, exist_ok=True)
        return v
    
    @validator('BACKUP_STORAGE_PATH')
    def validate_backup_path(cls, v):
        """التحقق من صحة مسار النسخ الاحتياطية"""
        v.mkdir(parents=True, exist_ok=True)
        return v
    
    @validator('DATABASE_URL')
    def validate_database_url(cls, v, values):
        """التحقق من صحة رابط قاعدة البيانات"""
        db_type = values.get('DATABASE_TYPE')
        if db_type == DatabaseType.SQLITE and not v.startswith('sqlite'):
            raise ValueError("SQLite URL يجب أن يبدأ بـ sqlite://")
        elif db_type == DatabaseType.POSTGRESQL and not v.startswith('postgresql'):
            raise ValueError("PostgreSQL URL يجب أن يبدأ بـ postgresql://")
        elif db_type == DatabaseType.MYSQL and not v.startswith('mysql'):
            raise ValueError("MySQL URL يجب أن يبدأ بـ mysql://")
        return v
    
    @validator('FALLBACK_AI_MODEL')
    def validate_fallback_model(cls, v, values):
        """التحقق من أن النموذج الاحتياطي مختلف عن الأساسي"""
        primary = values.get('PRIMARY_AI_MODEL')
        if v == primary:
            raise ValueError("النموذج الاحتياطي يجب أن يكون مختلفاً عن الأساسي")
        return v
    
    @root_validator
    def validate_hardware_constraints(cls, values):
        """تحسين الإعدادات بناءً على قدرات الأجهزة"""
        # الحصول على معلومات النظام
        available_memory = psutil.virtual_memory().total / (1024**3)  # GB
        cpu_count = psutil.cpu_count()
        
        # تحسين استخدام الذاكرة
        max_memory = values.get('MAX_MEMORY_GB', 8.0)
        if max_memory > available_memory * 0.8:
            values['MAX_MEMORY_GB'] = available_memory * 0.8
            logging.warning(f"تم تقليل استخدام الذاكرة إلى {values['MAX_MEMORY_GB']:.1f}GB")
        
        # تحسين استخدام المعالج
        max_cores = values.get('MAX_CPU_CORES', 4)
        if max_cores > cpu_count:
            values['MAX_CPU_CORES'] = cpu_count
            logging.warning(f"تم تقليل عدد المعالجات إلى {cpu_count}")
        
        return values
    
    def get_feature_flags(self) -> Dict[str, bool]:
        """الحصول على أعلام الميزات المفعلة"""
        return {
            "learning": self.ENABLE_LEARNING,
            "prediction": self.ENABLE_PREDICTION,
            "vision": self.ENABLE_VISION,
            "voice": self.ENABLE_VOICE,
            "multilingual": self.ENABLE_MULTILINGUAL,
            "emotional_ai": self.ENABLE_EMOTIONAL_AI,
            "ar_vr": self.ENABLE_AR_VR,
            "quantum": self.ENABLE_QUANTUM_FEATURES,
            "blockchain": self.ENABLE_BLOCKCHAIN,
            "iot": self.ENABLE_IOT_INTEGRATION,
            "edge_computing": self.ENABLE_EDGE_COMPUTING,
            "financial_advisor": self.ENABLE_FINANCIAL_ADVISOR,
            "health_monitor": self.ENABLE_HEALTH_MONITOR,
            "education": self.ENABLE_EDUCATION_TUTOR,
            "creative": self.ENABLE_CREATIVE_ASSISTANT,
            "business_intelligence": self.ENABLE_BUSINESS_INTELLIGENCE,
            "scientific_research": self.ENABLE_SCIENTIFIC_RESEARCH,
            "big_data": self.ENABLE_BIG_DATA_PROCESSING,
            "analytics": self.ENABLE_ANALYTICS
        }
    
    def get_ai_config(self) -> Dict[str, Union[str, int, float]]:
        """الحصول على تكوين نماذج الذكاء الاصطناعي"""
        return {
            "primary_model": self.PRIMARY_AI_MODEL.value,
            "fallback_model": self.FALLBACK_AI_MODEL.value,
            "max_tokens": self.MAX_TOKENS,
            "temperature": self.TEMPERATURE,
            "top_p": self.TOP_P,
            "frequency_penalty": self.FREQUENCY_PENALTY,
            "presence_penalty": self.PRESENCE_PENALTY
        }
    
    def get_database_config(self) -> Dict[str, Union[str, int, bool]]:
        """الحصول على تكوين قاعدة البيانات"""
        return {
            "type": self.DATABASE_TYPE.value,
            "url": self.DATABASE_URL,
            "pool_size": self.DATABASE_POOL_SIZE,
            "max_overflow": self.DATABASE_MAX_OVERFLOW,
            "echo": self.DATABASE_ECHO
        }
    
    def get_performance_config(self) -> Dict[str, Union[str, int, float, bool]]:
        """الحصول على تكوين الأداء"""
        return {
            "level": self.PERFORMANCE_LEVEL.value,
            "max_memory_gb": self.MAX_MEMORY_GB,
            "max_cpu_cores": self.MAX_CPU_CORES,
            "enable_gpu": self.ENABLE_GPU,
            "parallel_processing": self.ENABLE_PARALLEL_PROCESSING,
            "cache_size_mb": self.CACHE_SIZE_MB,
            "request_timeout": self.REQUEST_TIMEOUT_SECONDS
        }
    
    def get_security_config(self) -> Dict[str, Union[str, int, bool]]:
        """الحصول على تكوين الأمان"""
        return {
            "level": self.SECURITY_LEVEL.value,
            "encryption": self.ENABLE_ENCRYPTION,
            "algorithm": self.ENCRYPTION_ALGORITHM,
            "rate_limiting": self.ENABLE_RATE_LIMITING,
            "max_requests": self.MAX_REQUESTS_PER_MINUTE,
            "session_timeout": self.SESSION_TIMEOUT_MINUTES,
            "api_key_rotation": self.ENABLE_API_KEY_ROTATION
        }
    
    def is_development(self) -> bool:
        """فحص ما إذا كان في وضع التطوير"""
        return self.APP_ENVIRONMENT == "development" or self.APP_DEBUG
    
    def is_production(self) -> bool:
        """فحص ما إذا كان في وضع الإنتاج"""
        return self.APP_ENVIRONMENT == "production" and not self.APP_DEBUG
    
    def export_config(self, file_path: Optional[str] = None) -> Dict[str, Any]:
        """تصدير التكوين إلى ملف أو قاموس"""
        config_dict = self.dict()
        
        if file_path:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, ensure_ascii=False, indent=2, default=str)
        
        return config_dict
    
    def validate_api_keys(self) -> Dict[str, bool]:
        """التحقق من صحة مفاتيح API"""
        api_keys_status = {}
        
        # فحص المفاتيح المطلوبة
        api_keys_status['openai'] = bool(self.OPENAI_API_KEY and len(self.OPENAI_API_KEY) >= 20)
        
        # فحص المفاتيح الاختيارية
        api_keys_status['anthropic'] = bool(self.ANTHROPIC_API_KEY and len(self.ANTHROPIC_API_KEY) >= 10)
        api_keys_status['google'] = bool(self.GOOGLE_API_KEY and len(self.GOOGLE_API_KEY) >= 10)
        api_keys_status['azure'] = bool(self.AZURE_OPENAI_KEY and len(self.AZURE_OPENAI_KEY) >= 10)
        api_keys_status['huggingface'] = bool(self.HUGGINGFACE_API_KEY and len(self.HUGGINGFACE_API_KEY) >= 10)
        
        return api_keys_status

# إنشاء مثيل الإعدادات العامة
try:
    settings = Settings()
    
    # طباعة معلومات التهيئة
    print(f"✅ تم تحميل إعدادات {settings.APP_NAME} v{settings.APP_VERSION}")
    print(f"🌍 البيئة: {settings.APP_ENVIRONMENT}")
    print(f"🎯 مستوى الأداء: {settings.PERFORMANCE_LEVEL.value}")
    print(f"🔒 مستوى الأمان: {settings.SECURITY_LEVEL.value}")
    
    # فحص مفاتيح API
    api_status = settings.validate_api_keys()
    valid_keys = sum(api_status.values())
    print(f"🔑 مفاتيح API صالحة: {valid_keys}/{len(api_status)}")
    
    # تحذيرات إذا لزم الأمر
    if not api_status['openai']:
        print("⚠️ تحذير: مفتاح OpenAI غير صالح أو مفقود")
    
    if settings.is_development():
        print("🛠️ وضع التطوير مفعل")
    
except Exception as e:
    print(f"❌ خطأ في تحميل الإعدادات: {e}")
    print("💡 تأكد من وجود ملف .env مع الإعدادات المطلوبة")
    raise

# تصدير الكائنات المهمة
__all__ = [
    'settings', 'Settings', 'LogLevel', 'AIModel', 'DatabaseType',
    'PerformanceLevel', 'SecurityLevel'
]
