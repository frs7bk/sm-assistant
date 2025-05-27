
# config/settings.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, validator
from typing import Literal, Optional, Dict, List, Union
from pathlib import Path
import os
from enum import Enum

class LogLevel(str, Enum):
    """مستويات السجلات"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class AIModel(str, Enum):
    """نماذج الذكاء الاصطناعي المدعومة"""
    GPT4 = "gpt-4"
    GPT3_5 = "gpt-3.5-turbo"
    CLAUDE = "claude-3"
    GEMINI = "gemini-pro"

class DatabaseType(str, Enum):
    """أنواع قواعد البيانات"""
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    MONGODB = "mongodb"
    REDIS = "redis"

class Settings(BaseSettings):
    """
    إعدادات المساعد الذكي المتقدمة
    تدعم التحقق من الصحة والتكوين الديناميكي
    """
    
    # === الإعدادات الأساسية ===
    ASSISTANT_NAME: str = Field(default="المساعد الذكي المتطور", description="اسم المساعد")
    VERSION: str = Field(default="3.0.0", description="إصدار المساعد")
    DEFAULT_LANGUAGE: str = Field(default="ar", regex="^(ar|en|fr|es|de)$")
    TIMEZONE: str = Field(default="Asia/Riyadh", description="المنطقة الزمنية")
    
    # === مفاتيح API المطلوبة ===
    OPENAI_API_KEY: str = Field(..., min_length=20, description="مفتاح OpenAI مطلوب")
    
    # === مفاتيح API الاختيارية ===
    HUGGINGFACE_API_KEY: Optional[str] = Field(default=None, min_length=10)
    CLAUDE_API_KEY: Optional[str] = Field(default=None, min_length=10) 
    GOOGLE_API_KEY: Optional[str] = Field(default=None, min_length=10)
    AZURE_API_KEY: Optional[str] = Field(default=None, min_length=10)
    ANTHROPIC_API_KEY: Optional[str] = Field(default=None, min_length=10)
    
    # === إعدادات السجلات ===
    LOG_LEVEL: LogLevel = Field(default=LogLevel.INFO)
    LOG_FILE_PATH: Path = Field(default=Path("data/logs/assistant.log"))
    LOG_ROTATION_SIZE: str = Field(default="100MB", description="حجم دوران السجلات")
    LOG_RETENTION_DAYS: int = Field(default=30, ge=1, le=365)
    ENABLE_CONSOLE_LOGGING: bool = Field(default=True)
    ENABLE_FILE_LOGGING: bool = Field(default=True)
    
    # === إعدادات قاعدة البيانات ===
    DATABASE_TYPE: DatabaseType = Field(default=DatabaseType.SQLITE)
    DATABASE_URL: str = Field(default="sqlite:///data/assistant.db")
    DATABASE_POOL_SIZE: int = Field(default=10, ge=1, le=100)
    DATABASE_MAX_OVERFLOW: int = Field(default=20, ge=0, le=100)
    
    # === إعدادات نماذج الذكاء الاصطناعي ===
    PRIMARY_AI_MODEL: AIModel = Field(default=AIModel.GPT4)
    FALLBACK_AI_MODEL: AIModel = Field(default=AIModel.GPT3_5)
    MAX_TOKENS: int = Field(default=4000, ge=100, le=32000)
    TEMPERATURE: float = Field(default=0.7, ge=0.0, le=2.0)
    TOP_P: float = Field(default=0.9, ge=0.0, le=1.0)
    
    # === إعدادات الأداء ===
    MAX_MEMORY_USAGE: str = Field(default="8GB", description="الحد الأقصى لاستخدام الذاكرة")
    ENABLE_GPU: bool = Field(default=True, description="تفعيل معالج الرسوميات")
    PARALLEL_PROCESSING: bool = Field(default=True, description="المعالجة المتوازية")
    CACHE_SIZE: int = Field(default=1000, ge=10, le=10000)
    REQUEST_TIMEOUT: int = Field(default=30, ge=5, le=300)
    
    # === إعدادات الأمان ===
    ENABLE_ENCRYPTION: bool = Field(default=True, description="تفعيل التشفير")
    ENCRYPTION_KEY: Optional[str] = Field(default=None, min_length=32)
    ENABLE_RATE_LIMITING: bool = Field(default=True)
    MAX_REQUESTS_PER_MINUTE: int = Field(default=60, ge=1, le=1000)
    ENABLE_BIOMETRIC_AUTH: bool = Field(default=False)
    SESSION_TIMEOUT: int = Field(default=3600, ge=300, le=86400)
    
    # === إعدادات الميزات المتقدمة ===
    ENABLE_LEARNING: bool = Field(default=True, description="تفعيل التعلم المستمر")
    ENABLE_PREDICTION: bool = Field(default=True, description="تفعيل التنبؤات الذكية")
    ENABLE_VISION: bool = Field(default=True, description="تفعيل الرؤية الحاسوبية")
    ENABLE_VOICE: bool = Field(default=True, description="تفعيل المعالجة الصوتية")
    ENABLE_AR_VR: bool = Field(default=False, description="تفعيل الواقع المختلط")
    ENABLE_IOT: bool = Field(default=False, description="تفعيل إنترنت الأشياء")
    ENABLE_QUANTUM_FEATURES: bool = Field(default=False, description="الميزات الكمومية")
    
    # === إعدادات التطبيقات المتخصصة ===
    ENABLE_FINANCIAL_ADVISOR: bool = Field(default=True)
    ENABLE_HEALTH_MONITOR: bool = Field(default=True)
    ENABLE_GAMING_COACH: bool = Field(default=True)
    ENABLE_CREATIVE_AI: bool = Field(default=True)
    ENABLE_PROJECT_MANAGER: bool = Field(default=True)
    
    # === إعدادات البيانات الضخمة ===
    ENABLE_BIG_DATA: bool = Field(default=True)
    DASK_SCHEDULER_ADDRESS: Optional[str] = Field(default=None)
    SPARK_MASTER_URL: Optional[str] = Field(default="local[*]")
    MAX_DATASET_SIZE: str = Field(default="10GB")
    
    # === إعدادات الشبكة ===
    API_HOST: str = Field(default="0.0.0.0")
    API_PORT: int = Field(default=5000, ge=1000, le=65535)
    ENABLE_CORS: bool = Field(default=True)
    ALLOWED_ORIGINS: List[str] = Field(default=["*"])
    
    # === إعدادات المطورين ===
    DEBUG_MODE: bool = Field(default=False)
    ENABLE_PROFILING: bool = Field(default=False)
    ENABLE_METRICS: bool = Field(default=True)
    ENABLE_AUTO_BACKUP: bool = Field(default=True)
    BACKUP_INTERVAL_HOURS: int = Field(default=24, ge=1, le=168)
    
    # === تكوين Pydantic ===
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding='utf-8',
        case_sensitive=True,
        extra='forbid'
    )
    
    @validator('LOG_FILE_PATH')
    def validate_log_path(cls, v):
        """التحقق من صحة مسار السجلات"""
        log_dir = v.parent
        log_dir.mkdir(parents=True, exist_ok=True)
        return v
    
    @validator('DATABASE_URL')
    def validate_database_url(cls, v, values):
        """التحقق من صحة رابط قاعدة البيانات"""
        db_type = values.get('DATABASE_TYPE')
        if db_type == DatabaseType.SQLITE and not v.startswith('sqlite'):
            raise ValueError("SQLite URL يجب أن يبدأ بـ sqlite://")
        elif db_type == DatabaseType.POSTGRESQL and not v.startswith('postgresql'):
            raise ValueError("PostgreSQL URL يجب أن يبدأ بـ postgresql://")
        return v
    
    @validator('FALLBACK_AI_MODEL')
    def validate_fallback_model(cls, v, values):
        """التحقق من أن النموذج الاحتياطي مختلف عن الأساسي"""
        primary = values.get('PRIMARY_AI_MODEL')
        if v == primary:
            raise ValueError("النموذج الاحتياطي يجب أن يكون مختلفاً عن الأساسي")
        return v
    
    def get_feature_flags(self) -> Dict[str, bool]:
        """الحصول على إعدادات الميزات"""
        return {
            "learning": self.ENABLE_LEARNING,
            "prediction": self.ENABLE_PREDICTION,
            "vision": self.ENABLE_VISION,
            "voice": self.ENABLE_VOICE,
            "ar_vr": self.ENABLE_AR_VR,
            "iot": self.ENABLE_IOT,
            "quantum": self.ENABLE_QUANTUM_FEATURES,
            "financial": self.ENABLE_FINANCIAL_ADVISOR,
            "health": self.ENABLE_HEALTH_MONITOR,
            "gaming": self.ENABLE_GAMING_COACH,
            "creative": self.ENABLE_CREATIVE_AI,
            "project_mgmt": self.ENABLE_PROJECT_MANAGER,
            "big_data": self.ENABLE_BIG_DATA
        }
    
    def get_ai_config(self) -> Dict[str, Union[str, int, float]]:
        """الحصول على إعدادات الذكاء الاصطناعي"""
        return {
            "primary_model": self.PRIMARY_AI_MODEL.value,
            "fallback_model": self.FALLBACK_AI_MODEL.value,
            "max_tokens": self.MAX_TOKENS,
            "temperature": self.TEMPERATURE,
            "top_p": self.TOP_P
        }
    
    def is_development(self) -> bool:
        """فحص ما إذا كان في وضع التطوير"""
        return self.DEBUG_MODE
    
    def is_production(self) -> bool:
        """فحص ما إذا كان في وضع الإنتاج"""
        return not self.DEBUG_MODE

# إنشاء نسخة واحدة من الإعدادات
try:
    settings = Settings()
    print(f"✅ تم تحميل إعدادات {settings.ASSISTANT_NAME} v{settings.VERSION}")
except Exception as e:
    print(f"❌ خطأ في تحميل الإعدادات: {e}")
    print("💡 تأكد من وجود ملف .env مع المفاتيح المطلوبة")
    raise

# تصدير الإعدادات للاستخدام في المشروع
__all__ = ['settings', 'Settings', 'LogLevel', 'AIModel', 'DatabaseType']
