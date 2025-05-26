
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إعدادات المساعد الذكي الموحد
يحتوي على جميع الإعدادات والتكوينات المطلوبة
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# تحميل متغيرات البيئة
load_dotenv()

@dataclass
class AIModelsConfig:
    """إعدادات نماذج الذكاء الاصطناعي"""
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 2000
    
    # إعدادات نماذج Hugging Face
    bert_model: str = "bert-base-multilingual-cased"
    roberta_model: str = "cardiffnlp/twitter-roberta-base-emotion"
    wav2vec_model: str = "facebook/wav2vec2-base-960h"

@dataclass
class VoiceConfig:
    """إعدادات المعالجة الصوتية"""
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 1024
    language: str = "ar-SA"
    voice_speed: int = 150
    voice_volume: float = 0.9

@dataclass
class VisionConfig:
    """إعدادات الرؤية الحاسوبية"""
    camera_index: int = 0
    frame_width: int = 640
    frame_height: int = 480
    detection_confidence: float = 0.5
    tracking_enabled: bool = True

@dataclass
class LearningConfig:
    """إعدادات أنظمة التعلم"""
    memory_size: int = 1000
    learning_rate: float = 0.01
    reinforcement_decay: float = 0.95
    active_learning_threshold: float = 0.8

@dataclass
class SecurityConfig:
    """إعدادات الأمان"""
    encryption_key: Optional[str] = None
    max_session_duration: int = 3600  # ساعة واحدة
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # دقيقة واحدة

@dataclass
class DatabaseConfig:
    """إعدادات قواعد البيانات"""
    sqlite_path: str = "data/assistant.db"
    redis_host: str = "localhost"
    redis_port: int = 6379
    mongodb_uri: Optional[str] = None

@dataclass
class InterfaceConfig:
    """إعدادات الواجهات"""
    web_port: int = 5000
    web_host: str = "0.0.0.0"
    streamlit_port: int = 8501
    dash_port: int = 8050
    enable_voice: bool = True
    enable_vision: bool = True
    enable_web: bool = True

class UnifiedSettings:
    """إعدادات المساعد الموحدة"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.data_dir = self.project_root / "data"
        self.models_dir = self.data_dir / "models"
        self.logs_dir = self.data_dir / "logs"
        
        # إنشاء المجلدات إذا لم تكن موجودة
        self._create_directories()
        
        # تحميل الإعدادات
        self.ai_models = AIModelsConfig(
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        self.voice = VoiceConfig()
        self.vision = VisionConfig()
        self.learning = LearningConfig()
        self.security = SecurityConfig(
            encryption_key=os.getenv("ENCRYPTION_KEY")
        )
        self.database = DatabaseConfig()
        self.interface = InterfaceConfig()
        
        # إعدادات عامة
        self.debug_mode = os.getenv("DEBUG", "False").lower() == "true"
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.language = os.getenv("LANGUAGE", "ar")
        
    def _create_directories(self):
        """إنشاء المجلدات المطلوبة"""
        directories = [
            self.data_dir,
            self.models_dir,
            self.logs_dir,
            self.data_dir / "user_data",
            self.data_dir / "cache",
            self.data_dir / "backups"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_model_path(self, model_name: str) -> Path:
        """الحصول على مسار نموذج محدد"""
        return self.models_dir / model_name
    
    def get_log_path(self, log_name: str) -> Path:
        """الحصول على مسار ملف سجل محدد"""
        return self.logs_dir / log_name
    
    def validate_settings(self) -> Dict[str, bool]:
        """التحقق من صحة الإعدادات"""
        validation_results = {}
        
        # التحقق من API keys
        validation_results["openai_api_key"] = bool(self.ai_models.openai_api_key)
        validation_results["directories_exist"] = all([
            self.data_dir.exists(),
            self.models_dir.exists(),
            self.logs_dir.exists()
        ])
        
        # التحقق من إعدادات الشبكة
        validation_results["ports_available"] = all([
            1000 <= self.interface.web_port <= 65535,
            1000 <= self.interface.streamlit_port <= 65535,
            1000 <= self.interface.dash_port <= 65535
        ])
        
        return validation_results
    
    def to_dict(self) -> Dict[str, Any]:
        """تحويل الإعدادات إلى قاموس"""
        return {
            "ai_models": self.ai_models.__dict__,
            "voice": self.voice.__dict__,
            "vision": self.vision.__dict__,
            "learning": self.learning.__dict__,
            "security": {k: v for k, v in self.security.__dict__.items() if "key" not in k.lower()},
            "database": self.database.__dict__,
            "interface": self.interface.__dict__,
            "debug_mode": self.debug_mode,
            "log_level": self.log_level,
            "language": self.language
        }

# إنشاء مثيل عام للإعدادات
settings = UnifiedSettings()

# دالة مساعدة للوصول السريع للإعدادات
def get_settings() -> UnifiedSettings:
    """الحصول على مثيل الإعدادات"""
    return settings

def validate_environment() -> bool:
    """التحقق من البيئة والإعدادات"""
    validation = settings.validate_settings()
    
    all_valid = all(validation.values())
    
    if not all_valid:
        print("⚠️  تحذير: بعض الإعدادات غير صحيحة:")
        for key, valid in validation.items():
            if not valid:
                print(f"   ❌ {key}")
    else:
        print("✅ جميع الإعدادات صحيحة")
    
    return all_valid

if __name__ == "__main__":
    # اختبار الإعدادات
    print("🔧 اختبار إعدادات المساعد الذكي الموحد")
    print("=" * 50)
    
    validate_environment()
    
    print(f"\n📁 مجلد المشروع: {settings.project_root}")
    print(f"📊 مجلد البيانات: {settings.data_dir}")
    print(f"🤖 مجلد النماذج: {settings.models_dir}")
    print(f"📝 مجلد السجلات: {settings.logs_dir}")
    
    print(f"\n🌐 منافذ الواجهات:")
    print(f"   • الويب: {settings.interface.web_port}")
    print(f"   • Streamlit: {settings.interface.streamlit_port}")
    print(f"   • Dash: {settings.interface.dash_port}")
