
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ملف التكوين المتقدم للمساعد الذكي الموحد
يحتوي على جميع الإعدادات والتكوينات المتقدمة
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from dotenv import load_dotenv
import logging

# تحميل متغيرات البيئة
load_dotenv()

@dataclass
class AIModelConfig:
    """تكوين نماذج الذكاء الاصطناعي"""
    openai_api_key: str = ""
    huggingface_api_key: str = ""
    claude_api_key: str = ""
    model_name: str = "gpt-4"
    max_tokens: int = 2048
    temperature: float = 0.7
    timeout: int = 30
    retry_attempts: int = 3
    use_cache: bool = True
    cache_ttl: int = 3600

@dataclass
class DatabaseConfig:
    """تكوين قواعد البيانات"""
    sqlite_path: str = "data/assistant.db"
    redis_url: str = "redis://localhost:6379"
    mongodb_uri: str = "mongodb://localhost:27017/assistant"
    use_redis: bool = False
    use_mongodb: bool = False
    backup_interval: int = 24  # ساعات

@dataclass
class SecurityConfig:
    """تكوين الأمان"""
    encryption_key: str = ""
    jwt_secret: str = ""
    session_secret: str = ""
    max_login_attempts: int = 5
    session_timeout: int = 3600
    enable_encryption: bool = True
    secure_cookies: bool = True

@dataclass
class PerformanceConfig:
    """تكوين الأداء"""
    max_concurrent_requests: int = 10
    request_timeout: int = 30
    memory_limit_mb: int = 1024
    cpu_limit_percent: int = 80
    enable_caching: bool = True
    cache_size_mb: int = 256
    log_performance: bool = True

@dataclass
class VoiceConfig:
    """تكوين الصوت"""
    language: str = "ar-SA"
    speed: int = 150
    volume: float = 0.9
    voice_engine: str = "pyttsx3"
    enable_voice_recognition: bool = True
    enable_voice_synthesis: bool = True
    noise_reduction: bool = True

@dataclass
class VisionConfig:
    """تكوين الرؤية"""
    camera_index: int = 0
    detection_confidence: float = 0.5
    max_image_size_mb: int = 10
    supported_formats: List[str] = field(default_factory=lambda: ["jpg", "jpeg", "png", "bmp"])
    enable_face_recognition: bool = True
    enable_object_detection: bool = True
    enable_ocr: bool = True

@dataclass
class LearningConfig:
    """تكوين التعلم"""
    learning_rate: float = 0.01
    memory_size: int = 1000
    active_learning_threshold: float = 0.8
    enable_continuous_learning: bool = True
    save_interactions: bool = True
    model_update_interval: int = 7  # أيام

@dataclass
class AnalyticsConfig:
    """تكوين التحليلات"""
    enable_analytics: bool = True
    big_data_processing: bool = True
    enable_predictions: bool = True
    enable_recommendations: bool = True
    data_retention_days: int = 365
    anonymize_data: bool = True

@dataclass
class InterfaceConfig:
    """تكوين الواجهات"""
    web_port: int = 5000
    streamlit_port: int = 8501
    dash_port: int = 8050
    host: str = "0.0.0.0"
    enable_web_interface: bool = True
    enable_api: bool = True
    api_rate_limit: int = 1000  # طلبات في الساعة

class AdvancedConfig:
    """إدارة التكوين المتقدم"""
    
    def __init__(self, config_file: Optional[str] = None):
        """تهيئة التكوين"""
        self.config_file = config_file or "config/settings.yaml"
        self.config_dir = Path("config")
        self.data_dir = Path("data")
        
        # إنشاء المجلدات
        self.config_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)
        
        # تهيئة السجلات
        self.logger = logging.getLogger(__name__)
        
        # تحميل التكوين
        self.ai_models = AIModelConfig()
        self.database = DatabaseConfig()
        self.security = SecurityConfig()
        self.performance = PerformanceConfig()
        self.voice = VoiceConfig()
        self.vision = VisionConfig()
        self.learning = LearningConfig()
        self.analytics = AnalyticsConfig()
        self.interface = InterfaceConfig()
        
        # تحميل التكوين من الملفات
        self.load_config()
        
        # تحميل متغيرات البيئة
        self.load_environment_variables()
        
        # التحقق من صحة التكوين
        self.validate_config()
    
    def load_config(self):
        """تحميل التكوين من الملفات"""
        try:
            # تحميل من YAML
            yaml_file = Path(self.config_file)
            if yaml_file.exists():
                with open(yaml_file, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)
                    self.apply_config_data(config_data)
                    self.logger.info(f"تم تحميل التكوين من: {yaml_file}")
            
            # تحميل من JSON
            json_file = self.config_dir / "settings.json"
            if json_file.exists():
                with open(json_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                    self.apply_config_data(config_data)
                    self.logger.info(f"تم تحميل التكوين من: {json_file}")
                    
        except Exception as e:
            self.logger.warning(f"خطأ في تحميل التكوين: {e}")
    
    def load_environment_variables(self):
        """تحميل متغيرات البيئة"""
        try:
            # مفاتيح API
            self.ai_models.openai_api_key = os.getenv("OPENAI_API_KEY", "")
            self.ai_models.huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY", "")
            self.ai_models.claude_api_key = os.getenv("CLAUDE_API_KEY", "")
            
            # قواعد البيانات
            self.database.redis_url = os.getenv("REDIS_URL", self.database.redis_url)
            self.database.mongodb_uri = os.getenv("MONGODB_URI", self.database.mongodb_uri)
            
            # الأمان
            self.security.encryption_key = os.getenv("ENCRYPTION_KEY", "")
            self.security.jwt_secret = os.getenv("JWT_SECRET_KEY", "")
            self.security.session_secret = os.getenv("SESSION_SECRET", "")
            
            # الواجهات
            self.interface.web_port = int(os.getenv("WEB_PORT", self.interface.web_port))
            self.interface.host = os.getenv("HOST", self.interface.host)
            
            # الإعدادات العامة
            debug = os.getenv("DEBUG", "false").lower() == "true"
            if debug:
                logging.getLogger().setLevel(logging.DEBUG)
            
            self.logger.info("تم تحميل متغيرات البيئة")
            
        except Exception as e:
            self.logger.error(f"خطأ في تحميل متغيرات البيئة: {e}")
    
    def apply_config_data(self, config_data: Dict[str, Any]):
        """تطبيق بيانات التكوين"""
        try:
            # تطبيق تكوين نماذج الذكاء الاصطناعي
            if "ai_models" in config_data:
                ai_config = config_data["ai_models"]
                for key, value in ai_config.items():
                    if hasattr(self.ai_models, key):
                        setattr(self.ai_models, key, value)
            
            # تطبيق تكوين قاعدة البيانات
            if "database" in config_data:
                db_config = config_data["database"]
                for key, value in db_config.items():
                    if hasattr(self.database, key):
                        setattr(self.database, key, value)
            
            # تطبيق باقي التكوينات
            config_mappings = {
                "security": self.security,
                "performance": self.performance,
                "voice": self.voice,
                "vision": self.vision,
                "learning": self.learning,
                "analytics": self.analytics,
                "interface": self.interface
            }
            
            for section, config_obj in config_mappings.items():
                if section in config_data:
                    section_config = config_data[section]
                    for key, value in section_config.items():
                        if hasattr(config_obj, key):
                            setattr(config_obj, key, value)
                            
        except Exception as e:
            self.logger.error(f"خطأ في تطبيق التكوين: {e}")
    
    def validate_config(self):
        """التحقق من صحة التكوين"""
        errors = []
        
        # التحقق من مفاتيح API
        if not self.ai_models.openai_api_key:
            errors.append("مفتاح OpenAI API مفقود")
        
        # التحقق من مفاتيح الأمان
        if self.security.enable_encryption and not self.security.encryption_key:
            errors.append("مفتاح التشفير مفقود")
        
        # التحقق من حدود الأداء
        if self.performance.memory_limit_mb < 512:
            errors.append("حد الذاكرة منخفض جداً (أقل من 512 ميجابايت)")
        
        # التحقق من منافذ الواجهات
        if not (1024 <= self.interface.web_port <= 65535):
            errors.append("منفذ الويب غير صحيح")
        
        # عرض التحذيرات
        if errors:
            self.logger.warning("مشاكل في التكوين:")
            for error in errors:
                self.logger.warning(f"  - {error}")
        else:
            self.logger.info("التحقق من التكوين: ✅ نجح")
    
    def save_config(self):
        """حفظ التكوين الحالي"""
        try:
            config_data = {
                "ai_models": self.ai_models.__dict__,
                "database": self.database.__dict__,
                "security": {k: v for k, v in self.security.__dict__.items() 
                           if k not in ["encryption_key", "jwt_secret", "session_secret"]},
                "performance": self.performance.__dict__,
                "voice": self.voice.__dict__,
                "vision": self.vision.__dict__,
                "learning": self.learning.__dict__,
                "analytics": self.analytics.__dict__,
                "interface": self.interface.__dict__
            }
            
            # حفظ في YAML
            yaml_file = Path(self.config_file)
            with open(yaml_file, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
            
            self.logger.info(f"تم حفظ التكوين في: {yaml_file}")
            
        except Exception as e:
            self.logger.error(f"خطأ في حفظ التكوين: {e}")
    
    def get_config_summary(self) -> str:
        """الحصول على ملخص التكوين"""
        summary = f"""
🔧 ملخص تكوين المساعد الذكي
{'='*40}
🧠 نماذج الذكاء الاصطناعي:
   • النموذج الأساسي: {self.ai_models.model_name}
   • OpenAI: {'✅' if self.ai_models.openai_api_key else '❌'}
   • HuggingFace: {'✅' if self.ai_models.huggingface_api_key else '❌'}
   • Claude: {'✅' if self.ai_models.claude_api_key else '❌'}

💾 قواعد البيانات:
   • SQLite: ✅ {self.database.sqlite_path}
   • Redis: {'✅' if self.database.use_redis else '❌'}
   • MongoDB: {'✅' if self.database.use_mongodb else '❌'}

🔒 الأمان:
   • التشفير: {'✅' if self.security.enable_encryption else '❌'}
   • الجلسات الآمنة: {'✅' if self.security.secure_cookies else '❌'}

⚡ الأداء:
   • الحد الأقصى للطلبات: {self.performance.max_concurrent_requests}
   • حد الذاكرة: {self.performance.memory_limit_mb} ميجابايت
   • التخزين المؤقت: {'✅' if self.performance.enable_caching else '❌'}

🗣️ الصوت:
   • اللغة: {self.voice.language}
   • التعرف على الصوت: {'✅' if self.voice.enable_voice_recognition else '❌'}
   • تركيب الصوت: {'✅' if self.voice.enable_voice_synthesis else '❌'}

👁️ الرؤية:
   • كاميرا: كاميرا {self.vision.camera_index}
   • التعرف على الوجوه: {'✅' if self.vision.enable_face_recognition else '❌'}
   • اكتشاف الكائنات: {'✅' if self.vision.enable_object_detection else '❌'}

🧠 التعلم:
   • التعلم المستمر: {'✅' if self.learning.enable_continuous_learning else '❌'}
   • حفظ التفاعلات: {'✅' if self.learning.save_interactions else '❌'}

📊 التحليلات:
   • البيانات الضخمة: {'✅' if self.analytics.big_data_processing else '❌'}
   • التوقعات: {'✅' if self.analytics.enable_predictions else '❌'}
   • التوصيات: {'✅' if self.analytics.enable_recommendations else '❌'}

🌐 الواجهات:
   • الويب: {'✅' if self.interface.enable_web_interface else '❌'} (:{self.interface.web_port})
   • API: {'✅' if self.interface.enable_api else '❌'}
"""
        return summary
    
    def reset_to_defaults(self):
        """إعادة تعيين التكوين للإعدادات الافتراضية"""
        self.__init__()
        self.logger.info("تم إعادة تعيين التكوين للإعدادات الافتراضية")

# إنشاء مثيل عام للتكوين
config = AdvancedConfig()
