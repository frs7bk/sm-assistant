
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
نظام التكوين المتقدم للمساعد الذكي
يدير جميع إعدادات النظام بطريقة ديناميكية ومرنة
"""

import json
import yaml
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, asdict
from enum import Enum
import importlib.util

class ConfigType(Enum):
    """أنواع ملفات التكوين"""
    JSON = "json"
    YAML = "yaml"
    ENV = "env"
    PYTHON = "py"

@dataclass
class AIModelConfig:
    """تكوين نماذج الذكاء الاصطناعي"""
    openai_api_key: str = ""
    huggingface_api_key: str = ""
    model_cache_dir: str = "data/models"
    max_tokens: int = 2048
    temperature: float = 0.7
    enable_fine_tuning: bool = False
    custom_models: Dict[str, str] = None
    
    def __post_init__(self):
        if self.custom_models is None:
            self.custom_models = {}

@dataclass
class DatabaseConfig:
    """تكوين قواعد البيانات"""
    sqlite_path: str = "data/assistant.db"
    redis_url: str = "redis://localhost:6379"
    mongodb_uri: str = "mongodb://localhost:27017/assistant"
    backup_interval: int = 24  # ساعات
    enable_encryption: bool = True
    max_connections: int = 100

@dataclass
class SecurityConfig:
    """تكوين الأمان"""
    encryption_key: str = ""
    jwt_secret: str = ""
    session_secret: str = ""
    rate_limit_per_minute: int = 60
    enable_2fa: bool = False
    allowed_hosts: List[str] = None
    
    def __post_init__(self):
        if self.allowed_hosts is None:
            self.allowed_hosts = ["localhost", "127.0.0.1"]

@dataclass
class InterfaceConfig:
    """تكوين الواجهات"""
    enable_web: bool = True
    enable_voice: bool = True
    enable_vision: bool = True
    enable_api: bool = True
    web_port: int = 5000
    api_port: int = 8000
    host: str = "0.0.0.0"
    voice_language: str = "ar-SA"
    camera_index: int = 0

@dataclass
class LearningConfig:
    """تكوين التعلم الآلي"""
    enable_active_learning: bool = True
    enable_reinforcement: bool = True
    memory_size: int = 1000
    learning_rate: float = 0.01
    confidence_threshold: float = 0.8
    auto_save_interval: int = 300  # ثواني

@dataclass
class PerformanceConfig:
    """تكوين الأداء"""
    max_concurrent_requests: int = 10
    request_timeout: int = 30
    cache_ttl: int = 3600  # ثواني
    enable_gpu: bool = False
    max_memory_usage: str = "2GB"
    cleanup_interval: int = 7  # أيام

class AdvancedConfigManager:
    """مدير التكوين المتقدم"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # التكوينات
        self.ai_models = AIModelConfig()
        self.database = DatabaseConfig()
        self.security = SecurityConfig()
        self.interface = InterfaceConfig()
        self.learning = LearningConfig()
        self.performance = PerformanceConfig()
        
        # إعدادات عامة
        self.debug_mode = False
        self.log_level = "INFO"
        self.language = "ar"
        self.project_root = Path.cwd()
        
        # تحميل التكوينات
        self.load_all_configs()
    
    def load_all_configs(self):
        """تحميل جميع ملفات التكوين"""
        try:
            # تحميل من متغيرات البيئة
            self._load_from_env()
            
            # تحميل من ملفات JSON/YAML
            self._load_from_files()
            
            # التحقق من صحة التكوين
            self.validate_config()
            
            self.logger.info("✅ تم تحميل جميع التكوينات بنجاح")
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في تحميل التكوينات: {e}")
    
    def _load_from_env(self):
        """تحميل التكوين من متغيرات البيئة"""
        # تكوين الذكاء الاصطناعي
        self.ai_models.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.ai_models.huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY", "")
        
        # تكوين قواعد البيانات
        self.database.sqlite_path = os.getenv("DATABASE_URL", self.database.sqlite_path)
        self.database.redis_url = os.getenv("REDIS_URL", self.database.redis_url)
        self.database.mongodb_uri = os.getenv("MONGODB_URI", self.database.mongodb_uri)
        
        # تكوين الأمان
        self.security.encryption_key = os.getenv("ENCRYPTION_KEY", "")
        self.security.jwt_secret = os.getenv("JWT_SECRET_KEY", "")
        self.security.session_secret = os.getenv("SESSION_SECRET", "")
        
        # تكوين الواجهات
        self.interface.web_port = int(os.getenv("WEB_PORT", self.interface.web_port))
        self.interface.host = os.getenv("HOST", self.interface.host)
        self.interface.voice_language = os.getenv("VOICE_LANGUAGE", self.interface.voice_language)
        
        # إعدادات عامة
        self.debug_mode = os.getenv("DEBUG", "false").lower() == "true"
        self.log_level = os.getenv("LOG_LEVEL", self.log_level)
        self.language = os.getenv("LANGUAGE", self.language)
    
    def _load_from_files(self):
        """تحميل التكوين من الملفات"""
        config_files = {
            "ai_models.json": self.ai_models,
            "database.yaml": self.database,
            "security.json": self.security,
            "interface.yaml": self.interface,
            "learning.json": self.learning,
            "performance.yaml": self.performance
        }
        
        for filename, config_obj in config_files.items():
            file_path = self.config_dir / filename
            
            if file_path.exists():
                try:
                    if filename.endswith('.json'):
                        data = self._load_json(file_path)
                    elif filename.endswith('.yaml'):
                        data = self._load_yaml(file_path)
                    else:
                        continue
                    
                    # تحديث التكوين
                    for key, value in data.items():
                        if hasattr(config_obj, key):
                            setattr(config_obj, key, value)
                    
                    self.logger.info(f"تم تحميل {filename}")
                    
                except Exception as e:
                    self.logger.warning(f"خطأ في تحميل {filename}: {e}")
    
    def _load_json(self, file_path: Path) -> Dict[str, Any]:
        """تحميل ملف JSON"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _load_yaml(self, file_path: Path) -> Dict[str, Any]:
        """تحميل ملف YAML"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except ImportError:
            self.logger.warning("PyYAML غير مثبت - تخطي ملفات YAML")
            return {}
    
    def validate_config(self) -> List[str]:
        """التحقق من صحة التكوين"""
        issues = []
        
        # التحقق من مفاتيح API
        if not self.ai_models.openai_api_key:
            issues.append("openai_api_key")
        
        if not self.security.encryption_key:
            issues.append("encryption_key")
        
        if not self.security.jwt_secret:
            issues.append("jwt_secret")
        
        # التحقق من المسارات
        if not Path(self.ai_models.model_cache_dir).exists():
            Path(self.ai_models.model_cache_dir).mkdir(parents=True, exist_ok=True)
        
        # التحقق من المنافذ
        if not (1000 <= self.interface.web_port <= 65535):
            issues.append("web_port")
        
        if issues:
            self.logger.warning(f"⚠️ مشاكل في التكوين: {', '.join(issues)}")
        
        return issues
    
    def save_config(self, config_name: str, config_type: ConfigType = ConfigType.JSON):
        """حفظ تكوين معين"""
        try:
            config_map = {
                "ai_models": self.ai_models,
                "database": self.database,
                "security": self.security,
                "interface": self.interface,
                "learning": self.learning,
                "performance": self.performance
            }
            
            if config_name not in config_map:
                raise ValueError(f"تكوين غير معروف: {config_name}")
            
            config_obj = config_map[config_name]
            file_path = self.config_dir / f"{config_name}.{config_type.value}"
            
            data = asdict(config_obj)
            
            if config_type == ConfigType.JSON:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            elif config_type == ConfigType.YAML:
                with open(file_path, 'w', encoding='utf-8') as f:
                    yaml.dump(data, f, allow_unicode=True, default_flow_style=False)
            
            self.logger.info(f"تم حفظ تكوين {config_name}")
            
        except Exception as e:
            self.logger.error(f"خطأ في حفظ التكوين: {e}")
    
    def save_all_configs(self):
        """حفظ جميع التكوينات"""
        configs = ["ai_models", "database", "security", "interface", "learning", "performance"]
        
        for config_name in configs:
            self.save_config(config_name)
    
    def get_config_summary(self) -> Dict[str, Any]:
        """الحصول على ملخص التكوين"""
        return {
            "project_root": str(self.project_root),
            "debug_mode": self.debug_mode,
            "language": self.language,
            "ai_models_configured": bool(self.ai_models.openai_api_key),
            "security_configured": bool(self.security.encryption_key),
            "interfaces_enabled": {
                "web": self.interface.enable_web,
                "voice": self.interface.enable_voice,
                "vision": self.interface.enable_vision,
                "api": self.interface.enable_api
            },
            "learning_enabled": {
                "active": self.learning.enable_active_learning,
                "reinforcement": self.learning.enable_reinforcement
            }
        }
    
    def update_config(self, section: str, updates: Dict[str, Any]):
        """تحديث قسم من التكوين"""
        config_map = {
            "ai_models": self.ai_models,
            "database": self.database,
            "security": self.security,
            "interface": self.interface,
            "learning": self.learning,
            "performance": self.performance
        }
        
        if section not in config_map:
            raise ValueError(f"قسم غير معروف: {section}")
        
        config_obj = config_map[section]
        
        for key, value in updates.items():
            if hasattr(config_obj, key):
                setattr(config_obj, key, value)
                self.logger.info(f"تم تحديث {section}.{key}")
            else:
                self.logger.warning(f"خاصية غير معروفة: {section}.{key}")

# مثيل عام لمدير التكوين
config_manager = AdvancedConfigManager()

def get_config_manager() -> AdvancedConfigManager:
    """الحصول على مدير التكوين"""
    return config_manager

def get_config() -> AdvancedConfigManager:
    """اختصار للحصول على التكوين"""
    return config_manager
