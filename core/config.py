
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
نظام إعدادات مركزي للمساعد الذكي الموحد
"""

import os
from typing import Optional, List
from pydantic import BaseSettings, Field
from functools import lru_cache

class Settings(BaseSettings):
    """إعدادات التطبيق الرئيسية"""
    
    # إعدادات التطبيق العامة
    app_name: str = Field("المساعد الذكي الموحد", env="APP_NAME")
    app_version: str = Field("2.0.0", env="APP_VERSION")
    debug: bool = Field(False, env="DEBUG")
    
    # إعدادات الخادم
    host: str = Field("0.0.0.0", env="HOST")
    port: int = Field(5000, env="PORT")
    
    # إعدادات قاعدة البيانات
    database_url: Optional[str] = Field(None, env="DATABASE_URL")
    redis_url: Optional[str] = Field(None, env="REDIS_URL")
    
    # إعدادات الذكاء الاصطناعي
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    openai_model: str = Field("gpt-4", env="OPENAI_MODEL")
    openai_max_tokens: int = Field(1024, env="OPENAI_MAX_TOKENS")
    
    # إعدادات الصوت
    speech_recognition_service: str = Field("whisper", env="SPEECH_RECOGNITION_SERVICE")
    text_to_speech_service: str = Field("gtts", env="TEXT_TO_SPEECH_SERVICE")
    audio_sample_rate: int = Field(16000, env="AUDIO_SAMPLE_RATE")
    
    # إعدادات الأمان
    secret_key: str = Field("your-secret-key-here", env="SECRET_KEY")
    allowed_origins: List[str] = Field(["*"], env="ALLOWED_ORIGINS")
    
    # إعدادات التخزين
    data_directory: str = Field("data", env="DATA_DIRECTORY")
    models_directory: str = Field("data/models", env="MODELS_DIRECTORY")
    sessions_directory: str = Field("data/sessions", env="SESSIONS_DIRECTORY")
    
    # إعدادات الأداء
    max_concurrent_sessions: int = Field(100, env="MAX_CONCURRENT_SESSIONS")
    session_timeout: int = Field(3600, env="SESSION_TIMEOUT")  # بالثواني
    
    # إعدادات السجلات
    log_level: str = Field("INFO", env="LOG_LEVEL")
    log_file: Optional[str] = Field(None, env="LOG_FILE")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

class AIModelSettings(BaseSettings):
    """إعدادات نماذج الذكاء الاصطناعي"""
    
    # نماذج معالجة اللغة الطبيعية
    sentiment_model: str = Field(
        "cardiffnlp/twitter-xlm-roberta-base-sentiment",
        env="SENTIMENT_MODEL"
    )
    ner_model: str = Field(
        "CAMeL-Lab/bert-base-arabic-camelbert-mix-ner",
        env="NER_MODEL"
    )
    embedding_model: str = Field(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        env="EMBEDDING_MODEL"
    )
    
    # إعدادات الجهاز
    device: str = Field("auto", env="AI_DEVICE")  # auto, cpu, cuda
    use_gpu: bool = Field(True, env="USE_GPU")
    max_batch_size: int = Field(32, env="MAX_BATCH_SIZE")
    
    # إعدادات التحسين
    enable_model_caching: bool = Field(True, env="ENABLE_MODEL_CACHING")
    model_cache_size: int = Field(3, env="MODEL_CACHE_SIZE")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

class VoiceSettings(BaseSettings):
    """إعدادات الصوت والتعرف عليه"""
    
    # إعدادات تحويل الكلام إلى نص
    whisper_model: str = Field("base", env="WHISPER_MODEL")
    whisper_language: str = Field("ar", env="WHISPER_LANGUAGE")
    
    # إعدادات تحويل النص إلى كلام
    tts_language: str = Field("ar", env="TTS_LANGUAGE")
    tts_speed: float = Field(1.0, env="TTS_SPEED")
    tts_voice: str = Field("male", env="TTS_VOICE")
    
    # إعدادات الصوت المتقدمة
    noise_reduction: bool = Field(True, env="NOISE_REDUCTION")
    auto_gain_control: bool = Field(True, env="AUTO_GAIN_CONTROL")
    voice_activity_detection: bool = Field(True, env="VOICE_ACTIVITY_DETECTION")
    
    # إعدادات الضغط والجودة
    audio_compression: str = Field("opus", env="AUDIO_COMPRESSION")
    audio_quality: str = Field("medium", env="AUDIO_QUALITY")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

@lru_cache()
def get_settings() -> Settings:
    """الحصول على إعدادات التطبيق (مع التخزين المؤقت)"""
    return Settings()

@lru_cache()
def get_ai_settings() -> AIModelSettings:
    """الحصول على إعدادات الذكاء الاصطناعي"""
    return AIModelSettings()

@lru_cache()
def get_voice_settings() -> VoiceSettings:
    """الحصول على إعدادات الصوت"""
    return VoiceSettings()

def create_directories():
    """إنشاء المجلدات المطلوبة"""
    settings = get_settings()
    
    directories = [
        settings.data_directory,
        settings.models_directory,
        settings.sessions_directory,
        "logs",
        "frontend/static",
        "frontend/templates"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

if __name__ == "__main__":
    # إنشاء ملف .env إذا لم يكن موجوداً
    if not os.path.exists(".env"):
        env_content = """# إعدادات المساعد الذكي الموحد

# إعدادات التطبيق
APP_NAME=المساعد الذكي الموحد
APP_VERSION=2.0.0
DEBUG=False

# إعدادات الخادم
HOST=0.0.0.0
PORT=5000

# إعدادات الذكاء الاصطناعي
OPENAI_API_KEY=your-openai-api-key-here
OPENAI_MODEL=gpt-4
OPENAI_MAX_TOKENS=1024

# إعدادات الصوت
SPEECH_RECOGNITION_SERVICE=whisper
TEXT_TO_SPEECH_SERVICE=gtts
AUDIO_SAMPLE_RATE=16000

# إعدادات الأمان
SECRET_KEY=your-secret-key-here
ALLOWED_ORIGINS=*

# إعدادات الأداء
MAX_CONCURRENT_SESSIONS=100
SESSION_TIMEOUT=3600

# إعدادات السجلات
LOG_LEVEL=INFO
"""
        
        with open(".env", "w", encoding="utf-8") as f:
            f.write(env_content)
        
        print("✅ تم إنشاء ملف .env")
    
    # إنشاء المجلدات
    create_directories()
    print("✅ تم إنشاء المجلدات المطلوبة")
    
    # عرض الإعدادات الحالية
    settings = get_settings()
    print(f"📋 إعدادات التطبيق:")
    print(f"   • اسم التطبيق: {settings.app_name}")
    print(f"   • الإصدار: {settings.app_version}")
    print(f"   • المضيف: {settings.host}:{settings.port}")
    print(f"   • وضع التطوير: {settings.debug}")
