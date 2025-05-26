
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 إعدادات pytest وتركيبات الاختبار
pytest Configuration and Fixtures
"""

import pytest
import asyncio
import tempfile
import shutil
import sys
import logging
from pathlib import Path
from unittest.mock import Mock, patch

# إضافة مسار المشروع
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# إعداد السجلات للاختبارات
logging.basicConfig(
    level=logging.WARNING,  # تقليل الضوضاء أثناء الاختبارات
    format='%(levelname)s - %(name)s - %(message)s'
)

@pytest.fixture(scope="session")
def event_loop():
    """إنشاء حلقة أحداث للاختبارات غير المتزامنة"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def temp_directory():
    """إنشاء مجلد مؤقت للاختبارات"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture
def mock_config():
    """إعداد تهيئة وهمية للاختبارات"""
    config = Mock()
    config.ai_models = Mock()
    config.ai_models.device = "cpu"
    config.ai_models.max_batch_size = 1
    config.ai_models.use_gpu = False
    
    config.logging = Mock()
    config.logging.level = "WARNING"
    
    config.database = Mock()
    config.database.path = ":memory:"
    
    return config

@pytest.fixture
def mock_logger():
    """إنشاء مسجل وهمي للاختبارات"""
    logger = Mock()
    logger.info = Mock()
    logger.warning = Mock()
    logger.error = Mock()
    logger.debug = Mock()
    return logger

@pytest.fixture
def sample_user_data():
    """بيانات مستخدم نموذجية للاختبارات"""
    return {
        "user_id": "test_user_123",
        "preferences": {
            "language": "ar",
            "theme": "light",
            "voice_enabled": False
        },
        "interaction_history": [
            {
                "timestamp": "2025-01-26T20:00:00",
                "input": "مرحباً",
                "output": "أهلاً وسهلاً",
                "sentiment": "positive"
            },
            {
                "timestamp": "2025-01-26T20:01:00", 
                "input": "كيف يمكنك مساعدتي؟",
                "output": "يمكنني مساعدتك في العديد من المجالات",
                "sentiment": "neutral"
            }
        ],
        "learned_patterns": {
            "common_topics": ["تقنية", "تعليم", "عمل"],
            "preferred_response_style": "formal",
            "typical_session_length": 15
        }
    }

@pytest.fixture
def sample_ai_models():
    """نماذج ذكاء اصطناعي وهمية للاختبارات"""
    models = {}
    
    # نموذج NLP وهمي
    nlp_model = Mock()
    nlp_model.analyze = Mock(return_value={
        "sentiment": "positive",
        "confidence": 0.85,
        "entities": []
    })
    models["nlp"] = nlp_model
    
    # نموذج توليد وهمي
    generation_model = Mock()
    generation_model.generate = Mock(return_value={
        "text": "إجابة مولدة تلقائياً",
        "confidence": 0.90
    })
    models["generation"] = generation_model
    
    # نموذج رؤية وهمي
    vision_model = Mock()
    vision_model.analyze_image = Mock(return_value={
        "objects": ["شخص", "سيارة"],
        "scene": "شارع",
        "confidence": 0.75
    })
    models["vision"] = vision_model
    
    return models

@pytest.fixture
def mock_openai_client():
    """عميل OpenAI وهمي للاختبارات"""
    with patch('openai.OpenAI') as mock_client:
        # إعداد استجابات وهمية
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "استجابة GPT وهمية"
        
        mock_client.return_value.chat.completions.create.return_value = mock_response
        yield mock_client.return_value

@pytest.fixture
def test_database_path(temp_directory):
    """مسار قاعدة بيانات للاختبارات"""
    db_path = temp_directory / "test_database.db"
    yield str(db_path)

@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch, temp_directory):
    """إعداد بيئة الاختبار التلقائية"""
    # تعيين متغيرات البيئة للاختبارات
    monkeypatch.setenv("TESTING", "true")
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    monkeypatch.setenv("AI_DEVICE", "cpu")
    monkeypatch.setenv("USE_GPU", "false")
    
    # إنشاء مجلدات البيانات المؤقتة
    test_data_dir = temp_directory / "data"
    test_data_dir.mkdir(exist_ok=True)
    
    # إنشاء مجلدات فرعية
    for subdir in ["sessions", "user_data", "models", "logs", "cache"]:
        (test_data_dir / subdir).mkdir(exist_ok=True)
    
    # تحديد مسار البيانات للاختبارات
    monkeypatch.setenv("DATA_DIR", str(test_data_dir))

@pytest.fixture
def mock_torch():
    """PyTorch وهمي للاختبارات"""
    with patch('torch.cuda') as mock_cuda:
        mock_cuda.is_available.return_value = False
        mock_cuda.device_count.return_value = 0
        
        with patch('torch.device') as mock_device:
            mock_device.return_value = "cpu"
            yield mock_device

@pytest.fixture
def sample_text_data():
    """بيانات نصية نموذجية للاختبارات"""
    return {
        "arabic_texts": [
            "مرحباً بك في المساعد الذكي",
            "كيف يمكنني مساعدتك اليوم؟",
            "أشعر بالسعادة لوجودك هنا",
            "هذا نص تجريبي للاختبار"
        ],
        "english_texts": [
            "Hello and welcome to the AI assistant",
            "How can I help you today?",
            "I'm happy to have you here",
            "This is a test text for testing"
        ],
        "mixed_content": [
            "Hello مرحباً",
            "Thank you شكراً لك",
            "AI الذكاء الاصطناعي"
        ]
    }

@pytest.fixture
def performance_thresholds():
    """عتبات الأداء للاختبارات"""
    return {
        "max_response_time": 5.0,  # ثواني
        "max_memory_usage": 500,   # ميجابايت
        "min_accuracy": 0.7,       # 70%
        "max_error_rate": 0.1      # 10%
    }

# معالجات pytest
def pytest_collection_modifyitems(config, items):
    """تعديل عناصر الجمع لـ pytest"""
    for item in items:
        # إضافة علامات للاختبارات غير المتزامنة
        if asyncio.iscoroutinefunction(item.function):
            item.add_marker(pytest.mark.asyncio)

def pytest_runtest_setup(item):
    """إعداد قبل تشغيل كل اختبار"""
    # تنظيف الذاكرة قبل كل اختبار
    import gc
    gc.collect()

def pytest_runtest_teardown(item):
    """تنظيف بعد تشغيل كل اختبار"""
    # تنظيف الذاكرة بعد كل اختبار
    import gc
    gc.collect()

# إعدادات pytest
pytest_plugins = ["pytest_asyncio"]
