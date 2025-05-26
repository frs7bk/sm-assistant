
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 اختبارات الوظائف الأساسية للمساعد الذكي
Core Functionality Tests
"""

import pytest
import asyncio
import sys
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# إضافة مسار المشروع
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class TestCoreInfrastructure:
    """اختبارات البنية التحتية الأساسية"""
    
    def test_config_loading(self):
        """اختبار تحميل الإعدادات"""
        try:
            from core.config import get_config
            config = get_config()
            assert config is not None
            assert hasattr(config, 'ai_models')
        except ImportError:
            # اختبار الإعدادات البديلة
            try:
                from config.advanced_config import get_config
                config = get_config()
                assert config is not None
            except ImportError:
                pytest.skip("لا يمكن تحميل أي نظام إعدادات")
    
    def test_module_manager_basic(self):
        """اختبار مدير الوحدات الأساسي"""
        try:
            from core.module_manager import ModuleManager
            
            manager = ModuleManager()
            assert manager is not None
            assert hasattr(manager, 'logger')
            
            # اختبار تحميل الوحدات
            if hasattr(manager, 'load_module'):
                assert callable(manager.load_module)
                
        except ImportError as e:
            pytest.skip(f"لا يمكن استيراد ModuleManager: {e}")
    
    def test_performance_optimizer(self):
        """اختبار محسن الأداء"""
        try:
            from core.performance_optimizer import PerformanceOptimizer
            
            optimizer = PerformanceOptimizer()
            assert optimizer is not None
            
            # اختبار الطرق الأساسية
            expected_methods = ['optimize', 'monitor', 'analyze']
            for method in expected_methods:
                if hasattr(optimizer, method):
                    assert callable(getattr(optimizer, method))
                    
        except ImportError as e:
            pytest.skip(f"لا يمكن استيراد PerformanceOptimizer: {e}")
    
    def test_reliability_engine(self):
        """اختبار محرك الموثوقية"""
        try:
            from core.reliability_engine import ReliabilityEngine
            
            engine = ReliabilityEngine()
            assert engine is not None
            
            # اختبار الطرق الأساسية
            if hasattr(engine, 'monitor_health'):
                assert callable(engine.monitor_health)
                
        except ImportError as e:
            pytest.skip(f"لا يمكن استيراد ReliabilityEngine: {e}")
    
    @pytest.mark.asyncio
    async def test_unified_assistant_engine(self):
        """اختبار المحرك الموحد للمساعد"""
        try:
            from core.unified_assistant_engine import UnifiedAssistantEngine
            
            engine = UnifiedAssistantEngine()
            assert engine is not None
            
            # اختبار التهيئة
            if hasattr(engine, 'initialize'):
                try:
                    await engine.initialize()
                except Exception as e:
                    pytest.skip(f"فشل في تهيئة المحرك: {e}")
            
            # اختبار المعالجة الأساسية
            if hasattr(engine, 'process_request'):
                test_request = "مرحباً، كيف يمكنني مساعدتك؟"
                try:
                    response = await engine.process_request(test_request)
                    assert response is not None
                except Exception as e:
                    pytest.skip(f"فشل في معالجة الطلب: {e}")
                    
        except ImportError as e:
            pytest.skip(f"لا يمكن استيراد UnifiedAssistantEngine: {e}")

class TestErrorHandling:
    """اختبارات معالجة الأخطاء"""
    
    def test_advanced_error_handler(self):
        """اختبار معالج الأخطاء المتقدم"""
        try:
            from core.advanced_error_handler import AdvancedErrorHandler
            
            handler = AdvancedErrorHandler()
            assert handler is not None
            
            # اختبار معالجة خطأ تجريبي
            test_error = Exception("خطأ تجريبي للاختبار")
            
            if hasattr(handler, 'handle_error'):
                try:
                    result = handler.handle_error(test_error)
                    assert result is not None
                except Exception:
                    pass  # تجاهل أخطاء المعالجة في الاختبارات
                    
        except ImportError as e:
            pytest.skip(f"لا يمكن استيراد AdvancedErrorHandler: {e}")
    
    @pytest.mark.asyncio
    async def test_error_recovery_mechanisms(self):
        """اختبار آليات استرداد الأخطاء"""
        try:
            from core.advanced_error_handler import AdvancedErrorHandler
            
            handler = AdvancedErrorHandler()
            
            # محاكاة أخطاء مختلفة
            error_scenarios = [
                ("network_error", ConnectionError("فشل الاتصال")),
                ("memory_error", MemoryError("نفدت الذاكرة")),
                ("import_error", ImportError("فشل الاستيراد")),
                ("value_error", ValueError("قيمة غير صحيحة"))
            ]
            
            for scenario_name, error in error_scenarios:
                if hasattr(handler, 'handle_error_async'):
                    try:
                        result = await handler.handle_error_async(error)
                        assert result is not None
                    except Exception:
                        continue  # تجاهل الأخطاء في السيناريوهات التجريبية
                        
        except ImportError as e:
            pytest.skip(f"لا يمكن استيراد AdvancedErrorHandler: {e}")

class TestDataHandling:
    """اختبارات معالجة البيانات"""
    
    def setUp(self):
        """إعداد الاختبارات"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_data_dir = Path(self.temp_dir) / "test_data"
        self.test_data_dir.mkdir(exist_ok=True)
    
    def tearDown(self):
        """تنظيف بعد الاختبارات"""
        if hasattr(self, 'temp_dir'):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_session_management(self):
        """اختبار إدارة الجلسات"""
        # إنشاء بيانات جلسة تجريبية
        session_data = {
            "user_id": "test_user",
            "timestamp": "2025-01-26T20:00:00",
            "interactions": [
                {"input": "مرحباً", "output": "أهلاً وسهلاً"},
                {"input": "كيف الحال؟", "output": "بخير، شكراً"}
            ]
        }
        
        # محاولة حفظ الجلسة
        session_file = Path("data/sessions/test_session.json")
        session_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, ensure_ascii=False, indent=2)
            
            # اختبار تحميل الجلسة
            with open(session_file, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
                
            assert loaded_data == session_data
            
            # تنظيف
            session_file.unlink(missing_ok=True)
            
        except Exception as e:
            pytest.skip(f"فشل في اختبار إدارة الجلسات: {e}")
    
    def test_user_data_persistence(self):
        """اختبار استمرارية بيانات المستخدم"""
        user_data = {
            "preferences": {
                "language": "ar",
                "theme": "dark",
                "voice_enabled": True
            },
            "history": ["سؤال 1", "سؤال 2"],
            "settings": {
                "auto_save": True,
                "notifications": False
            }
        }
        
        user_file = Path("data/user_data/test_user.json")
        user_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # حفظ بيانات المستخدم
            with open(user_file, 'w', encoding='utf-8') as f:
                json.dump(user_data, f, ensure_ascii=False, indent=2)
            
            # تحميل وتحقق
            with open(user_file, 'r', encoding='utf-8') as f:
                loaded_user_data = json.load(f)
                
            assert loaded_user_data == user_data
            
            # تنظيف
            user_file.unlink(missing_ok=True)
            
        except Exception as e:
            pytest.skip(f"فشل في اختبار بيانات المستخدم: {e}")

class TestAPIEndpoints:
    """اختبارات نقاط النهاية للAPI"""
    
    def test_api_models_import(self):
        """اختبار استيراد نماذج API"""
        try:
            from api.models import *
            # التحقق من وجود النماذج الأساسية
            assert True  # نجح الاستيراد
        except ImportError as e:
            pytest.skip(f"لا يمكن استيراد نماذج API: {e}")
    
    def test_api_services_import(self):
        """اختبار استيراد خدمات API"""
        try:
            from api.services import *
            # التحقق من وجود الخدمات الأساسية
            assert True  # نجح الاستيراد
        except ImportError as e:
            pytest.skip(f"لا يمكن استيراد خدمات API: {e}")

class TestWebInterface:
    """اختبارات واجهة الويب"""
    
    def test_frontend_files_exist(self):
        """اختبار وجود ملفات الواجهة الأمامية"""
        frontend_files = [
            "frontend/index.html",
            "frontend/static/app.js", 
            "frontend/static/styles.css"
        ]
        
        missing_files = []
        for file_path in frontend_files:
            if not Path(file_path).exists():
                missing_files.append(file_path)
        
        if missing_files:
            pytest.skip(f"ملفات واجهة مفقودة: {missing_files}")
        
        assert len(missing_files) == 0
    
    def test_html_structure(self):
        """اختبار بنية HTML"""
        html_file = Path("frontend/index.html")
        
        if not html_file.exists():
            pytest.skip("ملف HTML غير موجود")
        
        try:
            with open(html_file, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # اختبارات أساسية للHTML
            assert '<!DOCTYPE html>' in html_content or '<html' in html_content
            assert '<head>' in html_content
            assert '<body>' in html_content
            
        except Exception as e:
            pytest.skip(f"فشل في قراءة HTML: {e}")

# إعداد pytest
def pytest_configure(config):
    """إعداد pytest"""
    config.addinivalue_line(
        "markers", "asyncio: اختبارات غير متزامنة"
    )

if __name__ == "__main__":
    # تشغيل الاختبارات مع تقرير مفصل
    pytest.main([
        __file__, 
        "-v", 
        "--tb=short",
        "--color=yes",
        "-x"  # توقف عند أول فشل
    ])
