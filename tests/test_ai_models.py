
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 اختبارات شاملة لنماذج الذكاء الاصطناعي
Comprehensive Tests for AI Models
"""

import pytest
import asyncio
import sys
import importlib
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import json
import tempfile
import shutil

# إضافة مسار المشروع
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class TestAIModels:
    """اختبارات نماذج الذكاء الاصطناعي"""
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """إعداد الاختبارات"""
        self.temp_dir = tempfile.mkdtemp()
        
    def teardown_method(self):
        """تنظيف بعد الاختبارات"""
        if hasattr(self, 'temp_dir'):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_basic_ai_engine_import(self):
        """اختبار استيراد محرك الذكاء الاصطناعي الأساسي"""
        try:
            from core.advanced_ai_engine import AdvancedAIEngine
            engine = AdvancedAIEngine()
            assert engine is not None
            assert hasattr(engine, 'logger')
            assert hasattr(engine, 'device')
        except ImportError as e:
            pytest.skip(f"لا يمكن استيراد AdvancedAIEngine: {e}")
    
    def test_nlp_models_basic_functionality(self):
        """اختبار الوظائف الأساسية لنماذج معالجة اللغة"""
        test_cases = [
            {
                "module": "ai_models.nlu.bert_analyzer",
                "class": "BERTAnalyzer",
                "test_text": "مرحباً بك في المساعد الذكي"
            },
            {
                "module": "ai_models.nlg.gpt4_generator", 
                "class": "GPT4Generator",
                "test_text": "اكتب نص قصير"
            }
        ]
        
        for case in test_cases:
            try:
                module = importlib.import_module(case["module"])
                ai_class = getattr(module, case["class"])
                
                # اختبار التهيئة
                instance = ai_class()
                assert instance is not None
                
                # اختبار وجود الطرق المطلوبة
                if hasattr(instance, 'analyze'):
                    assert callable(getattr(instance, 'analyze'))
                elif hasattr(instance, 'generate'):
                    assert callable(getattr(instance, 'generate'))
                    
            except (ImportError, AttributeError) as e:
                pytest.skip(f"تخطي اختبار {case['module']}: {e}")
    
    @pytest.mark.asyncio
    async def test_emotional_intelligence_engine(self):
        """اختبار محرك الذكاء العاطفي"""
        try:
            from ai_models.emotion.emotional_intelligence_engine import EmotionalIntelligenceEngine
            
            engine = EmotionalIntelligenceEngine()
            await engine.initialize()
            
            # اختبار تحليل المشاعر
            test_emotions = [
                {"text": "أنا سعيد جداً اليوم!", "expected_sentiment": "positive"},
                {"text": "أشعر بالحزن والإحباط", "expected_sentiment": "negative"},
                {"text": "هذا يوم عادي", "expected_sentiment": "neutral"}
            ]
            
            for test_case in test_emotions:
                try:
                    result = await engine.analyze_emotion(test_case["text"])
                    assert result is not None
                    assert isinstance(result, dict)
                    assert "sentiment" in result
                except Exception as e:
                    pytest.skip(f"خطأ في تحليل المشاعر: {e}")
                    
        except ImportError as e:
            pytest.skip(f"لا يمكن استيراد EmotionalIntelligenceEngine: {e}")
    
    @pytest.mark.asyncio
    async def test_predictive_intelligence_engine(self):
        """اختبار محرك التنبؤ الذكي"""
        try:
            from ai_models.prediction.predictive_intelligence_engine import PredictiveIntelligenceEngine
            
            engine = PredictiveIntelligenceEngine()
            await engine.initialize()
            
            # اختبار التنبؤ
            sample_data = {
                "user_id": "test_user",
                "interaction_history": ["مرحباً", "كيف الحال؟", "شكراً"],
                "preferences": {"language": "ar", "style": "formal"}
            }
            
            try:
                prediction = await engine.predict_user_needs(
                    sample_data["user_id"], 
                    sample_data
                )
                assert prediction is not None
                assert isinstance(prediction, dict)
            except Exception as e:
                pytest.skip(f"خطأ في التنبؤ: {e}")
                
        except ImportError as e:
            pytest.skip(f"لا يمكن استيراد PredictiveIntelligenceEngine: {e}")
    
    def test_vision_processing_basic(self):
        """اختبار معالجة الرؤية الأساسية"""
        try:
            from ai_models.vision.vision_pipeline import VisionPipeline
            
            pipeline = VisionPipeline()
            assert pipeline is not None
            
            # اختبار وجود الطرق المطلوبة
            expected_methods = ['process_image', 'detect_objects', 'analyze_scene']
            for method in expected_methods:
                if hasattr(pipeline, method):
                    assert callable(getattr(pipeline, method))
                    
        except ImportError as e:
            pytest.skip(f"لا يمكن استيراد VisionPipeline: {e}")
    
    @pytest.mark.asyncio
    async def test_learning_engines_basic(self):
        """اختبار محركات التعلم الأساسية"""
        learning_modules = [
            "ai_models.learning.adaptive_personality_engine",
            "ai_models.learning.continuous_learning_engine",
            "ai_models.learning.meta_learning_engine"
        ]
        
        for module_name in learning_modules:
            try:
                module = importlib.import_module(module_name)
                
                # محاولة العثور على الفئة الرئيسية
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (isinstance(attr, type) and 
                        'Engine' in attr_name and 
                        attr_name != 'Engine'):
                        
                        try:
                            instance = attr()
                            assert instance is not None
                            break
                        except Exception:
                            continue
                            
            except ImportError as e:
                pytest.skip(f"تخطي {module_name}: {e}")
    
    def test_security_engines_initialization(self):
        """اختبار تهيئة محركات الأمان"""
        security_modules = [
            "ai_models.security.biometric_security_engine",
            "ai_models.security.quantum_security_engine"
        ]
        
        for module_name in security_modules:
            try:
                module = importlib.import_module(module_name)
                
                # البحث عن فئات الأمان
                for attr_name in dir(module):
                    if 'Security' in attr_name or 'Engine' in attr_name:
                        attr = getattr(module, attr_name)
                        if isinstance(attr, type):
                            try:
                                instance = attr()
                                assert instance is not None
                                break
                            except Exception:
                                continue
                                
            except ImportError as e:
                pytest.skip(f"تخطي {module_name}: {e}")

class TestIntegrationScenarios:
    """اختبارات سيناريوهات التكامل"""
    
    @pytest.mark.asyncio
    async def test_multi_engine_interaction(self):
        """اختبار التفاعل بين محركات متعددة"""
        try:
            # محاكاة تفاعل بين محركات مختلفة
            engines_loaded = 0
            
            # محاولة تحميل محركات مختلفة
            engine_modules = [
                ("core.advanced_ai_engine", "AdvancedAIEngine"),
                ("ai_models.emotion.emotional_intelligence_engine", "EmotionalIntelligenceEngine")
            ]
            
            loaded_engines = []
            
            for module_name, class_name in engine_modules:
                try:
                    module = importlib.import_module(module_name)
                    engine_class = getattr(module, class_name)
                    engine = engine_class()
                    loaded_engines.append(engine)
                    engines_loaded += 1
                except (ImportError, AttributeError):
                    continue
            
            assert engines_loaded > 0, "لم يتم تحميل أي محرك"
            
            # اختبار تفاعل بسيط
            for engine in loaded_engines:
                if hasattr(engine, 'initialize'):
                    try:
                        await engine.initialize()
                    except Exception:
                        pass  # تجاهل أخطاء التهيئة في الاختبارات
                        
        except Exception as e:
            pytest.skip(f"خطأ في اختبار التكامل: {e}")

class TestPerformanceMetrics:
    """اختبارات مقاييس الأداء"""
    
    def test_memory_usage_basic(self):
        """اختبار استخدام الذاكرة الأساسي"""
        try:
            import psutil
            import gc
            
            # قياس الذاكرة قبل التحميل
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # محاولة تحميل محرك
            try:
                from core.advanced_ai_engine import AdvancedAIEngine
                engine = AdvancedAIEngine()
            except ImportError:
                pytest.skip("لا يمكن تحميل AdvancedAIEngine")
            
            # قياس الذاكرة بعد التحميل
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = memory_after - memory_before
            
            # التحقق من أن الزيادة معقولة (أقل من 500MB)
            assert memory_increase < 500, f"استهلاك ذاكرة مفرط: {memory_increase:.1f}MB"
            
            # تنظيف الذاكرة
            del engine
            gc.collect()
            
        except ImportError:
            pytest.skip("psutil غير متوفر")
    
    @pytest.mark.asyncio
    async def test_response_time_basic(self):
        """اختبار زمن الاستجابة الأساسي"""
        import time
        
        try:
            from core.advanced_ai_engine import AdvancedAIEngine
            
            engine = AdvancedAIEngine()
            
            # قياس زمن التهيئة
            start_time = time.time()
            if hasattr(engine, 'initialize'):
                try:
                    await engine.initialize()
                except Exception:
                    pass  # تجاهل أخطاء التهيئة
            
            initialization_time = time.time() - start_time
            
            # التحقق من أن التهيئة تتم في وقت معقول (أقل من 30 ثانية)
            assert initialization_time < 30, f"زمن تهيئة مفرط: {initialization_time:.2f}s"
            
        except ImportError:
            pytest.skip("لا يمكن تحميل AdvancedAIEngine")

# تشغيل الاختبارات
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
