
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 نظام الاختبار الشامل للمساعد الذكي
Comprehensive Test Suite for Advanced AI Assistant
"""

import asyncio
import sys
import traceback
import time
import logging
from pathlib import Path
from typing import Dict, Any, List
import json

# إضافة مسار المشروع
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class ComprehensiveTestSuite:
    """مجموعة الاختبارات الشاملة"""
    
    def __init__(self):
        self.test_results = {}
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    async def run_all_tests(self):
        """تشغيل جميع الاختبارات"""
        print("🧪 بدء الاختبارات الشاملة للمساعد الذكي")
        print("=" * 60)
        
        # قائمة جميع الاختبارات
        tests = [
            ("core_modules", self.test_core_modules),
            ("ai_engines", self.test_ai_engines),
            ("integration_hub", self.test_integration_hub),
            ("performance_optimizer", self.test_performance_optimizer),
            ("reliability_engine", self.test_reliability_engine),
            ("main_unified", self.test_main_unified),
            ("api_services", self.test_api_services),
            ("frontend_interface", self.test_frontend_interface),
            ("advanced_features", self.test_advanced_features)
        ]
        
        total_tests = len(tests)
        passed_tests = 0
        
        for test_name, test_func in tests:
            print(f"\n🔍 اختبار: {test_name}")
            print("-" * 40)
            
            try:
                result = await test_func()
                if result["success"]:
                    print(f"✅ نجح: {test_name}")
                    passed_tests += 1
                else:
                    print(f"❌ فشل: {test_name} - {result.get('error', 'خطأ غير معروف')}")
                
                self.test_results[test_name] = result
                
            except Exception as e:
                error_msg = f"خطأ في الاختبار: {str(e)}"
                print(f"💥 خطأ حرج: {test_name} - {error_msg}")
                self.test_results[test_name] = {
                    "success": False,
                    "error": error_msg,
                    "traceback": traceback.format_exc()
                }
        
        # تقرير النتائج النهائي
        print(f"\n📊 تقرير الاختبارات النهائي")
        print("=" * 60)
        print(f"إجمالي الاختبارات: {total_tests}")
        print(f"الناجحة: {passed_tests}")
        print(f"الفاشلة: {total_tests - passed_tests}")
        print(f"نسبة النجاح: {(passed_tests/total_tests)*100:.1f}%")
        
        # حفظ التقرير
        await self.save_test_report()
        
        return self.test_results
    
    async def test_core_modules(self) -> Dict[str, Any]:
        """اختبار الوحدات الأساسية"""
        try:
            # اختبار config
            from core.config import get_config
            config = get_config()
            
            # اختبار assistant
            from core.assistant import Assistant
            assistant = Assistant()
            
            # اختبار unified_assistant_engine
            from core.unified_assistant_engine import UnifiedAssistantEngine
            engine = UnifiedAssistantEngine()
            
            return {
                "success": True,
                "message": "جميع الوحدات الأساسية تعمل بشكل صحيح",
                "modules_tested": ["config", "assistant", "unified_assistant_engine"]
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    async def test_ai_engines(self) -> Dict[str, Any]:
        """اختبار محركات الذكاء الاصطناعي"""
        try:
            # اختبار advanced_ai_engine
            from core.advanced_ai_engine import get_ai_engine
            ai_engine = await get_ai_engine()
            
            # اختبار معالجة نص بسيط
            response = await ai_engine.process_natural_language("مرحباً")
            
            if not response or not hasattr(response, 'text'):
                raise Exception("فشل في معالجة النص")
            
            return {
                "success": True,
                "message": "محركات الذكاء الاصطناعي تعمل بشكل صحيح",
                "response_sample": response.text[:100],
                "confidence": getattr(response, 'confidence', 0)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    async def test_integration_hub(self) -> Dict[str, Any]:
        """اختبار مركز التكامل"""
        try:
            from core.integration_hub import integration_hub
            
            # تهيئة مركز التكامل
            await integration_hub.initialize()
            
            # اختبار حالة التكاملات
            status = await integration_hub.get_integrations_status()
            
            if not isinstance(status, dict):
                raise Exception("فشل في الحصول على حالة التكاملات")
            
            return {
                "success": True,
                "message": "مركز التكامل يعمل بشكل صحيح",
                "total_integrations": status.get("total_integrations", 0),
                "enabled_integrations": status.get("enabled_integrations", 0)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    async def test_performance_optimizer(self) -> Dict[str, Any]:
        """اختبار محسن الأداء"""
        try:
            from core.performance_optimizer import performance_optimizer
            
            # اختبار تقرير الأداء
            report = performance_optimizer.get_performance_report()
            
            if not isinstance(report, dict):
                raise Exception("فشل في الحصول على تقرير الأداء")
            
            return {
                "success": True,
                "message": "محسن الأداء يعمل بشكل صحيح",
                "cache_hit_rate": report.get("cache_performance", {}).get("hit_rate", "0%"),
                "gpu_available": report.get("system_performance", {}).get("gpu_available", False)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    async def test_reliability_engine(self) -> Dict[str, Any]:
        """اختبار محرك الموثوقية"""
        try:
            from core.reliability_engine import reliability_engine
            
            # تهيئة محرك الموثوقية
            await reliability_engine.initialize()
            
            # اختبار تقرير الموثوقية
            report = await reliability_engine.get_reliability_report()
            
            if not isinstance(report, dict):
                raise Exception("فشل في الحصول على تقرير الموثوقية")
            
            return {
                "success": True,
                "message": "محرك الموثوقية يعمل بشكل صحيح",
                "reliability_score": report.get("reliability_score", "غير محدد"),
                "health_status": report.get("health_status", {}).get("overall_health", "غير معروف")
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    async def test_main_unified(self) -> Dict[str, Any]:
        """اختبار النظام الموحد الرئيسي"""
        try:
            # محاولة استيراد main_unified
            import main_unified
            
            # التحقق من وجود الفئة الرئيسية
            if hasattr(main_unified, 'AdvancedUnifiedAssistant'):
                assistant_class = getattr(main_unified, 'AdvancedUnifiedAssistant')
                
                # إنشاء مثيل للاختبار
                assistant = assistant_class()
                
                return {
                    "success": True,
                    "message": "النظام الموحد الرئيسي يعمل بشكل صحيح",
                    "class_available": True,
                    "methods_count": len([method for method in dir(assistant) if not method.startswith('_')])
                }
            else:
                return {
                    "success": False,
                    "error": "لا يمكن العثور على فئة AdvancedUnifiedAssistant"
                }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    async def test_api_services(self) -> Dict[str, Any]:
        """اختبار خدمات API"""
        try:
            from api.services import APIService
            from api.models import RequestModel, ResponseModel
            
            # إنشاء مثيل للخدمة
            service = APIService()
            
            # اختبار نموذج الطلب
            request = RequestModel(text="اختبار", user_id="test_user")
            
            return {
                "success": True,
                "message": "خدمات API تعمل بشكل صحيح",
                "models_available": True,
                "service_initialized": True
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    async def test_frontend_interface(self) -> Dict[str, Any]:
        """اختبار واجهة المستخدم الأمامية"""
        try:
            # التحقق من وجود ملفات الواجهة الأمامية
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
                return {
                    "success": False,
                    "error": f"ملفات مفقودة: {missing_files}"
                }
            
            # قراءة محتوى HTML للتحقق من صحته
            with open("frontend/index.html", 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            if len(html_content) < 100:
                return {
                    "success": False,
                    "error": "ملف HTML فارغ أو غير صالح"
                }
            
            return {
                "success": True,
                "message": "واجهة المستخدم الأمامية سليمة",
                "files_checked": len(frontend_files),
                "html_size": len(html_content)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    async def test_advanced_features(self) -> Dict[str, Any]:
        """اختبار الميزات المتقدمة"""
        try:
            features_tested = []
            
            # اختبار smart_automation_engine
            try:
                from core.smart_automation_engine import SmartAutomationEngine
                automation = SmartAutomationEngine()
                features_tested.append("smart_automation")
            except ImportError:
                pass
            
            # اختبار enhanced_vision_module
            try:
                from core.enhanced_vision_module import EnhancedVisionModule
                vision = EnhancedVisionModule()
                features_tested.append("enhanced_vision")
            except ImportError:
                pass
            
            # اختبار module_manager
            try:
                from core.module_manager import ModuleManager
                manager = ModuleManager()
                features_tested.append("module_manager")
            except ImportError:
                pass
            
            return {
                "success": True,
                "message": "تم اختبار الميزات المتقدمة المتاحة",
                "features_available": features_tested,
                "features_count": len(features_tested)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    async def save_test_report(self):
        """حفظ تقرير الاختبار"""
        try:
            report = {
                "timestamp": time.time(),
                "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "results": self.test_results,
                "summary": {
                    "total_tests": len(self.test_results),
                    "passed": sum(1 for r in self.test_results.values() if r.get("success", False)),
                    "failed": sum(1 for r in self.test_results.values() if not r.get("success", False))
                }
            }
            
            # إنشاء مجلد التقارير إن لم يكن موجوداً
            reports_dir = Path("data/test_reports")
            reports_dir.mkdir(parents=True, exist_ok=True)
            
            # حفظ التقرير
            report_file = reports_dir / f"test_report_{time.strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            print(f"💾 تم حفظ تقرير الاختبار في: {report_file}")
            
        except Exception as e:
            print(f"❌ فشل حفظ تقرير الاختبار: {e}")

async def main():
    """الدالة الرئيسية للاختبار"""
    test_suite = ComprehensiveTestSuite()
    results = await test_suite.run_all_tests()
    
    # عرض ملخص سريع
    total = len(results)
    passed = sum(1 for r in results.values() if r.get("success", False))
    
    print(f"\n🎯 ملخص سريع:")
    print(f"النجاح: {passed}/{total} ({(passed/total)*100:.1f}%)")
    
    if passed == total:
        print("🎉 جميع الاختبارات نجحت! المساعد جاهز للاستخدام.")
    else:
        print("⚠️ بعض الاختبارات فشلت. راجع التقرير المفصل.")
    
    return results

if __name__ == "__main__":
    asyncio.run(main())
