
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
المساعد الذكي الموحد - النقطة الرئيسية للتشغيل
يدمج جميع الميزات والوحدات في نظام واحد متقدم
"""

import asyncio
import logging
import sys
import signal
from pathlib import Path
from typing import Optional

# إضافة مسار المشروع
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# الاستيرادات الأساسية
try:
    from config.settings import get_settings, validate_environment
    settings_available = True
except ImportError as e:
    print(f"⚠️ تعذر تحميل الإعدادات: {e}")
    settings_available = False

try:
    from core.module_manager import get_module_manager
    module_manager_available = True
except ImportError as e:
    print(f"⚠️ تعذر تحميل مدير الوحدات: {e}")
    module_manager_available = False

try:
    from core.unified_assistant_engine import get_assistant_engine
    assistant_engine_available = True
except ImportError as e:
    print(f"⚠️ تعذر تحميل محرك المساعد: {e}")
    assistant_engine_available = False

class UnifiedAssistantRunner:
    """مشغل المساعد الذكي الموحد"""
    
    def __init__(self):
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # المكونات الأساسية
        self.settings = None
        self.module_manager = None
        self.assistant_engine = None
        self.running = False
        
        # إعداد معالجات الإشارات
        self.setup_signal_handlers()
    
    def setup_logging(self):
        """إعداد نظام السجلات"""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # إعداد متقدم للسجلات
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('unified_assistant.log', encoding='utf-8')
            ]
        )
        
        # تخصيص مستويات السجلات
        logging.getLogger('transformers').setLevel(logging.WARNING)
        logging.getLogger('torch').setLevel(logging.WARNING)
    
    def setup_signal_handlers(self):
        """إعداد معالجات الإشارات للإيقاف الآمن"""
        def signal_handler(signum, frame):
            self.logger.info(f"تم استلام إشارة الإيقاف: {signum}")
            self.running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def initialize(self):
        """تهيئة جميع مكونات المساعد"""
        self.logger.info("🚀 بدء تهيئة المساعد الذكي الموحد...")
        
        try:
            # 1. تحميل الإعدادات
            if settings_available:
                self.settings = get_settings()
                if not validate_environment():
                    self.logger.warning("⚠️ بعض الإعدادات غير مكتملة")
                else:
                    self.logger.info("✅ تم تحميل الإعدادات بنجاح")
            else:
                self.logger.warning("⚠️ تشغيل بدون ملف الإعدادات")
            
            # 2. تهيئة مدير الوحدات
            if module_manager_available:
                self.module_manager = get_module_manager()
                successful, total = await self.module_manager.load_all_modules()
                
                if successful > 0:
                    self.logger.info(f"✅ تم تحميل {successful}/{total} وحدة")
                else:
                    self.logger.warning("⚠️ لم يتم تحميل أي وحدة - التشغيل في الوضع الأساسي")
            else:
                self.logger.warning("⚠️ مدير الوحدات غير متاح")
            
            # 3. تهيئة محرك المساعد
            if assistant_engine_available:
                self.assistant_engine = get_assistant_engine()
                await self.assistant_engine.initialize()
                self.logger.info("✅ تم تهيئة محرك المساعد")
            else:
                self.logger.warning("⚠️ محرك المساعد غير متاح - التشغيل في الوضع الأساسي")
            
            # 4. عرض تقرير التهيئة
            await self.show_initialization_report()
            
            self.running = True
            self.logger.info("🎉 تم تهيئة المساعد بنجاح!")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ فشل في التهيئة: {str(e)}")
            return False
    
    async def show_initialization_report(self):
        """عرض تقرير التهيئة"""
        print("\n" + "="*60)
        print("🤖 المساعد الذكي الموحد - تقرير التهيئة")
        print("="*60)
        
        # معلومات الإعدادات
        if self.settings:
            print(f"📁 مجلد المشروع: {self.settings.project_root}")
            print(f"🔧 وضع التطوير: {'نعم' if self.settings.debug_mode else 'لا'}")
            print(f"🌐 اللغة: {self.settings.language}")
        
        # معلومات الوحدات
        if self.module_manager:
            report = self.module_manager.get_status_report()
            print(f"\n📦 حالة الوحدات:")
            print(f"   • إجمالي: {report['total_modules']}")
            print(f"   • محملة: {report['status_counts']['loaded']}")
            print(f"   • فاشلة: {report['status_counts']['failed']}")
            
            # عرض الوحدات المحملة حسب النوع
            loaded_modules = {
                name: info for name, info in self.module_manager.modules.items()
                if info.status.value == "loaded"
            }
            
            if loaded_modules:
                print(f"\n✅ الوحدات المتاحة:")
                module_types = {}
                for name, info in loaded_modules.items():
                    if info.module_type not in module_types:
                        module_types[info.module_type] = []
                    module_types[info.module_type].append(name.split('.')[-1])
                
                for module_type, modules in module_types.items():
                    print(f"   🔹 {module_type}: {', '.join(modules)}")
        
        # الميزات المتاحة
        print(f"\n🎯 الميزات المتاحة:")
        features = [
            "معالجة اللغة الطبيعية",
            "التعلم النشط والتكيفي", 
            "نظام إدارة الوحدات المتقدم",
            "واجهة تفاعلية ذكية"
        ]
        
        if self.settings:
            if self.settings.ai_models.openai_api_key:
                features.append("تكامل GPT-4 المتقدم")
            if self.settings.interface.enable_voice:
                features.append("معالجة صوتية")
            if self.settings.interface.enable_vision:
                features.append("رؤية حاسوبية")
        
        for feature in features:
            print(f"   ✨ {feature}")
        
        print("\n" + "="*60)
    
    async def run(self):
        """تشغيل المساعد الرئيسي"""
        if not await self.initialize():
            self.logger.error("❌ فشل في التهيئة - إنهاء البرنامج")
            return
        
        try:
            # تشغيل الواجهة التفاعلية
            if self.assistant_engine:
                await self.assistant_engine.start_interactive_session()
            else:
                # وضع أساسي للتشغيل
                await self._basic_interactive_mode()
            
        except KeyboardInterrupt:
            self.logger.info("تم إيقاف المساعد بواسطة المستخدم")
        except Exception as e:
            self.logger.error(f"خطأ في تشغيل المساعد: {str(e)}")
        finally:
            await self.cleanup()
    
    async def cleanup(self):
        """تنظيف الموارد عند الإغلاق"""
        self.logger.info("🧹 تنظيف الموارد...")
        
        self.running = False
        
        # تنظيف المحرك
        if self.assistant_engine:
            # إضافة منطق التنظيف هنا
            pass
        
        # تنظيف الوحدات
        if self.module_manager:
            # إضافة منطق تنظيف الوحدات هنا
            pass
        
        self.logger.info("✅ تم تنظيف الموارد")
    
    async def run_web_interface(self):
        """تشغيل الواجهة الويب (مستقبلي)"""
        if not self.settings or not self.settings.interface.enable_web:
            self.logger.info("الواجهة الويب غير مفعلة")
            return
        
        self.logger.info(f"🌐 بدء الواجهة الويب على المنفذ {self.settings.interface.web_port}")
        # سيتم تطوير هذا لاحقاً
    
    async def _basic_interactive_mode(self):
        """وضع تفاعلي أساسي عندما لا تتوفر الميزات المتقدمة"""
        print("\n" + "="*60)
        print("🤖 المساعد الذكي - الوضع الأساسي")
        print("="*60)
        print("⚠️ بعض الميزات المتقدمة غير متاحة")
        print("💡 اكتب 'خروج' للخروج")
        print("="*60)
        
        while True:
            try:
                user_input = input("\n👤 أنت: ").strip()
                
                if user_input.lower() in ['خروج', 'quit', 'exit']:
                    print("👋 وداعاً!")
                    break
                
                if not user_input:
                    continue
                
                # استجابة أساسية
                if any(word in user_input.lower() for word in ["مرحبا", "أهلا", "سلام"]):
                    response = "أهلاً وسهلاً! كيف يمكنني مساعدتك؟"
                elif any(word in user_input.lower() for word in ["شكرا", "متشكر"]):
                    response = "عفواً! أسعدني أن أساعدك."
                elif any(word in user_input.lower() for word in ["وداعا", "مع السلامة"]):
                    response = "وداعاً! أتمنى لك يوماً سعيداً."
                else:
                    response = "أفهم ما تقوله. الميزات المتقدمة ستكون متاحة قريباً!"
                
                print(f"🤖 المساعد: {response}")
                
            except KeyboardInterrupt:
                print("\n👋 وداعاً!")
                break
            except Exception as e:
                print(f"❌ خطأ: {str(e)}")
    
    def get_status(self) -> dict:
        """الحصول على حالة المساعد"""
        status = {
            "running": self.running,
            "settings_loaded": self.settings is not None,
            "modules_loaded": 0,
            "engine_ready": self.assistant_engine is not None
        }
        
        if self.module_manager:
            report = self.module_manager.get_status_report()
            status["modules_loaded"] = report['status_counts']['loaded']
            status["total_modules"] = report['total_modules']
        
        return status

async def main():
    """الدالة الرئيسية"""
    print("🤖 بدء تشغيل المساعد الذكي الموحد...")
    
    runner = UnifiedAssistantRunner()
    
    try:
        await runner.run()
    except Exception as e:
        print(f"❌ خطأ فادح: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    # تشغيل المساعد
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 تم إيقاف المساعد")
    except Exception as e:
        print(f"❌ خطأ في التشغيل: {str(e)}")
        sys.exit(1)
