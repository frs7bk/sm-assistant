
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🤖 ملف التشغيل الرئيسي المتطور للمساعد الذكي الموحد
Advanced Startup Script for Unified AI Assistant v3.0.0
"""

import sys
import os
import asyncio
import argparse
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import time
from datetime import datetime

# إضافة المجلد الجذر للمسار
sys.path.insert(0, str(Path(__file__).parent))

class AdvancedStartupManager:
    """مدير التشغيل المتقدم"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.config = self._load_startup_config()
        self.services = {}
        self.startup_modes = {
            'interactive': 'وضع المحادثة التفاعلية',
            'web': 'واجهة الويب الكاملة', 
            'api': 'خادم API فقط',
            'unified': 'جميع الخدمات موحدة',
            'dev': 'وضع التطوير',
            'production': 'وضع الإنتاج'
        }
        
    def _setup_logging(self) -> logging.Logger:
        """إعداد نظام السجلات المتقدم"""
        log_dir = Path("data/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"startup_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        logger = logging.getLogger(__name__)
        logger.info("🚀 تهيئة نظام السجلات المتقدم")
        return logger
    
    def _load_startup_config(self) -> Dict:
        """تحميل إعدادات التشغيل"""
        config_file = Path("config/startup_config.json")
        
        default_config = {
            "auto_install_dependencies": True,
            "check_system_requirements": True,
            "enable_auto_updates": False,
            "max_startup_time": 60,
            "health_check_interval": 30,
            "services": {
                "core_engine": {"enabled": True, "priority": 1},
                "ai_models": {"enabled": True, "priority": 2},
                "analytics": {"enabled": True, "priority": 3},
                "web_interface": {"enabled": True, "priority": 4},
                "api_server": {"enabled": True, "priority": 5}
            }
        }
        
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    return {**default_config, **json.load(f)}
            except Exception as e:
                self.logger.warning(f"خطأ في تحميل الإعدادات: {e}")
        
        return default_config
    
    def check_system_requirements(self) -> bool:
        """فحص متطلبات النظام"""
        self.logger.info("🔍 فحص متطلبات النظام...")
        
        requirements_met = True
        
        # فحص Python
        python_version = sys.version_info
        if python_version.major < 3 or python_version.minor < 8:
            self.logger.error(f"❌ Python 3.8+ مطلوب. الحالي: {python_version.major}.{python_version.minor}")
            requirements_met = False
        else:
            self.logger.info(f"✅ Python {python_version.major}.{python_version.minor} - متوافق")
        
        # فحص المساحة المتاحة
        try:
            import shutil
            disk_usage = shutil.disk_usage(".")
            free_gb = disk_usage.free / (1024**3)
            if free_gb < 2:
                self.logger.warning(f"⚠️ مساحة قليلة متبقية: {free_gb:.1f}GB")
            else:
                self.logger.info(f"✅ مساحة كافية: {free_gb:.1f}GB")
        except Exception as e:
            self.logger.warning(f"⚠️ لا يمكن فحص المساحة: {e}")
        
        # فحص الذاكرة
        try:
            import psutil
            memory_gb = psutil.virtual_memory().total / (1024**3)
            if memory_gb < 4:
                self.logger.warning(f"⚠️ ذاكرة قليلة: {memory_gb:.1f}GB")
            else:
                self.logger.info(f"✅ ذاكرة كافية: {memory_gb:.1f}GB")
        except ImportError:
            self.logger.info("📦 سيتم تثبيت psutil...")
        
        return requirements_met
    
    def install_dependencies(self, package_type: str = "core") -> bool:
        """تثبيت التبعيات تلقائياً"""
        self.logger.info(f"📦 تثبيت تبعيات {package_type}...")
        
        requirements_files = {
            "core": "requirements-core.txt",
            "advanced": "requirements-advanced.txt", 
            "full": "requirements.txt"
        }
        
        req_file = requirements_files.get(package_type, "requirements-core.txt")
        
        if not Path(req_file).exists():
            self.logger.error(f"❌ ملف التبعيات غير موجود: {req_file}")
            return False
        
        try:
            # تثبيت التبعيات
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", req_file
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                self.logger.info(f"✅ تم تثبيت تبعيات {package_type} بنجاح")
                return True
            else:
                self.logger.error(f"❌ فشل تثبيت التبعيات: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error("⏰ انتهت مهلة تثبيت التبعيات")
            return False
        except Exception as e:
            self.logger.error(f"❌ خطأ في تثبيت التبعيات: {e}")
            return False
    
    async def start_service(self, service_name: str, service_config: Dict) -> bool:
        """تشغيل خدمة محددة"""
        if not service_config.get("enabled", False):
            return True
            
        self.logger.info(f"🚀 تشغيل خدمة: {service_name}")
        
        try:
            if service_name == "core_engine":
                from core.unified_assistant_engine import UnifiedAssistantEngine
                engine = UnifiedAssistantEngine()
                await engine.initialize()
                self.services[service_name] = engine
                
            elif service_name == "ai_models":
                from ai_models.learning.continuous_learning_engine import ContinuousLearningEngine
                learning_engine = ContinuousLearningEngine()
                await learning_engine.initialize()
                self.services[service_name] = learning_engine
                
            elif service_name == "analytics":
                from analytics.advanced_analytics_engine import AdvancedAnalyticsEngine
                analytics = AdvancedAnalyticsEngine()
                await analytics.initialize()
                self.services[service_name] = analytics
                
            elif service_name == "web_interface":
                # سيتم تشغيله في عملية منفصلة
                pass
                
            elif service_name == "api_server":
                # سيتم تشغيله في عملية منفصلة
                pass
            
            self.logger.info(f"✅ تم تشغيل {service_name} بنجاح")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ فشل تشغيل {service_name}: {e}")
            return False
    
    async def start_unified_mode(self):
        """تشغيل الوضع الموحد"""
        self.logger.info("🚀 بدء تشغيل الوضع الموحد...")
        
        # ترتيب الخدمات حسب الأولوية
        services = sorted(
            self.config["services"].items(),
            key=lambda x: x[1].get("priority", 999)
        )
        
        # تشغيل الخدمات تدريجياً
        for service_name, service_config in services:
            success = await self.start_service(service_name, service_config)
            if not success:
                self.logger.warning(f"⚠️ تعذر تشغيل {service_name}")
            
            # تأخير قصير بين الخدمات
            await asyncio.sleep(1)
        
        # تشغيل الواجهات في عمليات منفصلة
        await self._start_interfaces()
        
        self.logger.info("🎉 تم تشغيل جميع الخدمات بنجاح!")
    
    async def _start_interfaces(self):
        """تشغيل الواجهات في عمليات منفصلة"""
        try:
            # تشغيل خادم الويب
            if self.config["services"]["web_interface"]["enabled"]:
                web_process = subprocess.Popen([
                    sys.executable, "-m", "frontend.app", "--port", "5000"
                ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                self.logger.info("🌐 تم تشغيل خادم الويب على المنفذ 5000")
            
            # تشغيل خادم API
            if self.config["services"]["api_server"]["enabled"]:
                api_process = subprocess.Popen([
                    sys.executable, "-m", "api.main", "--port", "8000"
                ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                self.logger.info("📡 تم تشغيل خادم API على المنفذ 8000")
                
        except Exception as e:
            self.logger.error(f"❌ خطأ في تشغيل الواجهات: {e}")
    
    async def start_interactive_mode(self):
        """تشغيل الوضع التفاعلي"""
        self.logger.info("💬 بدء الوضع التفاعلي...")
        
        try:
            from core.unified_assistant_engine import UnifiedAssistantEngine
            engine = UnifiedAssistantEngine()
            await engine.initialize()
            await engine.start_interactive_session()
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في الوضع التفاعلي: {e}")
    
    async def health_check(self):
        """فحص صحة الخدمات"""
        while True:
            try:
                self.logger.info("🏥 فحص صحة الخدمات...")
                
                for service_name, service in self.services.items():
                    if hasattr(service, 'health_check'):
                        status = await service.health_check()
                        if status:
                            self.logger.info(f"✅ {service_name}: صحي")
                        else:
                            self.logger.warning(f"⚠️ {service_name}: مشكلة")
                
                await asyncio.sleep(self.config["health_check_interval"])
                
            except Exception as e:
                self.logger.error(f"❌ خطأ في فحص الصحة: {e}")
                await asyncio.sleep(30)

def main():
    """الدالة الرئيسية"""
    parser = argparse.ArgumentParser(description="المساعد الذكي المتقدم")
    parser.add_argument("--mode", choices=['interactive', 'web', 'api', 'unified', 'dev', 'production'], 
                       default="unified", help="وضع التشغيل")
    parser.add_argument("--install", choices=['core', 'advanced', 'full'], 
                       help="تثبيت التبعيات")
    parser.add_argument("--no-check", action="store_true", 
                       help="تخطي فحص متطلبات النظام")
    parser.add_argument("--port", type=int, default=5000, 
                       help="منفذ خادم الويب")
    parser.add_argument("--debug", action="store_true", 
                       help="وضع التصحيح")
    
    args = parser.parse_args()
    
    # إعداد مستوى السجلات
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # إنشاء مدير التشغيل
    startup_manager = AdvancedStartupManager()
    
    # طباعة رسالة الترحيب
    print("=" * 60)
    print("🤖 المساعد الذكي الموحد المتقدم v3.0.0")
    print("=" * 60)
    
    try:
        # فحص متطلبات النظام
        if not args.no_check:
            if not startup_manager.check_system_requirements():
                print("❌ متطلبات النظام غير مكتملة")
                return 1
        
        # تثبيت التبعيات إذا طُلب
        if args.install:
            if not startup_manager.install_dependencies(args.install):
                print(f"❌ فشل تثبيت تبعيات {args.install}")
                return 1
        
        # تشغيل الوضع المطلوب
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        if args.mode == "interactive":
            loop.run_until_complete(startup_manager.start_interactive_mode())
        elif args.mode == "unified":
            # تشغيل الوضع الموحد مع مراقبة الصحة
            async def run_with_health_check():
                await startup_manager.start_unified_mode()
                await startup_manager.health_check()
            
            loop.run_until_complete(run_with_health_check())
        else:
            print(f"🚀 تشغيل وضع: {args.mode}")
            # تنفيذ أوضاع أخرى
            
    except KeyboardInterrupt:
        print("\n👋 تم إيقاف المساعد بواسطة المستخدم")
        return 0
    except Exception as e:
        startup_manager.logger.error(f"❌ خطأ عام: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
