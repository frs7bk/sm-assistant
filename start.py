
#!/usr/bin/env python3
"""
🚀 سكريبت بدء التشغيل الموحد للمساعد الذكي المتقدم
"""

import os
import sys
import asyncio
import argparse
import subprocess
from pathlib import Path
from typing import List, Optional

# إضافة مسار المشروع
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def check_dependencies() -> bool:
    """فحص توفر المكتبات المطلوبة"""
    try:
        import torch
        import transformers
        import fastapi
        import redis
        print("✅ جميع المكتبات الأساسية متوفرة")
        return True
    except ImportError as e:
        print(f"❌ مكتبة مفقودة: {e}")
        print("يرجى تشغيل: pip install -r requirements-core.txt")
        return False

def setup_environment():
    """إعداد البيئة الأساسية"""
    # إنشاء المجلدات المطلوبة
    directories = [
        "data/sessions",
        "data/models", 
        "data/logs",
        "data/uploads",
        "data/cache"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    # نسخ ملف البيئة إذا لم يكن موجوداً
    if not Path(".env").exists() and Path(".env.example").exists():
        import shutil
        shutil.copy(".env.example", ".env")
        print("📋 تم إنشاء ملف .env من القالب")

def install_requirements(mode: str = "core"):
    """تثبيت المتطلبات حسب الوضع"""
    requirements_files = {
        "core": "requirements-core.txt",
        "advanced": "requirements-advanced.txt", 
        "full": "requirements.txt"
    }
    
    req_file = requirements_files.get(mode, "requirements.txt")
    
    if Path(req_file).exists():
        print(f"📦 تثبيت متطلبات الوضع: {mode}")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", req_file])
    else:
        print(f"⚠️ ملف المتطلبات غير موجود: {req_file}")

async def start_assistant(mode: str = "interactive"):
    """بدء تشغيل المساعد"""
    
    # التحقق من المتطلبات
    if not check_dependencies():
        return False
    
    # إعداد البيئة
    setup_environment()
    
    # اختيار وضع التشغيل
    if mode == "interactive":
        print("🤖 بدء الوضع التفاعلي...")
        from core.unified_assistant_engine import main
        await main()
        
    elif mode == "web":
        print("🌐 بدء واجهة الويب...")
        from interfaces.web.smart_web_interface import start_web_server
        await start_web_server()
        
    elif mode == "api":
        print("🔌 بدء خادم API...")
        from api.main import start_api_server
        await start_api_server()
        
    elif mode == "unified":
        print("🚀 بدء الوضع الموحد (جميع الخدمات)...")
        from main_unified import main
        await main()
        
    else:
        print(f"❌ وضع غير مدعوم: {mode}")
        return False
    
    return True

def main():
    """الوظيفة الرئيسية"""
    parser = argparse.ArgumentParser(
        description="🤖 المساعد الذكي الموحد المتقدم",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
أوضاع التشغيل:
  interactive    الوضع التفاعلي في الطرفية
  web           واجهة الويب
  api           خادم API فقط
  unified       جميع الخدمات (افتراضي)

أمثلة:
  python start.py                    # الوضع الموحد
  python start.py --mode interactive # الوضع التفاعلي
  python start.py --install core     # تثبيت المتطلبات الأساسية
        """
    )
    
    parser.add_argument(
        "--mode", 
        choices=["interactive", "web", "api", "unified"],
        default="unified",
        help="وضع التشغيل"
    )
    
    parser.add_argument(
        "--install",
        choices=["core", "advanced", "full"],
        help="تثبيت المتطلبات"
    )
    
    parser.add_argument(
        "--check",
        action="store_true",
        help="فحص المتطلبات فقط"
    )
    
    args = parser.parse_args()
    
    # تثبيت المتطلبات
    if args.install:
        install_requirements(args.install)
        return
    
    # فحص المتطلبات فقط
    if args.check:
        check_dependencies()
        return
    
    # بدء المساعد
    print("=" * 60)
    print("🤖 المساعد الذكي الموحد المتقدم v3.0.0")
    print("=" * 60)
    
    try:
        asyncio.run(start_assistant(args.mode))
    except KeyboardInterrupt:
        print("\n👋 تم إيقاف المساعد بنجاح")
    except Exception as e:
        print(f"❌ خطأ في التشغيل: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
