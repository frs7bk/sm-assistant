
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
سكريبت إنشاء حزمة ZIP للمساعد الذكي الموحد المطور
"""

import zipfile
import os
from pathlib import Path
from datetime import datetime
import json

def create_assistant_package():
    """إنشاء حزمة ZIP شاملة للمساعد"""
    
    # معلومات الحزمة
    package_info = {
        "name": "المساعد الذكي الموحد المتقدم",
        "version": "2.0.0",
        "created_date": datetime.now().isoformat(),
        "description": "مساعد ذكي متقدم مع قدرات الذكاء الاصطناعي والبيانات الضخمة",
        "features": [
            "ذكاء اصطناعي متقدم (GPT-4)",
            "معالجة البيانات الضخمة",
            "التعلم النشط والتكيفي",
            "رؤية حاسوبية ذكية",
            "معالجة اللغة الطبيعية",
            "واجهات متعددة",
            "تحليلات متقدمة",
            "أنظمة التوصية والتنبؤ"
        ],
        "requirements": "Python 3.8+",
        "author": "فريق الذكاء الاصطناعي المتقدم"
    }
    
    # اسم الملف المضغوط
    zip_filename = f"المساعد_الذكي_الموحد_المتقدم_v{package_info['version']}.zip"
    
    print(f"🚀 إنشاء حزمة المساعد الذكي: {zip_filename}")
    
    # الملفات والمجلدات المطلوبة
    files_to_include = [
        # الملفات الأساسية
        "main_unified.py",
        "requirements.txt",
        ".env.example",
        "README.md",
        
        # مجلد core
        "core/unified_assistant_engine.py",
        "core/advanced_ai_engine.py",
        "core/module_manager.py",
        "core/README.md",
        
        # مجلد config
        "config/advanced_config.py",
        "config/settings.py",
        "config/README.md",
        
        # مجلد ai_models
        "ai_models/nlu/bert_analyzer.py",
        "ai_models/nlu/gpt4_interface.py",
        "ai_models/nlu/roberta_embedder.py",
        "ai_models/nlu/wav2vec2_recognizer.py",
        "ai_models/nlu/README.md",
        
        "ai_models/nlg/gpt4_generator.py",
        "ai_models/nlg/fastspeech_tts.py",
        "ai_models/nlg/README.md",
        
        "ai_models/vision/vision_pipeline.py",
        "ai_models/vision/README.md",
        
        "ai_models/learning/active_learning.py",
        "ai_models/learning/few_shot_learner.py",
        "ai_models/learning/reinforcement_engine.py",
        "ai_models/learning/README.md",
        
        "ai_models/README.md",
        
        # مجلد analytics
        "analytics/big_data/dask_processor.py",
        "analytics/big_data/spark_processor.py",
        "analytics/big_data/README.md",
        
        "analytics/prediction/dl_predictor.py",
        "analytics/prediction/ml_predictor.py",
        "analytics/prediction/README.md",
        
        "analytics/recommendation/collaborative_filtering.py",
        "analytics/recommendation/content_based.py",
        "analytics/recommendation/README.md",
        
        "analytics/visualization/dash_dashboard.py",
        "analytics/visualization/README.md",
        
        "analytics/README.md",
        
        # مجلد tools
        "tools/project_organizer.py",
        "tools/advanced_analyzer.py",
        "tools/cleanup_duplicates.py",
        "tools/README.md",
        
        # مجلدات البيانات والاختبارات
        "data/README.md",
        "tests/README.md",
        "docs/README.md",
        
        # ملفات التكوين
        ".gitignore",
        "pyproject.toml"
    ]
    
    try:
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            
            # إضافة معلومات الحزمة
            zipf.writestr("package_info.json", json.dumps(package_info, ensure_ascii=False, indent=2))
            
            # إضافة الملفات
            added_files = 0
            missing_files = []
            
            for file_path in files_to_include:
                if Path(file_path).exists():
                    zipf.write(file_path)
                    added_files += 1
                    print(f"  ✅ تمت إضافة: {file_path}")
                else:
                    missing_files.append(file_path)
                    print(f"  ⚠️ ملف مفقود: {file_path}")
            
            # إضافة دليل التشغيل
            installation_guide = create_installation_guide()
            zipf.writestr("دليل_التشغيل.md", installation_guide)
            
            # إضافة مثال للاستخدام
            usage_example = create_usage_example()
            zipf.writestr("مثال_الاستخدام.py", usage_example)
            
            print(f"\n📦 تم إنشاء الحزمة بنجاح!")
            print(f"   📁 اسم الملف: {zip_filename}")
            print(f"   📊 الملفات المضافة: {added_files}")
            print(f"   ⚠️ الملفات المفقودة: {len(missing_files)}")
            print(f"   📏 حجم الملف: {Path(zip_filename).stat().st_size / 1024 / 1024:.2f} ميجابايت")
            
            if missing_files:
                print("\n⚠️ الملفات المفقودة:")
                for file_path in missing_files:
                    print(f"     - {file_path}")
    
    except Exception as e:
        print(f"❌ خطأ في إنشاء الحزمة: {e}")

def create_installation_guide():
    """إنشاء دليل التشغيل"""
    return """
# دليل تشغيل المساعد الذكي الموحد المتقدم

## 🚀 متطلبات التشغيل

### النظام
- Python 3.8 أو أحدث
- ذاكرة وصول عشوائي: 4 جيجابايت على الأقل (8 جيجابايت مستحسن)
- مساحة تخزين: 2 جيجابايت
- اتصال إنترنت للميزات السحابية

### مفاتيح API المطلوبة
- OpenAI API Key (للذكاء الاصطناعي)
- HuggingFace API Key (اختياري)
- Claude API Key (اختياري)

## 📥 خطوات التثبيت

### 1. استخراج الملفات
```bash
unzip المساعد_الذكي_الموحد_المتقدم_v2.0.0.zip
cd المساعد_الذكي_الموحد_المتقدم
```

### 2. تثبيت المتطلبات
```bash
pip install -r requirements.txt
```

### 3. إعداد متغيرات البيئة
```bash
cp .env.example .env
# قم بتحرير ملف .env وإضافة مفاتيح API الخاصة بك
```

### 4. تشغيل المساعد
```bash
python main_unified.py
```

## ⚙️ التكوين المتقدم

### إعداد قاعدة البيانات
المساعد يستخدم SQLite افتراضياً، لكن يمكن تكوين:
- Redis للتخزين المؤقت
- MongoDB للبيانات الضخمة

### تفعيل الميزات المتقدمة
```python
# في ملف config/advanced_config.py
enable_voice = True
enable_vision = True
enable_big_data = True
enable_learning = True
```

## 🎯 الاستخدام الأساسي

### بدء جلسة تفاعلية
```bash
python main_unified.py
```

### الأوامر الأساسية
- `مساعدة` - عرض دليل الأوامر
- `إحصائيات` - عرض إحصائيات الجلسة
- `تحليل` - تحليل البيانات الضخمة
- `توقع` - عمل توقعات ذكية
- `خروج` - إنهاء الجلسة

## 🔧 استكشاف الأخطاء

### مشاكل شائعة
1. **خطأ في مفاتيح API**: تأكد من صحة المفاتيح في ملف .env
2. **نقص الذاكرة**: قلل من حجم البيانات المعالجة
3. **مشاكل الشبكة**: تحقق من الاتصال بالإنترنت

### السجلات
يتم حفظ السجلات في: `data/logs/`

## 📞 الدعم
للحصول على الدعم أو الإبلاغ عن مشاكل، يرجى مراجعة الوثائق أو التواصل مع فريق التطوير.

## 📄 الترخيص
هذا المشروع مطور لأغراض تعليمية وبحثية.
"""

def create_usage_example():
    """إنشاء مثال للاستخدام"""
    return '''
#!/usr/bin/env python3
"""
مثال على استخدام المساعد الذكي الموحد المتقدم
"""

import asyncio
from main_unified import AdvancedUnifiedAssistant

async def example_usage():
    """مثال على الاستخدام"""
    
    # إنشاء المساعد
    assistant = AdvancedUnifiedAssistant()
    
    try:
        # تهيئة المحركات
        await assistant.initialize_engines()
        
        # مثال على التفاعل
        test_queries = [
            "مرحباً، كيف حالك؟",
            "ما هو الطقس اليوم؟",
            "ساعدني في تحليل البيانات",
            "ما هي توصياتك لي؟"
        ]
        
        print("🤖 اختبار المساعد الذكي")
        print("=" * 40)
        
        for query in test_queries:
            print(f"\\n👤 المستخدم: {query}")
            response = await assistant.process_user_input(query)
            print(f"🤖 المساعد: {response}")
        
        # عرض الإحصائيات
        print("\\n" + "=" * 40)
        stats = assistant.get_session_stats()
        print(stats)
        
    finally:
        # تنظيف الموارد
        await assistant.cleanup()

if __name__ == "__main__":
    asyncio.run(example_usage())
'''

if __name__ == "__main__":
    create_assistant_package()
