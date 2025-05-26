
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
سكريبت إنشاء حزمة ZIP محدثة للمساعد الذكي الموحد المتقدم
يحتوي على جميع الملفات والميزات الحديثة
"""

import zipfile
import os
import json
from pathlib import Path
from datetime import datetime

def create_complete_assistant_package():
    """إنشاء حزمة ZIP شاملة ومحدثة للمساعد"""
    
    # معلومات الحزمة المحدثة
    package_info = {
        "name": "المساعد الذكي الموحد المتقدم - النسخة الكاملة",
        "version": "3.0.0",
        "created_date": datetime.now().isoformat(),
        "description": "مساعد ذكي متقدم مع جميع الميزات المتطورة",
        "new_features": [
            "🧠 التعلم المستمر والتكيف",
            "🔮 الذكاء التنبؤي المتقدم", 
            "🌐 تكامل البيانات الضخمة",
            "🛡️ الأمان الكمومي المتقدم",
            "🎯 منصة إدارة المشاريع المتكاملة",
            "🎨 تحليل المشاعر والذكاء العاطفي",
            "🏠 تكامل إنترنت الأشياء الذكي",
            "💰 المستشار المالي الذكي",
            "🎮 مدرب الألعاب والتحليل",
            "🎥 خبير التصميم والفيديو",
            "👁️ الرؤية الحاسوبية المتطورة",
            "🗣️ معالجة الصوت المتقدمة"
        ],
        "ai_models": [
            "GPT-4 Interface",
            "BERT Analyzer", 
            "RoBERTa Embedder",
            "Wav2Vec2 Recognizer",
            "FastSpeech TTS",
            "Vision Pipeline",
            "Active Learning",
            "Few-Shot Learner",
            "Reinforcement Engine",
            "Deep Learning Predictor"
        ],
        "requirements": "Python 3.8+, 8GB RAM recommended",
        "author": "فريق الذكاء الاصطناعي المتقدم",
        "license": "Educational & Research Use"
    }
    
    # اسم الملف المضغوط المحدث
    zip_filename = f"المساعد_الذكي_المتكامل_النسخة_الكاملة_v{package_info['version']}.zip"
    
    print(f"🚀 إنشاء حزمة المساعد الذكي المتكاملة: {zip_filename}")
    print("="*60)
    
    # جمع جميع الملفات في المشروع
    files_to_include = []
    excluded_patterns = [
        '__pycache__', '.git', '.pytest_cache', 'node_modules',
        '.vscode', '.idea', '*.pyc', '*.pyo', '*.log',
        'temp', 'cache', '.DS_Store', 'Thumbs.db'
    ]
    
    def should_exclude(file_path):
        """فحص ما إذا كان يجب استبعاد الملف"""
        path_str = str(file_path)
        for pattern in excluded_patterns:
            if pattern in path_str:
                return True
        return False
    
    # البحث عن جميع الملفات
    project_root = Path(".")
    
    for file_path in project_root.rglob("*"):
        if file_path.is_file() and not should_exclude(file_path):
            # تحويل المسار النسبي
            relative_path = file_path.relative_to(project_root)
            files_to_include.append(relative_path)
    
    try:
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as zipf:
            
            # إضافة معلومات الحزمة
            zipf.writestr("📋_معلومات_الحزمة.json", 
                         json.dumps(package_info, ensure_ascii=False, indent=2))
            
            # إضافة جميع الملفات
            added_files = 0
            total_size = 0
            
            print("📁 إضافة الملفات:")
            print("-" * 40)
            
            for file_path in sorted(files_to_include):
                try:
                    if file_path.exists():
                        file_size = file_path.stat().st_size
                        zipf.write(file_path, file_path)
                        added_files += 1
                        total_size += file_size
                        
                        # عرض التقدم
                        if added_files % 50 == 0:
                            print(f"   ✅ تمت إضافة {added_files} ملف...")
                        
                except Exception as e:
                    print(f"   ⚠️ تجاهل الملف: {file_path} - {e}")
            
            # إضافة دليل التشغيل الشامل
            installation_guide = create_complete_installation_guide()
            zipf.writestr("📖_دليل_التشغيل_الكامل.md", installation_guide)
            
            # إضافة أمثلة الاستخدام
            usage_examples = create_usage_examples()
            zipf.writestr("💡_أمثلة_الاستخدام.py", usage_examples)
            
            # إضافة دليل الميزات
            features_guide = create_features_guide()
            zipf.writestr("🎯_دليل_الميزات.md", features_guide)
            
            # إضافة ملف الإعداد السريع
            quick_setup = create_quick_setup_script()
            zipf.writestr("⚡_الإعداد_السريع.py", quick_setup)
            
            # إحصائيات الحزمة
            package_size = Path(zip_filename).stat().st_size
            
            print("\n" + "="*60)
            print("🎉 تم إنشاء الحزمة الكاملة بنجاح!")
            print("="*60)
            print(f"📦 اسم الملف: {zip_filename}")
            print(f"📊 إجمالي الملفات: {added_files}")
            print(f"📏 حجم الحزمة: {package_size / 1024 / 1024:.2f} ميجابايت")
            print(f"📁 المحتوى الأصلي: {total_size / 1024 / 1024:.2f} ميجابايت")
            print(f"⚡ نسبة الضغط: {((total_size - package_size) / total_size * 100):.1f}%")
            print("="*60)
            
            return zip_filename
            
    except Exception as e:
        print(f"❌ خطأ في إنشاء الحزمة: {e}")
        return None

def create_complete_installation_guide():
    """إنشاء دليل التشغيل الكامل"""
    return """
# 🤖 دليل تشغيل المساعد الذكي الموحد المتقدم - النسخة الكاملة

## 🎯 نظرة عامة

هذا المساعد الذكي يحتوي على أحدث تقنيات الذكاء الاصطناعي وأكثر من 50 ميزة متقدمة!

## ⚡ الإعداد السريع (5 دقائق)

### 1. متطلبات النظام
```
- Python 3.8+ 
- ذاكرة وصول عشوائي: 8 جيجابايت (موصى به)
- مساحة تخزين: 5 جيجابايت 
- اتصال إنترنت سريع
```

### 2. التثبيت السريع
```bash
# استخراج الملفات
unzip المساعد_الذكي_المتكامل_النسخة_الكاملة_v3.0.0.zip
cd المساعد_الذكي_المتكامل_النسخة_الكاملة

# تشغيل الإعداد السريع
python ⚡_الإعداد_السريع.py

# تشغيل المساعد
python main_unified.py
```

## 🔑 إعداد مفاتيح API

### إنشاء ملف البيئة
```bash
cp .env.example .env
```

### المفاتيح المطلوبة:
```env
# أساسي (مطلوب)
OPENAI_API_KEY=your_openai_key_here

# متقدم (اختياري)
HUGGINGFACE_API_KEY=your_hf_key_here
CLAUDE_API_KEY=your_claude_key_here
GOOGLE_API_KEY=your_google_key_here
AZURE_API_KEY=your_azure_key_here
```

## 🚀 تشغيل الميزات المتقدمة

### تفعيل جميع الميزات
```python
# في config/advanced_config.py
ADVANCED_FEATURES = {
    "learning": True,           # التعلم المستمر
    "prediction": True,         # التنبؤ الذكي
    "big_data": True,          # البيانات الضخمة
    "quantum_security": True,   # الأمان الكمومي
    "ar_vr": True,             # الواقع المختلط
    "iot_integration": True,    # إنترنت الأشياء
    "emotional_ai": True,       # الذكاء العاطفي
    "financial_advisor": True,  # المستشار المالي
    "health_monitor": True,     # مراقب الصحة
    "creative_ai": True,        # الذكاء الإبداعي
}
```

## 🎮 أمثلة الاستخدام

### 1. التفاعل الأساسي
```python
python main_unified.py
# ثم اكتب: "مرحباً، ساعدني في تنظيم يومي"
```

### 2. تحليل البيانات الضخمة
```python
# في المحادثة
"تحليل البيانات"
# أو
"تحليل هذا الملف: data.csv"
```

### 3. التوقعات الذكية
```python
# في المحادثة  
"توقع احتياجاتي لهذا الأسبوع"
# أو
"ما هي توصياتك لتحسين إنتاجيتي؟"
```

## 🛠️ تخصيص المساعد

### تطوير الشخصية
```python
# في ai_models/learning/adaptive_personality_engine.py
personality_config = {
    "formal_level": 0.7,        # مستوى الرسمية
    "humor_level": 0.8,         # مستوى الفكاهة
    "detail_level": 0.9,        # مستوى التفصيل
    "proactive_level": 0.8,     # مستوى الاستباقية
}
```

### إضافة مهارات جديدة
```python
# في ai_models/learning/few_shot_learner.py
new_skill = {
    "name": "مهارة جديدة",
    "examples": ["مثال 1", "مثال 2"],
    "responses": ["استجابة 1", "استجابة 2"]
}
```

## 🔧 استكشاف الأخطاء

### مشاكل شائعة وحلولها:

1. **خطأ في تثبيت المكتبات**
```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

2. **نفاد الذاكرة**
```python
# في config/advanced_config.py
PERFORMANCE_SETTINGS = {
    "max_memory_usage": "4GB",
    "enable_memory_optimization": True,
    "use_lightweight_models": True
}
```

3. **بطء في الاستجابة**
```python
# تفعيل التسريع
ACCELERATION = {
    "use_gpu": True,
    "parallel_processing": True, 
    "cache_responses": True
}
```

## 📊 مراقبة الأداء

### عرض الإحصائيات
```python
# في المحادثة
"إحصائيات"
# أو  
"أداء النظام"
```

### تحليل السجلات
```bash
# عرض السجلات
tail -f data/logs/assistant_*.log

# تحليل الأداء
python tools/advanced_analyzer.py
```

## 🌐 النشر والمشاركة

### نشر المساعد كخدمة ويب
```bash
# تشغيل واجهة الويب
python interfaces/web/smart_web_interface.py

# النشر على السحابة
python deploy_assistant.py
```

## 🤝 المساهمة والتطوير

### إضافة ميزات جديدة
1. إنشاء مجلد جديد في `ai_models/`
2. إضافة الميزة في `core/module_manager.py`
3. تحديث التكوين في `config/advanced_config.py`

### اختبار الميزات
```bash
python -m pytest tests/ -v
```

## 📞 الدعم والمساعدة

- 📋 دليل الميزات: `🎯_دليل_الميزات.md`
- 💡 أمثلة الاستخدام: `💡_أمثلة_الاستخدام.py`
- 🐛 تقارير الأخطاء: `data/logs/`

## 📄 الترخيص والقيود

هذا المشروع مخصص للأغراض التعليمية والبحثية.
استخدم المساعد بمسؤولية واحترم خصوصية البيانات.

---
✨ **استمتع باستخدام أقوى مساعد ذكي تم تطويره!** ✨
"""

def create_usage_examples():
    """إنشاء أمثلة شاملة للاستخدام"""
    return '''
#!/usr/bin/env python3
"""
💡 أمثلة شاملة لاستخدام المساعد الذكي الموحد المتقدم
"""

import asyncio
import sys
from pathlib import Path

# إضافة مسار المشروع
sys.path.append(str(Path(__file__).parent))

from main_unified import AdvancedUnifiedAssistant

async def example_basic_interaction():
    """مثال على التفاعل الأساسي"""
    print("🤖 مثال: التفاعل الأساسي")
    print("=" * 40)
    
    assistant = AdvancedUnifiedAssistant()
    await assistant.initialize_engines()
    
    # أمثلة على الاستعلامات
    queries = [
        "مرحباً، كيف حالك؟",
        "ما هو الطقس اليوم؟", 
        "ساعدني في تنظيم مهامي",
        "احسب لي 15 × 24 + 100",
        "اشرح لي الذكاء الاصطناعي"
    ]
    
    for query in queries:
        print(f"\\n👤 المستخدم: {query}")
        response = await assistant.process_user_input(query)
        print(f"🤖 المساعد: {response}")
    
    await assistant.cleanup()

async def example_advanced_features():
    """مثال على الميزات المتقدمة"""
    print("\\n🚀 مثال: الميزات المتقدمة")
    print("=" * 40)
    
    assistant = AdvancedUnifiedAssistant()
    await assistant.initialize_engines()
    
    # تحليل البيانات الضخمة
    print("\\n📊 تحليل البيانات الضخمة:")
    data_analysis = await assistant.analyze_big_data()
    print(data_analysis)
    
    # التوقعات الذكية
    print("\\n🔮 التوقعات الذكية:")
    predictions = await assistant.make_predictions()
    print(predictions)
    
    # الإحصائيات
    print("\\n📈 إحصائيات الجلسة:")
    stats = assistant.get_session_stats()
    print(stats)
    
    await assistant.cleanup()

async def example_learning_adaptation():
    """مثال على التعلم والتكيف"""
    print("\\n🧠 مثال: التعلم والتكيف")
    print("=" * 40)
    
    assistant = AdvancedUnifiedAssistant()
    await assistant.initialize_engines()
    
    # تسلسل من التفاعلات لإظهار التعلم
    learning_sequence = [
        "أحب القهوة في الصباح",
        "أفضل العمل في المساء", 
        "أحتاج تذكيرات للاجتماعات",
        "ما هي توصياتك بناءً على تفضيلاتي؟"
    ]
    
    print("تسلسل التعلم:")
    for i, query in enumerate(learning_sequence, 1):
        print(f"\\n{i}. 👤 المستخدم: {query}")
        response = await assistant.process_user_input(query)
        print(f"   🤖 المساعد: {response}")
        
        # محاكاة التعلم من التفاعل
        if assistant.active_learning:
            assistant.active_learning.log_interaction(query, response)
    
    await assistant.cleanup()

async def example_specialized_modules():
    """مثال على الوحدات المتخصصة"""
    print("\\n🎯 مثال: الوحدات المتخصصة")
    print("=" * 40)
    
    assistant = AdvancedUnifiedAssistant()
    await assistant.initialize_engines()
    
    # اختبار وحدات مختلفة
    specialized_queries = [
        "حلل هذه الصورة",  # الرؤية الحاسوبية
        "اقترح استثمارات ذكية",  # المستشار المالي
        "راقب صحتي اليوم",  # مراقب الصحة
        "ساعدني في إدارة المشروع",  # إدارة المشاريع
        "حلل مشاعري من النص",  # تحليل المشاعر
    ]
    
    for query in specialized_queries:
        print(f"\\n🎯 اختبار: {query}")
        response = await assistant.process_user_input(query)
        print(f"📋 النتيجة: {response}")
    
    await assistant.cleanup()

async def demo_full_capabilities():
    """عرض توضيحي كامل للقدرات"""
    print("\\n" + "🌟" * 20)
    print("🎭 عرض توضيحي كامل لقدرات المساعد")
    print("🌟" * 20)
    
    try:
        # التفاعل الأساسي
        await example_basic_interaction()
        
        # الميزات المتقدمة
        await example_advanced_features()
        
        # التعلم والتكيف
        await example_learning_adaptation()
        
        # الوحدات المتخصصة
        await example_specialized_modules()
        
        print("\\n" + "✨" * 20)
        print("🎉 انتهى العرض التوضيحي بنجاح!")
        print("💡 يمكنك الآن استخدام المساعد بكامل قدراته")
        print("✨" * 20)
        
    except Exception as e:
        print(f"\\n❌ خطأ في العرض التوضيحي: {e}")

if __name__ == "__main__":
    print("🤖 أمثلة استخدام المساعد الذكي الموحد المتقدم")
    print("=" * 60)
    print("اختر ما تريد تجربته:")
    print("1. التفاعل الأساسي")
    print("2. الميزات المتقدمة") 
    print("3. التعلم والتكيف")
    print("4. الوحدات المتخصصة")
    print("5. عرض كامل لجميع القدرات")
    print("=" * 60)
    
    try:
        choice = input("اختر رقم (1-5): ").strip()
        
        if choice == "1":
            asyncio.run(example_basic_interaction())
        elif choice == "2":
            asyncio.run(example_advanced_features())
        elif choice == "3":
            asyncio.run(example_learning_adaptation())
        elif choice == "4":
            asyncio.run(example_specialized_modules())
        elif choice == "5":
            asyncio.run(demo_full_capabilities())
        else:
            print("❌ اختيار غير صحيح")
            
    except KeyboardInterrupt:
        print("\\n👋 تم إيقاف الأمثلة")
    except Exception as e:
        print(f"\\n❌ خطأ: {e}")
'''

def create_features_guide():
    """إنشاء دليل الميزات"""
    return """
# 🎯 دليل الميزات الشامل للمساعد الذكي

## 🧠 ميزات التعلم والذكاء

### 1. التعلم المستمر والتكيف
- **الذاكرة طويلة المدى**: يحفظ جميع التفاعلات ويتعلم منها
- **تطور الشخصية**: يطور شخصيته لتناسب أسلوبك  
- **التعلم من الأخطاء**: يحسن أداءه تلقائياً
- **التكيف مع المزاج**: يتعرف على حالتك النفسية

### 2. الذكاء التنبؤي المتقدم
- **تنبؤ الاحتياجات**: يتوقع ما تحتاجه قبل أن تطلبه
- **الجدولة الذكية**: ينظم يومك بناءً على عاداتك
- **إدارة الطاقة**: يقترح أوقات الراحة والعمل المثلى
- **تحليل الإنتاجية**: يحلل أدائك ويقترح تحسينات

## 🌐 ميزات التكامل والاتصال

### 3. تكامل البيانات الضخمة
- **تحليل أنماط السلوك** من مصادر متعددة
- **تتبع العادات الصحية** من الأجهزة القابلة للارتداء
- **تحليل أنماط الإنفاق** والميزانية
- **مراقبة الأهداف** طويلة المدى

### 4. منصة العمل الشاملة
- **إدارة المشاريع**: تخطيط وتتبع المهام المعقدة
- **التعاون الفريقي**: تنسيق العمل مع الآخرين
- **إدارة المعرفة**: بناء قاعدة معرفة شخصية ذكية
- **التطوير المهني**: تتبع المهارات ومسارات التعلم

## 🔮 ميزات مستقبلية متقدمة

### 5. الواقع المختلط والذكاء المكاني
- **الواقع المعزز**: عرض معلومات مفيدة في بيئتك
- **التنقل الذكي**: إرشادات مخصصة
- **إدارة المساحة**: تحسين ترتيب مكتبك/منزلك
- **التفاعل ثلاثي الأبعاد**: واجهات تفاعلية في الفضاء

### 6. الأمان والخصوصية المتقدمة
- **الحماية البيومترية**: بصمة، صوت، وجه، قزحية
- **التشفير الكمومي**: حماية فائقة للبيانات الحساسة
- **إدارة الهوية الرقمية**: حماية هويتك عبر الإنترنت
- **كشف التهديدات**: مراقبة أمنية استباقية

## 🎯 ميزات متخصصة حسب المجال

### 7. المستشار المالي الذكي
- **تحليل الاستثمارات** والمحافظ المالية
- **توقع اتجاهات السوق** باستخدام التعلم العميق
- **التخطيط المالي الشخصي** المخصص
- **تتبع النفقات والميزانية** الذكي

### 8. مراقب الصحة الذكي
- **تحليل البيانات الصحية** من الأجهزة المختلفة
- **توصيات التمارين** المخصصة
- **مراقبة الأنماط الصحية** والتنبيه للمخاطر
- **التذكير بالأدوية** والمواعيد الطبية

### 9. مدرب الألعاب والتحليل
- **تحليل أداء اللعب** في الألعاب المختلفة
- **استراتيجيات محسنة** باستخدام AI
- **تتبع التقدم** ونقاط القوة والضعف
- **توصيات للتحسين** والتطوير

### 10. خبير التصميم والفيديو
- **تحليل المحتوى البصري** والفيديوهات
- **اقتراحات التحسين** للتصميمات
- **أتمتة مهام التحرير** الأساسية
- **توليد أفكار إبداعية** للمحتوى

## 🏠 ميزات المنزل الذكي

### 11. تكامل إنترنت الأشياء
- **التحكم في الإضاءة** والمناخ تلقائياً
- **مراقبة الأمان** والحماية المنزلية
- **إدارة الطاقة** وتوفير الكهرباء
- **التسوق الذكي** وإدارة المخزون

### 12. المساعد الصوتي المتقدم
- **التعرف على الأصوات** المختلفة في المنزل
- **استجابة مخصصة** لكل فرد من العائلة
- **التحكم الصوتي** في جميع الأجهزة
- **المحادثات الطبيعية** متعددة اللغات

## 🎨 ميزات الإبداع والتطوير

### 13. الذكاء الإبداعي
- **توليد النصوص الإبداعية** والقصص
- **اقتراح الحلول المبتكرة** للمشاكل
- **تطوير الأفكار** وتحسينها
- **الإلهام الفني** والتصميمي

### 14. مطور البرمجيات الذكي
- **مراجعة الكود** وتحسين الأداء
- **اكتشاف الأخطاء** قبل حدوثها
- **اقتراح التحسينات** والممارسات الأفضل
- **توليد الوثائق** التقنية تلقائياً

## 📱 ميزات التفاعل والواجهات

### 15. واجهات متعددة الوسائط
- **واجهة الويب التفاعلية** مع لوحات معلومات ذكية
- **التفاعل الصوتي** المتقدم مع فهم السياق
- **واجهة الرسائل النصية** مع رموز تعبيرية ذكية
- **التكامل مع التطبيقات** المختلفة

### 16. التخصيص الشخصي
- **سمات مرئية قابلة للتخصيص** حسب الذوق
- **أنماط تفاعل متنوعة** (رسمي، ودود، مهني)
- **تخصيص الاستجابات** حسب الوقت والمكان
- **ذكريات شخصية** ومناسبات مهمة

## 🔬 ميزات البحث والتحليل

### 17. محرك البحث الذكي
- **البحث عبر مصادر متعددة** في نفس الوقت
- **تلخيص النتائج** بطريقة ذكية ومفيدة
- **التحقق من صحة المعلومات** من مصادر موثوقة
- **البحث بالصوت والصورة** والنص

### 18. محلل البيانات المتقدم
- **تحليل البيانات الضخمة** بسرعة فائقة
- **إنشاء الرسوم البيانية** التفاعلية
- **اكتشاف الأنماط المخفية** في البيانات
- **التوقعات الإحصائية** المتقدمة

## 🎓 ميزات التعليم والتطوير

### 19. المعلم الشخصي الذكي
- **خطط تعلم مخصصة** حسب أسلوبك
- **تتبع التقدم** وتقييم الأداء
- **اختبارات ذكية** وتقييمات تفاعلية
- **موارد تعليمية متنوعة** ومحدثة

### 20. مستشار التطوير المهني
- **تحليل المهارات الحالية** ونقاط القوة
- **اقتراح مسارات مهنية** مناسبة
- **تتبع الفرص** الوظيفية والتدريبية
- **بناء الشبكة المهنية** والتواصل

---

## 🚀 طريقة تفعيل الميزات

### تفعيل الميزات الأساسية:
```python
# في ملف التكوين
basic_features = [
    "learning", "prediction", "nlp", "voice", 
    "task_management", "data_analysis"
]
```

### تفعيل الميزات المتقدمة:
```python  
# في ملف التكوين المتقدم
advanced_features = [
    "quantum_security", "ar_vr", "iot_integration",
    "emotional_ai", "creative_ai", "predictive_analytics"
]
```

### تفعيل جميع الميزات:
```python
# تحذير: يتطلب موارد كبيرة
all_features = True
enable_experimental = True
max_performance_mode = True
```

---

💡 **نصيحة**: ابدأ بالميزات الأساسية ثم فعّل المتقدمة تدريجياً حسب احتياجاتك!
"""

def create_quick_setup_script():
    """إنشاء سكريبت الإعداد السريع"""
    return '''
#!/usr/bin/env python3
"""
⚡ سكريبت الإعداد السريع للمساعد الذكي
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def print_colored(message, color="white"):
    """طباعة ملونة"""
    colors = {
        "red": "\\033[91m",
        "green": "\\033[92m", 
        "yellow": "\\033[93m",
        "blue": "\\033[94m",
        "purple": "\\033[95m",
        "cyan": "\\033[96m",
        "white": "\\033[97m",
        "reset": "\\033[0m"
    }
    
    print(f"{colors.get(color, colors['white'])}{message}{colors['reset']}")

def check_python_version():
    """فحص إصدار Python"""
    print_colored("🔍 فحص إصدار Python...", "blue")
    
    if sys.version_info < (3, 8):
        print_colored("❌ يتطلب Python 3.8 أو أحدث", "red")
        return False
    
    print_colored(f"✅ Python {sys.version.split()[0]} - متوافق", "green")
    return True

def install_requirements():
    """تثبيت المتطلبات"""
    print_colored("📦 تثبيت المتطلبات...", "blue")
    
    requirements_file = Path("requirements.txt")
    
    if not requirements_file.exists():
        print_colored("⚠️ ملف requirements.txt غير موجود، إنشاء ملف أساسي...", "yellow")
        create_basic_requirements()
    
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], check=True, capture_output=True)
        
        print_colored("✅ تم تثبيت جميع المتطلبات", "green")
        return True
        
    except subprocess.CalledProcessError as e:
        print_colored(f"❌ خطأ في التثبيت: {e}", "red")
        return False

def create_basic_requirements():
    """إنشاء ملف متطلبات أساسي"""
    basic_requirements = """
# المتطلبات الأساسية
colorama>=0.4.4
python-dotenv>=0.19.0
requests>=2.25.1
numpy>=1.21.0
pandas>=1.3.0

# معالجة النصوص والذكاء الاصطناعي
openai>=0.27.0
transformers>=4.15.0
torch>=1.9.0
sentence-transformers>=2.1.0

# معالجة الصوت والصورة
pillow>=8.3.0
opencv-python>=4.5.0
librosa>=0.8.1
soundfile>=0.10.3

# واجهات الويب والتصور
flask>=2.0.0
dash>=2.0.0
plotly>=5.0.0
streamlit>=1.0.0

# قواعد البيانات والتخزين
sqlalchemy>=1.4.0
redis>=3.5.0
pymongo>=3.12.0

# أدوات التطوير
pytest>=6.2.0
black>=21.0.0
flake8>=3.9.0
""".strip()
    
    with open("requirements.txt", "w", encoding="utf-8") as f:
        f.write(basic_requirements)

def setup_environment():
    """إعداد البيئة"""
    print_colored("🔧 إعداد البيئة...", "blue")
    
    # إنشاء ملف .env إذا لم يكن موجوداً
    env_file = Path(".env")
    if not env_file.exists():
        env_example = Path(".env.example")
        if env_example.exists():
            import shutil
            shutil.copy(env_example, env_file)
            print_colored("✅ تم إنشاء ملف .env", "green")
        else:
            create_basic_env_file()
    
    # إنشاء المجلدات الأساسية
    essential_dirs = [
        "data/logs",
        "data/cache", 
        "data/models",
        "data/user_data",
        "temp"
    ]
    
    for dir_path in essential_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print_colored("✅ تم إعداد البيئة", "green")

def create_basic_env_file():
    """إنشاء ملف .env أساسي"""
    env_content = """
# مفاتيح API الأساسية
OPENAI_API_KEY=your_openai_key_here
HUGGINGFACE_API_KEY=your_hf_key_here

# إعدادات المساعد
ASSISTANT_NAME=المساعد الذكي
ASSISTANT_LANGUAGE=ar
DEBUG_MODE=False

# إعدادات الأداء
MAX_MEMORY_USAGE=4GB
ENABLE_GPU=True
PARALLEL_PROCESSING=True
""".strip()
    
    with open(".env", "w", encoding="utf-8") as f:
        f.write(env_content)
    
    print_colored("ℹ️ تم إنشاء ملف .env أساسي - لا تنس إضافة مفاتيح API", "yellow")

def run_initial_test():
    """تشغيل اختبار أولي"""
    print_colored("🧪 تشغيل اختبار أولي...", "blue")
    
    try:
        # اختبار استيراد المكتبات الأساسية
        import colorama
        import numpy
        import pandas
        
        print_colored("✅ جميع المكتبات الأساسية تعمل", "green")
        
        # اختبار تشغيل المساعد
        if Path("main_unified.py").exists():
            print_colored("✅ ملف المساعد الرئيسي موجود", "green")
            return True
        else:
            print_colored("⚠️ ملف المساعد الرئيسي غير موجود", "yellow")
            return False
            
    except ImportError as e:
        print_colored(f"❌ خطأ في الاستيراد: {e}", "red")
        return False

def main():
    """الدالة الرئيسية للإعداد"""
    print_colored("=" * 60, "cyan")
    print_colored("⚡ معالج الإعداد السريع للمساعد الذكي", "cyan")
    print_colored("=" * 60, "cyan")
    
    steps = [
        ("فحص إصدار Python", check_python_version),
        ("تثبيت المتطلبات", install_requirements), 
        ("إعداد البيئة", setup_environment),
        ("تشغيل اختبار أولي", run_initial_test)
    ]
    
    success_count = 0
    
    for step_name, step_func in steps:
        print_colored(f"\\n📋 {step_name}...", "purple")
        if step_func():
            success_count += 1
        else:
            print_colored(f"⚠️ فشل في: {step_name}", "yellow")
    
    print_colored("\\n" + "=" * 60, "cyan")
    
    if success_count == len(steps):
        print_colored("🎉 تم الإعداد بنجاح!", "green")
        print_colored("🚀 يمكنك الآن تشغيل المساعد:", "green")
        print_colored("   python main_unified.py", "white")
    else:
        print_colored(f"⚠️ نجح {success_count}/{len(steps)} خطوات", "yellow")
        print_colored("💡 تحقق من السجلات وأعد المحاولة", "yellow")
    
    print_colored("=" * 60, "cyan")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_colored("\\n⚠️ تم إيقاف الإعداد", "yellow")
    except Exception as e:
        print_colored(f"\\n❌ خطأ في الإعداد: {e}", "red")
'''

if __name__ == "__main__":
    zip_file = create_complete_assistant_package()
    if zip_file:
        print(f"\n🎁 الحزمة جاهزة: {zip_file}")
        print("📱 يمكنك الآن مشاركة هذا الملف مع الآخرين!")
