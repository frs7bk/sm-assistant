
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
سكريبت تنظيف الملفات المكررة في المشروع
"""

import os
import shutil
from pathlib import Path

class ProjectCleaner:
    """منظف المشروع لإزالة التكرارات"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.duplicates_found = []
        self.cleanup_log = []
    
    def identify_duplicates(self):
        """تحديد الملفات المكررة"""
        duplicates = {
            # ملفات ActiveLearning المكررة
            'active_learning': [
                'active_learning.py',
                'ai_models/learning/active_learning.py', 
                'learning/active_learning.py'
            ],
            
            # ملفات IFTTT المكررة
            'ifttt_integration': [
                'modules/iot.py',
                'modules/hybrid_integration.py'
            ],
            
            # ملفات المساعد المكررة في core
            'assistant_versions': [
                'core/assistant_updated.py',
                'core/assistant_updated_full_features.py',
                'core/assistant_updated_with_new_features.py',
                'core/assistant_updated_screen_interaction.py',
                'core/assistant_updated_screen_interaction_v2.py'
            ],
            
            # ملفات الجذر المكررة
            'root_duplicates': [
                'bert_analyzer.py',
                'gpt4_interface.py', 
                'roberta_embedder.py',
                'wav2vec2_recognizer.py',
                'vision_pipeline.py',
                'fastspeech_tts.py'
            ]
        }
        
        return duplicates
    
    def create_unified_structure(self):
        """إنشاء هيكل موحد للمشروع"""
        unified_structure = {
            'assistant/': {
                'core/': ['main_unified.py'],
                'modules/': {
                    'ai_models/': {
                        'nlu/': ['bert_analyzer.py', 'gpt4_interface.py', 'roberta_embedder.py', 'wav2vec2_recognizer.py'],
                        'nlg/': ['gpt4_generator.py', 'fastspeech_tts.py'],
                        'vision/': ['vision_pipeline.py'],
                        'learning/': ['active_learning.py', 'few_shot_learner.py', 'reinforcement_engine.py']
                    },
                    'interfaces/': {
                        'voice/': [],
                        'vision/': [],
                        'productivity/': []
                    },
                    'security/': [],
                    'analytics/': []
                }
            },
            'config/': ['settings.py', 'credentials.py'],
            'data/': ['models/', 'logs/', 'user_data/'],
            'tests/': [],
            'docs/': []
        }
        
        return unified_structure
    
    def backup_current_state(self):
        """نسخ احتياطي للحالة الحالية"""
        backup_dir = self.project_root / 'backup_before_cleanup'
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        
        # نسخ المجلدات المهمة
        important_dirs = ['core', 'modules', 'ai_models']
        backup_dir.mkdir()
        
        for dir_name in important_dirs:
            src_dir = self.project_root / dir_name
            if src_dir.exists():
                shutil.copytree(src_dir, backup_dir / dir_name)
        
        print(f"✅ تم إنشاء نسخة احتياطية في: {backup_dir}")
    
    def move_to_archive(self, file_paths):
        """نقل الملفات للأرشيف بدلاً من حذفها"""
        archive_dir = self.project_root / 'archived_files'
        archive_dir.mkdir(exist_ok=True)
        
        for file_path in file_paths:
            src = self.project_root / file_path
            if src.exists():
                dst = archive_dir / file_path
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))
                self.cleanup_log.append(f"نُقل للأرشيف: {file_path}")
    
    def generate_cleanup_report(self):
        """إنشاء تقرير التنظيف"""
        report_path = self.project_root / 'cleanup_report.md'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# تقرير تنظيف المشروع\n\n")
            f.write("## الملفات المنقولة للأرشيف\n\n")
            
            for log_entry in self.cleanup_log:
                f.write(f"- {log_entry}\n")
            
            f.write("\n## التوصيات\n\n")
            f.write("1. مراجعة الملفات في مجلد `archived_files`\n")
            f.write("2. حذف الملفات غير المطلوبة نهائياً\n") 
            f.write("3. اختبار المساعد الموحد\n")
            f.write("4. تحديث المسارات في الكود\n")
        
        print(f"📄 تم إنشاء تقرير التنظيف: {report_path}")

if __name__ == "__main__":
    cleaner = ProjectCleaner()
    
    print("🧹 بدء عملية تنظيف المشروع...")
    
    # إنشاء نسخة احتياطية
    cleaner.backup_current_state()
    
    # تحديد التكرارات
    duplicates = cleaner.identify_duplicates()
    
    print("\n📋 الملفات المكررة المكتشفة:")
    for category, files in duplicates.items():
        print(f"\n{category}:")
        for file in files:
            print(f"  - {file}")
    
    # إنشاء تقرير
    cleaner.generate_cleanup_report()
    
    print("\n✅ تم الانتهاء من تحليل التنظيف. راجع التقرير لاتخاذ الإجراءات.")
