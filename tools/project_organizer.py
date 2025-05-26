
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
أداة تنظيم المشروع المتقدمة
تقوم بتنظيف وإعادة هيكلة المشروع بطريقة احترافية
"""

import os
import shutil
import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
import hashlib
import subprocess

@dataclass
class FileInfo:
    """معلومات الملف"""
    path: Path
    size: int
    hash: str
    content_type: str
    is_duplicate: bool = False
    move_to: Optional[Path] = None

class ProjectOrganizer:
    """منظم المشروع المتقدم"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.logger = logging.getLogger(__name__)
        
        # الهيكل المستهدف
        self.target_structure = {
            "core": ["محرك المساعد الأساسي", "*.py"],
            "ai_models": {
                "nlu": ["معالجة اللغة الطبيعية", "*.py"],
                "nlg": ["توليد اللغة الطبيعية", "*.py"],
                "vision": ["الرؤية الحاسوبية", "*.py"],
                "learning": ["التعلم الآلي", "*.py"]
            },
            "analytics": {
                "big_data": ["معالجة البيانات الضخمة", "*.py"],
                "prediction": ["أنظمة التنبؤ", "*.py"],
                "recommendation": ["أنظمة التوصية", "*.py"],
                "visualization": ["التصور والرسوم البيانية", "*.py"]
            },
            "interfaces": {
                "web": ["واجهات الويب", "*.py", "*.html", "*.css", "*.js"],
                "voice": ["واجهات صوتية", "*.py"],
                "api": ["واجهات برمجية", "*.py"]
            },
            "config": ["ملفات التكوين", "*.py", "*.json", "*.yaml", "*.env"],
            "data": {
                "models": ["النماذج المدربة", "*.pth", "*.pkl", "*.h5"],
                "cache": ["ملفات التخزين المؤقت", "*"],
                "logs": ["ملفات السجلات", "*.log"],
                "user_data": ["بيانات المستخدمين", "*.json", "*.db"],
                "backups": ["النسخ الاحتياطية", "*"]
            },
            "tests": ["اختبارات الكود", "test_*.py", "*_test.py"],
            "docs": ["التوثيق", "*.md", "*.rst", "*.txt"],
            "tools": ["أدوات التطوير", "*.py"],
            "scripts": ["سكريبتات التشغيل", "*.py", "*.sh", "*.bat"]
        }
        
        # الملفات المكررة
        self.duplicates: Dict[str, List[FileInfo]] = {}
        
        # إحصائيات
        self.stats = {
            "files_moved": 0,
            "files_deleted": 0,
            "duplicates_found": 0,
            "directories_created": 0,
            "total_space_saved": 0
        }
    
    def organize_project(self):
        """تنظيم المشروع الكامل"""
        self.logger.info("🗂️ بدء تنظيم المشروع...")
        
        try:
            # المرحلة 1: تحليل الملفات
            self._analyze_files()
            
            # المرحلة 2: إنشاء الهيكل المستهدف
            self._create_target_structure()
            
            # المرحلة 3: نقل الملفات
            self._move_files()
            
            # المرحلة 4: حذف المكررات
            self._remove_duplicates()
            
            # المرحلة 5: تنظيف المجلدات الفارغة
            self._cleanup_empty_directories()
            
            # المرحلة 6: إنشاء ملفات README
            self._create_readme_files()
            
            # المرحلة 7: تحديث .gitignore
            self._update_gitignore()
            
            # المرحلة 8: إنشاء تقرير
            self._generate_report()
            
            self.logger.info("✅ تم تنظيم المشروع بنجاح")
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في تنظيم المشروع: {e}")
    
    def _analyze_files(self):
        """تحليل جميع الملفات في المشروع"""
        self.logger.info("📊 تحليل الملفات...")
        
        file_hashes: Dict[str, List[FileInfo]] = {}
        
        for file_path in self.project_root.rglob("*"):
            if file_path.is_file() and not self._should_ignore_file(file_path):
                try:
                    file_info = self._get_file_info(file_path)
                    
                    if file_info.hash in file_hashes:
                        file_hashes[file_info.hash].append(file_info)
                    else:
                        file_hashes[file_info.hash] = [file_info]
                        
                except Exception as e:
                    self.logger.warning(f"خطأ في تحليل {file_path}: {e}")
        
        # تحديد المكررات
        for file_hash, files in file_hashes.items():
            if len(files) > 1:
                self.duplicates[file_hash] = files
                for file_info in files:
                    file_info.is_duplicate = True
                
                self.stats["duplicates_found"] += len(files) - 1
        
        self.logger.info(f"🔍 تم العثور على {self.stats['duplicates_found']} ملف مكرر")
    
    def _get_file_info(self, file_path: Path) -> FileInfo:
        """الحصول على معلومات الملف"""
        stat = file_path.stat()
        
        # حساب hash
        with open(file_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        
        # تحديد نوع المحتوى
        content_type = self._determine_content_type(file_path)
        
        return FileInfo(
            path=file_path,
            size=stat.st_size,
            hash=file_hash,
            content_type=content_type
        )
    
    def _determine_content_type(self, file_path: Path) -> str:
        """تحديد نوع محتوى الملف"""
        suffix = file_path.suffix.lower()
        
        type_mapping = {
            '.py': 'python',
            '.js': 'javascript',
            '.html': 'html',
            '.css': 'css',
            '.json': 'json',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.md': 'markdown',
            '.txt': 'text',
            '.log': 'log',
            '.db': 'database',
            '.sqlite': 'database',
            '.pth': 'model',
            '.pkl': 'model',
            '.h5': 'model',
            '.jpg': 'image',
            '.png': 'image',
            '.gif': 'image',
            '.mp3': 'audio',
            '.wav': 'audio',
            '.mp4': 'video',
            '.avi': 'video'
        }
        
        return type_mapping.get(suffix, 'unknown')
    
    def _should_ignore_file(self, file_path: Path) -> bool:
        """تحديد ما إذا كان يجب تجاهل الملف"""
        ignore_patterns = [
            '.git', '__pycache__', '.pytest_cache',
            'node_modules', '.vscode', '.idea',
            '*.pyc', '*.pyo', '*.egg-info',
            '.env', '.DS_Store', 'Thumbs.db'
        ]
        
        path_str = str(file_path)
        
        for pattern in ignore_patterns:
            if pattern in path_str:
                return True
        
        return False
    
    def _create_target_structure(self):
        """إنشاء الهيكل المستهدف"""
        self.logger.info("🏗️ إنشاء الهيكل المستهدف...")
        
        self._create_structure_recursive(self.target_structure, self.project_root)
    
    def _create_structure_recursive(self, structure: Dict, base_path: Path):
        """إنشاء الهيكل بطريقة تكرارية"""
        for name, content in structure.items():
            dir_path = base_path / name
            
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)
                self.stats["directories_created"] += 1
                self.logger.info(f"📁 تم إنشاء المجلد: {dir_path}")
            
            if isinstance(content, dict):
                self._create_structure_recursive(content, dir_path)
    
    def _move_files(self):
        """نقل الملفات إلى المواقع الصحيحة"""
        self.logger.info("📦 نقل الملفات...")
        
        # قواعد النقل
        move_rules = {
            'assistant': ('core', 'الملفات الأساسية للمساعد'),
            'main': ('core', 'ملفات التشغيل الرئيسية'),
            'config': ('config', 'ملفات التكوين'),
            'settings': ('config', 'ملفات الإعدادات'),
            'bert_analyzer': ('ai_models/nlu', 'محلل BERT'),
            'gpt4_interface': ('ai_models/nlu', 'واجهة GPT-4'),
            'gpt4_generator': ('ai_models/nlg', 'مولد GPT-4'),
            'fastspeech_tts': ('ai_models/nlg', 'تحويل النص إلى كلام'),
            'vision_pipeline': ('ai_models/vision', 'خط معالجة الرؤية'),
            'active_learning': ('ai_models/learning', 'التعلم النشط'),
            'few_shot_learner': ('ai_models/learning', 'التعلم بالأمثلة القليلة'),
            'reinforcement_engine': ('ai_models/learning', 'محرك التعلم المعزز'),
            'dask_processor': ('analytics/big_data', 'معالج Dask'),
            'spark_processor': ('analytics/big_data', 'معالج Spark'),
            'ml_predictor': ('analytics/prediction', 'متنبئ التعلم الآلي'),
            'dl_predictor': ('analytics/prediction', 'متنبئ التعلم العميق'),
            'collaborative_filtering': ('analytics/recommendation', 'التصفية التعاونية'),
            'content_based': ('analytics/recommendation', 'التوصية المحتوى'),
            'dash_dashboard': ('analytics/visualization', 'لوحة Dash'),
            'test_': ('tests', 'ملفات الاختبار'),
            '_test': ('tests', 'ملفات الاختبار'),
            'cleanup': ('tools', 'أدوات التنظيف'),
            'analyzer': ('tools', 'أدوات التحليل')
        }
        
        # نقل الملفات من المجلد الجذر
        for file_path in self.project_root.glob("*.py"):
            if file_path.is_file():
                moved = False
                
                for pattern, (target_dir, description) in move_rules.items():
                    if pattern in file_path.stem.lower():
                        target_path = self.project_root / target_dir / file_path.name
                        
                        if not target_path.exists():
                            self._move_file_safely(file_path, target_path)
                            moved = True
                            break
                
                # إذا لم يتم نقل الملف، ضعه في مجلد misc
                if not moved and not self._is_important_root_file(file_path):
                    misc_dir = self.project_root / "misc"
                    misc_dir.mkdir(exist_ok=True)
                    target_path = misc_dir / file_path.name
                    
                    if not target_path.exists():
                        self._move_file_safely(file_path, target_path)
    
    def _is_important_root_file(self, file_path: Path) -> bool:
        """تحديد ما إذا كان الملف مهماً في المجلد الجذر"""
        important_files = [
            'main_unified.py', 'setup.py', 'manage.py',
            'requirements.txt', 'pyproject.toml', 'README.md'
        ]
        
        return file_path.name in important_files
    
    def _move_file_safely(self, source: Path, target: Path):
        """نقل الملف بأمان"""
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(target))
            self.stats["files_moved"] += 1
            self.logger.info(f"📦 تم نقل: {source.name} → {target.parent.name}/")
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في نقل {source}: {e}")
    
    def _remove_duplicates(self):
        """حذف الملفات المكررة"""
        if not self.duplicates:
            return
        
        self.logger.info("🗑️ حذف الملفات المكررة...")
        
        for file_hash, files in self.duplicates.items():
            # الاحتفاظ بأفضل نسخة (الأكبر أو في مكان أفضل)
            best_file = self._choose_best_duplicate(files)
            
            for file_info in files:
                if file_info != best_file:
                    try:
                        file_info.path.unlink()
                        self.stats["files_deleted"] += 1
                        self.stats["total_space_saved"] += file_info.size
                        self.logger.info(f"🗑️ تم حذف المكرر: {file_info.path}")
                        
                    except Exception as e:
                        self.logger.error(f"❌ خطأ في حذف {file_info.path}: {e}")
    
    def _choose_best_duplicate(self, files: List[FileInfo]) -> FileInfo:
        """اختيار أفضل نسخة من الملفات المكررة"""
        # الأولوية للملفات في مجلدات منظمة
        organized_dirs = ['core', 'ai_models', 'analytics', 'interfaces']
        
        for directory in organized_dirs:
            for file_info in files:
                if directory in str(file_info.path):
                    return file_info
        
        # إذا لم توجد في مجلدات منظمة، اختر الأكبر
        return max(files, key=lambda f: f.size)
    
    def _cleanup_empty_directories(self):
        """تنظيف المجلدات الفارغة"""
        self.logger.info("🧹 تنظيف المجلدات الفارغة...")
        
        # البحث عن المجلدات الفارغة
        empty_dirs = []
        
        for dir_path in self.project_root.rglob("*"):
            if dir_path.is_dir() and not self._should_ignore_file(dir_path):
                try:
                    # تحقق من وجود ملفات
                    if not any(dir_path.iterdir()):
                        empty_dirs.append(dir_path)
                except PermissionError:
                    continue
        
        # حذف المجلدات الفارغة
        for dir_path in empty_dirs:
            try:
                dir_path.rmdir()
                self.logger.info(f"🗑️ تم حذف المجلد الفارغ: {dir_path}")
            except Exception as e:
                self.logger.warning(f"⚠️ تعذر حذف {dir_path}: {e}")
    
    def _create_readme_files(self):
        """إنشاء ملفات README للمجلدات"""
        self.logger.info("📖 إنشاء ملفات README...")
        
        readme_content = {
            "core": "# المحرك الأساسي\nيحتوي على الملفات الأساسية لتشغيل المساعد الذكي",
            "ai_models": "# نماذج الذكاء الاصطناعي\nيحتوي على جميع نماذج الذكاء الاصطناعي والتعلم الآلي",
            "ai_models/nlu": "# معالجة اللغة الطبيعية\nنماذج فهم وتحليل اللغة الطبيعية",
            "ai_models/nlg": "# توليد اللغة الطبيعية\nنماذج توليد النصوص والاستجابات",
            "ai_models/vision": "# الرؤية الحاسوبية\nنماذج معالجة وتحليل الصور",
            "ai_models/learning": "# التعلم الآلي\nخوارزميات التعلم والتكيف",
            "analytics": "# التحليلات المتقدمة\nأدوات تحليل البيانات والإحصائيات",
            "analytics/big_data": "# البيانات الضخمة\nأدوات معالجة البيانات الضخمة",
            "analytics/prediction": "# أنظمة التنبؤ\nنماذج التنبؤ والتوقع",
            "analytics/recommendation": "# أنظمة التوصية\nخوارزميات التوصية الذكية",
            "analytics/visualization": "# التصور\nأدوات إنشاء الرسوم البيانية واللوحات",
            "config": "# ملفات التكوين\nجميع إعدادات وتكوينات النظام",
            "data": "# البيانات\nتخزين البيانات والنماذج والملفات",
            "tests": "# الاختبارات\nاختبارات الوحدة والتكامل",
            "tools": "# الأدوات\nأدوات التطوير والصيانة",
            "docs": "# التوثيق\nدليل المستخدم والتوثيق التقني"
        }
        
        for dir_path, content in readme_content.items():
            readme_file = self.project_root / dir_path / "README.md"
            
            if not readme_file.exists():
                try:
                    readme_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(readme_file, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    self.logger.info(f"📖 تم إنشاء README: {dir_path}")
                    
                except Exception as e:
                    self.logger.error(f"❌ خطأ في إنشاء README لـ {dir_path}: {e}")
    
    def _update_gitignore(self):
        """تحديث ملف .gitignore"""
        gitignore_path = self.project_root / ".gitignore"
        
        gitignore_content = """
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
env.bak/
venv.bak/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Project specific
data/cache/
data/logs/
*.log
.env
temp/
misc/

# Models and large files
*.pth
*.h5
*.pkl
data/models/*.bin

# Temporary files
*.tmp
*.temp
"""
        
        try:
            with open(gitignore_path, 'w', encoding='utf-8') as f:
                f.write(gitignore_content.strip())
            
            self.logger.info("📋 تم تحديث .gitignore")
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في تحديث .gitignore: {e}")
    
    def _generate_report(self):
        """إنشاء تقرير التنظيم"""
        report = {
            "تاريخ_التنظيم": str(Path.cwd()),
            "إحصائيات": self.stats,
            "الملفات_المكررة_المحذوفة": len(self.duplicates),
            "المساحة_المحررة_بالبايت": self.stats["total_space_saved"],
            "المساحة_المحررة_MB": round(self.stats["total_space_saved"] / (1024 * 1024), 2)
        }
        
        report_path = self.project_root / "organization_report.json"
        
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            self.logger.info("📊 تم إنشاء تقرير التنظيم")
            
            # طباعة الملخص
            print("\n" + "="*50)
            print("📊 ملخص تنظيم المشروع")
            print("="*50)
            print(f"📦 الملفات المنقولة: {self.stats['files_moved']}")
            print(f"🗑️ الملفات المحذوفة: {self.stats['files_deleted']}")
            print(f"📁 المجلدات المنشأة: {self.stats['directories_created']}")
            print(f"💾 المساحة المحررة: {report['المساحة_المحررة_MB']} MB")
            print("="*50)
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في إنشاء التقرير: {e}")

def main():
    """تشغيل منظم المشروع"""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    organizer = ProjectOrganizer()
    organizer.organize_project()

if __name__ == "__main__":
    main()
