
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧹 أداة التنظيف والتنظيم المتقدمة للمساعد الذكي
Advanced Cleanup and Organization Tool
"""

import os
import shutil
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

@dataclass
class FileInfo:
    """معلومات الملف"""
    path: str
    size: int
    modified_time: float
    hash_md5: str
    extension: str
    is_duplicate: bool = False
    keep_file: bool = True
    reason_for_removal: str = ""

@dataclass
class CleanupReport:
    """تقرير التنظيف"""
    total_files_scanned: int
    duplicates_found: int
    files_removed: int
    space_freed: int
    deprecated_files: int
    backup_created: bool
    cleanup_time: float
    errors: List[str]

class AdvancedProjectCleaner:
    """منظف المشروع المتقدم"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.logger = self._setup_logging()
        
        # قواعد التنظيف
        self.cleanup_rules = {
            # الملفات المكررة الواضحة
            "obvious_duplicates": [
                r"assistant_updated.*\.py$",
                r"assistant.*_v\d+\.py$",
                r".*_copy\.py$",
                r".*_backup\.py$",
                r".*_old\.py$"
            ],
            
            # ملفات مؤقتة
            "temp_files": [
                r".*\.tmp$",
                r".*\.temp$", 
                r".*~$",
                r"\.DS_Store$",
                r"Thumbs\.db$"
            ],
            
            # ملفات النظام المؤقتة
            "system_temp": [
                r"__pycache__",
                r"\.pyc$",
                r"\.pyo$",
                r"\.coverage$"
            ],
            
            # مجلدات مؤقتة
            "temp_directories": [
                "__pycache__",
                ".pytest_cache",
                ".coverage",
                "node_modules",
                ".git" # لا نحذفه، فقط نتجاهله
            ]
        }
        
        # الملفات المهمة التي يجب عدم حذفها
        self.protected_files = {
            "main.py",
            "main_unified.py", 
            "requirements.txt",
            "pyproject.toml",
            ".replit",
            ".env",
            ".env.example",
            "README.md"
        }
        
        # مجلدات محمية
        self.protected_dirs = {
            ".git",
            "data",
            "frontend",
            "api"
        }
        
        # إحصائيات
        self.stats = {
            "files_scanned": 0,
            "duplicates_found": 0,
            "files_removed": 0,
            "space_freed": 0,
            "errors": []
        }
    
    def _setup_logging(self) -> logging.Logger:
        """إعداد نظام السجلات"""
        logger = logging.getLogger("project_cleaner")
        logger.setLevel(logging.INFO)
        
        # إنشاء مجلد السجلات
        log_dir = Path("data/cleanup_logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # معالج الملف
        file_handler = logging.FileHandler(
            log_dir / f"cleanup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
            encoding='utf-8'
        )
        file_handler.setLevel(logging.INFO)
        
        # معالج وحدة التحكم
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # تنسيق السجلات
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def analyze_project_structure(self) -> Dict[str, Any]:
        """تحليل بنية المشروع"""
        print("🔍 تحليل بنية المشروع...")
        
        analysis = {
            "total_files": 0,
            "total_size": 0,
            "file_types": {},
            "largest_files": [],
            "potential_duplicates": [],
            "deprecated_patterns": [],
            "directory_structure": {}
        }
        
        # مسح جميع الملفات
        all_files = []
        for root, dirs, files in os.walk(self.project_root):
            # تجاهل المجلدات المحمية
            dirs[:] = [d for d in dirs if d not in self.protected_dirs]
            
            for file in files:
                file_path = Path(root) / file
                try:
                    file_stat = file_path.stat()
                    file_info = FileInfo(
                        path=str(file_path.relative_to(self.project_root)),
                        size=file_stat.st_size,
                        modified_time=file_stat.st_mtime,
                        hash_md5=self._get_file_hash(file_path),
                        extension=file_path.suffix.lower()
                    )
                    all_files.append(file_info)
                    
                    analysis["total_files"] += 1
                    analysis["total_size"] += file_info.size
                    
                    # تصنيف أنواع الملفات
                    ext = file_info.extension
                    analysis["file_types"][ext] = analysis["file_types"].get(ext, 0) + 1
                    
                except (PermissionError, FileNotFoundError) as e:
                    self.logger.warning(f"لا يمكن الوصول إلى الملف: {file_path} - {e}")
                    self.stats["errors"].append(str(e))
        
        # العثور على أكبر الملفات
        all_files.sort(key=lambda x: x.size, reverse=True)
        analysis["largest_files"] = [
            {"path": f.path, "size": f.size, "size_mb": f.size / 1024 / 1024}
            for f in all_files[:10]
        ]
        
        # العثور على الملفات المكررة المحتملة
        analysis["potential_duplicates"] = self._find_potential_duplicates(all_files)
        
        # العثور على الأنماط المتروكة
        analysis["deprecated_patterns"] = self._find_deprecated_patterns(all_files)
        
        self.logger.info(f"تم تحليل {analysis['total_files']} ملف بحجم إجمالي {analysis['total_size']/1024/1024:.2f} MB")
        
        return analysis
    
    def _get_file_hash(self, file_path: Path) -> str:
        """حساب hash للملف"""
        try:
            if file_path.stat().st_size > 50 * 1024 * 1024:  # تجاهل الملفات أكبر من 50MB
                return "large_file"
            
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception:
            return "error_hash"
    
    def _find_potential_duplicates(self, files: List[FileInfo]) -> List[Dict[str, Any]]:
        """العثور على الملفات المكررة المحتملة"""
        hash_groups = {}
        for file_info in files:
            if file_info.hash_md5 not in ["large_file", "error_hash"]:
                if file_info.hash_md5 not in hash_groups:
                    hash_groups[file_info.hash_md5] = []
                hash_groups[file_info.hash_md5].append(file_info)
        
        duplicates = []
        for hash_key, file_group in hash_groups.items():
            if len(file_group) > 1:
                duplicates.append({
                    "hash": hash_key,
                    "count": len(file_group),
                    "files": [{"path": f.path, "size": f.size} for f in file_group],
                    "total_waste": sum(f.size for f in file_group[1:])  # كل الملفات عدا الأول
                })
        
        return sorted(duplicates, key=lambda x: x["total_waste"], reverse=True)
    
    def _find_deprecated_patterns(self, files: List[FileInfo]) -> List[Dict[str, Any]]:
        """العثور على الأنماط المتروكة"""
        import re
        
        deprecated = []
        
        for pattern_type, patterns in self.cleanup_rules.items():
            matching_files = []
            for file_info in files:
                for pattern in patterns:
                    if re.search(pattern, file_info.path):
                        matching_files.append(file_info)
                        break
            
            if matching_files:
                deprecated.append({
                    "pattern_type": pattern_type,
                    "count": len(matching_files),
                    "files": [f.path for f in matching_files],
                    "total_size": sum(f.size for f in matching_files)
                })
        
        return deprecated
    
    def create_backup(self) -> bool:
        """إنشاء نسخة احتياطية"""
        try:
            backup_dir = Path("backup") / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"📦 إنشاء نسخة احتياطية في: {backup_dir}")
            
            # نسخ الملفات المهمة فقط
            important_files = [
                "main.py",
                "main_unified.py",
                "requirements.txt",
                "pyproject.toml",
                ".replit"
            ]
            
            for file_name in important_files:
                src_file = self.project_root / file_name
                if src_file.exists():
                    shutil.copy2(src_file, backup_dir / file_name)
            
            # نسخ المجلدات المهمة
            important_dirs = ["config", "data/sessions"]
            for dir_name in important_dirs:
                src_dir = self.project_root / dir_name
                if src_dir.exists():
                    shutil.copytree(src_dir, backup_dir / dir_name, dirs_exist_ok=True)
            
            self.logger.info(f"تم إنشاء النسخة الاحتياطية: {backup_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"فشل في إنشاء النسخة الاحتياطية: {e}")
            self.stats["errors"].append(f"Backup failed: {e}")
            return False
    
    def clean_duplicates(self, dry_run: bool = True) -> CleanupReport:
        """تنظيف الملفات المكررة"""
        start_time = time.time()
        
        print("🧹 بدء عملية التنظيف...")
        if dry_run:
            print("⚠️ وضع المحاكاة - لن يتم حذف أي ملفات")
        
        # تحليل المشروع
        analysis = self.analyze_project_structure()
        
        files_to_remove = []
        
        # معالجة الملفات المكررة
        for duplicate_group in analysis["potential_duplicates"]:
            files_in_group = duplicate_group["files"]
            if len(files_in_group) > 1:
                # ترتيب الملفات حسب الأولوية (الأحدث والأقصر اسماً يُحفظ)
                sorted_files = sorted(
                    files_in_group, 
                    key=lambda x: (len(x["path"]), -os.path.getmtime(self.project_root / x["path"]))
                )
                
                # حفظ الأول، حذف الباقي
                for file_info in sorted_files[1:]:
                    if not self._is_protected_file(file_info["path"]):
                        files_to_remove.append({
                            "path": file_info["path"],
                            "reason": "duplicate",
                            "size": file_info["size"]
                        })
        
        # معالجة الملفات المتروكة
        for deprecated_group in analysis["deprecated_patterns"]:
            if deprecated_group["pattern_type"] in ["obvious_duplicates", "temp_files"]:
                for file_path in deprecated_group["files"]:
                    if not self._is_protected_file(file_path):
                        files_to_remove.append({
                            "path": file_path,
                            "reason": deprecated_group["pattern_type"],
                            "size": next((f.size for f in analysis["largest_files"] if f["path"] == file_path), 0)
                        })
        
        # تنفيذ الحذف
        actually_removed = 0
        space_freed = 0
        
        if not dry_run:
            # إنشاء نسخة احتياطية أولاً
            backup_created = self.create_backup()
        else:
            backup_created = False
        
        for file_info in files_to_remove:
            file_path = self.project_root / file_info["path"]
            
            try:
                if file_path.exists():
                    if not dry_run:
                        if file_path.is_file():
                            file_path.unlink()
                        elif file_path.is_dir():
                            shutil.rmtree(file_path)
                        
                        actually_removed += 1
                        space_freed += file_info["size"]
                        self.logger.info(f"تم حذف: {file_info['path']} (السبب: {file_info['reason']})")
                    else:
                        print(f"سيتم حذف: {file_info['path']} (السبب: {file_info['reason']})")
                        
            except Exception as e:
                error_msg = f"فشل حذف {file_info['path']}: {e}"
                self.logger.error(error_msg)
                self.stats["errors"].append(error_msg)
        
        # إنشاء التقرير
        cleanup_time = time.time() - start_time
        
        report = CleanupReport(
            total_files_scanned=analysis["total_files"],
            duplicates_found=len(analysis["potential_duplicates"]),
            files_removed=actually_removed,
            space_freed=space_freed,
            deprecated_files=len(files_to_remove),
            backup_created=backup_created,
            cleanup_time=cleanup_time,
            errors=self.stats["errors"]
        )
        
        # عرض التقرير
        self._display_cleanup_report(report, dry_run)
        
        # حفظ التقرير
        self._save_cleanup_report(report, analysis)
        
        return report
    
    def _is_protected_file(self, file_path: str) -> bool:
        """فحص ما إذا كان الملف محمياً"""
        file_name = Path(file_path).name
        
        # الملفات المحمية بالاسم
        if file_name in self.protected_files:
            return True
        
        # المجلدات المحمية
        path_parts = Path(file_path).parts
        if any(part in self.protected_dirs for part in path_parts):
            return True
        
        # الملفات الحديثة (أقل من يوم)
        try:
            full_path = self.project_root / file_path
            if full_path.exists():
                modified_time = full_path.stat().st_mtime
                if time.time() - modified_time < 86400:  # 24 ساعة
                    return True
        except:
            pass
        
        return False
    
    def _display_cleanup_report(self, report: CleanupReport, dry_run: bool):
        """عرض تقرير التنظيف"""
        print("\n" + "="*60)
        print("🎯 تقرير التنظيف")
        print("="*60)
        print(f"إجمالي الملفات المفحوصة: {report.total_files_scanned:,}")
        print(f"الملفات المكررة المكتشفة: {report.duplicates_found}")
        print(f"الملفات {'المحددة للحذف' if dry_run else 'المحذوفة'}: {report.files_removed}")
        print(f"المساحة {'المحررة المحتملة' if dry_run else 'المحررة'}: {report.space_freed/1024/1024:.2f} MB")
        print(f"وقت المعالجة: {report.cleanup_time:.2f} ثانية")
        
        if report.backup_created:
            print("✅ تم إنشاء نسخة احتياطية")
        
        if report.errors:
            print(f"\n⚠️ الأخطاء ({len(report.errors)}):")
            for error in report.errors[:5]:  # عرض أول 5 أخطاء
                print(f"   • {error}")
            if len(report.errors) > 5:
                print(f"   ... و {len(report.errors) - 5} أخطاء أخرى")
        
        print("="*60)
    
    def _save_cleanup_report(self, report: CleanupReport, analysis: Dict[str, Any]):
        """حفظ تقرير التنظيف"""
        try:
            reports_dir = Path("data/cleanup_reports")
            reports_dir.mkdir(parents=True, exist_ok=True)
            
            report_data = {
                "timestamp": time.time(),
                "date": datetime.now().isoformat(),
                "cleanup_report": asdict(report),
                "project_analysis": analysis
            }
            
            report_file = reports_dir / f"cleanup_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, ensure_ascii=False, indent=2)
            
            print(f"💾 تم حفظ التقرير: {report_file}")
            
        except Exception as e:
            self.logger.error(f"فشل حفظ التقرير: {e}")
    
    def reorganize_structure(self):
        """إعادة تنظيم بنية المشروع"""
        print("📁 إعادة تنظيم بنية المشروع...")
        
        # خطة إعادة التنظيم
        reorganization_plan = {
            "consolidate_core": [
                "core/assistant.py",  # الملف الرئيسي
                "core/main_unified.py"  # النقطة الموحدة
            ],
            "clean_old_versions": [
                "core/assistant_updated*.py",
                "core/assistant_*_features.py"
            ],
            "organize_ai_models": [
                "ai_models/**/*.py"
            ]
        }
        
        # تنفيذ الخطة (في المستقبل)
        self.logger.info("خطة إعادة التنظيم محفوظة للتنفيذ المستقبلي")

def main():
    """الدالة الرئيسية"""
    print("🧹 أداة التنظيف والتنظيم المتقدمة")
    print("="*50)
    
    cleaner = AdvancedProjectCleaner()
    
    # تشغيل محاكاة أولاً
    print("\n📋 تشغيل تحليل أولي...")
    report = cleaner.clean_duplicates(dry_run=True)
    
    if report.files_removed > 0:
        response = input(f"\n❓ هل تريد المتابعة وحذف {report.files_removed} ملف؟ (y/N): ")
        if response.lower() in ['y', 'yes', 'نعم']:
            print("\n🗑️ تنفيذ التنظيف الفعلي...")
            final_report = cleaner.clean_duplicates(dry_run=False)
            print("✅ تم التنظيف بنجاح!")
        else:
            print("❌ تم إلغاء عملية التنظيف")
    else:
        print("✨ المشروع منظم بالفعل!")

if __name__ == "__main__":
    main()
