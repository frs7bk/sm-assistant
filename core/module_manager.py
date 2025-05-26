
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
مدير الوحدات المتقدم
يدير تحميل وتهيئة وتشغيل جميع وحدات المساعد
"""

import asyncio
import importlib
import inspect
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Type, Callable
from dataclasses import dataclass
from enum import Enum
import sys

# إضافة مسار المشروع
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class ModuleStatus(Enum):
    """حالات الوحدات"""
    NOT_LOADED = "not_loaded"
    LOADING = "loading"
    LOADED = "loaded"
    FAILED = "failed"
    DISABLED = "disabled"

@dataclass
class ModuleInfo:
    """معلومات الوحدة"""
    name: str
    module_type: str
    file_path: Path
    dependencies: List[str]
    status: ModuleStatus
    instance: Optional[Any] = None
    error_message: Optional[str] = None

class AdvancedModuleManager:
    """مدير الوحدات المتقدم"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.modules: Dict[str, ModuleInfo] = {}
        self.module_registry = {}
        self.load_order = []
        
        # مسارات الوحدات
        self.module_paths = {
            "ai_models": project_root / "ai_models",
            "analytics": project_root / "analytics", 
            "interfaces": project_root / "interfaces",
            "learning": project_root / "learning"
        }
        
    def discover_modules(self):
        """اكتشاف جميع الوحدات المتاحة"""
        self.logger.info("🔍 بدء اكتشاف الوحدات...")
        
        for module_type, path in self.module_paths.items():
            if path.exists():
                self._discover_modules_in_path(path, module_type)
        
        self.logger.info(f"✅ تم اكتشاف {len(self.modules)} وحدة")
    
    def _discover_modules_in_path(self, path: Path, module_type: str):
        """اكتشاف الوحدات في مسار محدد"""
        for file_path in path.rglob("*.py"):
            if file_path.name.startswith("__"):
                continue
                
            module_name = self._get_module_name(file_path, path)
            
            # تحليل التبعيات
            dependencies = self._analyze_dependencies(file_path)
            
            module_info = ModuleInfo(
                name=module_name,
                module_type=module_type,
                file_path=file_path,
                dependencies=dependencies,
                status=ModuleStatus.NOT_LOADED
            )
            
            self.modules[module_name] = module_info
    
    def _get_module_name(self, file_path: Path, base_path: Path) -> str:
        """استخراج اسم الوحدة من المسار"""
        relative_path = file_path.relative_to(project_root)
        module_name = str(relative_path).replace("/", ".").replace("\\", ".").replace(".py", "")
        return module_name
    
    def _analyze_dependencies(self, file_path: Path) -> List[str]:
        """تحليل تبعيات الوحدة"""
        dependencies = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # البحث عن استيرادات محلية
            for line in content.split('\n'):
                line = line.strip()
                if line.startswith(('from ai_models', 'from analytics', 'from learning')):
                    # استخراج اسم الوحدة المستوردة
                    parts = line.split()
                    if len(parts) >= 2:
                        module_path = parts[1]
                        dependencies.append(module_path)
                        
        except Exception as e:
            self.logger.warning(f"تعذر تحليل تبعيات {file_path}: {e}")
            
        return dependencies
    
    def calculate_load_order(self):
        """حساب ترتيب تحميل الوحدات بناءً على التبعيات"""
        self.logger.info("📊 حساب ترتيب تحميل الوحدات...")
        
        # خوارزمية الترتيب الطوبولوجي
        visited = set()
        temp_visited = set()
        self.load_order = []
        
        def visit(module_name: str):
            if module_name in temp_visited:
                # دورة في التبعيات
                self.logger.warning(f"⚠️ دورة في التبعيات تتضمن {module_name}")
                return
                
            if module_name in visited:
                return
                
            temp_visited.add(module_name)
            
            # زيارة التبعيات أولاً
            if module_name in self.modules:
                for dependency in self.modules[module_name].dependencies:
                    if dependency in self.modules:
                        visit(dependency)
            
            temp_visited.remove(module_name)
            visited.add(module_name)
            
            if module_name not in self.load_order:
                self.load_order.append(module_name)
        
        # زيارة جميع الوحدات
        for module_name in self.modules:
            visit(module_name)
        
        self.logger.info(f"✅ تم حساب ترتيب التحميل: {len(self.load_order)} وحدة")
    
    async def load_module(self, module_name: str) -> bool:
        """تحميل وحدة محددة"""
        if module_name not in self.modules:
            self.logger.error(f"❌ الوحدة غير موجودة: {module_name}")
            return False
            
        module_info = self.modules[module_name]
        
        if module_info.status == ModuleStatus.LOADED:
            return True
            
        if module_info.status == ModuleStatus.LOADING:
            # انتظار انتهاء التحميل
            while module_info.status == ModuleStatus.LOADING:
                await asyncio.sleep(0.1)
            return module_info.status == ModuleStatus.LOADED
        
        module_info.status = ModuleStatus.LOADING
        self.logger.info(f"📦 تحميل الوحدة: {module_name}")
        
        try:
            # تحميل التبعيات أولاً
            for dependency in module_info.dependencies:
                if dependency in self.modules:
                    success = await self.load_module(dependency)
                    if not success:
                        self.logger.warning(f"⚠️ فشل تحميل التبعية: {dependency}")
            
            # تحميل الوحدة
            spec = importlib.util.spec_from_file_location(
                module_name, module_info.file_path
            )
            
            if spec is None:
                raise ImportError(f"تعذر إنشاء مواصفات للوحدة {module_name}")
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # البحث عن الفئة الرئيسية
            main_class = self._find_main_class(module)
            
            if main_class:
                # إنشاء مثيل من الفئة
                instance = main_class()
                module_info.instance = instance
                
                # تهيئة الوحدة إذا كانت تدعم ذلك
                if hasattr(instance, 'initialize'):
                    await self._safe_call(instance.initialize)
            else:
                # تخزين الوحدة كما هي
                module_info.instance = module
            
            module_info.status = ModuleStatus.LOADED
            self.logger.info(f"✅ تم تحميل الوحدة: {module_name}")
            return True
            
        except Exception as e:
            module_info.status = ModuleStatus.FAILED
            module_info.error_message = str(e)
            self.logger.error(f"❌ فشل تحميل الوحدة {module_name}: {e}")
            return False
    
    def _find_main_class(self, module) -> Optional[Type]:
        """البحث عن الفئة الرئيسية في الوحدة"""
        for name, obj in inspect.getmembers(module):
            if (inspect.isclass(obj) and 
                obj.__module__ == module.__name__ and
                not name.startswith('_')):
                return obj
        return None
    
    async def _safe_call(self, method: Callable, *args, **kwargs):
        """استدعاء آمن للدوال"""
        try:
            if inspect.iscoroutinefunction(method):
                return await method(*args, **kwargs)
            else:
                return method(*args, **kwargs)
        except Exception as e:
            self.logger.error(f"خطأ في استدعاء {method.__name__}: {e}")
            return None
    
    async def load_all_modules(self):
        """تحميل جميع الوحدات بالترتيب الصحيح"""
        self.logger.info("🚀 بدء تحميل جميع الوحدات...")
        
        self.discover_modules()
        self.calculate_load_order()
        
        successful_loads = 0
        
        for module_name in self.load_order:
            success = await self.load_module(module_name)
            if success:
                successful_loads += 1
        
        self.logger.info(f"✅ تم تحميل {successful_loads}/{len(self.modules)} وحدة بنجاح")
        
        return successful_loads, len(self.modules)
    
    def get_module(self, module_name: str) -> Optional[Any]:
        """الحصول على مثيل الوحدة"""
        if module_name in self.modules:
            module_info = self.modules[module_name]
            if module_info.status == ModuleStatus.LOADED:
                return module_info.instance
        return None
    
    def get_modules_by_type(self, module_type: str) -> Dict[str, Any]:
        """الحصول على جميع الوحدات من نوع محدد"""
        result = {}
        for name, info in self.modules.items():
            if (info.module_type == module_type and 
                info.status == ModuleStatus.LOADED):
                result[name] = info.instance
        return result
    
    def get_status_report(self) -> Dict[str, Any]:
        """الحصول على تقرير حالة الوحدات"""
        status_counts = {}
        for status in ModuleStatus:
            status_counts[status.value] = 0
        
        failed_modules = []
        
        for name, info in self.modules.items():
            status_counts[info.status.value] += 1
            if info.status == ModuleStatus.FAILED:
                failed_modules.append({
                    "name": name,
                    "error": info.error_message
                })
        
        return {
            "total_modules": len(self.modules),
            "status_counts": status_counts,
            "failed_modules": failed_modules,
            "load_order": self.load_order
        }

# مثيل عام لمدير الوحدات
module_manager = AdvancedModuleManager()

def get_module_manager() -> AdvancedModuleManager:
    """الحصول على مدير الوحدات"""
    return module_manager

async def main():
    """دالة اختبار مدير الوحدات"""
    print("🔧 اختبار مدير الوحدات المتقدم")
    print("=" * 50)
    
    manager = get_module_manager()
    
    # تحميل جميع الوحدات
    successful, total = await manager.load_all_modules()
    
    # عرض التقرير
    report = manager.get_status_report()
    
    print(f"\n📊 تقرير حالة الوحدات:")
    print(f"   • إجمالي الوحدات: {report['total_modules']}")
    print(f"   • تم تحميلها: {report['status_counts']['loaded']}")
    print(f"   • فشل التحميل: {report['status_counts']['failed']}")
    
    if report['failed_modules']:
        print(f"\n❌ الوحدات الفاشلة:")
        for failed in report['failed_modules']:
            print(f"   • {failed['name']}: {failed['error']}")

if __name__ == "__main__":
    asyncio.run(main())
