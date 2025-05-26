
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
أداة تحليل المشروع المتقدمة
تقوم بتحليل شامل للكود وتقديم تقارير احترافية
"""

import os
import ast
import json
from pathlib import Path
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter

@dataclass
class FileAnalysis:
    """تحليل ملف واحد"""
    path: str
    lines_of_code: int
    functions: List[str]
    classes: List[str]
    imports: List[str]
    complexity_score: float
    duplicates_detected: List[str]
    quality_score: float

@dataclass
class ProjectAnalysis:
    """تحليل المشروع الكامل"""
    total_files: int
    total_lines: int
    programming_languages: Dict[str, int]
    duplicate_files: List[Tuple[str, str]]
    architectural_issues: List[str]
    recommendations: List[str]
    advanced_features: List[str]
    quality_metrics: Dict[str, float]

class AdvancedProjectAnalyzer:
    """محلل المشروع المتقدم"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.file_analyses = []
        self.duplicate_patterns = {}
        self.architecture_map = {}
        
    def analyze_project(self) -> ProjectAnalysis:
        """تحليل شامل للمشروع"""
        print("🔍 بدء التحليل المتقدم للمشروع...")
        
        # تحليل كل ملف
        self._analyze_all_files()
        
        # اكتشاف التكرارات
        duplicates = self._detect_duplicates()
        
        # تحليل البنية المعمارية
        architectural_issues = self._analyze_architecture()
        
        # اكتشاف الميزات المتقدمة
        advanced_features = self._detect_advanced_features()
        
        # حساب مقاييس الجودة
        quality_metrics = self._calculate_quality_metrics()
        
        # تقديم التوصيات
        recommendations = self._generate_recommendations()
        
        # إحصائيات اللغات البرمجية
        languages = self._analyze_languages()
        
        analysis = ProjectAnalysis(
            total_files=len(self.file_analyses),
            total_lines=sum(f.lines_of_code for f in self.file_analyses),
            programming_languages=languages,
            duplicate_files=duplicates,
            architectural_issues=architectural_issues,
            recommendations=recommendations,
            advanced_features=advanced_features,
            quality_metrics=quality_metrics
        )
        
        print("✅ تم الانتهاء من التحليل")
        return analysis
    
    def _analyze_all_files(self):
        """تحليل جميع الملفات في المشروع"""
        python_files = list(self.project_root.rglob("*.py"))
        
        for file_path in python_files:
            if self._should_analyze_file(file_path):
                analysis = self._analyze_file(file_path)
                if analysis:
                    self.file_analyses.append(analysis)
    
    def _should_analyze_file(self, file_path: Path) -> bool:
        """تحديد ما إذا كان يجب تحليل الملف"""
        # تجاهل ملفات معينة
        ignore_patterns = [
            '__pycache__',
            '.git',
            'venv',
            'env',
            '.pytest_cache'
        ]
        
        return not any(pattern in str(file_path) for pattern in ignore_patterns)
    
    def _analyze_file(self, file_path: Path) -> FileAnalysis:
        """تحليل ملف Python واحد"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # تحليل AST
            tree = ast.parse(content)
            
            # استخراج المعلومات
            functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
            imports = self._extract_imports(tree)
            
            # حساب المقاييس
            lines_of_code = len([line for line in content.split('\n') if line.strip() and not line.strip().startswith('#')])
            complexity_score = self._calculate_complexity(tree)
            quality_score = self._calculate_file_quality(content, tree)
            
            return FileAnalysis(
                path=str(file_path.relative_to(self.project_root)),
                lines_of_code=lines_of_code,
                functions=functions,
                classes=classes,
                imports=imports,
                complexity_score=complexity_score,
                duplicates_detected=[],
                quality_score=quality_score
            )
            
        except Exception as e:
            print(f"⚠️ خطأ في تحليل الملف {file_path}: {e}")
            return None
    
    def _extract_imports(self, tree) -> List[str]:
        """استخراج الاستيرادات من AST"""
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend([alias.name for alias in node.names])
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                imports.extend([f"{module}.{alias.name}" for alias in node.names])
        return imports
    
    def _calculate_complexity(self, tree) -> float:
        """حساب تعقد الكود"""
        complexity = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.For, ast.While, ast.Try)):
                complexity += 1
            elif isinstance(node, ast.FunctionDef):
                complexity += len(node.args.args)
        
        return complexity / max(1, len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]))
    
    def _calculate_file_quality(self, content: str, tree) -> float:
        """حساب جودة الملف"""
        score = 100.0
        
        # خصم نقاط للمشاكل
        lines = content.split('\n')
        
        # خطوط طويلة جداً
        long_lines = [line for line in lines if len(line) > 120]
        score -= len(long_lines) * 2
        
        # نقص التوثيق
        functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        documented_functions = 0
        for func in functions:
            if (func.body and isinstance(func.body[0], ast.Expr) 
                and isinstance(func.body[0].value, ast.Str)):
                documented_functions += 1
        
        if functions:
            doc_ratio = documented_functions / len(functions)
            score += doc_ratio * 20
        
        return max(0, min(100, score))
    
    def _detect_duplicates(self) -> List[Tuple[str, str]]:
        """اكتشاف الملفات المكررة"""
        duplicates = []
        
        # تجميع الملفات حسب الاسم
        name_groups = defaultdict(list)
        for analysis in self.file_analyses:
            filename = Path(analysis.path).name
            name_groups[filename].append(analysis.path)
        
        # البحث عن التكرارات
        for filename, paths in name_groups.items():
            if len(paths) > 1:
                for i, path1 in enumerate(paths):
                    for path2 in paths[i+1:]:
                        duplicates.append((path1, path2))
        
        return duplicates
    
    def _analyze_architecture(self) -> List[str]:
        """تحليل البنية المعمارية"""
        issues = []
        
        # فحص التنظيم
        root_python_files = [f for f in self.file_analyses if '/' not in f.path]
        if len(root_python_files) > 10:
            issues.append("عدد كبير من ملفات Python في المجلد الجذر")
        
        # فحص الاستيرادات الدائرية
        import_graph = self._build_import_graph()
        if self._has_circular_imports(import_graph):
            issues.append("اكتُشفت استيرادات دائرية")
        
        # فحص التسمية المتسقة
        naming_issues = self._check_naming_consistency()
        issues.extend(naming_issues)
        
        return issues
    
    def _build_import_graph(self) -> Dict[str, Set[str]]:
        """بناء رسم بياني للاستيرادات"""
        graph = defaultdict(set)
        
        for analysis in self.file_analyses:
            module_name = analysis.path.replace('/', '.').replace('.py', '')
            for imp in analysis.imports:
                if imp.startswith('.'):  # استيراد نسبي
                    continue
                graph[module_name].add(imp)
        
        return graph
    
    def _has_circular_imports(self, graph: Dict[str, Set[str]]) -> bool:
        """فحص الاستيرادات الدائرية"""
        # خوارزمية بسيطة لاكتشاف الدورات
        visited = set()
        rec_stack = set()
        
        def dfs(node):
            if node in rec_stack:
                return True
            if node in visited:
                return False
            
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in graph.get(node, []):
                if dfs(neighbor):
                    return True
            
            rec_stack.remove(node)
            return False
        
        return any(dfs(node) for node in graph if node not in visited)
    
    def _check_naming_consistency(self) -> List[str]:
        """فحص اتساق التسمية"""
        issues = []
        
        # فحص أسماء الملفات
        file_patterns = Counter()
        for analysis in self.file_analyses:
            filename = Path(analysis.path).name
            if '_' in filename:
                file_patterns['snake_case'] += 1
            elif any(c.isupper() for c in filename):
                file_patterns['camelCase'] += 1
            else:
                file_patterns['lowercase'] += 1
        
        if len(file_patterns) > 1:
            issues.append("عدم اتساق في تسمية الملفات")
        
        return issues
    
    def _detect_advanced_features(self) -> List[str]:
        """اكتشاف الميزات المتقدمة"""
        features = []
        
        # فحص مكتبات الذكاء الاصطناعي
        ai_libraries = ['transformers', 'torch', 'tensorflow', 'openai', 'deepface']
        found_ai = any(
            any(lib in imp for imp in analysis.imports)
            for analysis in self.file_analyses
            for lib in ai_libraries
        )
        if found_ai:
            features.append("نماذج الذكاء الاصطناعي المتقدمة")
        
        # فحص التعلم الآلي
        ml_libraries = ['sklearn', 'pandas', 'numpy', 'scipy']
        found_ml = any(
            any(lib in imp for imp in analysis.imports)
            for analysis in self.file_analyses
            for lib in ml_libraries
        )
        if found_ml:
            features.append("أدوات التعلم الآلي وتحليل البيانات")
        
        # فحص الواجهات المتقدمة
        ui_libraries = ['dash', 'streamlit', 'flask', 'fastapi']
        found_ui = any(
            any(lib in imp for imp in analysis.imports)
            for analysis in self.file_analyses
            for lib in ui_libraries
        )
        if found_ui:
            features.append("واجهات مستخدم تفاعلية")
        
        # فحص معالجة الصور والفيديو
        vision_libraries = ['cv2', 'PIL', 'skimage']
        found_vision = any(
            any(lib in imp for imp in analysis.imports)
            for analysis in self.file_analyses
            for lib in vision_libraries
        )
        if found_vision:
            features.append("معالجة الصور والرؤية الحاسوبية")
        
        # فحص البرمجة غير المتزامنة
        found_async = any(
            'async' in analysis.path or 'await' in analysis.path
            for analysis in self.file_analyses
        )
        if found_async:
            features.append("البرمجة غير المتزامنة المتقدمة")
        
        return features
    
    def _calculate_quality_metrics(self) -> Dict[str, float]:
        """حساب مقاييس الجودة"""
        if not self.file_analyses:
            return {}
        
        quality_scores = [f.quality_score for f in self.file_analyses]
        complexity_scores = [f.complexity_score for f in self.file_analyses]
        
        return {
            "متوسط_جودة_الكود": sum(quality_scores) / len(quality_scores),
            "متوسط_تعقد_الكود": sum(complexity_scores) / len(complexity_scores),
            "معدل_التوثيق": self._calculate_documentation_rate(),
            "تغطية_الاختبارات": self._estimate_test_coverage(),
        }
    
    def _calculate_documentation_rate(self) -> float:
        """حساب معدل التوثيق"""
        total_functions = sum(len(f.functions) for f in self.file_analyses)
        if total_functions == 0:
            return 0.0
        
        # تقدير بسيط - يحتاج تحسين
        documented_estimate = total_functions * 0.3  # افتراض 30%
        return (documented_estimate / total_functions) * 100
    
    def _estimate_test_coverage(self) -> float:
        """تقدير تغطية الاختبارات"""
        test_files = [f for f in self.file_analyses if 'test' in f.path.lower()]
        if not test_files:
            return 0.0
        
        # تقدير بسيط
        return min(len(test_files) / len(self.file_analyses) * 100, 100)
    
    def _analyze_languages(self) -> Dict[str, int]:
        """تحليل اللغات البرمجية المستخدمة"""
        languages = defaultdict(int)
        
        for file_path in self.project_root.rglob("*"):
            if file_path.is_file():
                suffix = file_path.suffix.lower()
                if suffix == '.py':
                    languages['Python'] += 1
                elif suffix == '.js':
                    languages['JavaScript'] += 1
                elif suffix == '.json':
                    languages['JSON'] += 1
                elif suffix == '.md':
                    languages['Markdown'] += 1
                elif suffix in ['.yml', '.yaml']:
                    languages['YAML'] += 1
                elif suffix == '.toml':
                    languages['TOML'] += 1
        
        return dict(languages)
    
    def _generate_recommendations(self) -> List[str]:
        """توليد التوصيات للتحسين"""
        recommendations = []
        
        # تحليل التكرارات
        if len(self._detect_duplicates()) > 5:
            recommendations.append("إزالة الملفات المكررة وتوحيد الكود")
        
        # تحليل البنية
        root_files = [f for f in self.file_analyses if '/' not in f.path]
        if len(root_files) > 8:
            recommendations.append("إعادة تنظيم الملفات في مجلدات مناسبة")
        
        # تحليل الجودة
        avg_quality = sum(f.quality_score for f in self.file_analyses) / len(self.file_analyses)
        if avg_quality < 70:
            recommendations.append("تحسين جودة الكود وإضافة التوثيق")
        
        # تحليل التعقد
        high_complexity_files = [f for f in self.file_analyses if f.complexity_score > 10]
        if high_complexity_files:
            recommendations.append("تبسيط الدوال المعقدة وتقسيمها لوحدات أصغر")
        
        # اختبارات
        test_coverage = self._estimate_test_coverage()
        if test_coverage < 30:
            recommendations.append("إضافة اختبارات شاملة للكود")
        
        # توثيق
        doc_rate = self._calculate_documentation_rate()
        if doc_rate < 50:
            recommendations.append("تحسين التوثيق وإضافة شروحات للدوال")
        
        return recommendations
    
    def generate_report(self, analysis: ProjectAnalysis) -> str:
        """إنشاء تقرير مفصل"""
        report = f"""
# 📊 تقرير التحليل المتقدم للمشروع

## 📈 إحصائيات عامة
- إجمالي الملفات: {analysis.total_files}
- إجمالي الأسطر: {analysis.total_lines:,}
- اللغات المستخدمة: {', '.join(analysis.programming_languages.keys())}

## 🎯 الميزات المتقدمة المكتشفة
"""
        for feature in analysis.advanced_features:
            report += f"✅ {feature}\n"
        
        report += f"""
## 📊 مقاييس الجودة
"""
        for metric, value in analysis.quality_metrics.items():
            report += f"- {metric}: {value:.1f}%\n"
        
        report += f"""
## ⚠️ المشاكل المكتشفة
"""
        for issue in analysis.architectural_issues:
            report += f"❌ {issue}\n"
        
        if analysis.duplicate_files:
            report += f"\n### 🔄 الملفات المكررة ({len(analysis.duplicate_files)})\n"
            for dup1, dup2 in analysis.duplicate_files[:5]:  # أول 5 فقط
                report += f"- {dup1} ↔️ {dup2}\n"
        
        report += f"""
## 💡 التوصيات للتحسين
"""
        for recommendation in analysis.recommendations:
            report += f"📝 {recommendation}\n"
        
        return report
    
    def save_analysis(self, analysis: ProjectAnalysis, filename: str = "project_analysis.json"):
        """حفظ التحليل في ملف JSON"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(asdict(analysis), f, ensure_ascii=False, indent=2)
        print(f"💾 تم حفظ التحليل في: {filename}")

def main():
    """الدالة الرئيسية"""
    analyzer = AdvancedProjectAnalyzer()
    analysis = analyzer.analyze_project()
    
    # إنشاء التقرير
    report = analyzer.generate_report(analysis)
    print(report)
    
    # حفظ التحليل
    analyzer.save_analysis(analysis)
    
    # حفظ التقرير
    with open("project_analysis_report.md", 'w', encoding='utf-8') as f:
        f.write(report)
    print("📄 تم حفظ التقرير في: project_analysis_report.md")

if __name__ == "__main__":
    main()
