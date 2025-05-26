
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
معالج البيانات الضخمة باستخدام Dask
يوفر قدرات متقدمة لمعالجة وتحليل البيانات الضخمة
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import json
from datetime import datetime, timedelta
import warnings

# تجاهل التحذيرات غير المهمة
warnings.filterwarnings('ignore')

try:
    import dask
    import dask.dataframe as dd
    import dask.array as da
    from dask.distributed import Client, as_completed
    from dask import delayed
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False
    print("⚠️ Dask غير متاح - سيتم استخدام Pandas العادي")

class DaskProcessor:
    """معالج البيانات الضخمة المتقدم"""
    
    def __init__(self):
        """تهيئة المعالج"""
        self.logger = logging.getLogger(__name__)
        self.client: Optional[Client] = None
        self.data_dir = Path("data")
        self.cache_dir = self.data_dir / "cache"
        self.results_dir = self.data_dir / "analysis_results"
        
        # إنشاء المجلدات
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # إحصائيات
        self.processing_stats = {
            "total_processed": 0,
            "processing_time": 0.0,
            "memory_used": 0,
            "cache_hits": 0
        }
        
        # تكوين Dask
        if DASK_AVAILABLE:
            dask.config.set({
                'dataframe.query-planning': True,
                'array.slicing.split_large_chunks': True
            })
    
    async def initialize(self):
        """تهيئة عميل Dask"""
        try:
            if DASK_AVAILABLE:
                self.client = Client(processes=False, silence_logs=False)
                self.logger.info(f"تم تهيئة عميل Dask: {self.client.dashboard_link}")
            else:
                self.logger.warning("Dask غير متاح - سيتم استخدام المعالجة التسلسلية")
        except Exception as e:
            self.logger.error(f"خطأ في تهيئة Dask: {e}")
    
    async def cleanup(self):
        """تنظيف الموارد"""
        try:
            if self.client:
                await self.client.close()
                self.logger.info("تم إغلاق عميل Dask")
        except Exception as e:
            self.logger.error(f"خطأ في تنظيف Dask: {e}")
    
    async def process_large_dataset(
        self, 
        file_path: Union[str, Path], 
        chunk_size: int = 10000,
        operations: List[str] = None
    ) -> Dict[str, Any]:
        """معالجة مجموعة بيانات كبيرة"""
        start_time = datetime.now()
        
        try:
            file_path = Path(file_path)
            
            # التحقق من وجود الملف
            if not file_path.exists():
                return {"error": f"الملف غير موجود: {file_path}"}
            
            # تحديد نوع الملف
            if file_path.suffix.lower() == '.csv':
                result = await self._process_csv_file(file_path, chunk_size, operations)
            elif file_path.suffix.lower() in ['.json', '.jsonl']:
                result = await self._process_json_file(file_path, operations)
            elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                result = await self._process_excel_file(file_path, operations)
            else:
                return {"error": f"نوع الملف غير مدعوم: {file_path.suffix}"}
            
            # حساب الوقت المستغرق
            processing_time = (datetime.now() - start_time).total_seconds()
            result["processing_time"] = processing_time
            
            # تحديث الإحصائيات
            self.processing_stats["total_processed"] += 1
            self.processing_stats["processing_time"] += processing_time
            
            return result
            
        except Exception as e:
            self.logger.error(f"خطأ في معالجة البيانات: {e}")
            return {"error": str(e)}
    
    async def _process_csv_file(
        self, 
        file_path: Path, 
        chunk_size: int,
        operations: List[str]
    ) -> Dict[str, Any]:
        """معالجة ملف CSV"""
        try:
            if DASK_AVAILABLE:
                # استخدام Dask للملفات الكبيرة
                df = dd.read_csv(str(file_path), blocksize=f"{chunk_size}KB")
                
                # العمليات الأساسية
                result = {
                    "file_info": {
                        "path": str(file_path),
                        "size_mb": file_path.stat().st_size / (1024 * 1024),
                        "partitions": df.npartitions
                    },
                    "basic_stats": {
                        "total_rows": len(df),
                        "total_columns": len(df.columns),
                        "columns": list(df.columns)
                    }
                }
                
                # تنفيذ العمليات المطلوبة
                if operations:
                    for operation in operations:
                        if operation == "describe":
                            result["description"] = df.describe().compute().to_dict()
                        elif operation == "null_counts":
                            result["null_counts"] = df.isnull().sum().compute().to_dict()
                        elif operation == "data_types":
                            result["data_types"] = df.dtypes.to_dict()
                        elif operation == "memory_usage":
                            result["memory_usage"] = df.memory_usage(deep=True).compute().to_dict()
                
            else:
                # استخدام Pandas العادي
                df = pd.read_csv(file_path, chunksize=chunk_size)
                
                # معالجة البيانات على دفعات
                total_rows = 0
                columns = None
                
                for chunk in df:
                    total_rows += len(chunk)
                    if columns is None:
                        columns = list(chunk.columns)
                
                result = {
                    "file_info": {
                        "path": str(file_path),
                        "size_mb": file_path.stat().st_size / (1024 * 1024)
                    },
                    "basic_stats": {
                        "total_rows": total_rows,
                        "total_columns": len(columns),
                        "columns": columns
                    }
                }
            
            return result
            
        except Exception as e:
            self.logger.error(f"خطأ في معالجة CSV: {e}")
            return {"error": str(e)}
    
    async def _process_json_file(self, file_path: Path, operations: List[str]) -> Dict[str, Any]:
        """معالجة ملف JSON"""
        try:
            # قراءة الملف
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.suffix.lower() == '.jsonl':
                    # JSON Lines
                    data = [json.loads(line) for line in f]
                else:
                    # JSON عادي
                    data = json.load(f)
            
            # تحليل البيانات
            if isinstance(data, list):
                result = {
                    "file_info": {
                        "path": str(file_path),
                        "size_mb": file_path.stat().st_size / (1024 * 1024),
                        "type": "array"
                    },
                    "basic_stats": {
                        "total_items": len(data),
                        "sample_keys": list(data[0].keys()) if data and isinstance(data[0], dict) else None
                    }
                }
            else:
                result = {
                    "file_info": {
                        "path": str(file_path),
                        "size_mb": file_path.stat().st_size / (1024 * 1024),
                        "type": "object"
                    },
                    "basic_stats": {
                        "keys": list(data.keys()) if isinstance(data, dict) else None
                    }
                }
            
            return result
            
        except Exception as e:
            self.logger.error(f"خطأ في معالجة JSON: {e}")
            return {"error": str(e)}
    
    async def _process_excel_file(self, file_path: Path, operations: List[str]) -> Dict[str, Any]:
        """معالجة ملف Excel"""
        try:
            # قراءة معلومات الأوراق
            excel_file = pd.ExcelFile(file_path)
            sheets = excel_file.sheet_names
            
            result = {
                "file_info": {
                    "path": str(file_path),
                    "size_mb": file_path.stat().st_size / (1024 * 1024),
                    "sheets": sheets
                },
                "sheets_data": {}
            }
            
            # معالجة كل ورقة
            for sheet in sheets[:5]:  # معالجة أول 5 أوراق فقط
                df = pd.read_excel(file_path, sheet_name=sheet)
                
                result["sheets_data"][sheet] = {
                    "rows": len(df),
                    "columns": len(df.columns),
                    "column_names": list(df.columns)
                }
            
            return result
            
        except Exception as e:
            self.logger.error(f"خطأ في معالجة Excel: {e}")
            return {"error": str(e)}
    
    async def analyze_sample_data(self) -> str:
        """تحليل بيانات عينة لأغراض التجربة"""
        try:
            # إنشاء بيانات عينة
            sample_data = await self._create_sample_data()
            
            # تحليل البيانات
            analysis_results = await self._analyze_dataframe(sample_data)
            
            # إنشاء تقرير
            report = self._generate_analysis_report(analysis_results)
            
            return report
            
        except Exception as e:
            self.logger.error(f"خطأ في تحليل البيانات العينة: {e}")
            return f"❌ خطأ في التحليل: {e}"
    
    async def _create_sample_data(self) -> pd.DataFrame:
        """إنشاء بيانات عينة للتجربة"""
        np.random.seed(42)
        
        # إنشاء بيانات متنوعة
        n_samples = 10000
        
        data = {
            'user_id': range(1, n_samples + 1),
            'age': np.random.randint(18, 80, n_samples),
            'income': np.random.normal(50000, 15000, n_samples),
            'education_level': np.random.choice(['ثانوي', 'جامعي', 'دراسات عليا'], n_samples),
            'city': np.random.choice(['الرياض', 'جدة', 'الدمام', 'مكة', 'المدينة'], n_samples),
            'satisfaction_score': np.random.uniform(1, 10, n_samples),
            'purchase_frequency': np.random.poisson(5, n_samples),
            'registration_date': pd.date_range('2020-01-01', periods=n_samples, freq='H')
        }
        
        return pd.DataFrame(data)
    
    async def _analyze_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """تحليل مفصل لإطار البيانات"""
        analysis = {}
        
        # الإحصائيات الأساسية
        analysis['basic_info'] = {
            'shape': df.shape,
            'memory_usage': df.memory_usage(deep=True).sum(),
            'dtypes': df.dtypes.to_dict()
        }
        
        # الإحصائيات الوصفية
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            analysis['descriptive_stats'] = df[numeric_columns].describe().to_dict()
        
        # تحليل القيم المفقودة
        analysis['missing_values'] = df.isnull().sum().to_dict()
        
        # تحليل البيانات الفئوية
        categorical_columns = df.select_dtypes(include=['object']).columns
        analysis['categorical_analysis'] = {}
        
        for col in categorical_columns:
            analysis['categorical_analysis'][col] = {
                'unique_values': df[col].nunique(),
                'most_frequent': df[col].mode().iloc[0] if not df[col].empty else None,
                'value_counts': df[col].value_counts().head().to_dict()
            }
        
        # تحليل التوزيعات
        analysis['distributions'] = {}
        for col in numeric_columns:
            analysis['distributions'][col] = {
                'mean': float(df[col].mean()),
                'median': float(df[col].median()),
                'std': float(df[col].std()),
                'skewness': float(df[col].skew()),
                'kurtosis': float(df[col].kurtosis())
            }
        
        # تحليل الارتباطات
        if len(numeric_columns) > 1:
            correlation_matrix = df[numeric_columns].corr()
            analysis['correlations'] = correlation_matrix.to_dict()
        
        return analysis
    
    def _generate_analysis_report(self, analysis: Dict[str, Any]) -> str:
        """إنشاء تقرير تحليل شامل"""
        report = f"""
📊 تقرير تحليل البيانات الضخمة
{'='*50}

📈 معلومات أساسية:
   • عدد الصفوف: {analysis['basic_info']['shape'][0]:,}
   • عدد الأعمدة: {analysis['basic_info']['shape'][1]}
   • استخدام الذاكرة: {analysis['basic_info']['memory_usage'] / 1024 / 1024:.2f} ميجابايت

🔍 تحليل القيم المفقودة:
"""
        
        missing_values = analysis.get('missing_values', {})
        for col, missing_count in missing_values.items():
            if missing_count > 0:
                report += f"   • {col}: {missing_count} قيمة مفقودة\n"
        
        if not any(missing_values.values()):
            report += "   ✅ لا توجد قيم مفقودة\n"
        
        # تحليل البيانات الرقمية
        if 'descriptive_stats' in analysis:
            report += f"\n📊 الإحصائيات الوصفية للأعمدة الرقمية:\n"
            for col, stats in analysis['descriptive_stats'].items():
                report += f"""   • {col}:
     - المتوسط: {stats['mean']:.2f}
     - الوسيط: {stats['50%']:.2f}
     - الانحراف المعياري: {stats['std']:.2f}
"""
        
        # تحليل البيانات الفئوية
        if 'categorical_analysis' in analysis:
            report += f"\n📋 تحليل البيانات الفئوية:\n"
            for col, cat_stats in analysis['categorical_analysis'].items():
                report += f"""   • {col}:
     - قيم فريدة: {cat_stats['unique_values']}
     - الأكثر تكراراً: {cat_stats['most_frequent']}
"""
        
        # الارتباطات
        if 'correlations' in analysis:
            report += f"\n🔗 أقوى الارتباطات:\n"
            correlations = analysis['correlations']
            
            # العثور على أقوى الارتباطات
            strong_correlations = []
            for col1 in correlations:
                for col2 in correlations[col1]:
                    if col1 != col2:
                        corr_value = correlations[col1][col2]
                        if abs(corr_value) > 0.5:
                            strong_correlations.append((col1, col2, corr_value))
            
            for col1, col2, corr in sorted(strong_correlations, key=lambda x: abs(x[2]), reverse=True)[:5]:
                report += f"   • {col1} ↔ {col2}: {corr:.3f}\n"
        
        # الخلاصة والتوصيات
        report += f"""
💡 خلاصة التحليل:
   • جودة البيانات: {'ممتازة' if not any(missing_values.values()) else 'جيدة مع بعض القيم المفقودة'}
   • تنوع البيانات: {'مرتفع' if len(analysis.get('categorical_analysis', {})) > 2 else 'متوسط'}
   • إمكانية التحليل: عالية ✅

📌 التوصيات:
   • يمكن استخدام هذه البيانات لبناء نماذج تنبؤية
   • الارتباطات القوية تشير إلى إمكانيات تحليل متقدمة
   • البيانات جاهزة للتحليلات الإحصائية المتطورة
"""
        
        return report
    
    async def process_user_behavior_data(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """معالجة بيانات سلوك المستخدم"""
        try:
            # تحويل البيانات إلى DataFrame
            df = pd.DataFrame([user_data])
            
            # تحليل البيانات
            analysis = {
                "user_profile": {
                    "interactions": user_data.get("interactions", 0),
                    "session_duration": user_data.get("session_duration", 0),
                    "preferences": user_data.get("preferences", {})
                },
                "behavior_patterns": await self._analyze_behavior_patterns(user_data),
                "recommendations": await self._generate_recommendations(user_data)
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"خطأ في معالجة بيانات المستخدم: {e}")
            return {"error": str(e)}
    
    async def _analyze_behavior_patterns(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """تحليل أنماط السلوك"""
        patterns = {}
        
        # تحليل نشاط المستخدم
        interactions = user_data.get("interactions", 0)
        if interactions > 0:
            patterns["activity_level"] = (
                "عالي" if interactions > 100 else
                "متوسط" if interactions > 50 else
                "منخفض"
            )
        
        # تحليل أوقات الاستخدام
        current_hour = datetime.now().hour
        if 6 <= current_hour < 12:
            patterns["usage_time"] = "صباحي"
        elif 12 <= current_hour < 18:
            patterns["usage_time"] = "بعد الظهر"
        elif 18 <= current_hour < 24:
            patterns["usage_time"] = "مسائي"
        else:
            patterns["usage_time"] = "ليلي"
        
        return patterns
    
    async def _generate_recommendations(self, user_data: Dict[str, Any]) -> List[str]:
        """توليد توصيات للمستخدم"""
        recommendations = []
        
        interactions = user_data.get("interactions", 0)
        
        if interactions < 10:
            recommendations.append("جرب استكشاف المزيد من الميزات المتاحة")
            recommendations.append("اطلع على دليل المستخدم للاستفادة القصوى")
        
        if interactions > 50:
            recommendations.append("يمكنك الآن استخدام الميزات المتقدمة")
            recommendations.append("جرب أنظمة التحليل والتوقع")
        
        # إضافة توصيات عامة
        recommendations.append("احرص على تحديث تفضيلاتك بانتظام")
        recommendations.append("استخدم ميزة التعلم النشط لتحسين الأداء")
        
        return recommendations

# اختبار المعالج
async def test_dask_processor():
    """اختبار معالج البيانات الضخمة"""
    processor = DaskProcessor()
    
    try:
        await processor.initialize()
        
        # اختبار تحليل البيانات العينة
        result = await processor.analyze_sample_data()
        print(result)
        
    finally:
        await processor.cleanup()

if __name__ == "__main__":
    asyncio.run(test_dask_processor())
