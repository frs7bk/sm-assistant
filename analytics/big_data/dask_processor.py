
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
معالج البيانات الضخمة باستخدام Dask
نظام متقدم للمعالجة المتوازية والموزعة
"""

import dask.dataframe as dd
import dask.array as da
import pandas as pd
import numpy as np
from dask.distributed import Client, as_completed
from dask.diagnostics import ProgressBar
import logging
from typing import Dict, List, Any, Optional
import asyncio
from pathlib import Path

class AdvancedDaskProcessor:
    """معالج البيانات الضخمة المتقدم باستخدام Dask"""
    
    def __init__(self, scheduler_address: Optional[str] = None):
        """تهيئة معالج Dask"""
        self.logger = logging.getLogger(__name__)
        
        # إعداد عميل Dask
        if scheduler_address:
            self.client = Client(scheduler_address)
        else:
            self.client = Client(processes=True, threads_per_worker=2)
        
        self.logger.info(f"تم تهيئة معالج Dask: {self.client.dashboard_link}")
    
    async def process_large_dataset(self, data_path: str, chunk_size: str = "100MB") -> Dict[str, Any]:
        """معالجة مجموعة بيانات ضخمة"""
        try:
            # قراءة البيانات بشكل كسول
            if data_path.endswith('.csv'):
                df = dd.read_csv(data_path, blocksize=chunk_size)
            elif data_path.endswith('.parquet'):
                df = dd.read_parquet(data_path)
            elif data_path.endswith('.json'):
                df = dd.read_json(data_path, blocksize=chunk_size)
            else:
                raise ValueError(f"تنسيق ملف غير مدعوم: {data_path}")
            
            # تحليل أساسي
            with ProgressBar():
                analysis_results = {
                    'total_rows': len(df),
                    'total_columns': len(df.columns),
                    'memory_usage': df.memory_usage(deep=True).sum().compute(),
                    'data_types': dict(df.dtypes),
                    'null_counts': df.isnull().sum().compute().to_dict(),
                    'basic_stats': df.describe().compute().to_dict()
                }
            
            self.logger.info(f"تم تحليل {analysis_results['total_rows']} صف")
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"خطأ في معالجة البيانات: {e}")
            return {}
    
    async def advanced_analytics(self, df: dd.DataFrame) -> Dict[str, Any]:
        """تحليلات متقدمة للبيانات"""
        try:
            results = {}
            
            # تحليل الارتباطات
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                with ProgressBar():
                    correlation_matrix = df[numeric_cols].corr().compute()
                    results['correlations'] = correlation_matrix.to_dict()
            
            # تحليل القيم المفقودة
            missing_data = df.isnull().sum().compute()
            results['missing_data_analysis'] = {
                'total_missing': int(missing_data.sum()),
                'missing_by_column': missing_data.to_dict(),
                'missing_percentage': (missing_data / len(df) * 100).to_dict()
            }
            
            # تحليل التوزيعات
            if len(numeric_cols) > 0:
                distributions = {}
                for col in numeric_cols[:5]:  # أول 5 أعمدة رقمية
                    with ProgressBar():
                        col_data = df[col].compute()
                        distributions[col] = {
                            'mean': float(col_data.mean()),
                            'std': float(col_data.std()),
                            'skewness': float(col_data.skew()),
                            'kurtosis': float(col_data.kurtosis())
                        }
                results['distributions'] = distributions
            
            return results
            
        except Exception as e:
            self.logger.error(f"خطأ في التحليلات المتقدمة: {e}")
            return {}
    
    async def machine_learning_pipeline(self, df: dd.DataFrame, target_column: str) -> Dict[str, Any]:
        """خط إنتاج تعلم آلي للبيانات الضخمة"""
        try:
            from dask_ml.model_selection import train_test_split
            from dask_ml.linear_model import LinearRegression
            from dask_ml.ensemble import RandomForestRegressor
            from dask_ml.preprocessing import StandardScaler
            
            # تحضير البيانات
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if target_column in numeric_cols:
                numeric_cols.remove(target_column)
            
            X = df[numeric_cols]
            y = df[target_column]
            
            # تقسيم البيانات
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # تطبيع البيانات
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # تدريب نماذج متعددة
            models = {
                'linear_regression': LinearRegression(),
                'random_forest': RandomForestRegressor(n_estimators=100, random_state=42)
            }
            
            results = {}
            
            for model_name, model in models.items():
                with ProgressBar():
                    # تدريب النموذج
                    model.fit(X_train_scaled, y_train)
                    
                    # التنبؤ والتقييم
                    train_score = model.score(X_train_scaled, y_train).compute()
                    test_score = model.score(X_test_scaled, y_test).compute()
                    
                    results[model_name] = {
                        'train_score': float(train_score),
                        'test_score': float(test_score)
                    }
            
            return results
            
        except Exception as e:
            self.logger.error(f"خطأ في خط الإنتاج: {e}")
            return {}
    
    async def time_series_analysis(self, df: dd.DataFrame, time_column: str, value_column: str) -> Dict[str, Any]:
        """تحليل السلاسل الزمنية"""
        try:
            # تحويل عمود الوقت
            df[time_column] = dd.to_datetime(df[time_column])
            df = df.set_index(time_column)
            
            # تجميع البيانات حسب فترات زمنية
            daily_data = df[value_column].resample('D').mean().compute()
            weekly_data = df[value_column].resample('W').mean().compute()
            monthly_data = df[value_column].resample('M').mean().compute()
            
            # اتجاهات وأنماط
            results = {
                'trend_analysis': {
                    'daily_trend': float(daily_data.pct_change().mean()),
                    'weekly_trend': float(weekly_data.pct_change().mean()),
                    'monthly_trend': float(monthly_data.pct_change().mean())
                },
                'seasonality': {
                    'daily_volatility': float(daily_data.std()),
                    'weekly_volatility': float(weekly_data.std()),
                    'monthly_volatility': float(monthly_data.std())
                },
                'summary_stats': {
                    'total_data_points': len(daily_data),
                    'date_range': {
                        'start': str(daily_data.index.min()),
                        'end': str(daily_data.index.max())
                    }
                }
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"خطأ في تحليل السلاسل الزمنية: {e}")
            return {}
    
    async def cluster_analysis(self, df: dd.DataFrame, n_clusters: int = 5) -> Dict[str, Any]:
        """تحليل التجميع للبيانات الضخمة"""
        try:
            from dask_ml.cluster import KMeans
            from dask_ml.preprocessing import StandardScaler
            
            # اختيار الأعمدة الرقمية
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            X = df[numeric_cols]
            
            # تطبيع البيانات
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # تطبيق K-Means
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            
            with ProgressBar():
                clusters = kmeans.fit_predict(X_scaled)
                cluster_centers = kmeans.cluster_centers_
            
            # تحليل المجموعات
            unique_clusters, counts = da.unique(clusters, return_counts=True)
            cluster_counts = dict(zip(unique_clusters.compute(), counts.compute()))
            
            results = {
                'n_clusters': n_clusters,
                'cluster_counts': cluster_counts,
                'cluster_centers': cluster_centers.tolist(),
                'total_samples': len(clusters)
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"خطأ في تحليل التجميع: {e}")
            return {}
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """إحصائيات الأداء"""
        try:
            return {
                'dashboard_link': self.client.dashboard_link,
                'workers': len(self.client.scheduler_info()['workers']),
                'total_cores': sum(w['ncores'] for w in self.client.scheduler_info()['workers'].values()),
                'total_memory': sum(w['memory_limit'] for w in self.client.scheduler_info()['workers'].values()),
                'active_tasks': len(self.client.processing()),
                'completed_tasks': self.client.call_stack(),
            }
        except Exception as e:
            self.logger.error(f"خطأ في إحصائيات الأداء: {e}")
            return {}
    
    def __del__(self):
        """تنظيف الموارد"""
        if hasattr(self, 'client'):
            self.client.close()

# مثال للاستخدام
async def main():
    """مثال شامل لاستخدام معالج البيانات الضخمة"""
    processor = AdvancedDaskProcessor()
    
    try:
        # مثال لمعالجة ملف CSV ضخم
        print("🔍 تحليل البيانات الضخمة...")
        
        # إنشاء بيانات تجريبية ضخمة
        import pandas as pd
        large_data = pd.DataFrame({
            'id': range(1000000),
            'value1': np.random.randn(1000000),
            'value2': np.random.randn(1000000) * 100,
            'category': np.random.choice(['A', 'B', 'C'], 1000000),
            'timestamp': pd.date_range('2020-01-01', periods=1000000, freq='1min')
        })
        large_data.to_csv('big_data_sample.csv', index=False)
        
        # تحليل البيانات
        analysis = await processor.process_large_dataset('big_data_sample.csv')
        print(f"📊 نتائج التحليل: {analysis}")
        
        # قراءة البيانات مرة أخرى للتحليلات المتقدمة
        df = dd.read_csv('big_data_sample.csv')
        
        # تحليلات متقدمة
        advanced_results = await processor.advanced_analytics(df)
        print(f"🔬 التحليلات المتقدمة: {advanced_results}")
        
        # تحليل السلاسل الزمنية
        ts_results = await processor.time_series_analysis(df, 'timestamp', 'value1')
        print(f"📈 تحليل السلاسل الزمنية: {ts_results}")
        
        # إحصائيات الأداء
        perf_stats = processor.get_performance_stats()
        print(f"⚡ إحصائيات الأداء: {perf_stats}")
        
    except Exception as e:
        print(f"❌ خطأ: {e}")
    
    finally:
        # تنظيف الملفات التجريبية
        Path('big_data_sample.csv').unlink(missing_ok=True)

if __name__ == "__main__":
    asyncio.run(main())
