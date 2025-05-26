
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
معالج البيانات الضخمة باستخدام Apache Spark
نظام متقدم للمعالجة الموزعة والتحليلات الضخمة
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
import logging
from typing import Dict, List, Any, Optional
import asyncio
import json
from pathlib import Path

class AdvancedSparkProcessor:
    """معالج البيانات الضخمة المتقدم باستخدام Apache Spark"""
    
    def __init__(self, app_name: str = "AdvancedDataProcessor", master: str = "local[*]"):
        """تهيئة جلسة Spark"""
        self.logger = logging.getLogger(__name__)
        
        # إعداد Spark
        self.spark = SparkSession.builder \
            .appName(app_name) \
            .master(master) \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
            .getOrCreate()
        
        # تقليل مستوى السجلات
        self.spark.sparkContext.setLogLevel("WARN")
        
        self.logger.info(f"تم تهيئة Spark: {self.spark.version}")
    
    async def process_large_dataset(self, data_path: str, format_type: str = "csv") -> Dict[str, Any]:
        """معالجة مجموعة بيانات ضخمة"""
        try:
            # قراءة البيانات
            if format_type == "csv":
                df = self.spark.read.option("header", "true").option("inferSchema", "true").csv(data_path)
            elif format_type == "parquet":
                df = self.spark.read.parquet(data_path)
            elif format_type == "json":
                df = self.spark.read.json(data_path)
            else:
                raise ValueError(f"تنسيق غير مدعوم: {format_type}")
            
            # تحليل أساسي
            total_rows = df.count()
            total_columns = len(df.columns)
            
            # معلومات الأعمدة
            column_info = []
            for col_name, col_type in df.dtypes:
                null_count = df.filter(col(col_name).isNull()).count()
                column_info.append({
                    'name': col_name,
                    'type': col_type,
                    'null_count': null_count,
                    'null_percentage': (null_count / total_rows) * 100 if total_rows > 0 else 0
                })
            
            # إحصائيات وصفية للأعمدة الرقمية
            numeric_columns = [col_name for col_name, col_type in df.dtypes 
                             if col_type in ['int', 'bigint', 'float', 'double']]
            
            descriptive_stats = {}
            if numeric_columns:
                stats_df = df.select(numeric_columns).describe()
                stats_rows = stats_df.collect()
                
                for col_name in numeric_columns:
                    descriptive_stats[col_name] = {
                        row['summary']: row[col_name] for row in stats_rows
                    }
            
            analysis_results = {
                'total_rows': total_rows,
                'total_columns': total_columns,
                'column_info': column_info,
                'descriptive_stats': descriptive_stats,
                'data_types_summary': dict(df.dtypes),
                'partitions': df.rdd.getNumPartitions()
            }
            
            self.logger.info(f"تم تحليل {total_rows} صف و {total_columns} عمود")
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"خطأ في معالجة البيانات: {e}")
            return {}
    
    async def advanced_analytics(self, df) -> Dict[str, Any]:
        """تحليلات متقدمة باستخدام Spark SQL"""
        try:
            # تسجيل البيانات كجدول مؤقت
            df.createOrReplaceTempView("data_table")
            
            results = {}
            
            # تحليل التوزيع للأعمدة الفئوية
            categorical_columns = [col_name for col_name, col_type in df.dtypes 
                                 if col_type == 'string']
            
            categorical_analysis = {}
            for col_name in categorical_columns[:5]:  # أول 5 أعمدة فئوية
                value_counts = df.groupBy(col_name).count().orderBy(desc("count")).limit(10)
                categorical_analysis[col_name] = [
                    {'value': row[col_name], 'count': row['count']} 
                    for row in value_counts.collect()
                ]
            
            results['categorical_analysis'] = categorical_analysis
            
            # تحليل الارتباطات
            numeric_columns = [col_name for col_name, col_type in df.dtypes 
                             if col_type in ['int', 'bigint', 'float', 'double']]
            
            if len(numeric_columns) >= 2:
                correlations = {}
                for i in range(min(5, len(numeric_columns))):
                    for j in range(i+1, min(5, len(numeric_columns))):
                        col1, col2 = numeric_columns[i], numeric_columns[j]
                        corr_value = df.stat.corr(col1, col2)
                        correlations[f"{col1}_vs_{col2}"] = corr_value
                
                results['correlations'] = correlations
            
            # تحليل القيم الشاذة (باستخدام IQR)
            outlier_analysis = {}
            for col_name in numeric_columns[:3]:  # أول 3 أعمدة رقمية
                quantiles = df.stat.approxQuantile(col_name, [0.25, 0.75], 0.05)
                if len(quantiles) == 2:
                    q1, q3 = quantiles
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    
                    outlier_count = df.filter(
                        (col(col_name) < lower_bound) | (col(col_name) > upper_bound)
                    ).count()
                    
                    outlier_analysis[col_name] = {
                        'q1': q1,
                        'q3': q3,
                        'iqr': iqr,
                        'outlier_count': outlier_count,
                        'outlier_percentage': (outlier_count / df.count()) * 100
                    }
            
            results['outlier_analysis'] = outlier_analysis
            
            return results
            
        except Exception as e:
            self.logger.error(f"خطأ في التحليلات المتقدمة: {e}")
            return {}
    
    async def machine_learning_pipeline(self, df, target_column: str) -> Dict[str, Any]:
        """خط إنتاج تعلم آلي متقدم"""
        try:
            from pyspark.ml.feature import VectorAssembler
            from pyspark.ml.regression import LinearRegression, RandomForestRegressor
            from pyspark.ml.evaluation import RegressionEvaluator
            
            # تحضير الميزات
            numeric_columns = [col_name for col_name, col_type in df.dtypes 
                             if col_type in ['int', 'bigint', 'float', 'double'] 
                             and col_name != target_column]
            
            if not numeric_columns:
                return {"error": "لا توجد أعمدة رقمية كافية للتدريب"}
            
            # تجميع الميزات
            assembler = VectorAssembler(inputCols=numeric_columns, outputCol="features")
            
            # تقسيم البيانات
            train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)
            
            # النماذج للاختبار
            models = {
                'linear_regression': LinearRegression(featuresCol="features", labelCol=target_column),
                'random_forest': RandomForestRegressor(featuresCol="features", labelCol=target_column, numTrees=10)
            }
            
            results = {}
            evaluator = RegressionEvaluator(labelCol=target_column, metricName="rmse")
            
            for model_name, model in models.items():
                try:
                    # إنشاء pipeline
                    pipeline = Pipeline(stages=[assembler, model])
                    
                    # تدريب النموذج
                    fitted_pipeline = pipeline.fit(train_data)
                    
                    # التنبؤ
                    train_predictions = fitted_pipeline.transform(train_data)
                    test_predictions = fitted_pipeline.transform(test_data)
                    
                    # التقييم
                    train_rmse = evaluator.evaluate(train_predictions)
                    test_rmse = evaluator.evaluate(test_predictions)
                    
                    # R² Score
                    evaluator_r2 = RegressionEvaluator(labelCol=target_column, metricName="r2")
                    train_r2 = evaluator_r2.evaluate(train_predictions)
                    test_r2 = evaluator_r2.evaluate(test_predictions)
                    
                    results[model_name] = {
                        'train_rmse': train_rmse,
                        'test_rmse': test_rmse,
                        'train_r2': train_r2,
                        'test_r2': test_r2,
                        'features_used': numeric_columns
                    }
                    
                except Exception as model_error:
                    results[model_name] = {"error": str(model_error)}
            
            return results
            
        except Exception as e:
            self.logger.error(f"خطأ في خط الإنتاج: {e}")
            return {"error": str(e)}
    
    async def time_series_analysis(self, df, time_column: str, value_column: str) -> Dict[str, Any]:
        """تحليل السلاسل الزمنية المتقدم"""
        try:
            # تحويل عمود الوقت
            df_time = df.withColumn(time_column, to_timestamp(col(time_column)))
            
            # تجميع البيانات حسب فترات زمنية
            daily_data = df_time.groupBy(date_format(col(time_column), "yyyy-MM-dd").alias("date")) \
                               .agg(avg(value_column).alias("avg_value"),
                                   count("*").alias("count"),
                                   stddev(value_column).alias("std_value"))
            
            weekly_data = df_time.groupBy(date_format(col(time_column), "yyyy-ww").alias("week")) \
                                .agg(avg(value_column).alias("avg_value"),
                                    count("*").alias("count"))
            
            # حساب الاتجاهات
            daily_stats = daily_data.agg(
                avg("avg_value").alias("overall_avg"),
                stddev("avg_value").alias("overall_std"),
                min("avg_value").alias("min_value"),
                max("avg_value").alias("max_value")
            ).collect()[0]
            
            # حساب معدل النمو
            daily_ordered = daily_data.orderBy("date")
            daily_with_lag = daily_ordered.withColumn(
                "prev_value", 
                lag("avg_value").over(Window.orderBy("date"))
            ).withColumn(
                "growth_rate",
                (col("avg_value") - col("prev_value")) / col("prev_value") * 100
            )
            
            growth_stats = daily_with_lag.agg(
                avg("growth_rate").alias("avg_growth_rate"),
                stddev("growth_rate").alias("growth_volatility")
            ).collect()[0]
            
            results = {
                'time_period': {
                    'total_days': daily_data.count(),
                    'total_weeks': weekly_data.count()
                },
                'value_statistics': {
                    'overall_average': float(daily_stats['overall_avg']) if daily_stats['overall_avg'] else 0,
                    'overall_std': float(daily_stats['overall_std']) if daily_stats['overall_std'] else 0,
                    'min_value': float(daily_stats['min_value']) if daily_stats['min_value'] else 0,
                    'max_value': float(daily_stats['max_value']) if daily_stats['max_value'] else 0
                },
                'trend_analysis': {
                    'avg_growth_rate': float(growth_stats['avg_growth_rate']) if growth_stats['avg_growth_rate'] else 0,
                    'growth_volatility': float(growth_stats['growth_volatility']) if growth_stats['growth_volatility'] else 0
                }
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"خطأ في تحليل السلاسل الزمنية: {e}")
            return {}
    
    async def advanced_clustering(self, df, n_clusters: int = 5) -> Dict[str, Any]:
        """تحليل التجميع المتقدم"""
        try:
            # اختيار الأعمدة الرقمية
            numeric_columns = [col_name for col_name, col_type in df.dtypes 
                             if col_type in ['int', 'bigint', 'float', 'double']]
            
            if len(numeric_columns) < 2:
                return {"error": "يجب وجود عمودين رقميين على الأقل للتجميع"}
            
            # تحضير البيانات
            assembler = VectorAssembler(inputCols=numeric_columns, outputCol="features")
            df_features = assembler.transform(df)
            
            # تطبيع البيانات
            scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
            scaler_model = scaler.fit(df_features)
            df_scaled = scaler_model.transform(df_features)
            
            # تطبيق K-Means
            kmeans = KMeans(k=n_clusters, featuresCol="scaled_features", seed=42)
            model = kmeans.fit(df_scaled)
            predictions = model.transform(df_scaled)
            
            # تحليل النتائج
            cluster_counts = predictions.groupBy("prediction").count().collect()
            cluster_distribution = {
                f"cluster_{row['prediction']}": row['count'] 
                for row in cluster_counts
            }
            
            # حساب مركز كل مجموعة
            cluster_centers = model.clusterCenters()
            
            # تحليل الميزات لكل مجموعة
            cluster_analysis = {}
            for i in range(n_clusters):
                cluster_data = predictions.filter(col("prediction") == i)
                
                if cluster_data.count() > 0:
                    cluster_stats = cluster_data.select(numeric_columns).describe().collect()
                    cluster_analysis[f"cluster_{i}"] = {
                        'size': cluster_data.count(),
                        'statistics': {
                            col_name: {
                                row['summary']: row[col_name] for row in cluster_stats
                            } for col_name in numeric_columns
                        }
                    }
            
            results = {
                'n_clusters': n_clusters,
                'cluster_distribution': cluster_distribution,
                'cluster_centers': [center.tolist() for center in cluster_centers],
                'cluster_analysis': cluster_analysis,
                'features_used': numeric_columns,
                'total_samples': df.count()
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"خطأ في تحليل التجميع: {e}")
            return {}
    
    def get_spark_info(self) -> Dict[str, Any]:
        """معلومات جلسة Spark"""
        try:
            return {
                'app_name': self.spark.sparkContext.appName,
                'spark_version': self.spark.version,
                'master': self.spark.sparkContext.master,
                'default_parallelism': self.spark.sparkContext.defaultParallelism,
                'total_cores': self.spark.sparkContext.getConf().get('spark.cores.max', 'غير محدد'),
                'driver_memory': self.spark.sparkContext.getConf().get('spark.driver.memory', 'غير محدد'),
                'executor_memory': self.spark.sparkContext.getConf().get('spark.executor.memory', 'غير محدد')
            }
        except Exception as e:
            self.logger.error(f"خطأ في معلومات Spark: {e}")
            return {}
    
    def __del__(self):
        """تنظيف الموارد"""
        if hasattr(self, 'spark'):
            self.spark.stop()

# مثال للاستخدام
async def main():
    """مثال شامل لاستخدام معالج Spark"""
    processor = AdvancedSparkProcessor()
    
    try:
        print("🚀 بدء معالجة البيانات الضخمة بـ Spark...")
        
        # إنشاء بيانات تجريبية
        import pandas as pd
        import numpy as np
        
        large_data = pd.DataFrame({
            'id': range(100000),
            'value1': np.random.randn(100000),
            'value2': np.random.randn(100000) * 100,
            'category': np.random.choice(['A', 'B', 'C', 'D'], 100000),
            'timestamp': pd.date_range('2020-01-01', periods=100000, freq='1H')
        })
        large_data.to_csv('spark_data_sample.csv', index=False)
        
        # قراءة البيانات
        df = processor.spark.read.option("header", "true").option("inferSchema", "true").csv('spark_data_sample.csv')
        
        # تحليل أساسي
        basic_analysis = await processor.process_large_dataset('spark_data_sample.csv')
        print(f"📊 التحليل الأساسي: {json.dumps(basic_analysis, indent=2, ensure_ascii=False)}")
        
        # تحليلات متقدمة
        advanced_analysis = await processor.advanced_analytics(df)
        print(f"🔬 التحليلات المتقدمة: {json.dumps(advanced_analysis, indent=2, ensure_ascii=False)}")
        
        # تحليل التجميع
        clustering_results = await processor.advanced_clustering(df, n_clusters=3)
        print(f"🎯 تحليل التجميع: {json.dumps(clustering_results, indent=2, ensure_ascii=False)}")
        
        # معلومات Spark
        spark_info = processor.get_spark_info()
        print(f"⚡ معلومات Spark: {json.dumps(spark_info, indent=2, ensure_ascii=False)}")
        
    except Exception as e:
        print(f"❌ خطأ: {e}")
    
    finally:
        # تنظيف الملفات
        Path('spark_data_sample.csv').unlink(missing_ok=True)

if __name__ == "__main__":
    asyncio.run(main())
