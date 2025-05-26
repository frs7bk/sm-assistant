
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 محرك تحسين الأداء والسرعة المتقدم
GPU Acceleration & Advanced Caching Engine
"""

import asyncio
import time
import logging
from typing import Dict, Any, Optional, Callable
from functools import wraps, lru_cache
import hashlib
import pickle
import redis
from pathlib import Path
import torch
import threading
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

class AdvancedCacheManager:
    """مدير الذاكرة التخزينية المتقدم"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.memory_cache = {}
        self.cache_stats = {"hits": 0, "misses": 0}
        
        # Redis cache (اختياري)
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
            self.redis_available = True
        except:
            self.redis_available = False
            self.logger.warning("Redis غير متاح - استخدام الذاكرة المحلية فقط")
    
    def cache_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """إنشاء مفتاح فريد للكاش"""
        key_data = f"{func_name}_{str(args)}_{str(sorted(kwargs.items()))}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def get(self, key: str) -> Optional[Any]:
        """استرجاع من الكاش"""
        # محاولة الذاكرة المحلية أولاً
        if key in self.memory_cache:
            self.cache_stats["hits"] += 1
            return self.memory_cache[key]
        
        # محاولة Redis
        if self.redis_available:
            try:
                data = self.redis_client.get(key)
                if data:
                    result = pickle.loads(data.encode('latin1'))
                    self.memory_cache[key] = result  # نسخ للذاكرة المحلية
                    self.cache_stats["hits"] += 1
                    return result
            except Exception as e:
                self.logger.warning(f"خطأ في Redis: {e}")
        
        self.cache_stats["misses"] += 1
        return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600):
        """حفظ في الكاش"""
        # حفظ في الذاكرة المحلية
        self.memory_cache[key] = value
        
        # حفظ في Redis
        if self.redis_available:
            try:
                serialized = pickle.dumps(value).decode('latin1')
                self.redis_client.setex(key, ttl, serialized)
            except Exception as e:
                self.logger.warning(f"خطأ في حفظ Redis: {e}")

class GPUAccelerator:
    """مسرع GPU للمعالجة المتقدمة"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gpu_available = torch.cuda.is_available()
        self.logger = logging.getLogger(__name__)
        
        if self.gpu_available:
            self.logger.info(f"🚀 GPU متاح: {torch.cuda.get_device_name()}")
        else:
            self.logger.info("💻 استخدام CPU للمعالجة")
    
    def accelerate_tensor_ops(self, data: torch.Tensor) -> torch.Tensor:
        """تسريع العمليات على GPU"""
        if self.gpu_available:
            return data.to(self.device)
        return data
    
    def optimize_memory(self):
        """تحسين استخدام ذاكرة GPU"""
        if self.gpu_available:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

class PerformanceMonitor:
    """مراقب الأداء في الوقت الفعلي"""
    
    def __init__(self):
        self.metrics = {
            "response_times": [],
            "memory_usage": [],
            "cpu_usage": [],
            "gpu_usage": [],
            "cache_hit_rate": 0.0
        }
        self.monitoring = False
        self.logger = logging.getLogger(__name__)
    
    def start_monitoring(self):
        """بدء مراقبة الأداء"""
        self.monitoring = True
        threading.Thread(target=self._monitor_loop, daemon=True).start()
    
    def _monitor_loop(self):
        """حلقة المراقبة"""
        import psutil
        
        while self.monitoring:
            try:
                # مراقبة CPU
                cpu_percent = psutil.cpu_percent(interval=1)
                self.metrics["cpu_usage"].append(cpu_percent)
                
                # مراقبة الذاكرة
                memory = psutil.virtual_memory()
                self.metrics["memory_usage"].append(memory.percent)
                
                # مراقبة GPU (إن وجد)
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100
                    self.metrics["gpu_usage"].append(gpu_memory)
                
                # الحفاظ على آخر 100 قراءة فقط
                for key in self.metrics:
                    if isinstance(self.metrics[key], list) and len(self.metrics[key]) > 100:
                        self.metrics[key] = self.metrics[key][-100:]
                
                time.sleep(5)  # مراقبة كل 5 ثوانٍ
                
            except Exception as e:
                self.logger.error(f"خطأ في مراقبة الأداء: {e}")
                time.sleep(10)

class PerformanceOptimizer:
    """محرك تحسين الأداء الشامل"""
    
    def __init__(self):
        self.cache_manager = AdvancedCacheManager()
        self.gpu_accelerator = GPUAccelerator()
        self.performance_monitor = PerformanceMonitor()
        self.thread_pool = ThreadPoolExecutor(max_workers=mp.cpu_count())
        self.logger = logging.getLogger(__name__)
        
        # بدء المراقبة
        self.performance_monitor.start_monitoring()
    
    def cached(self, ttl: int = 3600):
        """ديكوريتر للكاش المتقدم"""
        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # إنشاء مفتاح الكاش
                cache_key = self.cache_manager.cache_key(func.__name__, args, kwargs)
                
                # محاولة استرجاع من الكاش
                cached_result = await self.cache_manager.get(cache_key)
                if cached_result is not None:
                    return cached_result
                
                # تنفيذ الدالة
                start_time = time.time()
                
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = await asyncio.get_event_loop().run_in_executor(
                        self.thread_pool, func, *args, **kwargs
                    )
                
                # تسجيل وقت الاستجابة
                response_time = time.time() - start_time
                self.performance_monitor.metrics["response_times"].append(response_time)
                
                # حفظ في الكاش
                await self.cache_manager.set(cache_key, result, ttl)
                
                return result
            return wrapper
        return decorator
    
    def gpu_accelerated(self, func: Callable):
        """ديكوريتر لتسريع GPU"""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # نقل البيانات إلى GPU إن أمكن
            processed_args = []
            for arg in args:
                if isinstance(arg, torch.Tensor):
                    processed_args.append(self.gpu_accelerator.accelerate_tensor_ops(arg))
                else:
                    processed_args.append(arg)
            
            # تنفيذ الدالة
            if asyncio.iscoroutinefunction(func):
                result = await func(*processed_args, **kwargs)
            else:
                result = func(*processed_args, **kwargs)
            
            # تنظيف ذاكرة GPU
            self.gpu_accelerator.optimize_memory()
            
            return result
        return wrapper
    
    async def optimize_batch_processing(self, data_list: list, process_func: Callable, batch_size: int = 32):
        """معالجة الدفعات المحسنة"""
        results = []
        
        for i in range(0, len(data_list), batch_size):
            batch = data_list[i:i + batch_size]
            
            # معالجة متوازية للدفعة
            batch_tasks = [process_func(item) for item in batch]
            batch_results = await asyncio.gather(*batch_tasks)
            
            results.extend(batch_results)
        
        return results
    
    def get_performance_report(self) -> Dict[str, Any]:
        """تقرير الأداء المفصل"""
        cache_stats = self.cache_manager.cache_stats
        total_requests = cache_stats["hits"] + cache_stats["misses"]
        hit_rate = (cache_stats["hits"] / max(total_requests, 1)) * 100
        
        metrics = self.performance_monitor.metrics
        
        return {
            "cache_performance": {
                "hit_rate": f"{hit_rate:.1f}%",
                "total_hits": cache_stats["hits"],
                "total_misses": cache_stats["misses"]
            },
            "system_performance": {
                "avg_response_time": f"{sum(metrics['response_times'][-10:]) / max(len(metrics['response_times'][-10:]), 1):.3f}s",
                "current_cpu_usage": f"{metrics['cpu_usage'][-1] if metrics['cpu_usage'] else 0:.1f}%",
                "current_memory_usage": f"{metrics['memory_usage'][-1] if metrics['memory_usage'] else 0:.1f}%",
                "gpu_available": self.gpu_accelerator.gpu_available
            },
            "optimization_recommendations": self._get_optimization_recommendations()
        }
    
    def _get_optimization_recommendations(self) -> list:
        """اقتراحات تحسين الأداء"""
        recommendations = []
        
        cache_stats = self.cache_manager.cache_stats
        total_requests = cache_stats["hits"] + cache_stats["misses"]
        hit_rate = (cache_stats["hits"] / max(total_requests, 1)) * 100
        
        if hit_rate < 50:
            recommendations.append("زيادة مدة الكاش لتحسين معدل الإصابة")
        
        metrics = self.performance_monitor.metrics
        if metrics["cpu_usage"] and max(metrics["cpu_usage"][-10:]) > 80:
            recommendations.append("تحسين خوارزميات المعالجة لتقليل استخدام CPU")
        
        if not self.gpu_accelerator.gpu_available:
            recommendations.append("فكر في استخدام GPU لتسريع المعالجة")
        
        return recommendations

# مثيل عام للاستخدام
performance_optimizer = PerformanceOptimizer()
