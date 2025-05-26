
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
محرك التعلم الفوقي المتقدم
Meta-Learning Engine for Advanced AI
"""

import asyncio
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json
from datetime import datetime
import pickle
from dataclasses import dataclass
from sklearn.metrics import accuracy_score, f1_score
import copy

@dataclass
class LearningTask:
    """مهمة تعلم"""
    task_id: str
    task_type: str
    data: Any
    labels: Any
    metadata: Dict[str, Any]
    difficulty: float = 0.5
    priority: float = 0.5

@dataclass
class LearningEpisode:
    """حلقة تعلم"""
    episode_id: str
    tasks: List[LearningTask]
    performance: Dict[str, float]
    learned_patterns: List[str]
    timestamp: datetime

class MetaLearningNetwork(nn.Module):
    """شبكة التعلم الفوقي"""
    
    def __init__(self, input_dim: int = 512, hidden_dim: int = 256, output_dim: int = 128):
        super().__init__()
        
        # شبكة الاستخراج السريع للميزات
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # شبكة التكيف السريع
        self.adaptation_network = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # آلية الانتباه للمهام
        self.task_attention = nn.MultiheadAttention(
            output_dim, num_heads=8, dropout=0.1, batch_first=True
        )
        
        # ذاكرة النماذج المتعلمة
        self.model_memory = nn.Parameter(
            torch.randn(100, output_dim), requires_grad=True
        )
        
        # شبكة التنبؤ بالأداء
        self.performance_predictor = nn.Sequential(
            nn.Linear(output_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, support_set: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """المرور الأمامي"""
        # استخراج الميزات
        features = self.feature_extractor(x)
        
        # التكيف السريع مع المهمة الجديدة
        if support_set is not None:
            support_features = self.feature_extractor(support_set)
            adapted_features, _ = self.task_attention(
                features.unsqueeze(1), 
                support_features.unsqueeze(1), 
                support_features.unsqueeze(1)
            )
            features = adapted_features.squeeze(1)
        
        # تطبيق شبكة التكيف
        adapted_output = self.adaptation_network(features)
        
        # التنبؤ بالأداء
        memory_similarity = torch.matmul(features, self.model_memory.T)
        best_memory = self.model_memory[torch.argmax(memory_similarity, dim=1)]
        combined_features = torch.cat([features, best_memory], dim=1)
        performance_pred = self.performance_predictor(combined_features)
        
        return adapted_output, performance_pred

class MetaLearningEngine:
    """محرك التعلم الفوقي المتقدم"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # الشبكة الرئيسية
        self.meta_network = MetaLearningNetwork()
        self.meta_network.to(self.device)
        
        # المحسن
        self.optimizer = optim.Adam(self.meta_network.parameters(), lr=1e-3)
        
        # تاريخ التعلم
        self.learning_history: List[LearningEpisode] = []
        self.task_registry: Dict[str, LearningTask] = {}
        
        # إحصائيات الأداء
        self.performance_stats = {
            "total_episodes": 0,
            "successful_adaptations": 0,
            "average_adaptation_time": 0.0,
            "best_performance": 0.0,
            "learned_concepts": set()
        }
        
        # قاعدة المعرفة المكتسبة
        self.knowledge_base = {
            "patterns": {},
            "strategies": {},
            "concepts": {},
            "relationships": {}
        }
        
        # إعدادات التعلم
        self.config = {
            "adaptation_steps": 5,
            "meta_batch_size": 16,
            "inner_lr": 0.01,
            "outer_lr": 1e-3,
            "memory_size": 1000,
            "difficulty_threshold": 0.8
        }

    async def learn_from_interaction(
        self, 
        interaction_data: Dict[str, Any],
        user_feedback: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """التعلم من التفاعل"""
        
        try:
            # تحليل البيانات
            task = await self._create_learning_task(interaction_data)
            
            # التكيف السريع
            adaptation_result = await self._fast_adaptation(task)
            
            # التحديث الفوقي
            meta_update = await self._meta_update(task, adaptation_result)
            
            # حفظ التجربة
            episode = LearningEpisode(
                episode_id=f"episode_{len(self.learning_history)}",
                tasks=[task],
                performance=adaptation_result["performance"],
                learned_patterns=adaptation_result["patterns"],
                timestamp=datetime.now()
            )
            
            self.learning_history.append(episode)
            self.performance_stats["total_episodes"] += 1
            
            # تحديث قاعدة المعرفة
            await self._update_knowledge_base(task, adaptation_result)
            
            return {
                "adaptation_success": adaptation_result["success"],
                "performance_improvement": adaptation_result["improvement"],
                "new_patterns": adaptation_result["patterns"],
                "confidence": adaptation_result["confidence"],
                "metadata": {
                    "episode_id": episode.episode_id,
                    "processing_time": adaptation_result["time"],
                    "difficulty": task.difficulty
                }
            }
            
        except Exception as e:
            self.logger.error(f"خطأ في التعلم من التفاعل: {e}")
            return {"error": str(e), "success": False}

    async def _create_learning_task(self, interaction_data: Dict[str, Any]) -> LearningTask:
        """إنشاء مهمة تعلم من البيانات"""
        
        # تحليل نوع المهمة
        task_type = self._classify_task_type(interaction_data)
        
        # تقدير الصعوبة
        difficulty = self._estimate_difficulty(interaction_data)
        
        # حساب الأولوية
        priority = self._calculate_priority(interaction_data, difficulty)
        
        task = LearningTask(
            task_id=f"task_{datetime.now().timestamp()}",
            task_type=task_type,
            data=interaction_data.get("input"),
            labels=interaction_data.get("expected_output"),
            metadata={
                "context": interaction_data.get("context", {}),
                "user_profile": interaction_data.get("user_profile", {}),
                "environment": interaction_data.get("environment", {})
            },
            difficulty=difficulty,
            priority=priority
        )
        
        self.task_registry[task.task_id] = task
        return task

    def _classify_task_type(self, data: Dict[str, Any]) -> str:
        """تصنيف نوع المهمة"""
        
        if "image" in data or "vision" in data:
            return "vision"
        elif "audio" in data or "speech" in data:
            return "audio"
        elif "text" in data or "nlp" in data:
            return "nlp"
        elif "decision" in data or "planning" in data:
            return "reasoning"
        elif "prediction" in data or "forecast" in data:
            return "prediction"
        else:
            return "general"

    def _estimate_difficulty(self, data: Dict[str, Any]) -> float:
        """تقدير صعوبة المهمة"""
        
        difficulty_factors = []
        
        # عدد المتغيرات
        if isinstance(data.get("input"), (list, dict)):
            complexity = len(str(data["input"]))
            difficulty_factors.append(min(complexity / 1000, 1.0))
        
        # وجود سياق معقد
        if data.get("context") and len(str(data["context"])) > 500:
            difficulty_factors.append(0.7)
        
        # تعدد الأهداف
        if isinstance(data.get("expected_output"), list):
            difficulty_factors.append(0.6)
        
        return np.mean(difficulty_factors) if difficulty_factors else 0.5

    def _calculate_priority(self, data: Dict[str, Any], difficulty: float) -> float:
        """حساب أولوية المهمة"""
        
        priority_score = 0.5
        
        # أولوية المستخدم
        if data.get("user_priority"):
            priority_score += 0.3
        
        # الإلحاح الزمني
        if data.get("urgent"):
            priority_score += 0.2
        
        # التعقيد المناسب للتعلم
        if 0.3 <= difficulty <= 0.8:
            priority_score += 0.2
        
        return min(priority_score, 1.0)

    async def _fast_adaptation(self, task: LearningTask) -> Dict[str, Any]:
        """التكيف السريع مع المهمة"""
        
        start_time = datetime.now()
        
        try:
            # تحضير البيانات
            input_tensor = self._prepare_tensor(task.data)
            
            if input_tensor is None:
                return {
                    "success": False,
                    "error": "فشل في تحضير البيانات",
                    "time": 0,
                    "performance": {},
                    "patterns": [],
                    "confidence": 0.0,
                    "improvement": 0.0
                }
            
            # نسخ النموذج للتكيف
            adapted_model = copy.deepcopy(self.meta_network)
            adapted_optimizer = optim.SGD(adapted_model.parameters(), lr=self.config["inner_lr"])
            
            # خطوات التكيف
            initial_loss = float('inf')
            final_loss = float('inf')
            
            for step in range(self.config["adaptation_steps"]):
                output, performance_pred = adapted_model(input_tensor)
                
                # حساب الخسارة
                if task.labels is not None:
                    target_tensor = self._prepare_tensor(task.labels)
                    if target_tensor is not None:
                        loss = nn.MSELoss()(output, target_tensor)
                        
                        if step == 0:
                            initial_loss = loss.item()
                        
                        # التحسين
                        adapted_optimizer.zero_grad()
                        loss.backward()
                        adapted_optimizer.step()
                        
                        final_loss = loss.item()
            
            # تقييم الأداء
            performance = await self._evaluate_adaptation(adapted_model, task)
            
            # استخراج الأنماط المتعلمة
            patterns = await self._extract_learned_patterns(adapted_model, task)
            
            # حساب التحسن
            improvement = max(0, (initial_loss - final_loss) / max(initial_loss, 1e-6))
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "success": True,
                "performance": performance,
                "patterns": patterns,
                "confidence": performance.get("confidence", 0.5),
                "improvement": improvement,
                "time": processing_time,
                "adapted_weights": adapted_model.state_dict()
            }
            
        except Exception as e:
            self.logger.error(f"خطأ في التكيف السريع: {e}")
            return {
                "success": False,
                "error": str(e),
                "time": (datetime.now() - start_time).total_seconds(),
                "performance": {},
                "patterns": [],
                "confidence": 0.0,
                "improvement": 0.0
            }

    def _prepare_tensor(self, data: Any) -> Optional[torch.Tensor]:
        """تحضير البيانات كـ tensor"""
        
        try:
            if isinstance(data, str):
                # تحويل النص إلى تمثيل رقمي بسيط
                encoded = [ord(c) for c in data[:100]]  # أول 100 حرف
                padded = encoded + [0] * (512 - len(encoded))
                return torch.FloatTensor(padded[:512]).unsqueeze(0).to(self.device)
            
            elif isinstance(data, (list, tuple)):
                if len(data) > 0 and isinstance(data[0], (int, float)):
                    # بيانات رقمية
                    padded = list(data) + [0] * (512 - len(data))
                    return torch.FloatTensor(padded[:512]).unsqueeze(0).to(self.device)
            
            elif isinstance(data, dict):
                # تحويل القاموس إلى تمثيل رقمي
                values = [hash(str(v)) % 1000 for v in data.values()]
                padded = values + [0] * (512 - len(values))
                return torch.FloatTensor(padded[:512]).unsqueeze(0).to(self.device)
            
            elif isinstance(data, (int, float)):
                # قيمة مفردة
                repeated = [float(data)] * 512
                return torch.FloatTensor(repeated).unsqueeze(0).to(self.device)
            
            return None
            
        except Exception as e:
            self.logger.warning(f"فشل في تحضير tensor: {e}")
            return None

    async def _evaluate_adaptation(self, model: nn.Module, task: LearningTask) -> Dict[str, float]:
        """تقييم جودة التكيف"""
        
        try:
            model.eval()
            
            with torch.no_grad():
                input_tensor = self._prepare_tensor(task.data)
                if input_tensor is None:
                    return {"confidence": 0.0, "accuracy": 0.0}
                
                output, performance_pred = model(input_tensor)
                
                # تقييم الثقة
                confidence = float(performance_pred.mean())
                
                # تقييم دقة التنبؤ
                if task.labels is not None:
                    target_tensor = self._prepare_tensor(task.labels)
                    if target_tensor is not None:
                        mse = nn.MSELoss()(output, target_tensor)
                        accuracy = max(0, 1 - mse.item())
                    else:
                        accuracy = confidence
                else:
                    accuracy = confidence
                
                return {
                    "confidence": confidence,
                    "accuracy": accuracy,
                    "consistency": self._measure_consistency(output),
                    "complexity": self._measure_complexity(output)
                }
                
        except Exception as e:
            self.logger.warning(f"خطأ في تقييم التكيف: {e}")
            return {"confidence": 0.0, "accuracy": 0.0}

    def _measure_consistency(self, output: torch.Tensor) -> float:
        """قياس اتساق الإخراج"""
        
        try:
            variance = torch.var(output).item()
            return max(0, 1 - variance)
        except:
            return 0.5

    def _measure_complexity(self, output: torch.Tensor) -> float:
        """قياس تعقيد الإخراج"""
        
        try:
            # قياس التنوع في الإخراج
            unique_ratio = len(torch.unique(output)) / output.numel()
            return min(unique_ratio, 1.0)
        except:
            return 0.5

    async def _extract_learned_patterns(self, model: nn.Module, task: LearningTask) -> List[str]:
        """استخراج الأنماط المتعلمة"""
        
        patterns = []
        
        try:
            # تحليل الأوزان المتعلمة
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = torch.norm(param.grad).item()
                    if grad_norm > 0.1:  # تغيير كبير
                        patterns.append(f"weight_change_{name}")
            
            # تحليل نوع المهمة
            patterns.append(f"task_type_{task.task_type}")
            
            # تحليل مستوى الصعوبة
            if task.difficulty > 0.7:
                patterns.append("high_difficulty_adaptation")
            elif task.difficulty < 0.3:
                patterns.append("low_difficulty_adaptation")
            
            return patterns
            
        except Exception as e:
            self.logger.warning(f"خطأ في استخراج الأنماط: {e}")
            return []

    async def _meta_update(self, task: LearningTask, adaptation_result: Dict[str, Any]) -> Dict[str, Any]:
        """التحديث الفوقي للنموذج"""
        
        try:
            if not adaptation_result["success"]:
                return {"updated": False, "reason": "فشل التكيف"}
            
            # حساب التدرج الفوقي
            meta_loss = self._compute_meta_loss(task, adaptation_result)
            
            # تحديث النموذج الرئيسي
            self.optimizer.zero_grad()
            meta_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.meta_network.parameters(), 1.0)
            self.optimizer.step()
            
            return {
                "updated": True,
                "meta_loss": meta_loss.item(),
                "improvement": adaptation_result["improvement"]
            }
            
        except Exception as e:
            self.logger.error(f"خطأ في التحديث الفوقي: {e}")
            return {"updated": False, "reason": str(e)}

    def _compute_meta_loss(self, task: LearningTask, adaptation_result: Dict[str, Any]) -> torch.Tensor:
        """حساب خسارة التعلم الفوقي"""
        
        try:
            # خسارة الأداء
            performance_loss = 1.0 - adaptation_result["performance"].get("accuracy", 0.0)
            
            # خسارة الثقة
            confidence_loss = 1.0 - adaptation_result["confidence"]
            
            # خسارة التحسن
            improvement_loss = 1.0 - adaptation_result["improvement"]
            
            # الخسارة المدمجة
            total_loss = (performance_loss + confidence_loss + improvement_loss) / 3.0
            
            return torch.tensor(total_loss, requires_grad=True)
            
        except Exception as e:
            self.logger.warning(f"خطأ في حساب الخسارة الفوقية: {e}")
            return torch.tensor(1.0, requires_grad=True)

    async def _update_knowledge_base(self, task: LearningTask, result: Dict[str, Any]):
        """تحديث قاعدة المعرفة"""
        
        try:
            # إضافة الأنماط الجديدة
            for pattern in result.get("patterns", []):
                if pattern not in self.knowledge_base["patterns"]:
                    self.knowledge_base["patterns"][pattern] = {
                        "frequency": 1,
                        "success_rate": result["performance"].get("accuracy", 0.0),
                        "contexts": [task.task_type],
                        "first_seen": datetime.now().isoformat()
                    }
                else:
                    self.knowledge_base["patterns"][pattern]["frequency"] += 1
                    self.knowledge_base["patterns"][pattern]["contexts"].append(task.task_type)
            
            # إضافة الاستراتيجيات الناجحة
            if result["success"] and result["confidence"] > 0.7:
                strategy_key = f"{task.task_type}_strategy"
                self.knowledge_base["strategies"][strategy_key] = {
                    "adaptation_steps": self.config["adaptation_steps"],
                    "learning_rate": self.config["inner_lr"],
                    "success_rate": result["performance"].get("accuracy", 0.0),
                    "patterns": result.get("patterns", [])
                }
            
            # تحديث المفاهيم
            concept_key = f"concept_{task.task_type}_{task.difficulty:.1f}"
            self.knowledge_base["concepts"][concept_key] = {
                "examples": [task.task_id],
                "difficulty": task.difficulty,
                "success_rate": result["performance"].get("accuracy", 0.0)
            }
            
            self.performance_stats["learned_concepts"].add(concept_key)
            
        except Exception as e:
            self.logger.error(f"خطأ في تحديث قاعدة المعرفة: {e}")

    async def predict_adaptation_success(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """التنبؤ بنجاح التكيف"""
        
        try:
            # إنشاء مهمة مؤقتة
            temp_task = await self._create_learning_task(task_data)
            
            # البحث في قاعدة المعرفة
            similar_patterns = self._find_similar_patterns(temp_task)
            
            # تقدير احتمال النجاح
            success_probability = self._estimate_success_probability(temp_task, similar_patterns)
            
            # توقع وقت التكيف
            estimated_time = self._estimate_adaptation_time(temp_task)
            
            # اقتراح الاستراتيجية المثلى
            optimal_strategy = self._suggest_optimal_strategy(temp_task, similar_patterns)
            
            return {
                "success_probability": success_probability,
                "estimated_time": estimated_time,
                "confidence": min(success_probability * 1.2, 1.0),
                "optimal_strategy": optimal_strategy,
                "similar_cases": len(similar_patterns),
                "difficulty_assessment": temp_task.difficulty,
                "recommendations": self._generate_recommendations(temp_task)
            }
            
        except Exception as e:
            self.logger.error(f"خطأ في التنبؤ بنجاح التكيف: {e}")
            return {"error": str(e), "success_probability": 0.0}

    def _find_similar_patterns(self, task: LearningTask) -> List[str]:
        """البحث عن أنماط مشابهة"""
        
        similar_patterns = []
        
        for pattern, info in self.knowledge_base["patterns"].items():
            if task.task_type in info["contexts"]:
                similar_patterns.append(pattern)
        
        return similar_patterns

    def _estimate_success_probability(self, task: LearningTask, similar_patterns: List[str]) -> float:
        """تقدير احتمال النجاح"""
        
        if not similar_patterns:
            return 0.5  # لا توجد معلومات سابقة
        
        # حساب متوسط معدل النجاح للأنماط المشابهة
        success_rates = []
        for pattern in similar_patterns:
            pattern_info = self.knowledge_base["patterns"].get(pattern, {})
            success_rates.append(pattern_info.get("success_rate", 0.5))
        
        avg_success = np.mean(success_rates)
        
        # تعديل بناءً على الصعوبة
        difficulty_factor = 1.0 - (task.difficulty * 0.3)
        
        return min(avg_success * difficulty_factor, 1.0)

    def _estimate_adaptation_time(self, task: LearningTask) -> float:
        """تقدير وقت التكيف"""
        
        base_time = self.config["adaptation_steps"] * 0.1  # ثانية لكل خطوة
        
        # تعديل بناءً على الصعوبة
        difficulty_multiplier = 1.0 + task.difficulty
        
        # تعديل بناءً على نوع المهمة
        type_multipliers = {
            "vision": 1.5,
            "audio": 1.3,
            "nlp": 1.2,
            "reasoning": 2.0,
            "prediction": 1.1,
            "general": 1.0
        }
        
        type_multiplier = type_multipliers.get(task.task_type, 1.0)
        
        return base_time * difficulty_multiplier * type_multiplier

    def _suggest_optimal_strategy(self, task: LearningTask, similar_patterns: List[str]) -> Dict[str, Any]:
        """اقتراح الاستراتيجية المثلى"""
        
        strategy = {
            "adaptation_steps": self.config["adaptation_steps"],
            "learning_rate": self.config["inner_lr"],
            "focus_areas": []
        }
        
        # تعديل بناءً على الصعوبة
        if task.difficulty > 0.7:
            strategy["adaptation_steps"] *= 2
            strategy["learning_rate"] *= 0.5
            strategy["focus_areas"].append("gradual_learning")
        
        # تعديل بناءً على نوع المهمة
        if task.task_type == "reasoning":
            strategy["focus_areas"].append("attention_mechanisms")
        elif task.task_type == "vision":
            strategy["focus_areas"].append("feature_extraction")
        
        return strategy

    def _generate_recommendations(self, task: LearningTask) -> List[str]:
        """توليد التوصيات"""
        
        recommendations = []
        
        if task.difficulty > 0.8:
            recommendations.append("تقسيم المهمة إلى أجزاء أصغر")
            recommendations.append("زيادة عدد خطوات التكيف")
        
        if task.priority > 0.8:
            recommendations.append("إعطاء أولوية عالية للتعلم")
        
        if len(self.knowledge_base["patterns"]) < 10:
            recommendations.append("جمع المزيد من البيانات التدريبية")
        
        return recommendations

    async def get_learning_report(self) -> Dict[str, Any]:
        """الحصول على تقرير التعلم"""
        
        try:
            # إحصائيات أساسية
            success_rate = (
                self.performance_stats["successful_adaptations"] / 
                max(self.performance_stats["total_episodes"], 1)
            ) * 100
            
            # تحليل الأنماط
            pattern_analysis = {}
            for pattern, info in self.knowledge_base["patterns"].items():
                pattern_analysis[pattern] = {
                    "frequency": info["frequency"],
                    "success_rate": info["success_rate"],
                    "contexts": len(set(info["contexts"]))
                }
            
            # أفضل الاستراتيجيات
            best_strategies = sorted(
                self.knowledge_base["strategies"].items(),
                key=lambda x: x[1]["success_rate"],
                reverse=True
            )[:5]
            
            return {
                "performance_summary": {
                    "total_episodes": self.performance_stats["total_episodes"],
                    "success_rate": f"{success_rate:.1f}%",
                    "learned_concepts": len(self.performance_stats["learned_concepts"]),
                    "average_adaptation_time": self.performance_stats["average_adaptation_time"],
                    "best_performance": self.performance_stats["best_performance"]
                },
                "knowledge_base_summary": {
                    "total_patterns": len(self.knowledge_base["patterns"]),
                    "total_strategies": len(self.knowledge_base["strategies"]),
                    "total_concepts": len(self.knowledge_base["concepts"])
                },
                "pattern_analysis": pattern_analysis,
                "best_strategies": dict(best_strategies),
                "learning_trends": self._analyze_learning_trends(),
                "recommendations": self._get_system_recommendations()
            }
            
        except Exception as e:
            self.logger.error(f"خطأ في إنشاء تقرير التعلم: {e}")
            return {"error": str(e)}

    def _analyze_learning_trends(self) -> Dict[str, Any]:
        """تحليل اتجاهات التعلم"""
        
        if len(self.learning_history) < 2:
            return {"trend": "insufficient_data"}
        
        # تحليل تطور الأداء
        recent_performance = [
            episode.performance.get("confidence", 0.0) 
            for episode in self.learning_history[-10:]
        ]
        
        if len(recent_performance) > 1:
            trend_slope = np.polyfit(range(len(recent_performance)), recent_performance, 1)[0]
            
            if trend_slope > 0.01:
                trend = "improving"
            elif trend_slope < -0.01:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"
        
        return {
            "trend": trend,
            "recent_average": np.mean(recent_performance) if recent_performance else 0.0,
            "improvement_rate": trend_slope if 'trend_slope' in locals() else 0.0
        }

    def _get_system_recommendations(self) -> List[str]:
        """الحصول على توصيات النظام"""
        
        recommendations = []
        
        # توصيات بناءً على الأداء
        if self.performance_stats["total_episodes"] < 50:
            recommendations.append("جمع المزيد من البيانات التدريبية")
        
        success_rate = (
            self.performance_stats["successful_adaptations"] / 
            max(self.performance_stats["total_episodes"], 1)
        )
        
        if success_rate < 0.6:
            recommendations.append("تحسين خوارزميات التكيف")
            recommendations.append("زيادة عدد خطوات التدريب")
        
        # توصيات بناءً على التنوع
        if len(set(task.task_type for task in self.task_registry.values())) < 3:
            recommendations.append("تنويع أنواع المهام التدريبية")
        
        return recommendations

    async def save_learning_state(self, filepath: str):
        """حفظ حالة التعلم"""
        
        try:
            state = {
                "model_state": self.meta_network.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "performance_stats": self.performance_stats.copy(),
                "knowledge_base": self.knowledge_base.copy(),
                "config": self.config.copy(),
                "learning_history": [
                    {
                        "episode_id": ep.episode_id,
                        "performance": ep.performance,
                        "patterns": ep.learned_patterns,
                        "timestamp": ep.timestamp.isoformat(),
                        "task_count": len(ep.tasks)
                    }
                    for ep in self.learning_history
                ]
            }
            
            # تحويل sets إلى lists للـ JSON
            if "learned_concepts" in state["performance_stats"]:
                state["performance_stats"]["learned_concepts"] = list(
                    state["performance_stats"]["learned_concepts"]
                )
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"تم حفظ حالة التعلم في: {filepath}")
            
        except Exception as e:
            self.logger.error(f"خطأ في حفظ حالة التعلم: {e}")

    async def load_learning_state(self, filepath: str):
        """تحميل حالة التعلم"""
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            # تحميل النموذج
            self.meta_network.load_state_dict(state["model_state"])
            self.optimizer.load_state_dict(state["optimizer_state"])
            
            # تحميل الإحصائيات
            self.performance_stats = state["performance_stats"]
            
            # تحويل lists إلى sets
            if "learned_concepts" in self.performance_stats:
                self.performance_stats["learned_concepts"] = set(
                    self.performance_stats["learned_concepts"]
                )
            
            # تحميل قاعدة المعرفة
            self.knowledge_base = state["knowledge_base"]
            
            # تحميل التكوين
            self.config.update(state["config"])
            
            self.logger.info(f"تم تحميل حالة التعلم من: {filepath}")
            
        except Exception as e:
            self.logger.error(f"خطأ في تحميل حالة التعلم: {e}")

# إنشاء مثيل عام
meta_learning_engine = MetaLearningEngine()

async def get_meta_learning_engine() -> MetaLearningEngine:
    """الحصول على محرك التعلم الفوقي"""
    return meta_learning_engine

if __name__ == "__main__":
    async def test_meta_learning():
        """اختبار محرك التعلم الفوقي"""
        print("🧠 اختبار محرك التعلم الفوقي")
        print("=" * 50)
        
        engine = await get_meta_learning_engine()
        
        # بيانات تجريبية
        test_interactions = [
            {
                "input": "مرحباً، كيف حالك؟",
                "expected_output": "أهلاً! أنا بخير، شكراً لسؤالك",
                "context": {"user_mood": "friendly", "time": "morning"},
                "user_profile": {"preference": "casual"}
            },
            {
                "input": [1, 2, 3, 4, 5],
                "expected_output": [2, 4, 6, 8, 10],
                "context": {"task": "multiplication", "factor": 2}
            },
            {
                "input": {"image": "test.jpg", "task": "analyze"},
                "expected_output": {"objects": ["car", "tree"], "confidence": 0.9},
                "context": {"vision_task": True}
            }
        ]
        
        for i, interaction in enumerate(test_interactions):
            print(f"\n🔄 اختبار التفاعل {i+1}")
            result = await engine.learn_from_interaction(interaction)
            
            print(f"✅ نجح التكيف: {result.get('adaptation_success', False)}")
            print(f"📊 مستوى الثقة: {result.get('confidence', 0):.2f}")
            print(f"📈 تحسن الأداء: {result.get('performance_improvement', 0):.2f}")
            print(f"🔍 أنماط جديدة: {len(result.get('new_patterns', []))}")
        
        # تقرير التعلم
        print(f"\n📈 تقرير التعلم:")
        report = await engine.get_learning_report()
        
        print(f"📊 إجمالي الحلقات: {report['performance_summary']['total_episodes']}")
        print(f"✅ معدل النجاح: {report['performance_summary']['success_rate']}")
        print(f"🧠 المفاهيم المتعلمة: {report['performance_summary']['learned_concepts']}")
    
    asyncio.run(test_meta_learning())
