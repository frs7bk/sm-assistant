
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧬 محرك التعلم التطوري المتقدم
Evolutionary Learning Engine with Genetic Algorithms
"""

import asyncio
import numpy as np
import logging
import json
import random
import time
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import pickle
from pathlib import Path
import threading
import concurrent.futures

@dataclass
class Genome:
    """جينوم يمثل حلول ومعرفة المساعد"""
    genome_id: str
    genes: Dict[str, Any]  # الجينات (المعرفة، المهارات، السلوكيات)
    fitness_score: float
    generation: int
    parent_ids: List[str]
    mutation_rate: float
    creation_timestamp: datetime
    performance_history: List[float]

@dataclass
class Evolution:
    """تطور جيل واحد"""
    generation_id: int
    population_size: int
    best_fitness: float
    average_fitness: float
    diversity_index: float
    mutations_count: int
    crossovers_count: int
    timestamp: datetime
    dominant_traits: Dict[str, Any]

class GeneticOperations:
    """عمليات وراثية للتطوير"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def crossover(self, parent1: Genome, parent2: Genome) -> Tuple[Genome, Genome]:
        """تقاطع وراثي بين جينومين"""
        
        # إنشاء جينوم الأطفال
        child1_genes = {}
        child2_genes = {}
        
        # تبادل الجينات بشكل عشوائي
        for gene_name in set(parent1.genes.keys()) | set(parent2.genes.keys()):
            if random.random() < 0.5:
                # طفل 1 يأخذ من والد 1، طفل 2 من والد 2
                child1_genes[gene_name] = parent1.genes.get(gene_name, None)
                child2_genes[gene_name] = parent2.genes.get(gene_name, None)
            else:
                # العكس
                child1_genes[gene_name] = parent2.genes.get(gene_name, None)
                child2_genes[gene_name] = parent1.genes.get(gene_name, None)
        
        # إنشاء الجينومات الجديدة
        child1 = Genome(
            genome_id=f"gen_{int(time.time())}_{random.randint(1000, 9999)}",
            genes=child1_genes,
            fitness_score=0.0,
            generation=max(parent1.generation, parent2.generation) + 1,
            parent_ids=[parent1.genome_id, parent2.genome_id],
            mutation_rate=(parent1.mutation_rate + parent2.mutation_rate) / 2,
            creation_timestamp=datetime.now(),
            performance_history=[]
        )
        
        child2 = Genome(
            genome_id=f"gen_{int(time.time())}_{random.randint(1000, 9999)}",
            genes=child2_genes,
            fitness_score=0.0,
            generation=max(parent1.generation, parent2.generation) + 1,
            parent_ids=[parent1.genome_id, parent2.genome_id],
            mutation_rate=(parent1.mutation_rate + parent2.mutation_rate) / 2,
            creation_timestamp=datetime.now(),
            performance_history=[]
        )
        
        return child1, child2
    
    def mutate(self, genome: Genome) -> Genome:
        """طفرة وراثية في الجينوم"""
        
        mutated_genes = genome.genes.copy()
        mutations_applied = 0
        
        for gene_name, gene_value in mutated_genes.items():
            if random.random() < genome.mutation_rate:
                # تطبيق طفرة حسب نوع الجين
                mutated_genes[gene_name] = self._apply_mutation(gene_value)
                mutations_applied += 1
        
        # إنشاء جينوم جديد بالطفرات
        mutated_genome = Genome(
            genome_id=f"mut_{genome.genome_id}_{int(time.time())}",
            genes=mutated_genes,
            fitness_score=0.0,
            generation=genome.generation,
            parent_ids=[genome.genome_id],
            mutation_rate=min(0.1, genome.mutation_rate * 1.05),  # زيادة طفيفة في معدل الطفرة
            creation_timestamp=datetime.now(),
            performance_history=[]
        )
        
        self.logger.debug(f"تطبيق {mutations_applied} طفرة على الجينوم {genome.genome_id}")
        
        return mutated_genome
    
    def _apply_mutation(self, gene_value: Any) -> Any:
        """تطبيق طفرة على جين محدد"""
        
        if isinstance(gene_value, (int, float)):
            # طفرة رقمية
            mutation_strength = 0.1
            return gene_value * (1 + random.uniform(-mutation_strength, mutation_strength))
        
        elif isinstance(gene_value, str):
            # طفرة نصية (تعديل طفيف في النص)
            if len(gene_value) > 0:
                # إضافة تنويع طفيف
                variations = ["متطور", "محسن", "متقدم", "محدث"]
                return f"{gene_value}_{random.choice(variations)}"
            return gene_value
        
        elif isinstance(gene_value, list):
            # طفرة في قائمة (إضافة أو حذف عنصر)
            if len(gene_value) > 0 and random.random() < 0.5:
                # حذف عنصر
                return gene_value[:-1]
            else:
                # إضافة عنصر
                new_item = f"عنصر_جديد_{random.randint(1, 100)}"
                return gene_value + [new_item]
        
        elif isinstance(gene_value, dict):
            # طفرة في قاموس (تعديل قيمة)
            if gene_value:
                key = random.choice(list(gene_value.keys()))
                new_dict = gene_value.copy()
                new_dict[key] = self._apply_mutation(gene_value[key])
                return new_dict
            return gene_value
        
        else:
            # نوع غير معروف، إرجاع كما هو
            return gene_value

class FitnessEvaluator:
    """مقيم اللياقة الوراثية"""
    
    def __init__(self):
        self.performance_weights = {
            "response_quality": 0.3,
            "user_satisfaction": 0.25,
            "task_completion": 0.2,
            "learning_speed": 0.15,
            "adaptability": 0.1
        }
        self.logger = logging.getLogger(__name__)
    
    async def evaluate_fitness(self, genome: Genome, performance_data: Dict[str, Any]) -> float:
        """تقييم لياقة الجينوم"""
        
        fitness_score = 0.0
        
        try:
            # تقييم جودة الاستجابة
            response_quality = performance_data.get("response_quality", 0.5)
            fitness_score += response_quality * self.performance_weights["response_quality"]
            
            # تقييم رضا المستخدم
            user_satisfaction = performance_data.get("user_satisfaction", 0.5)
            fitness_score += user_satisfaction * self.performance_weights["user_satisfaction"]
            
            # تقييم إتمام المهام
            task_completion = performance_data.get("task_completion_rate", 0.5)
            fitness_score += task_completion * self.performance_weights["task_completion"]
            
            # تقييم سرعة التعلم
            learning_speed = performance_data.get("learning_improvement", 0.5)
            fitness_score += learning_speed * self.performance_weights["learning_speed"]
            
            # تقييم التكيف
            adaptability = self._evaluate_adaptability(genome, performance_data)
            fitness_score += adaptability * self.performance_weights["adaptability"]
            
            # تطبيق عوامل إضافية
            fitness_score = self._apply_bonus_factors(genome, fitness_score, performance_data)
            
            # تحديث نقاط اللياقة والتاريخ
            genome.fitness_score = fitness_score
            genome.performance_history.append(fitness_score)
            
            # الحفاظ على آخر 20 قياس فقط
            if len(genome.performance_history) > 20:
                genome.performance_history = genome.performance_history[-20:]
            
            return fitness_score
            
        except Exception as e:
            self.logger.error(f"خطأ في تقييم اللياقة: {e}")
            return 0.1  # نقاط أساسية في حالة الخطأ
    
    def _evaluate_adaptability(self, genome: Genome, performance_data: Dict[str, Any]) -> float:
        """تقييم قدرة التكيف"""
        
        # مراجعة تاريخ الأداء
        if len(genome.performance_history) < 2:
            return 0.5  # قيمة متوسطة للجينومات الجديدة
        
        # حساب تحسن الأداء عبر الوقت
        recent_performance = genome.performance_history[-5:]  # آخر 5 قياسات
        early_performance = genome.performance_history[:5]    # أول 5 قياسات
        
        if len(recent_performance) > 0 and len(early_performance) > 0:
            improvement = np.mean(recent_performance) - np.mean(early_performance)
            adaptability = min(1.0, max(0.0, 0.5 + improvement))
        else:
            adaptability = 0.5
        
        return adaptability
    
    def _apply_bonus_factors(self, genome: Genome, base_fitness: float, performance_data: Dict[str, Any]) -> float:
        """تطبيق عوامل المكافآت"""
        
        final_fitness = base_fitness
        
        # مكافأة الجينومات الناضجة (جيل عالي)
        if genome.generation > 5:
            final_fitness *= 1.05
        
        # مكافأة التنوع الجيني
        gene_diversity = len(genome.genes) / 20.0  # افتراض 20 جين كحد أقصى
        final_fitness *= (1 + gene_diversity * 0.1)
        
        # عقوبة للطفرات المفرطة
        if genome.mutation_rate > 0.05:
            final_fitness *= 0.95
        
        # مكافأة الاستقرار في الأداء
        if len(genome.performance_history) > 5:
            stability = 1 - np.std(genome.performance_history[-5:])
            final_fitness *= (1 + stability * 0.1)
        
        return min(1.0, max(0.0, final_fitness))

class EvolutionaryLearningEngine:
    """محرك التعلم التطوري الشامل"""
    
    def __init__(self, population_size: int = 20):
        self.population_size = population_size
        self.population: List[Genome] = []
        self.generation_count = 0
        self.evolution_history: List[Evolution] = []
        
        self.genetic_operations = GeneticOperations()
        self.fitness_evaluator = FitnessEvaluator()
        
        self.selection_pressure = 0.3  # نسبة الأفراد المختارين للتكاثر
        self.elite_size = 2  # عدد أفضل الأفراد المحفوظين دائماً
        
        self.logger = logging.getLogger(__name__)
        
        # معالجة متوازية
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        
        # تهيئة الجيل الأول
        self._initialize_population()
    
    def _initialize_population(self):
        """تهيئة الجيل الأول"""
        
        self.logger.info("🧬 تهيئة الجيل الأول من الجينومات")
        
        base_genes = {
            "language_skills": ["العربية", "الإنجليزية"],
            "response_style": "ودود_ومفيد",
            "learning_rate": 0.1,
            "creativity_level": 0.7,
            "analytical_depth": 0.8,
            "patience_level": 0.9,
            "humor_tendency": 0.3,
            "technical_knowledge": ["برمجة", "ذكاء_اصطناعي", "تقنية"],
            "emotional_intelligence": 0.8,
            "problem_solving_approach": "منهجي_وإبداعي",
            "memory_retention": 0.9,
            "context_awareness": 0.8,
            "user_adaptation": 0.7,
            "multitasking_ability": 0.6,
            "curiosity_level": 0.8
        }
        
        for i in range(self.population_size):
            # إنشاء تنويعات من الجينات الأساسية
            individual_genes = base_genes.copy()
            
            # إضافة تنويع عشوائي
            for gene_name, gene_value in individual_genes.items():
                if isinstance(gene_value, (int, float)):
                    # تنويع رقمي
                    variation = random.uniform(0.8, 1.2)
                    individual_genes[gene_name] = min(1.0, max(0.0, gene_value * variation))
                elif isinstance(gene_value, list):
                    # تنويع في القوائم
                    if random.random() < 0.3:
                        additional_items = [f"مهارة_إضافية_{random.randint(1, 100)}"]
                        individual_genes[gene_name] = gene_value + additional_items
            
            # إنشاء الجينوم
            genome = Genome(
                genome_id=f"init_gen0_{i}",
                genes=individual_genes,
                fitness_score=0.0,
                generation=0,
                parent_ids=[],
                mutation_rate=random.uniform(0.01, 0.03),
                creation_timestamp=datetime.now(),
                performance_history=[]
            )
            
            self.population.append(genome)
        
        self.logger.info(f"✅ تم إنشاء {len(self.population)} فرد في الجيل الأول")
    
    async def evolve_generation(self, performance_data: Dict[str, Dict[str, Any]]) -> Evolution:
        """تطوير جيل جديد"""
        
        self.logger.info(f"🔄 بدء تطوير الجيل {self.generation_count + 1}")
        
        # تقييم لياقة الجيل الحالي
        await self._evaluate_population_fitness(performance_data)
        
        # اختيار الأفراد للتكاثر
        selected_parents = self._select_parents()
        
        # إنشاء الجيل الجديد
        new_population = await self._create_new_generation(selected_parents)
        
        # تحليل التطور
        evolution_stats = self._analyze_evolution()
        
        # تحديث الجيل
        self.population = new_population
        self.generation_count += 1
        
        # حفظ إحصائيات التطور
        evolution = Evolution(
            generation_id=self.generation_count,
            population_size=len(self.population),
            best_fitness=evolution_stats["best_fitness"],
            average_fitness=evolution_stats["average_fitness"],
            diversity_index=evolution_stats["diversity_index"],
            mutations_count=evolution_stats["mutations_count"],
            crossovers_count=evolution_stats["crossovers_count"],
            timestamp=datetime.now(),
            dominant_traits=evolution_stats["dominant_traits"]
        )
        
        self.evolution_history.append(evolution)
        
        # الحفاظ على آخر 100 جيل فقط
        if len(self.evolution_history) > 100:
            self.evolution_history = self.evolution_history[-100:]
        
        self.logger.info(f"✅ تم تطوير الجيل {self.generation_count}")
        
        return evolution
    
    async def _evaluate_population_fitness(self, performance_data: Dict[str, Dict[str, Any]]):
        """تقييم لياقة جميع أفراد الجيل"""
        
        evaluation_tasks = []
        
        for genome in self.population:
            # استخدام بيانات الأداء الخاصة بالجينوم أو بيانات عامة
            genome_performance = performance_data.get(genome.genome_id, {
                "response_quality": random.uniform(0.3, 0.9),
                "user_satisfaction": random.uniform(0.4, 0.8),
                "task_completion_rate": random.uniform(0.5, 0.9),
                "learning_improvement": random.uniform(0.2, 0.7)
            })
            
            # إضافة مهمة التقييم
            task = self.fitness_evaluator.evaluate_fitness(genome, genome_performance)
            evaluation_tasks.append(task)
        
        # تشغيل التقييمات بشكل متوازي
        await asyncio.gather(*evaluation_tasks)
    
    def _select_parents(self) -> List[Genome]:
        """اختيار الأفراد للتكاثر"""
        
        # ترتيب السكان حسب اللياقة
        sorted_population = sorted(self.population, key=lambda g: g.fitness_score, reverse=True)
        
        # عدد الوالدين المختارين
        num_parents = max(2, int(self.population_size * self.selection_pressure))
        
        # اختيار النخبة
        elite_parents = sorted_population[:self.elite_size]
        
        # اختيار الباقي بشكل احتمالي حسب اللياقة
        remaining_slots = num_parents - len(elite_parents)
        
        if remaining_slots > 0:
            # حساب احتمالات الاختيار
            fitness_scores = [g.fitness_score for g in sorted_population[self.elite_size:]]
            total_fitness = sum(fitness_scores)
            
            if total_fitness > 0:
                probabilities = [score / total_fitness for score in fitness_scores]
                
                # اختيار عشوائي مرجح
                chosen_indices = np.random.choice(
                    len(fitness_scores),
                    size=remaining_slots,
                    replace=False,
                    p=probabilities
                )
                
                additional_parents = [sorted_population[self.elite_size + i] for i in chosen_indices]
            else:
                # إذا كانت جميع النقاط صفر، اختيار عشوائي
                additional_parents = random.sample(sorted_population[self.elite_size:], remaining_slots)
            
            elite_parents.extend(additional_parents)
        
        self.logger.debug(f"تم اختيار {len(elite_parents)} والد للتكاثر")
        
        return elite_parents
    
    async def _create_new_generation(self, parents: List[Genome]) -> List[Genome]:
        """إنشاء الجيل الجديد"""
        
        new_population = []
        mutations_count = 0
        crossovers_count = 0
        
        # الحفاظ على النخبة
        elite_count = min(self.elite_size, len(parents))
        new_population.extend(parents[:elite_count])
        
        # إنشاء الأفراد الجدد
        while len(new_population) < self.population_size:
            if len(parents) >= 2:
                # اختيار والدين عشوائياً
                parent1, parent2 = random.sample(parents, 2)
                
                # تقاطع وراثي
                child1, child2 = self.genetic_operations.crossover(parent1, parent2)
                crossovers_count += 1
                
                # طفرة محتملة
                if random.random() < 0.3:  # احتمال الطفرة
                    child1 = self.genetic_operations.mutate(child1)
                    mutations_count += 1
                
                if random.random() < 0.3:
                    child2 = self.genetic_operations.mutate(child2)
                    mutations_count += 1
                
                # إضافة الأطفال للجيل الجديد
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)
            else:
                # إذا لم يكن هناك والدين كافيين، إنشاء فرد عشوائي
                break
        
        # تسجيل إحصائيات العمليات الوراثية
        self.logger.debug(f"تم إجراء {crossovers_count} تقاطع و {mutations_count} طفرة")
        
        return new_population[:self.population_size]
    
    def _analyze_evolution(self) -> Dict[str, Any]:
        """تحليل إحصائيات التطور"""
        
        if not self.population:
            return {
                "best_fitness": 0,
                "average_fitness": 0,
                "diversity_index": 0,
                "mutations_count": 0,
                "crossovers_count": 0,
                "dominant_traits": {}
            }
        
        # حساب الإحصائيات الأساسية
        fitness_scores = [g.fitness_score for g in self.population]
        best_fitness = max(fitness_scores)
        average_fitness = sum(fitness_scores) / len(fitness_scores)
        
        # حساب مؤشر التنوع (بناءً على عدد الجينات الفريدة)
        all_genes = {}
        for genome in self.population:
            for gene_name, gene_value in genome.genes.items():
                if gene_name not in all_genes:
                    all_genes[gene_name] = set()
                all_genes[gene_name].add(str(gene_value))
        
        diversity_values = [len(unique_values) for unique_values in all_genes.values()]
        diversity_index = sum(diversity_values) / len(diversity_values) if diversity_values else 0
        
        # تحليل الصفات السائدة
        dominant_traits = {}
        for gene_name in all_genes:
            gene_values = [str(g.genes.get(gene_name, "")) for g in self.population]
            most_common = max(set(gene_values), key=gene_values.count)
            dominant_traits[gene_name] = most_common
        
        return {
            "best_fitness": best_fitness,
            "average_fitness": average_fitness,
            "diversity_index": diversity_index,
            "mutations_count": 0,  # سيتم تحديثها في create_new_generation
            "crossovers_count": 0,  # سيتم تحديثها في create_new_generation
            "dominant_traits": dominant_traits
        }
    
    def get_best_genome(self) -> Optional[Genome]:
        """الحصول على أفضل جينوم في الجيل الحالي"""
        if not self.population:
            return None
        
        return max(self.population, key=lambda g: g.fitness_score)
    
    def get_evolution_report(self) -> Dict[str, Any]:
        """تقرير شامل عن التطور"""
        
        best_genome = self.get_best_genome()
        
        report = {
            "current_generation": self.generation_count,
            "population_size": len(self.population),
            "total_evolutions": len(self.evolution_history),
            "best_individual": {
                "genome_id": best_genome.genome_id if best_genome else None,
                "fitness_score": best_genome.fitness_score if best_genome else 0,
                "generation": best_genome.generation if best_genome else 0,
                "key_traits": {}
            } if best_genome else None,
            "evolution_trends": [],
            "genetic_diversity": 0,
            "learning_progress": []
        }
        
        if best_genome:
            # أهم الصفات في أفضل جينوم
            key_genes = ["learning_rate", "creativity_level", "emotional_intelligence", "adaptability"]
            for gene in key_genes:
                if gene in best_genome.genes:
                    report["best_individual"]["key_traits"][gene] = best_genome.genes[gene]
        
        # اتجاهات التطور
        if len(self.evolution_history) > 1:
            recent_evolutions = self.evolution_history[-10:]  # آخر 10 أجيال
            
            fitness_trend = [e.best_fitness for e in recent_evolutions]
            diversity_trend = [e.diversity_index for e in recent_evolutions]
            
            report["evolution_trends"] = {
                "fitness_improvement": fitness_trend[-1] - fitness_trend[0] if len(fitness_trend) > 1 else 0,
                "diversity_change": diversity_trend[-1] - diversity_trend[0] if len(diversity_trend) > 1 else 0,
                "generations_analyzed": len(recent_evolutions)
            }
        
        # التنوع الجيني الحالي
        if self.population:
            unique_traits = set()
            for genome in self.population:
                for trait_value in genome.genes.values():
                    unique_traits.add(str(trait_value))
            
            max_possible_traits = len(self.population) * len(self.population[0].genes) if self.population else 1
            report["genetic_diversity"] = len(unique_traits) / max_possible_traits
        
        return report
    
    async def save_evolution_state(self, file_path: Optional[str] = None):
        """حفظ حالة التطور"""
        
        if not file_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f"data/evolution_state_{timestamp}.pkl"
        
        try:
            # تحضير البيانات للحفظ
            evolution_state = {
                "population": [asdict(genome) for genome in self.population],
                "generation_count": self.generation_count,
                "evolution_history": [asdict(evolution) for evolution in self.evolution_history],
                "population_size": self.population_size,
                "selection_pressure": self.selection_pressure,
                "elite_size": self.elite_size
            }
            
            # إنشاء المجلد إن لم يكن موجوداً
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            # حفظ الحالة
            with open(file_path, 'wb') as f:
                pickle.dump(evolution_state, f)
            
            self.logger.info(f"💾 تم حفظ حالة التطور في: {file_path}")
            
        except Exception as e:
            self.logger.error(f"خطأ في حفظ حالة التطور: {e}")
    
    async def load_evolution_state(self, file_path: str):
        """تحميل حالة التطور"""
        
        try:
            with open(file_path, 'rb') as f:
                evolution_state = pickle.load(f)
            
            # استعادة الجيل
            self.population = []
            for genome_data in evolution_state["population"]:
                # تحويل datetime من string إن كان مطلوباً
                if isinstance(genome_data["creation_timestamp"], str):
                    genome_data["creation_timestamp"] = datetime.fromisoformat(genome_data["creation_timestamp"])
                
                genome = Genome(**genome_data)
                self.population.append(genome)
            
            # استعادة الإحصائيات
            self.generation_count = evolution_state["generation_count"]
            self.population_size = evolution_state["population_size"]
            
            # استعادة تاريخ التطور
            self.evolution_history = []
            for evolution_data in evolution_state["evolution_history"]:
                if isinstance(evolution_data["timestamp"], str):
                    evolution_data["timestamp"] = datetime.fromisoformat(evolution_data["timestamp"])
                
                evolution = Evolution(**evolution_data)
                self.evolution_history.append(evolution)
            
            self.logger.info(f"📁 تم تحميل حالة التطور من: {file_path}")
            
        except Exception as e:
            self.logger.error(f"خطأ في تحميل حالة التطور: {e}")

# مثيل عام للاستخدام
evolutionary_engine = EvolutionaryLearningEngine()

async def get_evolutionary_engine() -> EvolutionaryLearningEngine:
    """الحصول على محرك التعلم التطوري"""
    return evolutionary_engine

# مثال على الاستخدام
async def example_evolutionary_usage():
    """مثال على استخدام التعلم التطوري"""
    engine = await get_evolutionary_engine()
    
    # محاكاة بيانات أداء للاختبار
    performance_data = {}
    for genome in engine.population[:5]:  # أول 5 أفراد فقط للاختبار
        performance_data[genome.genome_id] = {
            "response_quality": random.uniform(0.5, 0.9),
            "user_satisfaction": random.uniform(0.4, 0.8),
            "task_completion_rate": random.uniform(0.6, 0.95),
            "learning_improvement": random.uniform(0.3, 0.7)
        }
    
    # تطوير جيل جديد
    evolution = await engine.evolve_generation(performance_data)
    
    print(f"🧬 تطوير الجيل {evolution.generation_id}")
    print(f"🏆 أفضل لياقة: {evolution.best_fitness:.3f}")
    print(f"📊 متوسط اللياقة: {evolution.average_fitness:.3f}")
    print(f"🌈 مؤشر التنوع: {evolution.diversity_index:.3f}")
    
    # عرض أفضل فرد
    best_genome = engine.get_best_genome()
    if best_genome:
        print(f"🥇 أفضل فرد: {best_genome.genome_id}")
        print(f"   • اللياقة: {best_genome.fitness_score:.3f}")
        print(f"   • الجيل: {best_genome.generation}")

if __name__ == "__main__":
    asyncio.run(example_evolutionary_usage())
