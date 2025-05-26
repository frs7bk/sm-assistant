
"""
محرك الأتمتة الذكية للمساعد
Smart Automation Engine for Assistant
"""

import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import threading
import queue

class SmartAutomationEngine:
    """
    محرك أتمتة ذكي يتعلم من سلوك المستخدم
    """
    
    def __init__(self):
        self.user_patterns = {}
        self.automation_rules = []
        self.context_history = []
        self.learning_enabled = True
        self.prediction_queue = queue.Queue()
        
    def learn_user_behavior(self, action: str, context: Dict, timestamp: datetime = None):
        """
        تعلم سلوك المستخدم من الأفعال والسياق
        Learn user behavior from actions and context
        """
        if not self.learning_enabled:
            return
            
        if timestamp is None:
            timestamp = datetime.now()
            
        # إنشاء نمط للسلوك
        pattern_key = self._create_pattern_key(action, context)
        
        if pattern_key not in self.user_patterns:
            self.user_patterns[pattern_key] = {
                'frequency': 0,
                'times': [],
                'contexts': [],
                'confidence': 0.0
            }
        
        # تحديث النمط
        pattern = self.user_patterns[pattern_key]
        pattern['frequency'] += 1
        pattern['times'].append(timestamp)
        pattern['contexts'].append(context)
        
        # حساب الثقة بناءً على التكرار والانتظام
        pattern['confidence'] = self._calculate_pattern_confidence(pattern)
        
        # إضافة إلى تاريخ السياق
        self.context_history.append({
            'action': action,
            'context': context,
            'timestamp': timestamp
        })
        
        # الاحتفاظ بآخر 1000 إدخال فقط
        if len(self.context_history) > 1000:
            self.context_history = self.context_history[-1000:]
    
    def predict_next_action(self, current_context: Dict) -> Optional[Dict]:
        """
        التنبؤ بالعمل التالي بناءً على السياق الحالي
        Predict next action based on current context
        """
        best_prediction = None
        highest_probability = 0.0
        
        for pattern_key, pattern in self.user_patterns.items():
            if pattern['confidence'] < 0.3:  # تجاهل الأنماط ضعيفة الثقة
                continue
                
            # حساب احتمالية حدوث هذا النمط
            probability = self._calculate_action_probability(pattern, current_context)
            
            if probability > highest_probability:
                highest_probability = probability
                action = self._extract_action_from_pattern_key(pattern_key)
                best_prediction = {
                    'action': action,
                    'probability': probability,
                    'suggested_time': self._predict_optimal_time(pattern),
                    'context_match': self._calculate_context_similarity(pattern, current_context)
                }
        
        return best_prediction if highest_probability > 0.5 else None
    
    def create_automation_rule(self, name: str, condition: Dict, action: Dict, priority: int = 1):
        """
        إنشاء قاعدة أتمتة جديدة
        Create new automation rule
        """
        rule = {
            'name': name,
            'condition': condition,
            'action': action,
            'priority': priority,
            'enabled': True,
            'created_at': datetime.now(),
            'execution_count': 0,
            'last_executed': None
        }
        
        self.automation_rules.append(rule)
        # ترتيب القواعد حسب الأولوية
        self.automation_rules.sort(key=lambda x: x['priority'], reverse=True)
    
    def evaluate_automation_rules(self, current_context: Dict) -> List[Dict]:
        """
        تقييم قواعد الأتمتة وإرجاع الإجراءات المطلوبة
        Evaluate automation rules and return required actions
        """
        triggered_actions = []
        
        for rule in self.automation_rules:
            if not rule['enabled']:
                continue
                
            if self._evaluate_condition(rule['condition'], current_context):
                # التحقق من عدم تنفيذ القاعدة مؤخراً (تجنب التكرار)
                if self._should_execute_rule(rule):
                    triggered_actions.append({
                        'rule_name': rule['name'],
                        'action': rule['action'],
                        'priority': rule['priority'],
                        'execution_time': datetime.now()
                    })
                    
                    # تحديث بيانات التنفيذ
                    rule['execution_count'] += 1
                    rule['last_executed'] = datetime.now()
        
        return triggered_actions
    
    def adaptive_scheduling(self, task: str, flexibility: float = 0.5) -> datetime:
        """
        جدولة تكيفية للمهام بناءً على أنماط المستخدم
        Adaptive task scheduling based on user patterns
        """
        # البحث عن أنماط مشابهة في التاريخ
        similar_patterns = self._find_similar_task_patterns(task)
        
        if not similar_patterns:
            # إذا لم توجد أنماط، اقترح وقت افتراضي
            return datetime.now() + timedelta(hours=1)
        
        # تحليل الأوقات المفضلة
        preferred_times = []
        for pattern in similar_patterns:
            for timestamp in pattern['times']:
                preferred_times.append(timestamp.hour)
        
        # حساب الوقت الأمثل
        if preferred_times:
            optimal_hour = max(set(preferred_times), key=preferred_times.count)
            
            # إنشاء وقت مقترح
            suggested_time = datetime.now().replace(hour=optimal_hour, minute=0, second=0)
            
            # إذا كان الوقت قد مضى اليوم، اجعله للغد
            if suggested_time <= datetime.now():
                suggested_time += timedelta(days=1)
            
            # تطبيق المرونة
            if flexibility > 0:
                variation_minutes = int(flexibility * 60)
                import random
                suggested_time += timedelta(minutes=random.randint(-variation_minutes, variation_minutes))
            
            return suggested_time
        
        return datetime.now() + timedelta(hours=1)
    
    def generate_proactive_suggestions(self, current_context: Dict) -> List[Dict]:
        """
        إنتاج اقتراحات استباقية بناءً على السياق
        Generate proactive suggestions based on context
        """
        suggestions = []
        
        # البحث عن أنماط متكررة في هذا الوقت/السياق
        current_time = datetime.now()
        similar_contexts = self._find_contexts_by_time_and_situation(current_time, current_context)
        
        for context in similar_contexts:
            if context['frequency'] > 3:  # تكرر أكثر من 3 مرات
                suggestion = {
                    'type': 'routine_suggestion',
                    'action': context['action'],
                    'confidence': context['confidence'],
                    'reason': f"عادة ما تقوم بـ {context['action']} في هذا الوقت",
                    'suggested_time': current_time,
                    'priority': self._calculate_suggestion_priority(context)
                }
                suggestions.append(suggestion)
        
        # اقتراحات بناءً على التحليل التنبؤي
        prediction = self.predict_next_action(current_context)
        if prediction and prediction['probability'] > 0.7:
            suggestions.append({
                'type': 'predictive_suggestion',
                'action': prediction['action'],
                'confidence': prediction['probability'],
                'reason': "اقتراح تنبؤي بناءً على سلوكك السابق",
                'suggested_time': prediction['suggested_time'],
                'priority': 'medium'
            })
        
        # ترتيب الاقتراحات حسب الأولوية
        suggestions.sort(key=lambda x: x.get('priority', 'low'), reverse=True)
        
        return suggestions
    
    # دوال مساعدة داخلية
    def _create_pattern_key(self, action: str, context: Dict) -> str:
        """إنشاء مفتاح فريد للنمط"""
        context_str = json.dumps(context, sort_keys=True, default=str)
        return f"{action}:{hash(context_str)}"
    
    def _calculate_pattern_confidence(self, pattern: Dict) -> float:
        """حساب مستوى الثقة في النمط"""
        frequency = pattern['frequency']
        time_regularity = self._calculate_time_regularity(pattern['times'])
        
        # الثقة تزيد مع التكرار والانتظام
        confidence = min(1.0, (frequency / 10) * 0.7 + time_regularity * 0.3)
        return confidence
    
    def _calculate_time_regularity(self, timestamps: List[datetime]) -> float:
        """حساب انتظام التوقيتات"""
        if len(timestamps) < 2:
            return 0.0
        
        # حساب الفترات بين التوقيتات
        intervals = []
        for i in range(1, len(timestamps)):
            interval = (timestamps[i] - timestamps[i-1]).total_seconds()
            intervals.append(interval)
        
        if not intervals:
            return 0.0
        
        # حساب الانحراف المعياري للفترات
        mean_interval = sum(intervals) / len(intervals)
        variance = sum((x - mean_interval) ** 2 for x in intervals) / len(intervals)
        std_dev = variance ** 0.5
        
        # كلما قل الانحراف، زادت الانتظام
        regularity = max(0.0, 1.0 - (std_dev / mean_interval) if mean_interval > 0 else 0.0)
        return min(1.0, regularity)
    
    def _calculate_action_probability(self, pattern: Dict, current_context: Dict) -> float:
        """حساب احتمالية حدوث الفعل"""
        context_similarity = self._calculate_context_similarity(pattern, current_context)
        time_probability = self._calculate_time_probability(pattern)
        
        # دمج العوامل المختلفة
        probability = (context_similarity * 0.6 + 
                      time_probability * 0.3 + 
                      pattern['confidence'] * 0.1)
        
        return probability
    
    def _calculate_context_similarity(self, pattern: Dict, current_context: Dict) -> float:
        """حساب تشابه السياق"""
        if not pattern['contexts']:
            return 0.0
        
        # مقارنة السياق الحالي مع أحدث سياقات النمط
        recent_contexts = pattern['contexts'][-5:]  # آخر 5 سياقات
        
        similarities = []
        for context in recent_contexts:
            similarity = self._compare_contexts(context, current_context)
            similarities.append(similarity)
        
        return sum(similarities) / len(similarities) if similarities else 0.0
    
    def _compare_contexts(self, context1: Dict, context2: Dict) -> float:
        """مقارنة سياقين وإرجاع درجة التشابه"""
        common_keys = set(context1.keys()) & set(context2.keys())
        if not common_keys:
            return 0.0
        
        matches = 0
        for key in common_keys:
            if context1[key] == context2[key]:
                matches += 1
        
        return matches / len(common_keys)
    
    def _calculate_time_probability(self, pattern: Dict) -> float:
        """حساب احتمالية الوقت"""
        if not pattern['times']:
            return 0.0
        
        current_time = datetime.now()
        current_hour = current_time.hour
        current_day = current_time.weekday()
        
        # تحليل الساعات المفضلة
        hours = [t.hour for t in pattern['times']]
        hour_matches = hours.count(current_hour)
        hour_probability = hour_matches / len(hours) if hours else 0.0
        
        # تحليل الأيام المفضلة
        days = [t.weekday() for t in pattern['times']]
        day_matches = days.count(current_day)
        day_probability = day_matches / len(days) if days else 0.0
        
        return (hour_probability * 0.7 + day_probability * 0.3)
    
    def _extract_action_from_pattern_key(self, pattern_key: str) -> str:
        """استخراج الفعل من مفتاح النمط"""
        return pattern_key.split(':')[0]
    
    def _predict_optimal_time(self, pattern: Dict) -> datetime:
        """التنبؤ بالوقت الأمثل للفعل"""
        if not pattern['times']:
            return datetime.now() + timedelta(hours=1)
        
        # حساب متوسط الوقت المفضل
        hours = [t.hour for t in pattern['times']]
        avg_hour = sum(hours) / len(hours)
        
        # إنشاء وقت مقترح
        suggested_time = datetime.now().replace(hour=int(avg_hour), minute=0, second=0)
        
        if suggested_time <= datetime.now():
            suggested_time += timedelta(days=1)
        
        return suggested_time
    
    def _evaluate_condition(self, condition: Dict, context: Dict) -> bool:
        """تقييم شرط قاعدة الأتمتة"""
        for key, expected_value in condition.items():
            if key not in context:
                return False
            
            if isinstance(expected_value, dict) and 'operator' in expected_value:
                # شروط متقدمة مع مشغلات
                operator = expected_value['operator']
                value = expected_value['value']
                
                if operator == 'equals' and context[key] != value:
                    return False
                elif operator == 'greater_than' and context[key] <= value:
                    return False
                elif operator == 'less_than' and context[key] >= value:
                    return False
                elif operator == 'contains' and value not in str(context[key]):
                    return False
            else:
                # شرط بسيط
                if context[key] != expected_value:
                    return False
        
        return True
    
    def _should_execute_rule(self, rule: Dict) -> bool:
        """تحديد ما إذا كان يجب تنفيذ القاعدة"""
        if rule['last_executed'] is None:
            return True
        
        # تجنب التنفيذ المتكرر
        time_since_last = datetime.now() - rule['last_executed']
        min_interval = timedelta(minutes=5)  # الحد الأدنى بين التنفيذات
        
        return time_since_last >= min_interval
    
    def _find_similar_task_patterns(self, task: str) -> List[Dict]:
        """البحث عن أنماط مهام مشابهة"""
        similar_patterns = []
        
        for pattern_key, pattern in self.user_patterns.items():
            action = self._extract_action_from_pattern_key(pattern_key)
            if task.lower() in action.lower() or action.lower() in task.lower():
                similar_patterns.append(pattern)
        
        return similar_patterns
    
    def _find_contexts_by_time_and_situation(self, target_time: datetime, current_context: Dict) -> List[Dict]:
        """البحث عن سياقات مشابهة بالوقت والموقف"""
        similar_contexts = {}
        target_hour = target_time.hour
        target_day = target_time.weekday()
        
        for entry in self.context_history:
            if (entry['timestamp'].hour == target_hour and 
                entry['timestamp'].weekday() == target_day):
                
                action = entry['action']
                if action not in similar_contexts:
                    similar_contexts[action] = {
                        'action': action,
                        'frequency': 0,
                        'confidence': 0.0,
                        'contexts': []
                    }
                
                similar_contexts[action]['frequency'] += 1
                similar_contexts[action]['contexts'].append(entry['context'])
        
        # حساب الثقة لكل سياق
        for context in similar_contexts.values():
            context['confidence'] = min(1.0, context['frequency'] / 10)
        
        return list(similar_contexts.values())
    
    def _calculate_suggestion_priority(self, context: Dict) -> str:
        """حساب أولوية الاقتراح"""
        confidence = context.get('confidence', 0.0)
        frequency = context.get('frequency', 0)
        
        if confidence > 0.8 and frequency > 5:
            return 'high'
        elif confidence > 0.5 and frequency > 2:
            return 'medium'
        else:
            return 'low'
