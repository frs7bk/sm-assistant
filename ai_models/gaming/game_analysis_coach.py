
"""
مدرب الألعاب الذكي مع تحليل متقدم واستراتيجيات
Intelligent Game Coach with Advanced Analysis and Strategies
"""

import cv2
import numpy as np
import json
from typing import Dict, List, Any, Optional, Tuple
import time
from datetime import datetime
import threading
import queue
import requests
import base64
from dataclasses import dataclass

@dataclass
class GameSession:
    """بيانات جلسة اللعب"""
    game_name: str
    start_time: datetime
    player_performance: Dict[str, float]
    achievements: List[str]
    areas_for_improvement: List[str]
    strategies_suggested: List[str]
    emotional_state: str

class IntelligentGameCoach:
    """
    مدرب ألعاب ذكي يحلل الألعاب ويقدم الاستراتيجيات والتشجيع
    """
    
    def __init__(self):
        print("🎮 تهيئة المدرب الذكي للألعاب...")
        
        # قاعدة بيانات الألعاب والاستراتيجيات
        self.game_database = {
            'fps_games': {
                'keywords': ['shooting', 'fps', 'counter', 'call of duty', 'valorant'],
                'strategies': {
                    'aiming': [
                        "حافظ على استقرار اليد أثناء التصويب",
                        "استخدم تدريبات التصويب اليومية",
                        "اضبط حساسية الماوس للحصول على دقة أفضل"
                    ],
                    'positioning': [
                        "تجنب البقاء في المناطق المكشوفة",
                        "استخدم الغطاء بذكاء",
                        "احتفظ بخطوط الانسحاب"
                    ],
                    'team_play': [
                        "تواصل مع فريقك باستمرار",
                        "تبادل المعلومات حول مواقع الأعداء",
                        "ادعم زملاءك في الفريق"
                    ]
                },
                'common_mistakes': [
                    "الاندفاع بدون تفكير",
                    "تجاهل الخريطة الصغيرة",
                    "عدم إعادة التحميل في الوقت المناسب"
                ]
            },
            'strategy_games': {
                'keywords': ['strategy', 'civilization', 'age of empires', 'chess'],
                'strategies': {
                    'resource_management': [
                        "خطط لاستخدام الموارد على المدى الطويل",
                        "لا تهدر الموارد في البداية",
                        "وازن بين الاقتصاد والجيش"
                    ],
                    'expansion': [
                        "توسع بحذر وبشكل محكم",
                        "أمن خطوط الإمداد",
                        "لا تتوسع أسرع من قدرتك على الدفاع"
                    ],
                    'technology': [
                        "استثمر في التقنيات المفيدة",
                        "لا تتأخر في التطوير التقني",
                        "خطط لشجرة التقنيات مسبقاً"
                    ]
                }
            },
            'puzzle_games': {
                'keywords': ['puzzle', 'tetris', 'candy crush', 'match'],
                'strategies': {
                    'pattern_recognition': [
                        "ابحث عن الأنماط المتكررة",
                        "فكر بعدة خطوات مقدماً",
                        "تعلم من الأخطاء السابقة"
                    ],
                    'time_management': [
                        "لا تتسرع في اتخاذ القرارات",
                        "استغل الوقت المتاح بحكمة",
                        "مارس تحت ضغط الوقت"
                    ]
                }
            },
            'racing_games': {
                'keywords': ['racing', 'car', 'formula', 'need for speed'],
                'strategies': {
                    'racing_line': [
                        "تعلم خط السباق الأمثل",
                        "ادخل المنعطفات من الخارج",
                        "اخرج من المنعطفات بسرعة"
                    ],
                    'braking': [
                        "اكبح قبل المنعطف وليس أثناءه",
                        "استخدم الكبح التدريجي",
                        "تعلم نقاط الكبح لكل منعطف"
                    ]
                }
            }
        }
        
        # إعدادات التحليل
        self.analysis_settings = {
            'track_mouse_movement': True,
            'analyze_reaction_time': True,
            'monitor_game_screen': True,
            'detect_emotions': True,
            'performance_tracking': True
        }
        
        # متغيرات التتبع
        self.current_session = None
        self.performance_history = []
        self.real_time_feedback = True
        
        # نماذج التشجيع
        self.encouragement_phrases = {
            'good_performance': [
                "أداء ممتاز! استمر هكذا! 🔥",
                "رائع! أنت تتحسن باستمرار! 🌟",
                "أحسنت! هذا ما أتحدث عنه! 💪",
                "مذهل! أنت تلعب كالمحترفين! 🏆"
            ],
            'needs_improvement': [
                "لا تقلق، كل محترف بدأ من هنا! 💪",
                "أنت تتعلم! الممارسة تؤدي للإتقان! 📈",
                "استمر في المحاولة، أنت على الطريق الصحيح! 🎯",
                "كل خطأ هو فرصة للتعلم! 🧠"
            ],
            'motivational': [
                "أؤمن بك! يمكنك تحقيق ذلك! 🚀",
                "لا تستسلم! النجاح قريب! ⭐",
                "أنت أقوى مما تعتقد! 💎",
                "كل لحظة تمرين تجعلك أفضل! 📊"
            ]
        }
        
    def start_game_session(self, game_name: str) -> str:
        """بدء جلسة تحليل لعبة جديدة"""
        print(f"🎮 بدء جلسة تحليل للعبة: {game_name}")
        
        self.current_session = GameSession(
            game_name=game_name,
            start_time=datetime.now(),
            player_performance={
                'accuracy': 0.0,
                'reaction_time': 0.0,
                'decision_making': 0.0,
                'consistency': 0.0,
                'improvement_rate': 0.0
            },
            achievements=[],
            areas_for_improvement=[],
            strategies_suggested=[],
            emotional_state='neutral'
        )
        
        # كشف نوع اللعبة
        game_type = self._detect_game_type(game_name)
        print(f"🎯 نوع اللعبة المكتشف: {game_type}")
        
        # تقديم استراتيجيات البداية
        initial_strategies = self._get_initial_strategies(game_type)
        self.current_session.strategies_suggested.extend(initial_strategies)
        
        return f"تم بدء جلسة تحليل {game_name}. أنا جاهز لمساعدتك!"
    
    def analyze_gameplay_screen(self, screen_capture: np.ndarray) -> Dict[str, Any]:
        """تحليل شاشة اللعب في الوقت الفعلي"""
        if not self.current_session:
            return {"error": "لا توجد جلسة نشطة"}
        
        analysis_result = {
            'timestamp': datetime.now().isoformat(),
            'screen_analysis': {},
            'performance_indicators': {},
            'suggestions': [],
            'encouragement': ""
        }
        
        # تحليل محتوى الشاشة
        screen_analysis = self._analyze_screen_content(screen_capture)
        analysis_result['screen_analysis'] = screen_analysis
        
        # تحليل مؤشرات الأداء
        performance = self._analyze_performance_indicators(screen_analysis)
        analysis_result['performance_indicators'] = performance
        
        # تحديث أداء الجلسة
        self._update_session_performance(performance)
        
        # تقديم اقتراحات في الوقت الفعلي
        suggestions = self._generate_real_time_suggestions(screen_analysis, performance)
        analysis_result['suggestions'] = suggestions
        
        # تقديم التشجيع
        encouragement = self._generate_encouragement(performance)
        analysis_result['encouragement'] = encouragement
        
        return analysis_result
    
    def _detect_game_type(self, game_name: str) -> str:
        """كشف نوع اللعبة"""
        game_name_lower = game_name.lower()
        
        for game_type, info in self.game_database.items():
            for keyword in info['keywords']:
                if keyword in game_name_lower:
                    return game_type
        
        return 'general'
    
    def _get_initial_strategies(self, game_type: str) -> List[str]:
        """الحصول على استراتيجيات البداية"""
        if game_type in self.game_database:
            strategies = []
            for category, strategy_list in self.game_database[game_type]['strategies'].items():
                strategies.extend(strategy_list[:2])  # أول استراتيجيتين من كل فئة
            return strategies
        
        return [
            "ركز على تعلم الأساسيات أولاً",
            "مارس بانتظام لتحسين مهاراتك",
            "تعلم من اللاعبين المحترفين"
        ]
    
    def _analyze_screen_content(self, screen: np.ndarray) -> Dict[str, Any]:
        """تحليل محتوى الشاشة"""
        analysis = {
            'ui_elements': [],
            'game_state': 'unknown',
            'health_status': 1.0,
            'score_trend': 'stable',
            'action_density': 0.0,
            'visual_complexity': 0.0
        }
        
        # تحليل الألوان للكشف عن حالة اللعبة
        hsv = cv2.cvtColor(screen, cv2.COLOR_BGR2HSV)
        
        # كشف اللون الأحمر (قد يشير للضرر أو التحذير)
        red_mask = cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
        red_percentage = np.sum(red_mask > 0) / (screen.shape[0] * screen.shape[1])
        
        if red_percentage > 0.1:
            analysis['health_status'] = max(0.0, 1.0 - red_percentage * 2)
            analysis['game_state'] = 'danger'
        
        # تحليل كثافة الحركة
        gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (screen.shape[0] * screen.shape[1])
        analysis['action_density'] = edge_density
        
        # تحليل التعقيد البصري
        analysis['visual_complexity'] = self._calculate_visual_complexity(screen)
        
        return analysis
    
    def _analyze_performance_indicators(self, screen_analysis: Dict) -> Dict[str, float]:
        """تحليل مؤشرات الأداء"""
        performance = {
            'current_focus': 0.8,  # مستوى التركيز الحالي
            'stress_level': 0.0,   # مستوى التوتر
            'efficiency': 0.7,     # كفاءة الأداء
            'adaptability': 0.6    # القدرة على التكيف
        }
        
        # تحليل مستوى التوتر بناءً على حالة اللعبة
        if screen_analysis['game_state'] == 'danger':
            performance['stress_level'] = min(1.0, performance['stress_level'] + 0.3)
        
        # تحليل التركيز بناءً على كثافة الحركة
        action_density = screen_analysis['action_density']
        if action_density > 0.5:
            performance['current_focus'] = max(0.0, performance['current_focus'] - 0.2)
        
        return performance
    
    def _update_session_performance(self, current_performance: Dict[str, float]):
        """تحديث أداء الجلسة"""
        if not self.current_session:
            return
        
        # تحديث مؤشرات الأداء بناءً على القراءات الحالية
        session_perf = self.current_session.player_performance
        
        # حساب متوسط متحرك للأداء
        alpha = 0.3  # معامل التنعيم
        for key in current_performance:
            if key in session_perf:
                session_perf[key] = (alpha * current_performance[key] + 
                                   (1 - alpha) * session_perf[key])
    
    def _generate_real_time_suggestions(self, screen_analysis: Dict, 
                                      performance: Dict) -> List[str]:
        """توليد اقتراحات في الوقت الفعلي"""
        suggestions = []
        
        # اقتراحات بناءً على حالة اللعبة
        if screen_analysis['game_state'] == 'danger':
            suggestions.append("⚠️ احذر! ابحث عن مكان آمن!")
            suggestions.append("🛡️ استخدم العلاج إذا كان متوفراً")
        
        # اقتراحات بناءً على الأداء
        if performance['stress_level'] > 0.7:
            suggestions.append("😌 خذ نفساً عميقاً واهدأ")
            suggestions.append("🧘 ركز على استراتيجيتك")
        
        if performance['current_focus'] < 0.5:
            suggestions.append("👁️ ركز على الهدف الأساسي")
            suggestions.append("🎯 تجنب التشتت")
        
        # اقتراحات خاصة بنوع اللعبة
        if self.current_session:
            game_type = self._detect_game_type(self.current_session.game_name)
            if game_type in self.game_database:
                common_mistakes = self.game_database[game_type].get('common_mistakes', [])
                if common_mistakes and len(suggestions) < 3:
                    suggestions.append(f"💡 تذكر: {common_mistakes[0]}")
        
        return suggestions[:3]  # حد أقصى 3 اقتراحات
    
    def _generate_encouragement(self, performance: Dict) -> str:
        """توليد رسائل التشجيع"""
        avg_performance = np.mean(list(performance.values()))
        
        if avg_performance > 0.8:
            phrases = self.encouragement_phrases['good_performance']
        elif avg_performance > 0.5:
            phrases = self.encouragement_phrases['motivational']
        else:
            phrases = self.encouragement_phrases['needs_improvement']
        
        import random
        return random.choice(phrases)
    
    def get_detailed_strategy(self, area: str) -> Dict[str, Any]:
        """الحصول على استراتيجية مفصلة لمجال معين"""
        if not self.current_session:
            return {"error": "لا توجد جلسة نشطة"}
        
        game_type = self._detect_game_type(self.current_session.game_name)
        
        strategy_guide = {
            'area': area,
            'game_type': game_type,
            'strategies': [],
            'practice_exercises': [],
            'expected_improvement_time': "1-2 أسابيع",
            'difficulty_level': "متوسط"
        }
        
        # استراتيجيات خاصة بنوع اللعبة
        if game_type in self.game_database:
            game_strategies = self.game_database[game_type]['strategies']
            
            if area in game_strategies:
                strategy_guide['strategies'] = game_strategies[area]
            else:
                # إذا لم يوجد المجال، أعط استراتيجيات عامة
                all_strategies = []
                for strategies in game_strategies.values():
                    all_strategies.extend(strategies)
                strategy_guide['strategies'] = all_strategies[:5]
        
        # تمارين التطبيق
        strategy_guide['practice_exercises'] = self._generate_practice_exercises(area, game_type)
        
        return strategy_guide
    
    def _generate_practice_exercises(self, area: str, game_type: str) -> List[str]:
        """توليد تمارين التطبيق"""
        exercises = {
            'aiming': [
                "مارس على أهداف ثابتة لمدة 10 دقائق يومياً",
                "استخدم خرائط التدريب المخصصة",
                "مارس مع حساسيات مختلفة للماوس"
            ],
            'reaction_time': [
                "استخدم تطبيقات تدريب وقت الاستجابة",
                "مارس ألعاب التركيز السريع",
                "تدرب على اتخاذ قرارات سريعة"
            ],
            'strategy': [
                "ادرس استراتيجيات اللاعبين المحترفين",
                "حلل أخطاءك بعد كل مباراة",
                "تدرب على سيناريوهات مختلفة"
            ]
        }
        
        return exercises.get(area, [
            "مارس بانتظام",
            "شاهد فيديوهات تعليمية",
            "تدرب مع لاعبين أفضل منك"
        ])
    
    def end_session_analysis(self) -> Dict[str, Any]:
        """إنهاء الجلسة وتقديم التحليل النهائي"""
        if not self.current_session:
            return {"error": "لا توجد جلسة نشطة"}
        
        session_duration = (datetime.now() - self.current_session.start_time).total_seconds() / 60
        
        final_analysis = {
            'session_summary': {
                'game': self.current_session.game_name,
                'duration_minutes': session_duration,
                'overall_performance': self._calculate_overall_performance(),
                'achievements': self.current_session.achievements,
                'improvement_areas': self.current_session.areas_for_improvement
            },
            'detailed_metrics': dict(self.current_session.player_performance),
            'recommendations': self._generate_final_recommendations(),
            'next_session_goals': self._set_next_session_goals()
        }
        
        # حفظ في التاريخ
        self.performance_history.append(final_analysis)
        
        # إعادة تعيين الجلسة
        self.current_session = None
        
        return final_analysis
    
    def _calculate_overall_performance(self) -> float:
        """حساب الأداء الإجمالي"""
        if not self.current_session:
            return 0.0
        
        performance_values = list(self.current_session.player_performance.values())
        return np.mean(performance_values) if performance_values else 0.0
    
    def _generate_final_recommendations(self) -> List[str]:
        """توليد التوصيات النهائية"""
        recommendations = []
        
        if not self.current_session:
            return recommendations
        
        performance = self.current_session.player_performance
        
        # توصيات بناءً على نقاط الضعف
        if performance.get('accuracy', 0) < 0.6:
            recommendations.append("🎯 ركز على تحسين الدقة من خلال التدريب المنتظم")
        
        if performance.get('reaction_time', 0) < 0.5:
            recommendations.append("⚡ تدرب على تحسين وقت الاستجابة")
        
        if performance.get('decision_making', 0) < 0.6:
            recommendations.append("🧠 طور مهارات اتخاذ القرار السريع")
        
        # توصيات عامة
        recommendations.extend([
            "📚 ادرس استراتيجيات اللاعبين المحترفين",
            "🤝 العب مع فريق لتطوير التعاون",
            "📊 راجع إحصائياتك بانتظام"
        ])
        
        return recommendations[:5]
    
    def _set_next_session_goals(self) -> List[str]:
        """تحديد أهداف الجلسة القادمة"""
        goals = []
        
        if not self.current_session:
            return goals
        
        performance = self.current_session.player_performance
        
        # أهداف بناءً على الأداء الحالي
        lowest_skill = min(performance, key=performance.get)
        goals.append(f"تحسين {lowest_skill} بنسبة 10%")
        
        goals.extend([
            "تطبيق استراتيجية جديدة",
            "تحقيق نتيجة أفضل من الجلسة السابقة",
            "التركيز على التحسن المستمر"
        ])
        
        return goals
    
    def _calculate_visual_complexity(self, screen: np.ndarray) -> float:
        """حساب التعقيد البصري للشاشة"""
        gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        
        # حساب التدرج
        gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        
        # حساب التعقيد كمتوسط قوة التدرج
        complexity = np.mean(gradient_magnitude) / 255.0
        
        return min(1.0, complexity)

# مثال على الاستخدام
if __name__ == "__main__":
    coach = IntelligentGameCoach()
    
    # بدء جلسة
    session_msg = coach.start_game_session("Counter-Strike: Global Offensive")
    print(session_msg)
    
    # محاكاة تحليل اللعب
    print("\n🎮 محاكاة جلسة تحليل...")
    
    # إنشاء شاشة وهمية
    dummy_screen = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    
    # تحليل الأداء
    analysis = coach.analyze_gameplay_screen(dummy_screen)
    print(f"\n📊 تحليل الأداء:")
    print(f"التشجيع: {analysis['encouragement']}")
    print(f"الاقتراحات: {analysis['suggestions']}")
    
    # الحصول على استراتيجية مفصلة
    strategy = coach.get_detailed_strategy("aiming")
    print(f"\n🎯 استراتيجية التصويب:")
    for i, strat in enumerate(strategy['strategies'][:3], 1):
        print(f"{i}. {strat}")
    
    # إنهاء الجلسة
    final_analysis = coach.end_session_analysis()
    print(f"\n📈 التحليل النهائي:")
    print(f"الأداء الإجمالي: {final_analysis['session_summary']['overall_performance']:.2f}")
    print(f"أهداف الجلسة القادمة: {final_analysis['next_session_goals'][:2]}")
