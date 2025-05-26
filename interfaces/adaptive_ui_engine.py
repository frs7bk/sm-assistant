
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎨 محرك الواجهة التكيفية وتجربة المستخدم المتقدمة
Adaptive UI & Advanced User Experience Engine
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import colorsys

@dataclass
class UserPreferences:
    """تفضيلات المستخدم"""
    theme: str = "auto"  # light, dark, auto
    language: str = "ar"
    font_size: str = "medium"  # small, medium, large, extra-large
    animation_speed: str = "normal"  # slow, normal, fast, none
    accessibility_features: List[str] = None
    color_scheme: str = "blue"
    layout_density: str = "comfortable"  # compact, comfortable, spacious
    voice_enabled: bool = True
    notifications_enabled: bool = True
    
    def __post_init__(self):
        if self.accessibility_features is None:
            self.accessibility_features = []

@dataclass
class UIState:
    """حالة الواجهة الحالية"""
    current_view: str
    sidebar_collapsed: bool = False
    chat_mode: str = "normal"  # normal, focused, minimal
    active_widgets: List[str] = None
    window_size: Dict[str, int] = None
    last_interaction: datetime = None
    
    def __post_init__(self):
        if self.active_widgets is None:
            self.active_widgets = []
        if self.window_size is None:
            self.window_size = {"width": 1200, "height": 800}
        if self.last_interaction is None:
            self.last_interaction = datetime.now()

@dataclass
class InteractionPattern:
    """نمط تفاعل المستخدم"""
    action_type: str
    frequency: int
    time_of_day: str
    context: Dict[str, Any]
    success_rate: float

class AdaptiveThemeEngine:
    """محرك الثيمات التكيفية"""
    
    def __init__(self):
        self.themes = {
            "light": {
                "primary": "#667eea",
                "secondary": "#764ba2", 
                "background": "#ffffff",
                "surface": "#f8f9fa",
                "text": "#333333",
                "accent": "#ff6b6b"
            },
            "dark": {
                "primary": "#bb86fc",
                "secondary": "#03dac6",
                "background": "#121212",
                "surface": "#1e1e1e",
                "text": "#ffffff",
                "accent": "#cf6679"
            },
            "high_contrast": {
                "primary": "#ffffff",
                "secondary": "#000000",
                "background": "#000000",
                "surface": "#333333",
                "text": "#ffffff",
                "accent": "#ffff00"
            }
        }
        
        self.color_schemes = {
            "blue": {"hue": 220, "saturation": 0.7},
            "green": {"hue": 120, "saturation": 0.6},
            "purple": {"hue": 280, "saturation": 0.8},
            "orange": {"hue": 30, "saturation": 0.9},
            "pink": {"hue": 330, "saturation": 0.7}
        }
    
    def generate_adaptive_theme(self, preferences: UserPreferences, context: Dict[str, Any]) -> Dict[str, Any]:
        """توليد ثيم تكيفي"""
        base_theme = preferences.theme
        
        # التكيف حسب الوقت
        current_hour = datetime.now().hour
        if base_theme == "auto":
            if 6 <= current_hour <= 18:
                base_theme = "light"
            else:
                base_theme = "dark"
        
        # تطبيق مخطط الألوان المفضل
        theme_colors = self.themes[base_theme].copy()
        color_scheme = self.color_schemes.get(preferences.color_scheme, self.color_schemes["blue"])
        
        # تعديل الألوان الأساسية
        primary_rgb = self._hsl_to_rgb(color_scheme["hue"], color_scheme["saturation"], 0.5)
        theme_colors["primary"] = self._rgb_to_hex(primary_rgb)
        
        # إضافة متغيرات CSS
        css_variables = {}
        for key, value in theme_colors.items():
            css_variables[f"--color-{key}"] = value
        
        # إعدادات الخط
        font_sizes = {
            "small": "0.875rem",
            "medium": "1rem", 
            "large": "1.125rem",
            "extra-large": "1.25rem"
        }
        css_variables["--font-size-base"] = font_sizes[preferences.font_size]
        
        # إعدادات الحركة
        animation_speeds = {
            "slow": "0.5s",
            "normal": "0.3s",
            "fast": "0.1s",
            "none": "0s"
        }
        css_variables["--animation-duration"] = animation_speeds[preferences.animation_speed]
        
        return {
            "theme_name": f"{base_theme}_{preferences.color_scheme}",
            "colors": theme_colors,
            "css_variables": css_variables,
            "accessibility_features": preferences.accessibility_features
        }
    
    def _hsl_to_rgb(self, h: float, s: float, l: float) -> tuple:
        """تحويل HSL إلى RGB"""
        h = h / 360.0
        r, g, b = colorsys.hls_to_rgb(h, l, s)
        return (int(r * 255), int(g * 255), int(b * 255))
    
    def _rgb_to_hex(self, rgb: tuple) -> str:
        """تحويل RGB إلى HEX"""
        return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"

class UIPersonalizationEngine:
    """محرك تخصيص الواجهة الذكي"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.interaction_patterns: Dict[str, List[InteractionPattern]] = {}
        self.user_preferences: Dict[str, UserPreferences] = {}
        self.ui_states: Dict[str, UIState] = {}
        
    def learn_from_interaction(self, user_id: str, action: str, context: Dict[str, Any], success: bool):
        """التعلم من تفاعلات المستخدم"""
        if user_id not in self.interaction_patterns:
            self.interaction_patterns[user_id] = []
        
        current_hour = datetime.now().hour
        time_of_day = "morning" if 6 <= current_hour < 12 else \
                     "afternoon" if 12 <= current_hour < 18 else \
                     "evening" if 18 <= current_hour < 22 else "night"
        
        # البحث عن نمط مشابه
        existing_pattern = None
        for pattern in self.interaction_patterns[user_id]:
            if (pattern.action_type == action and 
                pattern.time_of_day == time_of_day and
                pattern.context.get("view") == context.get("view")):
                existing_pattern = pattern
                break
        
        if existing_pattern:
            # تحديث النمط الموجود
            existing_pattern.frequency += 1
            existing_pattern.success_rate = (
                (existing_pattern.success_rate * (existing_pattern.frequency - 1) + (1 if success else 0)) /
                existing_pattern.frequency
            )
        else:
            # إنشاء نمط جديد
            new_pattern = InteractionPattern(
                action_type=action,
                frequency=1,
                time_of_day=time_of_day,
                context=context.copy(),
                success_rate=1.0 if success else 0.0
            )
            self.interaction_patterns[user_id].append(new_pattern)
    
    def get_personalized_layout(self, user_id: str, current_context: Dict[str, Any]) -> Dict[str, Any]:
        """الحصول على تخطيط مخصص للمستخدم"""
        if user_id not in self.interaction_patterns:
            return self._get_default_layout()
        
        patterns = self.interaction_patterns[user_id]
        current_hour = datetime.now().hour
        time_of_day = "morning" if 6 <= current_hour < 12 else \
                     "afternoon" if 12 <= current_hour < 18 else \
                     "evening" if 18 <= current_hour < 22 else "night"
        
        # العثور على الأنماط الأكثر صلة
        relevant_patterns = [
            p for p in patterns 
            if p.time_of_day == time_of_day and p.success_rate > 0.7
        ]
        
        # ترتيب حسب التكرار
        relevant_patterns.sort(key=lambda x: x.frequency, reverse=True)
        
        # توليد التخطيط المخصص
        layout = {
            "sidebar_position": "right" if any(p.action_type == "sidebar_right" for p in relevant_patterns[:3]) else "left",
            "chat_position": "center",
            "quick_actions": self._get_frequent_actions(relevant_patterns[:5]),
            "widget_priority": self._get_widget_priorities(relevant_patterns),
            "suggested_shortcuts": self._get_suggested_shortcuts(relevant_patterns)
        }
        
        return layout
    
    def _get_default_layout(self) -> Dict[str, Any]:
        """التخطيط الافتراضي"""
        return {
            "sidebar_position": "right",
            "chat_position": "center", 
            "quick_actions": ["مساعدة", "إعدادات", "إحصائيات"],
            "widget_priority": ["chat", "suggestions", "status"],
            "suggested_shortcuts": ["Ctrl+Enter", "Ctrl+/", "Ctrl+K"]
        }
    
    def _get_frequent_actions(self, patterns: List[InteractionPattern]) -> List[str]:
        """الحصول على الأعمال الأكثر تكراراً"""
        action_counts = {}
        for pattern in patterns:
            action = pattern.action_type
            action_counts[action] = action_counts.get(action, 0) + pattern.frequency
        
        sorted_actions = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)
        return [action for action, _ in sorted_actions[:5]]
    
    def _get_widget_priorities(self, patterns: List[InteractionPattern]) -> List[str]:
        """أولويات العناصر"""
        widget_usage = {}
        for pattern in patterns:
            widget = pattern.context.get("widget", "unknown")
            widget_usage[widget] = widget_usage.get(widget, 0) + pattern.frequency
        
        sorted_widgets = sorted(widget_usage.items(), key=lambda x: x[1], reverse=True)
        return [widget for widget, _ in sorted_widgets if widget != "unknown"]
    
    def _get_suggested_shortcuts(self, patterns: List[InteractionPattern]) -> List[str]:
        """اقتراح اختصارات مفيدة"""
        shortcuts = ["Ctrl+Enter: إرسال", "Ctrl+/: مساعدة", "Ctrl+K: بحث سريع"]
        
        # إضافة اختصارات مخصصة بناءً على الاستخدام
        frequent_actions = self._get_frequent_actions(patterns)
        if "voice_command" in frequent_actions:
            shortcuts.append("Space: تسجيل صوتي")
        if "image_upload" in frequent_actions:
            shortcuts.append("Ctrl+U: رفع صورة")
        
        return shortcuts

class QuickSetupWizard:
    """معالج الإعداد السريع"""
    
    def __init__(self):
        self.setup_steps = [
            {
                "id": "welcome",
                "title": "مرحباً بك!",
                "description": "دعنا نقوم بإعداد المساعد الذكي حسب تفضيلاتك",
                "type": "info"
            },
            {
                "id": "language",
                "title": "اللغة المفضلة",
                "description": "اختر لغة التفاعل المفضلة",
                "type": "choice",
                "options": ["العربية", "English", "Français"]
            },
            {
                "id": "theme",
                "title": "المظهر",
                "description": "اختر المظهر المناسب لك",
                "type": "choice",
                "options": ["فاتح", "داكن", "تلقائي"]
            },
            {
                "id": "accessibility",
                "title": "ميزات الوصول",
                "description": "فعل الميزات التي تحتاجها",
                "type": "multiple_choice",
                "options": ["خط كبير", "تباين عالي", "قارئ الشاشة", "تشغيل بالصوت"]
            },
            {
                "id": "usage_type", 
                "title": "نوع الاستخدام",
                "description": "كيف تخطط لاستخدام المساعد؟",
                "type": "choice",
                "options": ["شخصي", "عمل", "تعليم", "ترفيه"]
            },
            {
                "id": "complete",
                "title": "تم الإعداد!",
                "description": "المساعد الذكي جاهز للاستخدام حسب تفضيلاتك",
                "type": "completion"
            }
        ]
    
    def get_setup_progress(self, completed_steps: List[str]) -> Dict[str, Any]:
        """تقدم الإعداد"""
        total_steps = len(self.setup_steps)
        completed_count = len(completed_steps)
        progress_percentage = (completed_count / total_steps) * 100
        
        current_step = None
        for step in self.setup_steps:
            if step["id"] not in completed_steps:
                current_step = step
                break
        
        return {
            "progress_percentage": progress_percentage,
            "completed_steps": completed_count,
            "total_steps": total_steps,
            "current_step": current_step,
            "remaining_steps": total_steps - completed_count
        }
    
    def apply_setup_choices(self, choices: Dict[str, Any]) -> UserPreferences:
        """تطبيق خيارات الإعداد"""
        preferences = UserPreferences()
        
        # تحويل الخيارات إلى تفضيلات
        if "language" in choices:
            lang_map = {"العربية": "ar", "English": "en", "Français": "fr"}
            preferences.language = lang_map.get(choices["language"], "ar")
        
        if "theme" in choices:
            theme_map = {"فاتح": "light", "داكن": "dark", "تلقائي": "auto"}
            preferences.theme = theme_map.get(choices["theme"], "auto")
        
        if "accessibility" in choices:
            accessibility_map = {
                "خط كبير": "large_font",
                "تباين عالي": "high_contrast",
                "قارئ الشاشة": "screen_reader",
                "تشغيل بالصوت": "voice_control"
            }
            preferences.accessibility_features = [
                accessibility_map[feature] for feature in choices["accessibility"]
                if feature in accessibility_map
            ]
            
            if "large_font" in preferences.accessibility_features:
                preferences.font_size = "large"
        
        return preferences

class AdaptiveUIEngine:
    """محرك الواجهة التكيفية الرئيسي"""
    
    def __init__(self):
        self.theme_engine = AdaptiveThemeEngine()
        self.personalization_engine = UIPersonalizationEngine()
        self.setup_wizard = QuickSetupWizard()
        self.logger = logging.getLogger(__name__)
        
        # مسار ملفات المستخدمين
        self.user_data_path = Path("data/user_preferences")
        self.user_data_path.mkdir(parents=True, exist_ok=True)
    
    async def initialize_user_session(self, user_id: str) -> Dict[str, Any]:
        """تهيئة جلسة المستخدم"""
        # تحميل التفضيلات المحفوظة
        preferences = await self._load_user_preferences(user_id)
        
        # توليد الثيم التكيفي
        context = {"time": datetime.now().hour, "device": "desktop"}
        theme = self.theme_engine.generate_adaptive_theme(preferences, context)
        
        # الحصول على التخطيط المخصص
        layout = self.personalization_engine.get_personalized_layout(user_id, context)
        
        # إنشاء حالة الواجهة
        ui_state = UIState(
            current_view="main",
            chat_mode="normal",
            active_widgets=["chat", "sidebar", "status"]
        )
        
        self.personalization_engine.ui_states[user_id] = ui_state
        
        return {
            "user_id": user_id,
            "preferences": asdict(preferences),
            "theme": theme,
            "layout": layout,
            "ui_state": asdict(ui_state),
            "setup_required": len(preferences.accessibility_features) == 0 and preferences.theme == "auto"
        }
    
    async def update_user_interaction(self, user_id: str, action: str, context: Dict[str, Any], success: bool = True):
        """تحديث نمط تفاعل المستخدم"""
        self.personalization_engine.learn_from_interaction(user_id, action, context, success)
        
        # حفظ التفضيلات المحدثة
        if user_id in self.personalization_engine.user_preferences:
            await self._save_user_preferences(user_id, self.personalization_engine.user_preferences[user_id])
    
    async def get_adaptive_suggestions(self, user_id: str, current_context: Dict[str, Any]) -> List[str]:
        """اقتراحات تكيفية للمستخدم"""
        if user_id not in self.personalization_engine.interaction_patterns:
            return [
                "جرب قول 'مساعدة' للحصول على الأوامر",
                "يمكنك رفع صورة للتحليل",
                "استخدم الأوامر الصوتية للتفاعل السريع"
            ]
        
        patterns = self.personalization_engine.interaction_patterns[user_id]
        current_hour = datetime.now().hour
        
        # اقتراحات مخصصة حسب الوقت والسياق
        suggestions = []
        
        if 9 <= current_hour <= 17:  # ساعات العمل
            work_patterns = [p for p in patterns if "work" in str(p.context)]
            if work_patterns:
                suggestions.append("هل تريد مراجعة مهام اليوم؟")
                suggestions.append("يمكنني مساعدتك في تنظيم جدولك")
        
        elif 18 <= current_hour <= 22:  # المساء
            suggestions.append("كيف كان يومك؟ هل تريد ملخصاً؟")
            suggestions.append("وقت الاسترخاء - هل تريد اقتراحات ترفيه؟")
        
        # إضافة اقتراحات عامة مخصصة
        frequent_actions = self.personalization_engine._get_frequent_actions(patterns[:10])
        if "voice_command" in frequent_actions:
            suggestions.append("جرب الأوامر الصوتية للتفاعل الأسرع")
        
        return suggestions[:3]  # أقصى 3 اقتراحات
    
    async def _load_user_preferences(self, user_id: str) -> UserPreferences:
        """تحميل تفضيلات المستخدم"""
        prefs_file = self.user_data_path / f"{user_id}_preferences.json"
        
        try:
            if prefs_file.exists():
                with open(prefs_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return UserPreferences(**data)
        except Exception as e:
            self.logger.warning(f"فشل تحميل تفضيلات {user_id}: {e}")
        
        # إرجاع تفضيلات افتراضية
        preferences = UserPreferences()
        self.personalization_engine.user_preferences[user_id] = preferences
        return preferences
    
    async def _save_user_preferences(self, user_id: str, preferences: UserPreferences):
        """حفظ تفضيلات المستخدم"""
        prefs_file = self.user_data_path / f"{user_id}_preferences.json"
        
        try:
            with open(prefs_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(preferences), f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"فشل حفظ تفضيلات {user_id}: {e}")
    
    def get_ui_analytics(self, user_id: str) -> Dict[str, Any]:
        """تحليلات واجهة المستخدم"""
        if user_id not in self.personalization_engine.interaction_patterns:
            return {"message": "لا توجد بيانات كافية"}
        
        patterns = self.personalization_engine.interaction_patterns[user_id]
        
        # تحليل أوقات الاستخدام
        time_usage = {}
        for pattern in patterns:
            time_usage[pattern.time_of_day] = time_usage.get(pattern.time_of_day, 0) + pattern.frequency
        
        # الإجراءات الأكثر استخداماً
        action_usage = {}
        for pattern in patterns:
            action_usage[pattern.action_type] = action_usage.get(pattern.action_type, 0) + pattern.frequency
        
        return {
            "total_interactions": sum(p.frequency for p in patterns),
            "most_active_time": max(time_usage, key=time_usage.get) if time_usage else "غير محدد",
            "favorite_actions": sorted(action_usage.items(), key=lambda x: x[1], reverse=True)[:5],
            "average_success_rate": sum(p.success_rate for p in patterns) / len(patterns),
            "personalization_level": min(len(patterns) / 50, 1.0) * 100  # نسبة التخصيص
        }

# مثيل عام للاستخدام
adaptive_ui_engine = AdaptiveUIEngine()
