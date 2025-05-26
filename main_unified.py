
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
نقطة الدخول الموحدة للمساعد الذكي المتقدم
"""

import logging
import asyncio
import sys
import os
from pathlib import Path

# إضافة مسار المشروع
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from modules.intent_context import IntentClassifier, ContextMemory
from modules.user_manager import UserManager
from modules.background_tasks import BackgroundTaskManager
from modules.vision.recognition_3d.recognition_3d_engine import VisionIntelligenceEngine
from modules.voice_emotion.emotion_recognizer import EmotionRecognizer
from modules.productivity import *
from modules.security.smart_security import SecurityMonitor
from modules.analytics.behavior_predictor import BehaviorPredictor
from modules.reminder_scheduler import ReminderScheduler

class UnifiedAdvancedAssistant:
    """المساعد الذكي المتقدم الموحد"""
    
    def __init__(self):
        """تهيئة المساعد بجميع الوحدات المتقدمة"""
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # الوحدات الأساسية
        self.intent_classifier = IntentClassifier()
        self.context_memory = ContextMemory()
        self.user_manager = UserManager()
        self.security_monitor = SecurityMonitor()
        
        # الوحدات المتقدمة
        self.vision_engine = VisionIntelligenceEngine()
        self.emotion_recognizer = EmotionRecognizer()
        self.behavior_predictor = BehaviorPredictor()
        self.reminder_scheduler = ReminderScheduler()
        self.background_tasks = BackgroundTaskManager()
        
        # حالة المساعد
        self.current_user = None
        self.session_active = False
        
        self.logger.info("تم تهيئة المساعد الذكي المتقدم بنجاح")
    
    def setup_logging(self):
        """إعداد نظام السجلات"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('assistant.log', encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    async def process_command(self, command: str, user_id: str = None) -> dict:
        """معالجة الأوامر بطريقة متقدمة"""
        try:
            # فحص الأمان
            if self.security_monitor.is_suspicious(command):
                self.logger.warning(f"أمر مشبوه تم اكتشافه: {command}")
                return {
                    "status": "blocked",
                    "message": "تم حجب الأمر لأسباب أمنية",
                    "command": command
                }
            
            # تصنيف القصد
            intent = self.intent_classifier.classify(command)
            self.logger.info(f"تم تصنيف القصد: {intent}")
            
            # تحديث السياق
            if user_id:
                self.context_memory.update_context(user_id, "last_command", command)
                self.context_memory.update_context(user_id, "last_intent", intent)
            
            # معالجة الأوامر المختلفة
            if intent == "reminder":
                return await self.handle_reminder(command, user_id)
            elif intent == "focus_mode":
                return await self.handle_focus_mode(user_id)
            elif intent == "analyze_emotion":
                return await self.handle_emotion_analysis(user_id)
            elif intent == "productivity":
                return await self.handle_productivity_task(command, user_id)
            else:
                return await self.handle_general_query(command, user_id)
                
        except Exception as e:
            self.logger.error(f"خطأ في معالجة الأمر: {str(e)}")
            return {
                "status": "error",
                "message": f"حدث خطأ أثناء معالجة الأمر: {str(e)}",
                "command": command
            }
    
    async def handle_reminder(self, command: str, user_id: str) -> dict:
        """معالجة أوامر التذكير"""
        # استخراج التوقيت والرسالة من الأمر
        # هذا مثال بسيط - يحتاج لمعالجة NLP أكثر تطوراً
        if "في" in command:
            parts = command.split("في")
            if len(parts) >= 2:
                reminder_text = parts[0].replace("ذكرني", "").strip()
                time_text = parts[1].strip()
                
                # إضافة التذكير
                self.reminder_scheduler.add_reminder(time_text, reminder_text)
                
                return {
                    "status": "success",
                    "message": f"تم إضافة التذكير: {reminder_text} في {time_text}",
                    "reminder": reminder_text,
                    "time": time_text
                }
        
        return {
            "status": "error",
            "message": "لم أتمكن من فهم صيغة التذكير. حاول: 'ذكرني بالاجتماع في 15:00'"
        }
    
    async def handle_focus_mode(self, user_id: str) -> dict:
        """تفعيل وضع التركيز"""
        self.context_memory.update_context(user_id, "focus_mode", True)
        
        return {
            "status": "success",
            "message": "تم تفعيل وضع التركيز. سأقلل من الإشعارات والمقاطعات.",
            "mode": "focus_activated"
        }
    
    async def handle_emotion_analysis(self, user_id: str) -> dict:
        """تحليل المشاعر الحالية للمستخدم"""
        try:
            # هذا يحتاج لكاميرا أو صوت فعلي
            # في الوقت الحالي سنستخدم بيانات وهمية
            analysis_result = {
                "visual_emotion": "محايد",
                "confidence": 0.8,
                "suggestions": [
                    "يبدو أنك في حالة محايدة",
                    "هل تحتاج مساعدة في شيء معين؟"
                ]
            }
            
            return {
                "status": "success",
                "message": "تم تحليل المشاعر",
                "analysis": analysis_result
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"فشل تحليل المشاعر: {str(e)}"
            }
    
    async def handle_productivity_task(self, command: str, user_id: str) -> dict:
        """معالجة مهام الإنتاجية"""
        if "slack" in command.lower():
            # مثال لإرسال رسالة Slack
            return {
                "status": "info",
                "message": "يحتاج إعداد مفاتيح Slack API لتفعيل هذه الميزة"
            }
        elif "trello" in command.lower():
            return {
                "status": "info", 
                "message": "يحتاج إعداد مفاتيح Trello API لتفعيل هذه الميزة"
            }
        else:
            return {
                "status": "info",
                "message": "الميزات الإنتاجية المتاحة: Slack, Trello, Jira"
            }
    
    async def handle_general_query(self, command: str, user_id: str) -> dict:
        """معالجة الاستفسارات العامة"""
        return {
            "status": "success",
            "message": f"تم استلام استفسارك: {command}",
            "suggestion": "يمكنني مساعدتك في التذكيرات، تحليل المشاعر، والمهام الإنتاجية"
        }
    
    async def start_session(self, user_id: str = None):
        """بدء جلسة مع المساعد"""
        self.session_active = True
        self.current_user = user_id
        
        if user_id:
            self.logger.info(f"بدء جلسة للمستخدم: {user_id}")
        
        # بدء المهام الخلفية
        self.reminder_scheduler.run_scheduler()
        
        print("🤖 مرحباً! أنا مساعدك الذكي المتقدم")
        print("📝 يمكنني مساعدتك في:")
        print("   • إدارة التذكيرات")
        print("   • تحليل المشاعر") 
        print("   • المهام الإنتاجية")
        print("   • وضع التركيز")
        print("   • والمزيد...")
        print("\n💬 اكتب أمرك أو 'exit' للخروج")
        
        return {
            "status": "session_started",
            "user_id": user_id,
            "features": [
                "reminders", "emotion_analysis", 
                "productivity", "focus_mode"
            ]
        }
    
    async def run_interactive_session(self):
        """تشغيل جلسة تفاعلية"""
        await self.start_session()
        
        try:
            while self.session_active:
                user_input = input("\n👤 أدخل أمرك: ").strip()
                
                if user_input.lower() in ['exit', 'خروج', 'quit']:
                    break
                
                if not user_input:
                    continue
                
                result = await self.process_command(user_input, self.current_user)
                
                print(f"\n🤖 {result.get('message', 'تم معالجة الأمر')}")
                
                if result.get('status') == 'error':
                    print(f"❌ خطأ: {result.get('message')}")
                elif result.get('status') == 'success':
                    print(f"✅ {result.get('message')}")
                
        except KeyboardInterrupt:
            print("\n\n👋 وداعاً!")
        except Exception as e:
            self.logger.error(f"خطأ في الجلسة التفاعلية: {str(e)}")
        finally:
            self.session_active = False

async def main():
    """الدالة الرئيسية"""
    assistant = UnifiedAdvancedAssistant()
    await assistant.run_interactive_session()

if __name__ == "__main__":
    asyncio.run(main())
