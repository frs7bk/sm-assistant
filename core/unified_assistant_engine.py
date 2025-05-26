
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
محرك المساعد الذكي الموحد
يدمج جميع الوحدات والميزات في نظام واحد متقدم
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
from datetime import datetime
import threading
from dataclasses import dataclass
import queue
import sys

# إضافة مسار المشروع
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from core.advanced_ai_engine import get_ai_engine, AIResponse
    from core.module_manager import get_module_manager
    AI_ENGINE_AVAILABLE = True
except ImportError:
    AI_ENGINE_AVAILABLE = False

@dataclass
class ConversationTurn:
    """دورة المحادثة"""
    timestamp: datetime
    user_input: str
    assistant_response: str
    confidence: float
    context: Dict[str, Any]
    metadata: Dict[str, Any]

class UnifiedAssistantEngine:
    """محرك المساعد الذكي الموحد"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # حالة المحرك
        self.is_running = False
        self.is_initialized = False
        
        # المكونات الأساسية
        self.ai_engine = None
        self.module_manager = None
        
        # سياق المحادثة
        self.conversation_history = []
        self.current_session = {
            "start_time": datetime.now(),
            "user_id": "default",
            "context": {},
            "turn_count": 0
        }
        
        # الواجهات النشطة
        self.active_interfaces = {
            "text": True,
            "voice": False,
            "web": False,
            "api": False
        }
        
        # إحصائيات الجلسة
        self.session_stats = {
            "total_interactions": 0,
            "successful_responses": 0,
            "error_count": 0,
            "avg_confidence": 0.0,
            "session_duration": 0.0
        }
        
        # قائمة انتظار المهام
        self.task_queue = queue.Queue()
        self.workers = []
        
    async def initialize(self):
        """تهيئة محرك المساعد"""
        self.logger.info("🚀 تهيئة محرك المساعد الموحد...")
        
        try:
            # تهيئة محرك الذكاء الاصطناعي
            if AI_ENGINE_AVAILABLE:
                self.ai_engine = await get_ai_engine()
                self.logger.info("✅ تم تهيئة محرك الذكاء الاصطناعي")
            else:
                self.logger.warning("⚠️ محرك الذكاء الاصطناعي غير متاح")
            
            # تهيئة مدير الوحدات
            self.module_manager = get_module_manager()
            if not self.module_manager:
                self.logger.warning("⚠️ مدير الوحدات غير متاح")
            
            # تهيئة العمال للمعالجة المتوازية
            self._initialize_workers()
            
            self.is_initialized = True
            self.logger.info("✅ تم تهيئة محرك المساعد بنجاح")
            
        except Exception as e:
            self.logger.error(f"❌ فشل تهيئة محرك المساعد: {e}")
            raise
    
    def _initialize_workers(self):
        """تهيئة العمال للمعالجة المتوازية"""
        num_workers = 2
        
        for i in range(num_workers):
            worker = threading.Thread(
                target=self._worker_thread,
                name=f"AssistantWorker-{i}",
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
        
        self.logger.info(f"تم تشغيل {num_workers} عامل للمعالجة")
    
    def _worker_thread(self):
        """خيط العامل للمعالجة"""
        while True:
            try:
                task = self.task_queue.get(timeout=1)
                if task is None:  # إشارة الإيقاف
                    break
                
                # تنفيذ المهمة
                task_func, args, kwargs = task
                task_func(*args, **kwargs)
                
                self.task_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"خطأ في العامل: {e}")
    
    async def process_input(
        self, 
        user_input: str, 
        input_type: str = "text",
        user_id: str = "default",
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """معالجة إدخال المستخدم"""
        
        start_time = time.time()
        
        try:
            self.logger.info(f"📝 معالجة الإدخال: {user_input[:50]}...")
            
            # تحديث سياق الجلسة
            self.current_session["user_id"] = user_id
            self.current_session["turn_count"] += 1
            
            if context:
                self.current_session["context"].update(context)
            
            # معالجة الإدخال حسب النوع
            if input_type == "text":
                response = await self._process_text_input(user_input, user_id)
            elif input_type == "voice":
                response = await self._process_voice_input(user_input, user_id)
            elif input_type == "image":
                response = await self._process_image_input(user_input, user_id)
            else:
                response = await self._process_generic_input(user_input, user_id)
            
            # إنشاء دورة المحادثة
            turn = ConversationTurn(
                timestamp=datetime.now(),
                user_input=user_input,
                assistant_response=response.get("text", ""),
                confidence=response.get("confidence", 0.0),
                context=self.current_session["context"].copy(),
                metadata={
                    "input_type": input_type,
                    "processing_time": time.time() - start_time,
                    "user_id": user_id
                }
            )
            
            # إضافة إلى التاريخ
            self.conversation_history.append(turn)
            
            # تحديث الإحصائيات
            self._update_session_stats(turn)
            
            # إضافة معلومات إضافية للاستجابة
            response.update({
                "turn_id": len(self.conversation_history),
                "session_info": {
                    "turn_count": self.current_session["turn_count"],
                    "session_duration": (datetime.now() - self.current_session["start_time"]).total_seconds()
                },
                "processing_time": time.time() - start_time
            })
            
            return response
            
        except Exception as e:
            self.logger.error(f"❌ خطأ في معالجة الإدخال: {e}")
            self.session_stats["error_count"] += 1
            
            return {
                "text": "عذراً، حدث خطأ في المعالجة. يرجى المحاولة مرة أخرى.",
                "confidence": 0.0,
                "intent": "error",
                "suggestions": ["إعادة المحاولة", "تبسيط السؤال"],
                "error": str(e)
            }
    
    async def _process_text_input(self, text: str, user_id: str) -> Dict[str, Any]:
        """معالجة النص"""
        
        if self.ai_engine:
            # استخدام محرك الذكاء الاصطناعي المتقدم
            ai_response = await self.ai_engine.process_natural_language(
                text, user_id, self.current_session["context"]
            )
            
            return {
                "text": ai_response.text,
                "confidence": ai_response.confidence,
                "intent": ai_response.intent,
                "emotions": ai_response.emotions,
                "entities": ai_response.entities,
                "suggestions": ai_response.suggestions,
                "metadata": ai_response.metadata
            }
        else:
            # معالجة أساسية
            return await self._basic_text_processing(text)
    
    async def _basic_text_processing(self, text: str) -> Dict[str, Any]:
        """معالجة نصية أساسية"""
        
        # تحليل أساسي للنص
        text_lower = text.lower()
        
        # كشف القصد الأساسي
        if any(word in text_lower for word in ["مرحبا", "أهلا", "سلام"]):
            intent = "greeting"
            response = "أهلاً وسهلاً! كيف يمكنني مساعدتك؟"
            confidence = 0.9
        elif any(word in text_lower for word in ["شكرا", "متشكر", "أشكرك"]):
            intent = "thanks"
            response = "عفواً! أسعدني أن أساعدك."
            confidence = 0.9
        elif any(word in text_lower for word in ["وداعا", "مع السلامة"]):
            intent = "goodbye"
            response = "وداعاً! أتمنى لك يوماً سعيداً."
            confidence = 0.9
        elif "؟" in text or any(word in text_lower for word in ["كيف", "ماذا", "متى", "أين"]):
            intent = "question"
            response = "سؤال جيد! أحاول أن أجد أفضل إجابة لك..."
            confidence = 0.7
        else:
            intent = "general"
            response = "أفهم ما تقوله. كيف يمكنني مساعدتك بشكل أفضل؟"
            confidence = 0.5
        
        return {
            "text": response,
            "confidence": confidence,
            "intent": intent,
            "emotions": {"neutral": 1.0},
            "entities": [],
            "suggestions": [
                "هل تريد المزيد من المساعدة؟",
                "هل لديك أسئلة أخرى؟"
            ]
        }
    
    async def _process_voice_input(self, audio_data: str, user_id: str) -> Dict[str, Any]:
        """معالجة الصوت"""
        # مؤقتاً نعامل الصوت كنص
        return await self._process_text_input(audio_data, user_id)
    
    async def _process_image_input(self, image_path: str, user_id: str) -> Dict[str, Any]:
        """معالجة الصور"""
        
        if self.ai_engine:
            try:
                analysis = await self.ai_engine.analyze_image(image_path)
                
                if "error" in analysis:
                    return {
                        "text": f"عذراً، لم أستطع تحليل الصورة: {analysis['error']}",
                        "confidence": 0.0,
                        "intent": "error"
                    }
                
                faces_count = analysis.get("faces_detected", 0)
                
                if faces_count > 0:
                    response = f"أرى {faces_count} وجه في الصورة."
                else:
                    response = "لا أرى وجوه في هذه الصورة."
                
                return {
                    "text": response,
                    "confidence": 0.8,
                    "intent": "image_analysis",
                    "analysis_results": analysis
                }
                
            except Exception as e:
                return {
                    "text": f"خطأ في تحليل الصورة: {str(e)}",
                    "confidence": 0.0,
                    "intent": "error"
                }
        else:
            return {
                "text": "عذراً، تحليل الصور غير متاح حالياً.",
                "confidence": 0.0,
                "intent": "unavailable"
            }
    
    async def _process_generic_input(self, input_data: str, user_id: str) -> Dict[str, Any]:
        """معالجة عامة للإدخال"""
        return await self._process_text_input(input_data, user_id)
    
    def _update_session_stats(self, turn: ConversationTurn):
        """تحديث إحصائيات الجلسة"""
        self.session_stats["total_interactions"] += 1
        
        if turn.confidence > 0.5:
            self.session_stats["successful_responses"] += 1
        
        # حساب متوسط الثقة
        total = self.session_stats["total_interactions"]
        current_avg = self.session_stats["avg_confidence"]
        new_avg = (current_avg * (total - 1) + turn.confidence) / total
        self.session_stats["avg_confidence"] = new_avg
        
        # مدة الجلسة
        self.session_stats["session_duration"] = (
            datetime.now() - self.current_session["start_time"]
        ).total_seconds()
    
    async def start_interactive_session(self):
        """بدء جلسة تفاعلية"""
        self.logger.info("🎯 بدء الجلسة التفاعلية")
        
        if not self.is_initialized:
            await self.initialize()
        
        self.is_running = True
        
        print("\n" + "="*60)
        print("🤖 أهلاً بك في المساعد الذكي الموحد!")
        print("="*60)
        print("💡 نصائح:")
        print("   • اكتب 'خروج' أو 'quit' للخروج")
        print("   • اكتب 'إحصائيات' لعرض إحصائيات الجلسة")
        print("   • اكتب 'مساعدة' لعرض الأوامر المتاحة")
        print("="*60)
        
        while self.is_running:
            try:
                # الحصول على إدخال المستخدم
                user_input = input("\n👤 أنت: ").strip()
                
                if not user_input:
                    continue
                
                # أوامر خاصة
                if user_input.lower() in ['خروج', 'quit', 'exit']:
                    await self._handle_exit()
                    break
                
                elif user_input.lower() in ['إحصائيات', 'stats']:
                    self._display_session_stats()
                    continue
                
                elif user_input.lower() in ['مساعدة', 'help']:
                    self._display_help()
                    continue
                
                elif user_input.lower() in ['تنظيف', 'clear']:
                    self._clear_conversation()
                    continue
                
                # معالجة الإدخال
                print("🤖 المساعد: يفكر...")
                
                response = await self.process_input(user_input)
                
                # عرض الاستجابة
                print(f"\n🤖 المساعد: {response['text']}")
                
                # عرض معلومات إضافية إذا كانت متاحة
                if response.get('confidence', 0) < 0.7:
                    print(f"   💭 (مستوى الثقة: {response['confidence']:.1%})")
                
                if response.get('suggestions'):
                    print("   💡 اقتراحات:")
                    for suggestion in response['suggestions'][:2]:
                        print(f"      • {suggestion}")
                
            except KeyboardInterrupt:
                await self._handle_exit()
                break
            except Exception as e:
                self.logger.error(f"خطأ في الجلسة التفاعلية: {e}")
                print(f"\n❌ حدث خطأ: {str(e)}")
    
    def _display_session_stats(self):
        """عرض إحصائيات الجلسة"""
        print("\n📊 إحصائيات الجلسة:")
        print(f"   • إجمالي التفاعلات: {self.session_stats['total_interactions']}")
        print(f"   • الاستجابات الناجحة: {self.session_stats['successful_responses']}")
        print(f"   • متوسط الثقة: {self.session_stats['avg_confidence']:.1%}")
        print(f"   • مدة الجلسة: {self.session_stats['session_duration']:.1f} ثانية")
        print(f"   • عدد الأخطاء: {self.session_stats['error_count']}")
    
    def _display_help(self):
        """عرض المساعدة"""
        print("\n❓ الأوامر المتاحة:")
        print("   • خروج / quit - إنهاء الجلسة")
        print("   • إحصائيات / stats - عرض إحصائيات الجلسة")
        print("   • مساعدة / help - عرض هذه المساعدة")
        print("   • تنظيف / clear - مسح تاريخ المحادثة")
        
        if self.ai_engine:
            print("\n🎯 الميزات المتاحة:")
            print("   • معالجة اللغة الطبيعية المتقدمة")
            print("   • تحليل المشاعر والكيانات")
            print("   • ذاكرة المحادثة الذكية")
            print("   • اقتراحات تفاعلية")
    
    def _clear_conversation(self):
        """مسح تاريخ المحادثة"""
        self.conversation_history.clear()
        self.current_session["turn_count"] = 0
        print("✅ تم مسح تاريخ المحادثة")
    
    async def _handle_exit(self):
        """معالجة الخروج"""
        print("\n👋 شكراً لاستخدام المساعد الذكي!")
        
        # عرض ملخص الجلسة
        if self.session_stats["total_interactions"] > 0:
            print("\n📈 ملخص الجلسة:")
            self._display_session_stats()
        
        # حفظ البيانات
        await self._save_session_data()
        
        self.is_running = False
    
    async def _save_session_data(self):
        """حفظ بيانات الجلسة"""
        try:
            if self.ai_engine:
                await self.ai_engine.save_memory()
            
            # حفظ تاريخ المحادثة
            session_data = {
                "session_id": self.current_session["start_time"].isoformat(),
                "user_id": self.current_session["user_id"],
                "stats": self.session_stats,
                "conversation_count": len(self.conversation_history)
            }
            
            sessions_dir = Path("data/sessions")
            sessions_dir.mkdir(parents=True, exist_ok=True)
            
            session_file = sessions_dir / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info("تم حفظ بيانات الجلسة")
            
        except Exception as e:
            self.logger.error(f"خطأ في حفظ بيانات الجلسة: {e}")
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """الحصول على ملخص المحادثة"""
        if not self.conversation_history:
            return {"message": "لا يوجد محادثات"}
        
        # تحليل المحادثة
        total_turns = len(self.conversation_history)
        avg_confidence = sum(turn.confidence for turn in self.conversation_history) / total_turns
        
        # استخراج الموضوعات الرئيسية
        user_inputs = [turn.user_input for turn in self.conversation_history]
        
        return {
            "total_turns": total_turns,
            "avg_confidence": avg_confidence,
            "session_duration": self.session_stats["session_duration"],
            "recent_topics": user_inputs[-5:] if len(user_inputs) > 5 else user_inputs,
            "overall_satisfaction": "جيد" if avg_confidence > 0.7 else "متوسط"
        }

# مثيل عام لمحرك المساعد
assistant_engine = UnifiedAssistantEngine()

def get_assistant_engine() -> UnifiedAssistantEngine:
    """الحصول على محرك المساعد"""
    return assistant_engine

if __name__ == "__main__":
    async def main():
        """تشغيل المساعد"""
        engine = get_assistant_engine()
        await engine.start_interactive_session()
    
    asyncio.run(main())
