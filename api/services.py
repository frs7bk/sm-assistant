
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
طبقة الخدمات للمساعد الذكي الموحد
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, Optional
from fastapi import WebSocket
import speech_recognition as sr
import tempfile
import os

from core.unified_assistant_engine import get_assistant_engine
from api.models import ChatRequest, ChatResponse, VoiceSessionState, AssistantState
from core.config import get_settings, get_voice_settings

logger = logging.getLogger(__name__)

class ChatService:
    """خدمة المحادثة النصية"""
    
    def __init__(self):
        self.settings = get_settings()
        self.assistant = None
    
    async def initialize(self):
        """تهيئة الخدمة"""
        self.assistant = get_assistant_engine()
        logger.info("✅ تم تهيئة خدمة المحادثة")
    
    async def process_message(
        self,
        message: str,
        user_id: str = "default",
        context: Optional[Dict[str, Any]] = None
    ) -> ChatResponse:
        """معالجة رسالة نصية"""
        
        start_time = time.time()
        
        try:
            # معالجة الرسالة باستخدام محرك المساعد
            response = await self.assistant.process_input(
                user_input=message,
                input_type="text",
                user_id=user_id,
                context=context
            )
            
            # تحويل الاستجابة إلى نموذج API
            api_response = ChatResponse(
                text=response.get("text", ""),
                confidence=response.get("confidence", 0.0),
                intent=response.get("intent", "unknown"),
                emotions=response.get("emotions", {"neutral": 1.0}),
                entities=response.get("entities", []),
                suggestions=response.get("suggestions", []),
                metadata=response.get("metadata", {}),
                processing_time=time.time() - start_time
            )
            
            logger.info(f"✅ تم معالجة رسالة المستخدم {user_id}")
            return api_response
            
        except Exception as e:
            logger.error(f"❌ خطأ في معالجة الرسالة: {e}")
            
            # استجابة احتياطية
            return ChatResponse(
                text="عذراً، حدث خطأ في المعالجة. يرجى المحاولة مرة أخرى.",
                confidence=0.0,
                intent="error",
                emotions={"neutral": 1.0},
                entities=[],
                suggestions=["إعادة المحاولة", "تبسيط السؤال"],
                metadata={"error": str(e)},
                processing_time=time.time() - start_time
            )

class VoiceService:
    """خدمة المحادثة الصوتية"""
    
    def __init__(self):
        self.settings = get_settings()
        self.voice_settings = get_voice_settings()
        self.assistant = None
        self.recognizer = sr.Recognizer()
        self.active_sessions: Dict[str, VoiceSessionState] = {}
    
    async def initialize(self):
        """تهيئة الخدمة"""
        self.assistant = get_assistant_engine()
        logger.info("✅ تم تهيئة خدمة الصوت")
    
    async def handle_voice_session(self, websocket: WebSocket, user_id: str):
        """معالجة جلسة صوتية عبر WebSocket"""
        
        # إنشاء حالة الجلسة
        session_state = VoiceSessionState(user_id=user_id)
        self.active_sessions[user_id] = session_state
        
        # إرسال رسالة ترحيب
        await self._send_message(websocket, {
            "type": "session_started",
            "message": "مرحباً! يمكنك التحدث الآن",
            "state": session_state.state
        })
        
        try:
            while True:
                # استقبال رسالة من العميل
                message = await websocket.receive_text()
                data = json.loads(message)
                
                await self._handle_voice_message(websocket, user_id, data)
                
        except Exception as e:
            logger.error(f"خطأ في الجلسة الصوتية: {e}")
            await self._send_error(websocket, str(e))
        
        finally:
            # تنظيف الجلسة
            if user_id in self.active_sessions:
                del self.active_sessions[user_id]
    
    async def _handle_voice_message(self, websocket: WebSocket, user_id: str, data: Dict[str, Any]):
        """معالجة رسالة صوتية"""
        
        message_type = data.get("type")
        session_state = self.active_sessions.get(user_id)
        
        if not session_state:
            await self._send_error(websocket, "جلسة غير صالحة")
            return
        
        if message_type == "start_recording":
            await self._start_recording(websocket, session_state)
        
        elif message_type == "audio_chunk":
            await self._process_audio_chunk(websocket, session_state, data)
        
        elif message_type == "stop_recording":
            await self._stop_recording(websocket, session_state)
        
        elif message_type == "text_input":
            await self._process_text_input(websocket, session_state, data.get("text", ""))
        
        else:
            await self._send_error(websocket, f"نوع رسالة غير معروف: {message_type}")
    
    async def _start_recording(self, websocket: WebSocket, session_state: VoiceSessionState):
        """بدء التسجيل"""
        session_state.is_recording = True
        session_state.state = AssistantState.LISTENING
        session_state.current_transcript = ""
        
        await self._send_message(websocket, {
            "type": "recording_started",
            "message": "جاري التسجيل...",
            "state": session_state.state
        })
    
    async def _process_audio_chunk(self, websocket: WebSocket, session_state: VoiceSessionState, data: Dict[str, Any]):
        """معالجة قطعة صوتية"""
        
        if not session_state.is_recording:
            return
        
        # هنا يمكن إضافة منطق معالجة الصوت المباشر
        # مثل تحويل الصوت إلى نص تدريجياً
        
        await self._send_message(websocket, {
            "type": "audio_processing",
            "message": "جاري معالجة الصوت...",
            "state": session_state.state
        })
    
    async def _stop_recording(self, websocket: WebSocket, session_state: VoiceSessionState):
        """إيقاف التسجيل ومعالجة النص"""
        
        session_state.is_recording = False
        session_state.state = AssistantState.PROCESSING
        
        await self._send_message(websocket, {
            "type": "recording_stopped",
            "message": "جاري تحويل الصوت إلى نص...",
            "state": session_state.state
        })
        
        # محاكاة تحويل الصوت إلى نص
        # في التطبيق الحقيقي، هنا سيتم استخدام Whisper أو خدمة أخرى
        transcript = "مرحباً، كيف يمكنني مساعدتك؟"  # نص تجريبي
        
        await self._send_message(websocket, {
            "type": "transcript_ready",
            "transcript": transcript,
            "state": session_state.state
        })
        
        # معالجة النص
        await self._process_text_input(websocket, session_state, transcript)
    
    async def _process_text_input(self, websocket: WebSocket, session_state: VoiceSessionState, text: str):
        """معالجة النص وتوليد الاستجابة"""
        
        if not text.strip():
            await self._send_error(websocket, "نص فارغ")
            return
        
        session_state.state = AssistantState.PROCESSING
        session_state.current_transcript = text
        
        await self._send_message(websocket, {
            "type": "processing_text",
            "text": text,
            "message": "جاري معالجة طلبك...",
            "state": session_state.state
        })
        
        try:
            # معالجة النص باستخدام محرك المساعد
            response = await self.assistant.process_input(
                user_input=text,
                input_type="voice",
                user_id=session_state.user_id
            )
            
            response_text = response.get("text", "عذراً، لم أستطع فهم طلبك")
            
            # إرسال الاستجابة النصية
            await self._send_message(websocket, {
                "type": "response_ready",
                "text": response_text,
                "confidence": response.get("confidence", 0.0),
                "intent": response.get("intent", "unknown"),
                "suggestions": response.get("suggestions", []),
                "state": AssistantState.SPEAKING
            })
            
            # تحويل النص إلى صوت (محاكاة)
            session_state.state = AssistantState.SPEAKING
            session_state.is_speaking = True
            
            await self._send_message(websocket, {
                "type": "tts_started",
                "message": "جاري تحويل النص إلى صوت...",
                "state": session_state.state
            })
            
            # محاكاة وقت التحدث
            await asyncio.sleep(2)
            
            session_state.is_speaking = False
            session_state.state = AssistantState.IDLE
            
            await self._send_message(websocket, {
                "type": "tts_completed",
                "message": "تم الانتهاء من التحدث",
                "state": session_state.state
            })
            
        except Exception as e:
            logger.error(f"خطأ في معالجة النص: {e}")
            await self._send_error(websocket, f"خطأ في المعالجة: {str(e)}")
            session_state.state = AssistantState.ERROR
    
    async def _send_message(self, websocket: WebSocket, data: Dict[str, Any]):
        """إرسال رسالة عبر WebSocket"""
        try:
            await websocket.send_text(json.dumps(data, ensure_ascii=False, default=str))
        except Exception as e:
            logger.error(f"خطأ في إرسال الرسالة: {e}")
    
    async def _send_error(self, websocket: WebSocket, error_message: str):
        """إرسال رسالة خطأ"""
        await self._send_message(websocket, {
            "type": "error",
            "message": error_message,
            "state": AssistantState.ERROR
        })
