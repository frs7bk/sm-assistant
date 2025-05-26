
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
نماذج البيانات للAPI باستخدام Pydantic
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum

class MessageType(str, Enum):
    """أنواع الرسائل"""
    TEXT = "text"
    VOICE = "voice"
    IMAGE = "image"

class AssistantState(str, Enum):
    """حالات المساعد"""
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    ERROR = "error"

class ChatRequest(BaseModel):
    """طلب محادثة نصية"""
    message: str = Field(..., description="نص الرسالة")
    user_id: str = Field("default", description="معرف المستخدم")
    message_type: MessageType = Field(MessageType.TEXT, description="نوع الرسالة")
    context: Optional[Dict[str, Any]] = Field(None, description="سياق المحادثة")
    
    class Config:
        schema_extra = {
            "example": {
                "message": "مرحباً، كيف يمكنني مساعدتك؟",
                "user_id": "user123",
                "message_type": "text",
                "context": {"previous_topic": "weather"}
            }
        }

class ChatResponse(BaseModel):
    """استجابة المحادثة"""
    text: str = Field(..., description="نص الاستجابة")
    confidence: float = Field(..., description="مستوى الثقة", ge=0.0, le=1.0)
    intent: str = Field(..., description="القصد المكتشف")
    emotions: Dict[str, float] = Field(..., description="المشاعر المكتشفة")
    entities: List[Dict[str, Any]] = Field(..., description="الكيانات المستخرجة")
    suggestions: List[str] = Field(..., description="الاقتراحات")
    metadata: Dict[str, Any] = Field(..., description="بيانات إضافية")
    processing_time: float = Field(..., description="وقت المعالجة بالثواني")
    timestamp: datetime = Field(default_factory=datetime.now, description="وقت الاستجابة")
    
    class Config:
        schema_extra = {
            "example": {
                "text": "أهلاً وسهلاً! كيف يمكنني مساعدتك اليوم؟",
                "confidence": 0.95,
                "intent": "greeting",
                "emotions": {"happy": 0.8, "neutral": 0.2},
                "entities": [],
                "suggestions": ["هل تحتاج مساعدة؟", "أخبرني عن يومك"],
                "metadata": {"model_used": "advanced_ai_engine"},
                "processing_time": 0.234,
                "timestamp": "2025-01-26T12:00:00Z"
            }
        }

class VoiceMessage(BaseModel):
    """رسالة صوتية عبر WebSocket"""
    type: str = Field(..., description="نوع الرسالة")
    data: Optional[Dict[str, Any]] = Field(None, description="بيانات الرسالة")
    user_id: str = Field(..., description="معرف المستخدم")
    timestamp: datetime = Field(default_factory=datetime.now)

class VoiceSessionState(BaseModel):
    """حالة الجلسة الصوتية"""
    user_id: str
    state: AssistantState = AssistantState.IDLE
    is_recording: bool = False
    is_speaking: bool = False
    current_transcript: str = ""
    session_start: datetime = Field(default_factory=datetime.now)
    last_activity: datetime = Field(default_factory=datetime.now)

class AudioChunk(BaseModel):
    """قطعة صوتية"""
    data: bytes = Field(..., description="البيانات الصوتية")
    sample_rate: int = Field(16000, description="معدل العينات")
    channels: int = Field(1, description="عدد القنوات")
    format: str = Field("wav", description="تنسيق الصوت")

class HealthResponse(BaseModel):
    """استجابة فحص الصحة"""
    status: str = Field(..., description="حالة التطبيق")
    version: str = Field(..., description="إصدار التطبيق")
    assistant_initialized: bool = Field(..., description="هل تم تهيئة المساعد")
    ai_engine_available: bool = Field(..., description="هل محرك الذكاء الاصطناعي متاح")
    timestamp: datetime = Field(default_factory=datetime.now)

class StatsResponse(BaseModel):
    """استجابة الإحصائيات"""
    active_connections: int = Field(..., description="عدد الاتصالات النشطة")
    session_stats: Dict[str, Any] = Field(..., description="إحصائيات الجلسة")
    conversation_summary: Dict[str, Any] = Field(..., description="ملخص المحادثة")
    ai_performance: Optional[Dict[str, Any]] = Field(None, description="أداء الذكاء الاصطناعي")

class ErrorResponse(BaseModel):
    """استجابة الخطأ"""
    error: str = Field(..., description="رسالة الخطأ")
    error_code: Optional[str] = Field(None, description="رمز الخطأ")
    details: Optional[Dict[str, Any]] = Field(None, description="تفاصيل إضافية")
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        schema_extra = {
            "example": {
                "error": "حدث خطأ في معالجة الطلب",
                "error_code": "PROCESSING_ERROR",
                "details": {"step": "text_analysis"},
                "timestamp": "2025-01-26T12:00:00Z"
            }
        }
