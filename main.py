
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
نقطة الدخول الرئيسية للمساعد الذكي الموحد
تطبيق FastAPI احترافي جاهز للإنتاج
"""

import uvicorn
import asyncio
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from typing import Dict, Any, Optional
import json
from pathlib import Path

# إعداد المسارات
from core.unified_assistant_engine import get_assistant_engine
from api.models import ChatRequest, ChatResponse
from api.services import ChatService, VoiceService
from core.config import get_settings

# إعداد السجلات
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# إنشاء التطبيق
app = FastAPI(
    title="المساعد الذكي الموحد",
    description="مساعد ذكي متطور يدعم المحادثات النصية والصوتية",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# إعداد CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# إعداد الملفات الثابتة
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")

# الخدمات
settings = get_settings()
chat_service = ChatService()
voice_service = VoiceService()

# إدارة الاتصالات النشطة
active_connections: Dict[str, WebSocket] = {}

@app.on_event("startup")
async def startup_event():
    """تهيئة التطبيق عند البدء"""
    logger.info("🚀 بدء تشغيل المساعد الذكي الموحد...")
    
    # تهيئة محرك المساعد
    assistant = get_assistant_engine()
    await assistant.initialize()
    
    # تهيئة الخدمات
    await chat_service.initialize()
    await voice_service.initialize()
    
    logger.info("✅ تم تشغيل التطبيق بنجاح")

@app.on_event("shutdown")
async def shutdown_event():
    """تنظيف الموارد عند الإغلاق"""
    logger.info("🔄 إيقاف التطبيق...")
    
    # إغلاق الاتصالات النشطة
    for connection in active_connections.values():
        await connection.close()
    
    # حفظ البيانات
    assistant = get_assistant_engine()
    if assistant.ai_engine:
        await assistant.ai_engine.save_memory()
    
    logger.info("✅ تم إيقاف التطبيق بنجاح")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """الصفحة الرئيسية"""
    html_file = Path("frontend/index.html")
    if html_file.exists():
        return html_file.read_text(encoding='utf-8')
    
    return """
    <!DOCTYPE html>
    <html dir="rtl">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>المساعد الذكي الموحد</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; min-height: 100vh; }
            .container { max-width: 800px; margin: 0 auto; text-align: center; }
            h1 { font-size: 3rem; margin-bottom: 1rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }
            .status { background: rgba(255,255,255,0.1); padding: 20px; border-radius: 10px; margin: 20px 0; }
            .btn { background: rgba(255,255,255,0.2); border: none; padding: 15px 30px; border-radius: 25px; color: white; font-size: 1.1rem; cursor: pointer; margin: 10px; transition: all 0.3s; }
            .btn:hover { background: rgba(255,255,255,0.3); transform: translateY(-2px); }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤖 المساعد الذكي الموحد</h1>
            <div class="status">
                <h2>✅ التطبيق يعمل بنجاح</h2>
                <p>مرحباً بك في النسخة الاحترافية الجديدة</p>
            </div>
            <button class="btn" onclick="window.location.href='/api/docs'">📚 وثائق API</button>
            <button class="btn" onclick="window.location.href='/chat'">💬 واجهة المحادثة</button>
        </div>
    </body>
    </html>
    """

@app.get("/health")
async def health_check():
    """فحص صحة التطبيق"""
    assistant = get_assistant_engine()
    
    return {
        "status": "healthy",
        "version": "2.0.0",
        "assistant_initialized": assistant.is_initialized,
        "ai_engine_available": assistant.ai_engine is not None,
        "timestamp": "2025-01-26T12:00:00Z"
    }

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """نقطة نهاية المحادثة النصية"""
    try:
        response = await chat_service.process_message(
            message=request.message,
            user_id=request.user_id,
            context=request.context
        )
        return response
    
    except Exception as e:
        logger.error(f"خطأ في معالجة الرسالة: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/voice/{user_id}")
async def voice_websocket_endpoint(websocket: WebSocket, user_id: str):
    """نقطة نهاية WebSocket للمحادثة الصوتية"""
    await websocket.accept()
    active_connections[user_id] = websocket
    
    logger.info(f"🔊 اتصال صوتي جديد: {user_id}")
    
    try:
        await voice_service.handle_voice_session(websocket, user_id)
    
    except WebSocketDisconnect:
        logger.info(f"📴 انقطع الاتصال الصوتي: {user_id}")
    
    except Exception as e:
        logger.error(f"خطأ في الجلسة الصوتية: {e}")
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": "حدث خطأ في الجلسة الصوتية"
        }))
    
    finally:
        if user_id in active_connections:
            del active_connections[user_id]

@app.get("/api/stats")
async def get_stats():
    """إحصائيات التطبيق"""
    assistant = get_assistant_engine()
    
    stats = {
        "active_connections": len(active_connections),
        "session_stats": assistant.session_stats,
        "conversation_summary": assistant.get_conversation_summary()
    }
    
    if assistant.ai_engine:
        stats["ai_performance"] = assistant.ai_engine.get_performance_report()
    
    return stats

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=5000,
        reload=True,
        log_level="info"
    )
