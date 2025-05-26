
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌐 API الرئيسي للمساعد الذكي الموحد
Main API for Unified AI Assistant
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
import asyncio
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
import uuid

# إضافة مسار المشروع
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# الاستيرادات المحلية
try:
    from core.unified_assistant_engine import UnifiedAssistantEngine
    from core.advanced_error_handler import AdvancedErrorHandler
    from api.models import *
    from api.services import *
except ImportError as e:
    logging.warning(f"فشل في استيراد بعض الوحدات: {e}")

# إعداد التطبيق
app = FastAPI(
    title="🤖 المساعد الذكي الموحد",
    description="API متقدم للمساعد الذكي مع قدرات متعددة",
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

# إعداد الأمان
security = HTTPBearer(auto_error=False)

# متغيرات عامة
assistant_engine: Optional[UnifiedAssistantEngine] = None
error_handler = AdvancedErrorHandler()
active_connections: Dict[str, WebSocket] = {}

# نماذج البيانات
class ChatRequest(BaseModel):
    message: str = Field(..., description="رسالة المستخدم")
    user_id: Optional[str] = Field(None, description="معرف المستخدم")
    session_id: Optional[str] = Field(None, description="معرف الجلسة")
    context: Optional[Dict[str, Any]] = Field(None, description="سياق إضافي")

class ChatResponse(BaseModel):
    response: str = Field(..., description="رد المساعد")
    session_id: str = Field(..., description="معرف الجلسة")
    timestamp: str = Field(..., description="وقت الاستجابة")
    metadata: Optional[Dict[str, Any]] = Field(None, description="بيانات إضافية")

class SystemStatus(BaseModel):
    status: str = Field(..., description="حالة النظام")
    version: str = Field(..., description="إصدار النظام")
    uptime: float = Field(..., description="وقت التشغيل")
    active_sessions: int = Field(..., description="الجلسات النشطة")
    memory_usage: Optional[float] = Field(None, description="استخدام الذاكرة")

class CapabilitiesResponse(BaseModel):
    capabilities: List[str] = Field(..., description="قائمة القدرات المتاحة")
    ai_models: List[str] = Field(..., description="النماذج المحملة")
    features: Dict[str, bool] = Field(..., description="الميزات المتاحة")

# بدء التشغيل
@app.on_event("startup")
async def startup_event():
    """تهيئة التطبيق عند البدء"""
    global assistant_engine
    
    try:
        logging.info("🚀 بدء تشغيل API المساعد الذكي...")
        
        # تهيئة محرك المساعد
        assistant_engine = UnifiedAssistantEngine()
        await assistant_engine.initialize()
        
        logging.info("✅ تم تشغيل API بنجاح")
        
    except Exception as e:
        logging.error(f"❌ فشل في تشغيل API: {e}")
        # لا نوقف التطبيق، نتيح العمل في وضع محدود

# إغلاق التطبيق
@app.on_event("shutdown") 
async def shutdown_event():
    """تنظيف عند إغلاق التطبيق"""
    global assistant_engine
    
    try:
        logging.info("🛑 بدء إغلاق API...")
        
        # إغلاق الاتصالات النشطة
        for connection in active_connections.values():
            try:
                await connection.close()
            except:
                pass
        
        # إغلاق محرك المساعد
        if assistant_engine and hasattr(assistant_engine, 'cleanup'):
            await assistant_engine.cleanup()
            
        logging.info("✅ تم إغلاق API بنجاح")
        
    except Exception as e:
        logging.error(f"❌ خطأ أثناء الإغلاق: {e}")

# نقاط النهاية الأساسية

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """عرض الواجهة الأمامية"""
    try:
        frontend_path = Path("frontend/index.html")
        if frontend_path.exists():
            with open(frontend_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            return """
            <!DOCTYPE html>
            <html dir="rtl" lang="ar">
            <head>
                <meta charset="UTF-8">
                <title>المساعد الذكي الموحد</title>
                <style>
                    body { font-family: Arial, sans-serif; text-align: center; padding: 50px; }
                    .container { max-width: 600px; margin: 0 auto; }
                    .status { color: #28a745; font-size: 1.2em; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>🤖 المساعد الذكي الموحد</h1>
                    <p class="status">✅ API يعمل بنجاح</p>
                    <p>للوصول إلى التوثيق: <a href="/api/docs">/api/docs</a></p>
                    <p>للواجهة التفاعلية: <a href="/chat">/chat</a></p>
                </div>
            </body>
            </html>
            """
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطأ في تحميل الواجهة: {str(e)}")

@app.get("/api/status", response_model=SystemStatus)
async def get_system_status():
    """الحصول على حالة النظام"""
    try:
        import psutil
        import time
        
        # حساب وقت التشغيل (تقريبي)
        uptime = time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0
        
        # استخدام الذاكرة
        memory_usage = None
        try:
            process = psutil.Process()
            memory_usage = process.memory_info().rss / 1024 / 1024  # MB
        except:
            pass
        
        return SystemStatus(
            status="running" if assistant_engine else "limited",
            version="2.0.0",
            uptime=uptime,
            active_sessions=len(active_connections),
            memory_usage=memory_usage
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطأ في الحصول على حالة النظام: {str(e)}")

@app.get("/api/capabilities", response_model=CapabilitiesResponse)
async def get_capabilities():
    """الحصول على قدرات النظام"""
    try:
        capabilities = [
            "معالجة اللغة الطبيعية",
            "توليد النصوص",
            "تحليل المشاعر",
            "الذكاء العاطفي",
            "التعلم المستمر",
            "التنبؤ الذكي",
            "معالجة الرؤية",
            "الأتمتة الذكية"
        ]
        
        ai_models = []
        features = {}
        
        if assistant_engine:
            # الحصول على النماذج المحملة
            if hasattr(assistant_engine, 'get_loaded_models'):
                ai_models = assistant_engine.get_loaded_models()
            
            # الحصول على الميزات المتاحة
            if hasattr(assistant_engine, 'get_available_features'):
                features = assistant_engine.get_available_features()
        
        return CapabilitiesResponse(
            capabilities=capabilities,
            ai_models=ai_models,
            features=features
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطأ في الحصول على القدرات: {str(e)}")

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """نقطة نهاية الدردشة الرئيسية"""
    try:
        # التحقق من توفر المحرك
        if not assistant_engine:
            raise HTTPException(
                status_code=503, 
                detail="المساعد غير متاح حالياً. يعمل النظام في وضع محدود."
            )
        
        # إنشاء معرف جلسة إذا لم يكن موجوداً
        session_id = request.session_id or str(uuid.uuid4())
        
        # معالجة الطلب
        response_data = await assistant_engine.process_request(
            message=request.message,
            user_id=request.user_id,
            session_id=session_id,
            context=request.context or {}
        )
        
        # إنشاء الاستجابة
        return ChatResponse(
            response=response_data.get("response", "عذراً، لم أتمكن من معالجة طلبك."),
            session_id=session_id,
            timestamp=datetime.now().isoformat(),
            metadata=response_data.get("metadata", {})
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"خطأ في معالجة الدردشة: {e}")
        raise HTTPException(status_code=500, detail=f"خطأ في معالجة الطلب: {str(e)}")

# WebSocket للدردشة المباشرة
@app.websocket("/api/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """نقطة نهاية WebSocket للدردشة المباشرة"""
    await websocket.accept()
    active_connections[session_id] = websocket
    
    try:
        while True:
            # استقبال الرسالة
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            if not assistant_engine:
                await websocket.send_text(json.dumps({
                    "error": "المساعد غير متاح حالياً",
                    "timestamp": datetime.now().isoformat()
                }))
                continue
            
            # معالجة الرسالة
            try:
                response = await assistant_engine.process_request(
                    message=message_data.get("message", ""),
                    user_id=message_data.get("user_id"),
                    session_id=session_id,
                    context=message_data.get("context", {})
                )
                
                # إرسال الاستجابة
                await websocket.send_text(json.dumps({
                    "response": response.get("response", ""),
                    "session_id": session_id,
                    "timestamp": datetime.now().isoformat(),
                    "metadata": response.get("metadata", {})
                }))
                
            except Exception as e:
                await websocket.send_text(json.dumps({
                    "error": f"خطأ في معالجة الرسالة: {str(e)}",
                    "timestamp": datetime.now().isoformat()
                }))
                
    except WebSocketDisconnect:
        if session_id in active_connections:
            del active_connections[session_id]
    except Exception as e:
        logging.error(f"خطأ في WebSocket: {e}")
        if session_id in active_connections:
            del active_connections[session_id]

# نقاط نهاية إضافية
@app.get("/api/sessions/{session_id}/history")
async def get_session_history(session_id: str):
    """الحصول على تاريخ الجلسة"""
    try:
        if assistant_engine and hasattr(assistant_engine, 'get_session_history'):
            history = await assistant_engine.get_session_history(session_id)
            return {"session_id": session_id, "history": history}
        else:
            return {"session_id": session_id, "history": []}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطأ في الحصول على التاريخ: {str(e)}")

@app.delete("/api/sessions/{session_id}")
async def clear_session(session_id: str):
    """مسح بيانات الجلسة"""
    try:
        if assistant_engine and hasattr(assistant_engine, 'clear_session'):
            await assistant_engine.clear_session(session_id)
        
        # إزالة الاتصال إذا كان موجوداً
        if session_id in active_connections:
            del active_connections[session_id]
            
        return {"message": "تم مسح الجلسة بنجاح", "session_id": session_id}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطأ في مسح الجلسة: {str(e)}")

@app.get("/chat", response_class=HTMLResponse)
async def chat_interface():
    """واجهة الدردشة التفاعلية"""
    return """
    <!DOCTYPE html>
    <html dir="rtl" lang="ar">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>💬 دردشة المساعد الذكي</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f5f5f5; }
            .chat-container { max-width: 800px; margin: 20px auto; background: white; border-radius: 10px; overflow: hidden; box-shadow: 0 4px 20px rgba(0,0,0,0.1); }
            .chat-header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; text-align: center; }
            .chat-messages { height: 400px; overflow-y: auto; padding: 20px; background: #fafafa; }
            .message { margin: 10px 0; padding: 10px 15px; border-radius: 15px; max-width: 80%; }
            .user-message { background: #007bff; color: white; margin-right: auto; text-align: left; }
            .bot-message { background: #e9ecef; color: #333; margin-left: auto; text-align: right; }
            .chat-input { display: flex; padding: 20px; background: white; border-top: 1px solid #dee2e6; }
            .chat-input input { flex: 1; padding: 12px; border: 1px solid #ced4da; border-radius: 25px; font-size: 16px; }
            .chat-input button { margin-right: 10px; padding: 12px 20px; background: #28a745; color: white; border: none; border-radius: 25px; cursor: pointer; }
            .chat-input button:hover { background: #218838; }
            .status { text-align: center; padding: 10px; color: #6c757d; font-size: 14px; }
        </style>
    </head>
    <body>
        <div class="chat-container">
            <div class="chat-header">
                <h1>🤖 المساعد الذكي الموحد</h1>
                <p>أهلاً بك! كيف يمكنني مساعدتك اليوم؟</p>
            </div>
            <div class="chat-messages" id="messages">
                <div class="message bot-message">
                    مرحباً! أنا مساعدك الذكي. يمكنني مساعدتك في العديد من المجالات. ما الذي تود أن تسأل عنه؟
                </div>
            </div>
            <div class="status" id="status">متصل</div>
            <div class="chat-input">
                <button onclick="sendMessage()">إرسال</button>
                <input type="text" id="messageInput" placeholder="اكتب رسالتك هنا..." onkeypress="handleKeyPress(event)">
            </div>
        </div>
        
        <script>
            const messagesContainer = document.getElementById('messages');
            const messageInput = document.getElementById('messageInput');
            const statusElement = document.getElementById('status');
            
            function addMessage(content, isUser = false) {
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
                messageDiv.textContent = content;
                messagesContainer.appendChild(messageDiv);
                messagesContainer.scrollTop = messagesContainer.scrollHeight;
            }
            
            async function sendMessage() {
                const message = messageInput.value.trim();
                if (!message) return;
                
                addMessage(message, true);
                messageInput.value = '';
                statusElement.textContent = 'يكتب...';
                
                try {
                    const response = await fetch('/api/chat', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ message: message })
                    });
                    
                    const data = await response.json();
                    addMessage(data.response || 'عذراً، حدث خطأ في معالجة طلبك.');
                    statusElement.textContent = 'متصل';
                    
                } catch (error) {
                    addMessage('عذراً، حدث خطأ في الاتصال. يرجى المحاولة مرة أخرى.');
                    statusElement.textContent = 'خطأ في الاتصال';
                }
            }
            
            function handleKeyPress(event) {
                if (event.key === 'Enter') {
                    sendMessage();
                }
            }
            
            // التركيز على حقل الإدخال
            messageInput.focus();
        </script>
    </body>
    </html>
    """

# معالج الأخطاء العام
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """معالج الأخطاء العام"""
    logging.error(f"خطأ غير متوقع: {exc}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "حدث خطأ داخلي في الخادم",
            "message": "نعتذر عن الإزعاج. يرجى المحاولة مرة أخرى لاحقاً.",
            "timestamp": datetime.now().isoformat()
        }
    )

# إعداد وقت البدء
@app.middleware("http")
async def add_startup_time(request, call_next):
    if not hasattr(app.state, 'start_time'):
        import time
        app.state.start_time = time.time()
    response = await call_next(request)
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=5000,
        reload=True,
        log_level="info"
    )
