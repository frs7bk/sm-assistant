
# 📚 توثيق API المساعد الذكي الموحد

## 🌟 نظرة عامة

هذا التوثيق يغطي جميع نقاط النهاية والوظائف المتاحة في API المساعد الذكي الموحد.

## 🚀 البدء السريع

### متطلبات النظام
- Python 3.8+
- FastAPI
- المتطلبات المذكورة في `requirements.txt`

### تشغيل الخادم
```bash
python api/main.py
```

الخادم سيعمل على: `http://localhost:5000`

## 📋 نقاط النهاية الأساسية

### 1. الصفحة الرئيسية
```
GET /
```
عرض الواجهة الأمامية للمساعد.

### 2. حالة النظام
```
GET /api/status
```

**الاستجابة:**
```json
{
  "status": "running",
  "version": "2.0.0", 
  "uptime": 3600.5,
  "active_sessions": 12,
  "memory_usage": 256.7
}
```

### 3. قدرات النظام
```
GET /api/capabilities
```

**الاستجابة:**
```json
{
  "capabilities": [
    "معالجة اللغة الطبيعية",
    "توليد النصوص",
    "تحليل المشاعر"
  ],
  "ai_models": [
    "bert-base-arabic",
    "gpt-4",
    "emotion-analyzer"
  ],
  "features": {
    "voice_processing": true,
    "image_analysis": true,
    "real_time_learning": true
  }
}
```

## 💬 نقاط النهاية للدردشة

### 1. الدردشة العادية
```
POST /api/chat
```

**البيانات المرسلة:**
```json
{
  "message": "مرحباً، كيف يمكنك مساعدتي؟",
  "user_id": "user123",
  "session_id": "session456", 
  "context": {
    "previous_topic": "programming",
    "user_mood": "curious"
  }
}
```

**الاستجابة:**
```json
{
  "response": "أهلاً وسهلاً! يمكنني مساعدتك في البرمجة والتقنية وأمور أخرى كثيرة.",
  "session_id": "session456",
  "timestamp": "2025-01-26T20:30:00",
  "metadata": {
    "confidence": 0.95,
    "processing_time": 0.12,
    "model_used": "gpt-4-advanced"
  }
}
```

### 2. الدردشة المباشرة (WebSocket)
```
WS /api/ws/{session_id}
```

**رسالة مرسلة:**
```json
{
  "message": "ما هو الذكاء الاصطناعي؟",
  "user_id": "user123",
  "context": {}
}
```

**رسالة مستقبلة:**
```json
{
  "response": "الذكاء الاصطناعي هو...",
  "session_id": "session456",
  "timestamp": "2025-01-26T20:30:15",
  "metadata": {
    "real_time": true
  }
}
```

## 📝 إدارة الجلسات

### 1. تاريخ الجلسة
```
GET /api/sessions/{session_id}/history
```

**الاستجابة:**
```json
{
  "session_id": "session456",
  "history": [
    {
      "timestamp": "2025-01-26T20:25:00",
      "user_message": "مرحباً",
      "bot_response": "أهلاً وسهلاً",
      "sentiment": "positive"
    }
  ]
}
```

### 2. مسح الجلسة
```
DELETE /api/sessions/{session_id}
```

**الاستجابة:**
```json
{
  "message": "تم مسح الجلسة بنجاح",
  "session_id": "session456"
}
```

## 🔐 الأمان والتوثيق

### Bearer Token (اختياري)
```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
     -X POST http://localhost:5000/api/chat
```

## 📊 مراقبة الأداء

### مقاييس مهمة:
- **زمن الاستجابة**: < 2 ثانية عادة
- **استخدام الذاكرة**: مراقب باستمرار
- **الجلسات النشطة**: معروضة في `/api/status`

## 🚨 معالجة الأخطاء

### أكواد الأخطاء الشائعة:

| الكود | الوصف | الحل |
|-------|--------|------|
| 400 | طلب غير صحيح | تحقق من صيغة البيانات |
| 503 | الخدمة غير متاحة | المساعد في وضع محدود |
| 500 | خطأ داخلي | اتصل بالدعم التقني |

### مثال على استجابة خطأ:
```json
{
  "error": "حدث خطأ داخلي في الخادم",
  "message": "نعتذر عن الإزعاج. يرجى المحاولة مرة أخرى لاحقاً.",
  "timestamp": "2025-01-26T20:30:00"
}
```

## 🔧 أمثلة عملية

### Python
```python
import requests

# إرسال رسالة
response = requests.post(
    'http://localhost:5000/api/chat',
    json={
        'message': 'اشرح لي البرمجة الكائنية',
        'user_id': 'python_learner'
    }
)

data = response.json()
print(data['response'])
```

### JavaScript
```javascript
// إرسال رسالة
async function sendMessage(message) {
    const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            message: message,
            user_id: 'web_user'
        })
    });
    
    const data = await response.json();
    return data.response;
}
```

### WebSocket (JavaScript)
```javascript
// اتصال WebSocket
const ws = new WebSocket('ws://localhost:5000/api/ws/my_session');

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('استجابة المساعد:', data.response);
};

// إرسال رسالة
ws.send(JSON.stringify({
    message: 'مرحباً!',
    user_id: 'websocket_user'
}));
```

## 🎯 أفضل الممارسات

### 1. إدارة الجلسات
- استخدم معرف جلسة ثابت للحفاظ على السياق
- امسح الجلسات غير المستخدمة

### 2. معالجة الأخطاء
- تحقق دائماً من استجابة الخادم
- اعتمد على آليات إعادة المحاولة

### 3. الأداء
- استخدم WebSocket للدردشة المباشرة
- راقب استخدام الذاكرة

## 🔄 التحديثات والإصدارات

### الإصدار الحالي: 2.0.0
- دعم WebSocket
- توثيق تلقائي
- معالجة أخطاء محسنة
- مراقبة أداء

### الميزات القادمة:
- دعم الملفات متعددة الوسائط
- API للتحليلات المتقدمة
- تكامل مع خدمات خارجية

## 📞 الدعم والمساعدة

- **التوثيق التفاعلي**: `/api/docs`
- **ReDoc**: `/api/redoc`
- **واجهة الدردشة**: `/chat`

---

💡 **نصيحة**: استخدم `/api/docs` للحصول على واجهة تفاعلية لاختبار جميع نقاط النهاية!
