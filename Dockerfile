
# الصورة الأساسية مع Python 3.11
FROM python:3.11-slim

# تعيين متغيرات البيئة
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# تحديث النظام وتثبيت المكتبات المطلوبة
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    cmake \
    libportaudio2 \
    libportaudiocpp0 \
    portaudio19-dev \
    python3-pyaudio \
    espeak \
    espeak-data \
    libespeak1 \
    libespeak-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    libfontconfig1 \
    libxrender1 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# إنشاء مجلد العمل
WORKDIR /app

# نسخ ملفات المتطلبات
COPY requirements.txt .

# تثبيت المكتبات Python
RUN pip install --no-cache-dir -r requirements.txt

# نسخ كامل المشروع
COPY . .

# إنشاء مجلدات البيانات
RUN mkdir -p data/sessions data/models data/logs

# تعيين الصلاحيات
RUN chmod +x main_unified.py

# المنفذ المستخدم
EXPOSE 5000

# الأمر الافتراضي لتشغيل المساعد
CMD ["python", "main_unified.py"]
