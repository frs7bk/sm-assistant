
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
واجهة ويب ذكية متقدمة للمساعد
تدعم التفاعل الصوتي والمرئي والنصي
"""

import asyncio
import logging
import json
import base64
import io
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import threading

# مكتبات الويب
from flask import Flask, render_template, request, jsonify, session, send_file
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_cors import CORS
import uuid

# مكتبات معالجة الوسائط
from PIL import Image
import numpy as np

class SmartWebInterface:
    """واجهة ويب ذكية"""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 5000):
        self.logger = logging.getLogger(__name__)
        self.host = host
        self.port = port
        
        # إعداد Flask
        self.app = Flask(__name__, 
                        template_folder="templates",
                        static_folder="static")
        self.app.secret_key = "smart_assistant_2024"
        
        # إعداد CORS
        CORS(self.app, origins="*")
        
        # إعداد SocketIO للتفاعل المباشر
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # جلسات المستخدمين النشطة
        self.active_sessions = {}
        
        # إحصائيات الواجهة
        self.interface_stats = {
            "total_connections": 0,
            "active_users": 0,
            "messages_processed": 0,
            "voice_interactions": 0,
            "image_uploads": 0,
            "start_time": datetime.now()
        }
        
        # تهيئة المسارات والأحداث
        self._setup_routes()
        self._setup_socket_events()
        
        # إنشاء المجلدات المطلوبة
        self._create_directories()
        
        # إنشاء ملفات الويب الأساسية
        self._create_web_files()
    
    def _create_directories(self):
        """إنشاء المجلدات المطلوبة"""
        directories = [
            "interfaces/web/templates",
            "interfaces/web/static/css",
            "interfaces/web/static/js",
            "interfaces/web/static/images",
            "data/uploads",
            "data/chat_logs"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def _create_web_files(self):
        """إنشاء ملفات الويب الأساسية"""
        
        # إنشاء ملف HTML الرئيسي
        html_template = """
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>المساعد الذكي - واجهة متقدمة</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <header class="header">
            <div class="header-content">
                <div class="logo">
                    <i class="fas fa-robot"></i>
                    <h1>المساعد الذكي</h1>
                </div>
                <div class="status-indicator">
                    <span class="status-dot" id="connectionStatus"></span>
                    <span id="statusText">متصل</span>
                </div>
            </div>
        </header>

        <!-- Main Content -->
        <main class="main-content">
            <!-- Chat Container -->
            <div class="chat-container">
                <div class="chat-messages" id="chatMessages">
                    <div class="welcome-message">
                        <div class="message assistant-message">
                            <div class="message-content">
                                <i class="fas fa-robot"></i>
                                <div class="text">
                                    <h3>أهلاً بك في المساعد الذكي!</h3>
                                    <p>يمكنني مساعدتك في:</p>
                                    <ul>
                                        <li>الإجابة على الأسئلة</li>
                                        <li>تحليل الصور والفيديوهات</li>
                                        <li>التحكم في التطبيقات</li>
                                        <li>البحث والتصفح</li>
                                    </ul>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Input Container -->
                <div class="input-container">
                    <div class="input-group">
                        <div class="input-tools">
                            <button class="tool-btn" id="voiceBtn" title="التسجيل الصوتي">
                                <i class="fas fa-microphone"></i>
                            </button>
                            <button class="tool-btn" id="imageBtn" title="رفع صورة">
                                <i class="fas fa-image"></i>
                            </button>
                            <button class="tool-btn" id="cameraBtn" title="كاميرا">
                                <i class="fas fa-camera"></i>
                            </button>
                            <input type="file" id="imageInput" accept="image/*" style="display: none;">
                        </div>
                        
                        <div class="text-input-container">
                            <textarea 
                                id="messageInput" 
                                placeholder="اكتب رسالتك هنا..."
                                rows="1"
                            ></textarea>
                            <button class="send-btn" id="sendBtn">
                                <i class="fas fa-paper-plane"></i>
                            </button>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Sidebar -->
            <aside class="sidebar">
                <div class="sidebar-section">
                    <h3><i class="fas fa-chart-line"></i> الإحصائيات</h3>
                    <div class="stats-grid">
                        <div class="stat-item">
                            <span class="stat-value" id="messagesCount">0</span>
                            <span class="stat-label">الرسائل</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-value" id="confidenceLevel">95%</span>
                            <span class="stat-label">مستوى الثقة</span>
                        </div>
                    </div>
                </div>

                <div class="sidebar-section">
                    <h3><i class="fas fa-cog"></i> الإعدادات</h3>
                    <div class="settings-group">
                        <label class="setting-item">
                            <span>الأوامر الصوتية</span>
                            <input type="checkbox" id="voiceCommands" checked>
                        </label>
                        <label class="setting-item">
                            <span>الإشعارات</span>
                            <input type="checkbox" id="notifications" checked>
                        </label>
                        <label class="setting-item">
                            <span>الوضع المظلم</span>
                            <input type="checkbox" id="darkMode">
                        </label>
                    </div>
                </div>

                <div class="sidebar-section">
                    <h3><i class="fas fa-history"></i> المحادثات الأخيرة</h3>
                    <div class="recent-chats" id="recentChats">
                        <!-- سيتم إضافة المحادثات هنا ديناميكياً -->
                    </div>
                </div>
            </aside>
        </main>

        <!-- Loading Overlay -->
        <div class="loading-overlay" id="loadingOverlay">
            <div class="loading-spinner">
                <i class="fas fa-robot"></i>
                <span>المساعد يفكر...</span>
            </div>
        </div>
    </div>

    <script src="{{ url_for('static', filename='js/app.js') }}"></script>
</body>
</html>
        """
        
        template_path = Path("interfaces/web/templates/index.html")
        template_path.parent.mkdir(parents=True, exist_ok=True)
        with open(template_path, 'w', encoding='utf-8') as f:
            f.write(html_template)
        
        # إنشاء ملف CSS
        css_content = """
/* Global Styles */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    height: 100vh;
    overflow: hidden;
    direction: rtl;
    color: #333;
}

.container {
    display: flex;
    flex-direction: column;
    height: 100vh;
}

/* Header */
.header {
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(10px);
    padding: 1rem 2rem;
    box-shadow: 0 2px 20px rgba(0, 0, 0, 0.1);
}

.header-content {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.logo {
    display: flex;
    align-items: center;
    gap: 1rem;
}

.logo i {
    font-size: 2rem;
    color: #667eea;
}

.logo h1 {
    font-size: 1.5rem;
    font-weight: 700;
    color: #333;
}

.status-indicator {
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.status-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: #4CAF50;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.5; }
    100% { opacity: 1; }
}

/* Main Content */
.main-content {
    display: flex;
    flex: 1;
    overflow: hidden;
}

/* Chat Container */
.chat-container {
    flex: 1;
    display: flex;
    flex-direction: column;
    background: rgba(255, 255, 255, 0.9);
    margin: 1rem;
    border-radius: 20px;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
}

.chat-messages {
    flex: 1;
    overflow-y: auto;
    padding: 2rem;
    scroll-behavior: smooth;
}

/* Messages */
.message {
    margin-bottom: 2rem;
    opacity: 0;
    animation: fadeInUp 0.5s forwards;
}

@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.user-message {
    text-align: left;
}

.assistant-message {
    text-align: right;
}

.message-content {
    display: inline-flex;
    align-items: flex-start;
    gap: 1rem;
    max-width: 70%;
    padding: 1.5rem;
    border-radius: 20px;
    position: relative;
}

.user-message .message-content {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    margin-left: auto;
}

.assistant-message .message-content {
    background: rgba(102, 126, 234, 0.1);
    color: #333;
    border: 1px solid rgba(102, 126, 234, 0.2);
}

.message-content i {
    font-size: 1.5rem;
    margin-top: 0.2rem;
    flex-shrink: 0;
}

.message-content .text {
    flex: 1;
}

.message-content h3 {
    margin-bottom: 0.5rem;
    color: #667eea;
}

.message-content ul {
    margin: 1rem 0;
    padding-right: 1rem;
}

.message-content li {
    margin-bottom: 0.5rem;
}

/* Input Container */
.input-container {
    padding: 1.5rem 2rem;
    background: rgba(255, 255, 255, 0.95);
    border-top: 1px solid rgba(0, 0, 0, 0.05);
}

.input-group {
    display: flex;
    gap: 1rem;
    align-items: flex-end;
}

.input-tools {
    display: flex;
    gap: 0.5rem;
}

.tool-btn {
    width: 45px;
    height: 45px;
    border: none;
    border-radius: 50%;
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    cursor: pointer;
    transition: all 0.3s ease;
    display: flex;
    align-items: center;
    justify-content: center;
}

.tool-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
}

.tool-btn.active {
    background: #e74c3c;
    animation: pulse 1s infinite;
}

.text-input-container {
    flex: 1;
    display: flex;
    gap: 1rem;
    align-items: flex-end;
}

#messageInput {
    flex: 1;
    border: 2px solid rgba(102, 126, 234, 0.2);
    border-radius: 25px;
    padding: 1rem 1.5rem;
    font-size: 1rem;
    resize: none;
    outline: none;
    transition: all 0.3s ease;
    max-height: 120px;
    font-family: inherit;
}

#messageInput:focus {
    border-color: #667eea;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
}

.send-btn {
    width: 45px;
    height: 45px;
    border: none;
    border-radius: 50%;
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    cursor: pointer;
    transition: all 0.3s ease;
    display: flex;
    align-items: center;
    justify-content: center;
}

.send-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
}

/* Sidebar */
.sidebar {
    width: 300px;
    background: rgba(255, 255, 255, 0.9);
    backdrop-filter: blur(10px);
    padding: 2rem;
    overflow-y: auto;
    border-left: 1px solid rgba(0, 0, 0, 0.05);
}

.sidebar-section {
    margin-bottom: 2rem;
}

.sidebar-section h3 {
    color: #667eea;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.stats-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
}

.stat-item {
    text-align: center;
    padding: 1rem;
    background: rgba(102, 126, 234, 0.1);
    border-radius: 10px;
}

.stat-value {
    display: block;
    font-size: 1.5rem;
    font-weight: bold;
    color: #667eea;
}

.stat-label {
    font-size: 0.9rem;
    color: #666;
}

.settings-group {
    display: flex;
    flex-direction: column;
    gap: 1rem;
}

.setting-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.5rem 0;
    cursor: pointer;
}

.setting-item input[type="checkbox"] {
    width: 20px;
    height: 20px;
    accent-color: #667eea;
}

.recent-chats {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
}

.chat-item {
    padding: 0.75rem;
    background: rgba(102, 126, 234, 0.05);
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.3s ease;
    border-right: 3px solid transparent;
}

.chat-item:hover {
    background: rgba(102, 126, 234, 0.1);
    border-right-color: #667eea;
}

/* Loading Overlay */
.loading-overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.7);
    display: none;
    align-items: center;
    justify-content: center;
    z-index: 9999;
}

.loading-spinner {
    text-align: center;
    color: white;
}

.loading-spinner i {
    font-size: 3rem;
    margin-bottom: 1rem;
    animation: spin 2s linear infinite;
}

@keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}

/* Dark Mode */
body.dark-mode {
    background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
    color: #fff;
}

.dark-mode .header,
.dark-mode .chat-container,
.dark-mode .sidebar {
    background: rgba(45, 45, 45, 0.9);
    color: #fff;
}

.dark-mode .message-content {
    background: rgba(60, 60, 60, 0.8);
    color: #fff;
}

/* Responsive Design */
@media (max-width: 768px) {
    .main-content {
        flex-direction: column;
    }
    
    .sidebar {
        width: 100%;
        height: auto;
        border-left: none;
        border-top: 1px solid rgba(0, 0, 0, 0.05);
    }
    
    .input-group {
        flex-direction: column;
        gap: 1rem;
    }
    
    .message-content {
        max-width: 90%;
    }
}

/* Scrollbar Styling */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: rgba(0, 0, 0, 0.05);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb {
    background: rgba(102, 126, 234, 0.3);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: rgba(102, 126, 234, 0.5);
}
        """
        
        css_path = Path("interfaces/web/static/css/style.css")
        css_path.parent.mkdir(parents=True, exist_ok=True)
        with open(css_path, 'w', encoding='utf-8') as f:
            f.write(css_content)
        
        # إنشاء ملف JavaScript
        js_content = """
// المتغيرات العامة
let socket = null;
let isRecording = false;
let mediaRecorder = null;
let audioChunks = [];
let messageCount = 0;

// تهيئة التطبيق
document.addEventListener('DOMContentLoaded', function() {
    initializeSocket();
    setupEventListeners();
    initializeComponents();
});

// تهيئة اتصال Socket.IO
function initializeSocket() {
    socket = io();
    
    socket.on('connect', function() {
        updateConnectionStatus(true);
        console.log('متصل بالخادم');
    });
    
    socket.on('disconnect', function() {
        updateConnectionStatus(false);
        console.log('انقطع الاتصال بالخادم');
    });
    
    socket.on('assistant_response', function(data) {
        hideLoading();
        displayMessage(data.text, 'assistant', data);
        updateStats(data);
    });
    
    socket.on('error', function(error) {
        hideLoading();
        displayMessage('عذراً، حدث خطأ: ' + error.message, 'assistant', {error: true});
    });
}

// إعداد مستمعي الأحداث
function setupEventListeners() {
    const messageInput = document.getElementById('messageInput');
    const sendBtn = document.getElementById('sendBtn');
    const voiceBtn = document.getElementById('voiceBtn');
    const imageBtn = document.getElementById('imageBtn');
    const imageInput = document.getElementById('imageInput');
    const darkModeToggle = document.getElementById('darkMode');
    
    // إرسال الرسالة
    sendBtn.addEventListener('click', sendMessage);
    messageInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    
    // تسجيل صوتي
    voiceBtn.addEventListener('click', toggleVoiceRecording);
    
    // رفع صورة
    imageBtn.addEventListener('click', () => imageInput.click());
    imageInput.addEventListener('change', handleImageUpload);
    
    // الوضع المظلم
    darkModeToggle.addEventListener('change', toggleDarkMode);
    
    // تعديل حجم textarea تلقائياً
    messageInput.addEventListener('input', autoResizeTextarea);
}

// تهيئة المكونات
function initializeComponents() {
    // تحديث الإحصائيات
    updateMessageCount();
    
    // تحميل المحادثات الأخيرة
    loadRecentChats();
    
    // تطبيق الإعدادات المحفوظة
    applyStoredSettings();
}

// إرسال رسالة
function sendMessage() {
    const messageInput = document.getElementById('messageInput');
    const message = messageInput.value.trim();
    
    if (!message) return;
    
    // عرض رسالة المستخدم
    displayMessage(message, 'user');
    
    // إرسال للخادم
    showLoading();
    socket.emit('user_message', {
        text: message,
        timestamp: new Date().toISOString(),
        session_id: getSessionId()
    });
    
    // مسح الإدخال
    messageInput.value = '';
    autoResizeTextarea.call(messageInput);
}

// عرض رسالة في المحادثة
function displayMessage(text, sender, data = {}) {
    const chatMessages = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;
    
    const icon = sender === 'user' ? 'fas fa-user' : 'fas fa-robot';
    const timestamp = new Date().toLocaleTimeString('ar-SA');
    
    let additionalContent = '';
    
    // إضافة محتوى إضافي للاستجابات المتقدمة
    if (data.confidence) {
        additionalContent += `<div class="confidence-indicator">مستوى الثقة: ${Math.round(data.confidence * 100)}%</div>`;
    }
    
    if (data.suggestions && data.suggestions.length > 0) {
        additionalContent += '<div class="suggestions"><h4>اقتراحات:</h4><ul>';
        data.suggestions.forEach(suggestion => {
            additionalContent += `<li class="suggestion-item" onclick="sendSuggestion('${suggestion}')">${suggestion}</li>`;
        });
        additionalContent += '</ul></div>';
    }
    
    messageDiv.innerHTML = `
        <div class="message-content">
            <i class="${icon}"></i>
            <div class="text">
                <div class="message-text">${text}</div>
                ${additionalContent}
                <div class="message-time">${timestamp}</div>
            </div>
        </div>
    `;
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    updateMessageCount();
}

// إرسال اقتراح
function sendSuggestion(suggestion) {
    document.getElementById('messageInput').value = suggestion;
    sendMessage();
}

// تسجيل صوتي
async function toggleVoiceRecording() {
    const voiceBtn = document.getElementById('voiceBtn');
    
    if (!isRecording) {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);
            audioChunks = [];
            
            mediaRecorder.ondataavailable = function(event) {
                audioChunks.push(event.data);
            };
            
            mediaRecorder.onstop = function() {
                const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                sendAudioMessage(audioBlob);
            };
            
            mediaRecorder.start();
            isRecording = true;
            voiceBtn.classList.add('active');
            voiceBtn.innerHTML = '<i class="fas fa-stop"></i>';
            
        } catch (error) {
            console.error('خطأ في الوصول للميكروفون:', error);
            alert('لا يمكن الوصول للميكروفون');
        }
    } else {
        mediaRecorder.stop();
        isRecording = false;
        voiceBtn.classList.remove('active');
        voiceBtn.innerHTML = '<i class="fas fa-microphone"></i>';
        
        // إيقاف المسارات
        mediaRecorder.stream.getTracks().forEach(track => track.stop());
    }
}

// إرسال رسالة صوتية
function sendAudioMessage(audioBlob) {
    const reader = new FileReader();
    reader.onload = function() {
        const base64Audio = reader.result.split(',')[1];
        
        showLoading();
        socket.emit('voice_message', {
            audio_data: base64Audio,
            timestamp: new Date().toISOString(),
            session_id: getSessionId()
        });
        
        displayMessage('🎤 رسالة صوتية', 'user');
    };
    reader.readAsDataURL(audioBlob);
}

// رفع صورة
function handleImageUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    // التحقق من نوع الملف
    if (!file.type.startsWith('image/')) {
        alert('يرجى اختيار ملف صورة صحيح');
        return;
    }
    
    // التحقق من حجم الملف (5MB)
    if (file.size > 5 * 1024 * 1024) {
        alert('حجم الصورة كبير جداً (الحد الأقصى 5MB)');
        return;
    }
    
    const reader = new FileReader();
    reader.onload = function() {
        const base64Image = reader.result.split(',')[1];
        
        showLoading();
        socket.emit('image_message', {
            image_data: base64Image,
            filename: file.name,
            timestamp: new Date().toISOString(),
            session_id: getSessionId()
        });
        
        // عرض الصورة في المحادثة
        displayImageMessage(reader.result, file.name);
    };
    reader.readAsDataURL(file);
}

// عرض صورة في المحادثة
function displayImageMessage(imageSrc, filename) {
    const chatMessages = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message user-message';
    
    messageDiv.innerHTML = `
        <div class="message-content">
            <i class="fas fa-user"></i>
            <div class="text">
                <div class="image-message">
                    <img src="${imageSrc}" alt="${filename}" style="max-width: 300px; border-radius: 10px;">
                    <div class="image-filename">${filename}</div>
                </div>
                <div class="message-time">${new Date().toLocaleTimeString('ar-SA')}</div>
            </div>
        </div>
    `;
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    updateMessageCount();
}

// تعديل حجم textarea تلقائياً
function autoResizeTextarea() {
    this.style.height = 'auto';
    this.style.height = this.scrollHeight + 'px';
}

// إظهار/إخفاء التحميل
function showLoading() {
    document.getElementById('loadingOverlay').style.display = 'flex';
}

function hideLoading() {
    document.getElementById('loadingOverlay').style.display = 'none';
}

// تحديث حالة الاتصال
function updateConnectionStatus(connected) {
    const statusDot = document.getElementById('connectionStatus');
    const statusText = document.getElementById('statusText');
    
    if (connected) {
        statusDot.style.background = '#4CAF50';
        statusText.textContent = 'متصل';
    } else {
        statusDot.style.background = '#f44336';
        statusText.textContent = 'غير متصل';
    }
}

// تحديث عداد الرسائل
function updateMessageCount() {
    messageCount++;
    document.getElementById('messagesCount').textContent = messageCount;
}

// تحديث الإحصائيات
function updateStats(data) {
    if (data.confidence) {
        const confidencePercent = Math.round(data.confidence * 100);
        document.getElementById('confidenceLevel').textContent = confidencePercent + '%';
    }
}

// تبديل الوضع المظلم
function toggleDarkMode() {
    document.body.classList.toggle('dark-mode');
    localStorage.setItem('darkMode', document.body.classList.contains('dark-mode'));
}

// تطبيق الإعدادات المحفوظة
function applyStoredSettings() {
    // الوضع المظلم
    if (localStorage.getItem('darkMode') === 'true') {
        document.body.classList.add('dark-mode');
        document.getElementById('darkMode').checked = true;
    }
    
    // الأوامر الصوتية
    const voiceCommands = localStorage.getItem('voiceCommands');
    if (voiceCommands !== null) {
        document.getElementById('voiceCommands').checked = voiceCommands === 'true';
    }
    
    // الإشعارات
    const notifications = localStorage.getItem('notifications');
    if (notifications !== null) {
        document.getElementById('notifications').checked = notifications === 'true';
    }
}

// حفظ الإعدادات
function saveSettings() {
    localStorage.setItem('voiceCommands', document.getElementById('voiceCommands').checked);
    localStorage.setItem('notifications', document.getElementById('notifications').checked);
}

// تحميل المحادثات الأخيرة
function loadRecentChats() {
    // هذه وظيفة تجريبية - ستكون متصلة بقاعدة البيانات لاحقاً
    const recentChats = document.getElementById('recentChats');
    const sampleChats = [
        'كيف أتعلم البرمجة؟',
        'ما هو أفضل نظام غذائي؟',
        'تحليل صورة المشروع'
    ];
    
    sampleChats.forEach(chat => {
        const chatItem = document.createElement('div');
        chatItem.className = 'chat-item';
        chatItem.textContent = chat;
        chatItem.onclick = function() {
            document.getElementById('messageInput').value = chat;
        };
        recentChats.appendChild(chatItem);
    });
}

// الحصول على معرف الجلسة
function getSessionId() {
    let sessionId = localStorage.getItem('sessionId');
    if (!sessionId) {
        sessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
        localStorage.setItem('sessionId', sessionId);
    }
    return sessionId;
}

// حفظ الإعدادات عند التغيير
document.addEventListener('change', function(e) {
    if (e.target.type === 'checkbox' && e.target.id !== 'darkMode') {
        saveSettings();
    }
});
        """
        
        js_path = Path("interfaces/web/static/js/app.js")
        js_path.parent.mkdir(parents=True, exist_ok=True)
        with open(js_path, 'w', encoding='utf-8') as f:
            f.write(js_content)
    
    def _setup_routes(self):
        """إعداد مسارات Flask"""
        
        @self.app.route('/')
        def index():
            """الصفحة الرئيسية"""
            return render_template('index.html')
        
        @self.app.route('/api/status')
        def api_status():
            """حالة النظام"""
            return jsonify({
                "status": "active",
                "stats": self.interface_stats,
                "active_sessions": len(self.active_sessions)
            })
        
        @self.app.route('/api/download-chat/<session_id>')
        def download_chat(session_id):
            """تحميل محادثة"""
            try:
                chat_file = Path(f"data/chat_logs/{session_id}.json")
                if chat_file.exists():
                    return send_file(chat_file, as_attachment=True)
                else:
                    return jsonify({"error": "المحادثة غير موجودة"}), 404
            except Exception as e:
                return jsonify({"error": str(e)}), 500
    
    def _setup_socket_events(self):
        """إعداد أحداث SocketIO"""
        
        @self.socketio.on('connect')
        def handle_connect():
            """اتصال جديد"""
            session_id = request.sid
            self.active_sessions[session_id] = {
                "connected_at": datetime.now(),
                "message_count": 0,
                "user_info": {}
            }
            
            self.interface_stats["total_connections"] += 1
            self.interface_stats["active_users"] = len(self.active_sessions)
            
            self.logger.info(f"اتصال جديد: {session_id}")
            
            # إرسال رسالة ترحيب
            emit('assistant_response', {
                "text": "مرحباً! كيف يمكنني مساعدتك اليوم؟",
                "confidence": 1.0,
                "suggestions": [
                    "ما هي قدراتك؟",
                    "ساعدني في تحليل صورة",
                    "أريد فتح تطبيق معين"
                ]
            })
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """قطع الاتصال"""
            session_id = request.sid
            if session_id in self.active_sessions:
                # حفظ محادثة الجلسة
                self._save_session_chat(session_id)
                del self.active_sessions[session_id]
            
            self.interface_stats["active_users"] = len(self.active_sessions)
            self.logger.info(f"قطع الاتصال: {session_id}")
        
        @self.socketio.on('user_message')
        def handle_user_message(data):
            """رسالة نصية من المستخدم"""
            asyncio.create_task(self._process_user_message(data, request.sid))
        
        @self.socketio.on('voice_message')
        def handle_voice_message(data):
            """رسالة صوتية من المستخدم"""
            asyncio.create_task(self._process_voice_message(data, request.sid))
        
        @self.socketio.on('image_message')
        def handle_image_message(data):
            """صورة من المستخدم"""
            asyncio.create_task(self._process_image_message(data, request.sid))
    
    async def _process_user_message(self, data: Dict[str, Any], session_id: str):
        """معالجة رسالة المستخدم"""
        try:
            user_text = data.get("text", "")
            
            # تحديث إحصائيات الجلسة
            if session_id in self.active_sessions:
                self.active_sessions[session_id]["message_count"] += 1
            
            self.interface_stats["messages_processed"] += 1
            
            # محاكاة معالجة الرسالة (ستكون متصلة بالمحرك الرئيسي)
            await asyncio.sleep(1)  # محاكاة وقت المعالجة
            
            # إرسال استجابة
            response = {
                "text": f"شكراً لرسالتك: '{user_text}'. أعمل على تطوير قدراتي لتقديم إجابات أفضل!",
                "confidence": 0.8,
                "intent": "general",
                "suggestions": [
                    "أخبرني المزيد",
                    "ما رأيك في هذا؟",
                    "هل يمكنك المساعدة أكثر؟"
                ]
            }
            
            self.socketio.emit('assistant_response', response, room=session_id)
            
            # حفظ المحادثة
            self._log_conversation(session_id, user_text, response["text"])
            
        except Exception as e:
            self.logger.error(f"خطأ في معالجة رسالة المستخدم: {e}")
            self.socketio.emit('error', {"message": str(e)}, room=session_id)
    
    async def _process_voice_message(self, data: Dict[str, Any], session_id: str):
        """معالجة رسالة صوتية"""
        try:
            audio_data = data.get("audio_data", "")
            
            self.interface_stats["voice_interactions"] += 1
            
            # محاكاة تحويل الصوت إلى نص
            await asyncio.sleep(2)
            
            response = {
                "text": "تم استلام الرسالة الصوتية! أعمل على تطوير ميزة التعرف على الكلام.",
                "confidence": 0.7,
                "intent": "voice_recognition",
                "suggestions": [
                    "جرب إرسال رسالة نصية",
                    "ما رأيك في الواجهة؟"
                ]
            }
            
            self.socketio.emit('assistant_response', response, room=session_id)
            
        except Exception as e:
            self.logger.error(f"خطأ في معالجة الرسالة الصوتية: {e}")
            self.socketio.emit('error', {"message": str(e)}, room=session_id)
    
    async def _process_image_message(self, data: Dict[str, Any], session_id: str):
        """معالجة صورة"""
        try:
            image_data = data.get("image_data", "")
            filename = data.get("filename", "image.jpg")
            
            self.interface_stats["image_uploads"] += 1
            
            # حفظ الصورة
            upload_dir = Path("data/uploads")
            upload_dir.mkdir(parents=True, exist_ok=True)
            
            image_path = upload_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}"
            
            # فك تشفير Base64 وحفظ الصورة
            image_bytes = base64.b64decode(image_data)
            with open(image_path, 'wb') as f:
                f.write(image_bytes)
            
            # محاكاة تحليل الصورة
            await asyncio.sleep(3)
            
            response = {
                "text": f"تم استلام الصورة '{filename}' وحفظها بنجاح! أعمل على تطوير ميزات تحليل الصور المتقدمة.",
                "confidence": 0.9,
                "intent": "image_analysis",
                "suggestions": [
                    "حلل محتوى الصورة",
                    "ما رأيك في الصورة؟",
                    "اشرح ما تراه"
                ],
                "image_info": {
                    "filename": filename,
                    "saved_path": str(image_path),
                    "size": len(image_bytes)
                }
            }
            
            self.socketio.emit('assistant_response', response, room=session_id)
            
        except Exception as e:
            self.logger.error(f"خطأ في معالجة الصورة: {e}")
            self.socketio.emit('error', {"message": str(e)}, room=session_id)
    
    def _log_conversation(self, session_id: str, user_message: str, assistant_response: str):
        """تسجيل المحادثة"""
        try:
            log_dir = Path("data/chat_logs")
            log_dir.mkdir(parents=True, exist_ok=True)
            
            log_file = log_dir / f"{session_id}.json"
            
            conversation_entry = {
                "timestamp": datetime.now().isoformat(),
                "user_message": user_message,
                "assistant_response": assistant_response
            }
            
            # قراءة المحادثات الموجودة أو إنشاء قائمة جديدة
            if log_file.exists():
                with open(log_file, 'r', encoding='utf-8') as f:
                    conversations = json.load(f)
            else:
                conversations = []
            
            conversations.append(conversation_entry)
            
            # حفظ المحادثات المحدثة
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(conversations, f, ensure_ascii=False, indent=2)
            
        except Exception as e:
            self.logger.error(f"خطأ في تسجيل المحادثة: {e}")
    
    def _save_session_chat(self, session_id: str):
        """حفظ محادثة الجلسة عند الإغلاق"""
        try:
            if session_id in self.active_sessions:
                session_info = self.active_sessions[session_id]
                
                session_summary = {
                    "session_id": session_id,
                    "connected_at": session_info["connected_at"].isoformat(),
                    "disconnected_at": datetime.now().isoformat(),
                    "message_count": session_info["message_count"],
                    "duration_minutes": (datetime.now() - session_info["connected_at"]).total_seconds() / 60
                }
                
                # حفظ ملخص الجلسة
                summary_dir = Path("data/session_summaries")
                summary_dir.mkdir(parents=True, exist_ok=True)
                
                summary_file = summary_dir / f"{session_id}_summary.json"
                with open(summary_file, 'w', encoding='utf-8') as f:
                    json.dump(session_summary, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            self.logger.error(f"خطأ في حفظ ملخص الجلسة: {e}")
    
    def start_server(self, debug: bool = False):
        """تشغيل الخادم"""
        self.logger.info(f"🌐 بدء واجهة الويب على {self.host}:{self.port}")
        
        self.socketio.run(
            self.app,
            host=self.host,
            port=self.port,
            debug=debug,
            allow_unsafe_werkzeug=True
        )
    
    def get_interface_stats(self) -> Dict[str, Any]:
        """الحصول على إحصائيات الواجهة"""
        uptime = datetime.now() - self.interface_stats["start_time"]
        
        return {
            **self.interface_stats,
            "uptime_minutes": uptime.total_seconds() / 60,
            "average_messages_per_user": (
                self.interface_stats["messages_processed"] / max(self.interface_stats["total_connections"], 1)
            )
        }

# إنشاء مثيل عام
web_interface = SmartWebInterface()

def get_web_interface() -> SmartWebInterface:
    """الحصول على واجهة الويب"""
    return web_interface

if __name__ == "__main__":
    # تشغيل واجهة الويب
    interface = get_web_interface()
    interface.start_server(debug=True)
