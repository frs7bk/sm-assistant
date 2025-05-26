
/**
 * المساعد الذكي الموحد - المنطق الأمامي
 * تطبيق JavaScript احترافي مع إدارة حالة متقدمة
 */

class AssistantApp {
    constructor() {
        // إعدادات التطبيق
        this.config = {
            apiUrl: window.location.origin,
            wsUrl: `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}`,
            userId: 'user' + Math.random().toString(36).substr(2, 9),
            audioSampleRate: 16000,
            maxRetries: 3
        };

        // حالة التطبيق
        this.state = {
            currentMode: 'text', // text | voice
            isConnected: false,
            isRecording: false,
            isProcessing: false,
            isSpeaking: false,
            messages: [],
            currentTranscript: '',
            stats: {
                totalMessages: 0,
                avgResponseTime: 0,
                successRate: 100
            }
        };

        // اتصال WebSocket
        this.websocket = null;
        this.audioContext = null;
        this.mediaRecorder = null;
        this.audioChunks = [];

        // عناصر DOM
        this.elements = {};

        // ربط الطرق
        this.handleSendMessage = this.handleSendMessage.bind(this);
        this.handleKeyPress = this.handleKeyPress.bind(this);
        this.handleStartRecording = this.handleStartRecording.bind(this);
        this.handleStopRecording = this.handleStopRecording.bind(this);
        this.switchToTextMode = this.switchToTextMode.bind(this);
        this.switchToVoiceMode = this.switchToVoiceMode.bind(this);

        this.init();
    }

    /**
     * تهيئة التطبيق
     */
    async init() {
        console.log('🚀 تهيئة المساعد الذكي الموحد...');
        
        // ربط عناصر DOM
        this.bindElements();
        
        // ربط الأحداث
        this.bindEvents();
        
        // فحص الصحة
        await this.checkHealth();
        
        // تهيئة الوضع الافتراضي
        this.switchToTextMode();
        
        console.log('✅ تم تهيئة التطبيق بنجاح');
    }

    /**
     * ربط عناصر DOM
     */
    bindElements() {
        this.elements = {
            // Mode selector
            textModeBtn: document.getElementById('textModeBtn'),
            voiceModeBtn: document.getElementById('voiceModeBtn'),
            
            // Chat interface
            chatInterface: document.getElementById('chatInterface'),
            chatMessages: document.getElementById('chatMessages'),
            messageInput: document.getElementById('messageInput'),
            sendBtn: document.getElementById('sendBtn'),
            
            // Voice interface
            voiceInterface: document.getElementById('voiceInterface'),
            audioCanvas: document.getElementById('audioCanvas'),
            voiceStatus: document.getElementById('voiceStatus'),
            transcriptText: document.getElementById('transcriptText'),
            startRecordingBtn: document.getElementById('startRecordingBtn'),
            stopRecordingBtn: document.getElementById('stopRecordingBtn'),
            assistantResponse: document.getElementById('assistantResponse'),
            
            // Status bar
            connectionIcon: document.getElementById('connectionIcon'),
            connectionStatus: document.getElementById('connectionStatus'),
            responseTime: document.getElementById('responseTime'),
            confidenceLevel: document.getElementById('confidenceLevel'),
            
            // Loading
            loadingOverlay: document.getElementById('loadingOverlay'),
            
            // Modals
            settingsModal: document.getElementById('settingsModal'),
            statsModal: document.getElementById('statsModal'),
            settingsBtn: document.getElementById('settingsBtn'),
            statsBtn: document.getElementById('statsBtn'),
            closeSettingsBtn: document.getElementById('closeSettingsBtn'),
            closeStatsBtn: document.getElementById('closeStatsBtn'),
            
            // Settings
            userIdInput: document.getElementById('userIdInput'),
            speechSpeedSlider: document.getElementById('speechSpeedSlider'),
            speechSpeedValue: document.getElementById('speechSpeedValue'),
            enableAudioCheckbox: document.getElementById('enableAudioCheckbox'),
            saveSettingsBtn: document.getElementById('saveSettingsBtn'),
            cancelSettingsBtn: document.getElementById('cancelSettingsBtn'),
            
            // Stats
            totalMessages: document.getElementById('totalMessages'),
            avgResponseTime: document.getElementById('avgResponseTime'),
            successRate: document.getElementById('successRate')
        };
    }

    /**
     * ربط الأحداث
     */
    bindEvents() {
        // Mode switching
        this.elements.textModeBtn?.addEventListener('click', this.switchToTextMode);
        this.elements.voiceModeBtn?.addEventListener('click', this.switchToVoiceMode);
        
        // Chat events
        this.elements.sendBtn?.addEventListener('click', this.handleSendMessage);
        this.elements.messageInput?.addEventListener('keypress', this.handleKeyPress);
        
        // Voice events
        this.elements.startRecordingBtn?.addEventListener('click', this.handleStartRecording);
        this.elements.stopRecordingBtn?.addEventListener('click', this.handleStopRecording);
        
        // Modal events
        this.elements.settingsBtn?.addEventListener('click', () => this.showModal('settings'));
        this.elements.statsBtn?.addEventListener('click', () => this.showModal('stats'));
        this.elements.closeSettingsBtn?.addEventListener('click', () => this.hideModal('settings'));
        this.elements.closeStatsBtn?.addEventListener('click', () => this.hideModal('stats'));
        
        // Settings events
        this.elements.speechSpeedSlider?.addEventListener('input', (e) => {
            this.elements.speechSpeedValue.textContent = e.target.value + 'x';
        });
        this.elements.saveSettingsBtn?.addEventListener('click', this.saveSettings.bind(this));
        this.elements.cancelSettingsBtn?.addEventListener('click', () => this.hideModal('settings'));
        
        // Close modals on backdrop click
        [this.elements.settingsModal, this.elements.statsModal].forEach(modal => {
            modal?.addEventListener('click', (e) => {
                if (e.target === modal) {
                    this.hideModal(modal.id.replace('Modal', ''));
                }
            });
        });
        
        // Window events
        window.addEventListener('beforeunload', () => {
            this.cleanup();
        });
    }

    /**
     * فحص صحة التطبيق
     */
    async checkHealth() {
        try {
            const response = await fetch(`${this.config.apiUrl}/health`);
            const data = await response.json();
            
            if (data.status === 'healthy') {
                this.updateConnectionStatus(true);
                console.log('✅ التطبيق يعمل بصحة جيدة');
            } else {
                this.updateConnectionStatus(false);
                console.warn('⚠️ مشكلة في صحة التطبيق');
            }
        } catch (error) {
            console.error('❌ خطأ في فحص الصحة:', error);
            this.updateConnectionStatus(false);
        }
    }

    /**
     * تحديث حالة الاتصال
     */
    updateConnectionStatus(connected) {
        this.state.isConnected = connected;
        
        if (this.elements.connectionIcon && this.elements.connectionStatus) {
            this.elements.connectionIcon.className = connected 
                ? 'fas fa-circle status-connected' 
                : 'fas fa-circle status-disconnected';
            this.elements.connectionStatus.textContent = connected ? 'متصل' : 'غير متصل';
        }
    }

    /**
     * التبديل إلى الوضع النصي
     */
    switchToTextMode() {
        this.state.currentMode = 'text';
        
        // تحديث أزرار الوضع
        this.elements.textModeBtn?.classList.add('active');
        this.elements.voiceModeBtn?.classList.remove('active');
        
        // إظهار/إخفاء الواجهات
        this.elements.chatInterface?.classList.remove('hidden');
        this.elements.voiceInterface?.classList.add('hidden');
        
        // إغلاق اتصال WebSocket إذا كان مفتوحاً
        this.closeWebSocket();
        
        // التركيز على حقل الإدخال
        this.elements.messageInput?.focus();
        
        console.log('📝 تم التبديل إلى الوضع النصي');
    }

    /**
     * التبديل إلى الوضع الصوتي
     */
    async switchToVoiceMode() {
        this.state.currentMode = 'voice';
        
        // تحديث أزرار الوضع
        this.elements.textModeBtn?.classList.remove('active');
        this.elements.voiceModeBtn?.classList.add('active');
        
        // إظهار/إخفاء الواجهات
        this.elements.chatInterface?.classList.add('hidden');
        this.elements.voiceInterface?.classList.remove('hidden');
        
        // تهيئة WebSocket
        await this.initializeWebSocket();
        
        // تهيئة Audio Context
        await this.initializeAudio();
        
        console.log('🎤 تم التبديل إلى الوضع الصوتي');
    }

    /**
     * معالجة إرسال الرسالة النصية
     */
    async handleSendMessage() {
        const message = this.elements.messageInput?.value.trim();
        if (!message) return;

        // تنظيف حقل الإدخال
        this.elements.messageInput.value = '';
        
        // إضافة رسالة المستخدم للشاشة
        this.addMessage('user', message);
        
        // تعطيل زر الإرسال
        this.setLoadingState(true);
        
        try {
            const startTime = Date.now();
            
            const response = await fetch(`${this.config.apiUrl}/api/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: message,
                    user_id: this.config.userId,
                    context: {}
                })
            });

            if (!response.ok) {
                throw new Error(`خطأ في الخادم: ${response.status}`);
            }

            const data = await response.json();
            const responseTime = Date.now() - startTime;
            
            // إضافة رد المساعد
            this.addMessage('assistant', data.text, {
                confidence: data.confidence,
                suggestions: data.suggestions,
                responseTime: responseTime
            });
            
            // تحديث الإحصائيات
            this.updateStats(responseTime, data.confidence);
            
        } catch (error) {
            console.error('❌ خطأ في إرسال الرسالة:', error);
            this.addMessage('assistant', 'عذراً، حدث خطأ في الاتصال. يرجى المحاولة مرة أخرى.', {
                confidence: 0,
                isError: true
            });
        } finally {
            this.setLoadingState(false);
        }
    }

    /**
     * معالجة الضغط على Enter
     */
    handleKeyPress(event) {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            this.handleSendMessage();
        }
    }

    /**
     * بدء التسجيل الصوتي
     */
    async handleStartRecording() {
        if (!this.audioContext) {
            await this.initializeAudio();
        }

        try {
            // طلب إذن الميكروفون
            const stream = await navigator.mediaDevices.getUserMedia({ 
                audio: {
                    sampleRate: this.config.audioSampleRate,
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true
                } 
            });

            // إنشاء MediaRecorder
            this.mediaRecorder = new MediaRecorder(stream);
            this.audioChunks = [];

            this.mediaRecorder.ondataavailable = (event) => {
                this.audioChunks.push(event.data);
            };

            this.mediaRecorder.onstop = () => {
                this.processAudioRecording();
            };

            // بدء التسجيل
            this.mediaRecorder.start();
            this.state.isRecording = true;

            // تحديث الواجهة
            this.updateVoiceUI('recording');
            
            // إرسال إشارة بدء التسجيل عبر WebSocket
            this.sendWebSocketMessage({
                type: 'start_recording',
                user_id: this.config.userId
            });

            console.log('🎤 بدء التسجيل الصوتي');

        } catch (error) {
            console.error('❌ خطأ في بدء التسجيل:', error);
            this.showError('لا يمكن الوصول إلى الميكروفون. يرجى التحقق من الأذونات.');
        }
    }

    /**
     * إيقاف التسجيل الصوتي
     */
    handleStopRecording() {
        if (this.mediaRecorder && this.state.isRecording) {
            this.mediaRecorder.stop();
            this.state.isRecording = false;
            
            // إيقاف تدفق الصوت
            this.mediaRecorder.stream.getTracks().forEach(track => track.stop());
            
            // تحديث الواجهة
            this.updateVoiceUI('processing');
            
            console.log('⏹️ تم إيقاف التسجيل');
        }
    }

    /**
     * معالجة التسجيل الصوتي
     */
    async processAudioRecording() {
        if (this.audioChunks.length === 0) return;

        try {
            // تحويل الصوت إلى Blob
            const audioBlob = new Blob(this.audioChunks, { type: 'audio/wav' });
            
            // إرسال إشارة إيقاف التسجيل عبر WebSocket
            this.sendWebSocketMessage({
                type: 'stop_recording',
                user_id: this.config.userId,
                audio_size: audioBlob.size
            });

            console.log('📤 تم إرسال التسجيل للمعالجة');

        } catch (error) {
            console.error('❌ خطأ في معالجة التسجيل:', error);
            this.updateVoiceUI('error');
            this.showError('خطأ في معالجة التسجيل الصوتي');
        }
    }

    /**
     * تهيئة WebSocket
     */
    async initializeWebSocket() {
        if (this.websocket) {
            this.websocket.close();
        }

        try {
            const wsUrl = `${this.config.wsUrl}/ws/voice/${this.config.userId}`;
            this.websocket = new WebSocket(wsUrl);

            this.websocket.onopen = () => {
                console.log('🔌 تم إنشاء اتصال WebSocket');
                this.updateConnectionStatus(true);
            };

            this.websocket.onmessage = (event) => {
                this.handleWebSocketMessage(event);
            };

            this.websocket.onclose = () => {
                console.log('📴 تم إغلاق اتصال WebSocket');
                this.updateConnectionStatus(false);
            };

            this.websocket.onerror = (error) => {
                console.error('❌ خطأ في WebSocket:', error);
                this.updateConnectionStatus(false);
            };

        } catch (error) {
            console.error('❌ خطأ في تهيئة WebSocket:', error);
        }
    }

    /**
     * معالجة رسائل WebSocket
     */
    handleWebSocketMessage(event) {
        try {
            const data = JSON.parse(event.data);
            console.log('📨 رسالة WebSocket:', data);

            switch (data.type) {
                case 'session_started':
                    this.updateVoiceStatus(data.message);
                    break;

                case 'recording_started':
                    this.updateVoiceStatus(data.message);
                    break;

                case 'recording_stopped':
                    this.updateVoiceStatus(data.message);
                    break;

                case 'transcript_ready':
                    this.updateTranscript(data.transcript);
                    break;

                case 'response_ready':
                    this.showAssistantResponse(data);
                    break;

                case 'error':
                    this.showError(data.message);
                    this.updateVoiceUI('idle');
                    break;

                default:
                    console.log('نوع رسالة غير معروف:', data.type);
            }

        } catch (error) {
            console.error('❌ خطأ في معالجة رسالة WebSocket:', error);
        }
    }

    /**
     * إرسال رسالة عبر WebSocket
     */
    sendWebSocketMessage(data) {
        if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
            this.websocket.send(JSON.stringify(data));
        }
    }

    /**
     * إغلاق WebSocket
     */
    closeWebSocket() {
        if (this.websocket) {
            this.websocket.close();
            this.websocket = null;
        }
    }

    /**
     * تهيئة Audio Context
     */
    async initializeAudio() {
        try {
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            console.log('🔊 تم تهيئة Audio Context');
        } catch (error) {
            console.error('❌ خطأ في تهيئة Audio Context:', error);
        }
    }

    /**
     * إضافة رسالة للمحادثة
     */
    addMessage(sender, text, options = {}) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}`;

        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        messageContent.textContent = text;

        const messageTime = document.createElement('div');
        messageTime.className = 'message-time';
        messageTime.textContent = new Date().toLocaleTimeString('ar-SA');

        messageDiv.appendChild(messageContent);
        messageDiv.appendChild(messageTime);

        // إضافة الاقتراحات للمساعد
        if (sender === 'assistant' && options.suggestions && options.suggestions.length > 0) {
            const suggestionsDiv = document.createElement('div');
            suggestionsDiv.className = 'message-suggestions';

            options.suggestions.forEach(suggestion => {
                const suggestionBtn = document.createElement('button');
                suggestionBtn.className = 'suggestion-btn';
                suggestionBtn.textContent = suggestion;
                suggestionBtn.addEventListener('click', () => {
                    this.elements.messageInput.value = suggestion;
                    this.handleSendMessage();
                });
                suggestionsDiv.appendChild(suggestionBtn);
            });

            messageDiv.appendChild(suggestionsDiv);
        }

        this.elements.chatMessages?.appendChild(messageDiv);
        this.scrollToBottom();

        // تحديث الإحصائيات
        this.state.messages.push({ sender, text, timestamp: new Date(), options });
        this.state.stats.totalMessages++;
    }

    /**
     * تمرير الصفحة لأسفل
     */
    scrollToBottom() {
        if (this.elements.chatMessages) {
            this.elements.chatMessages.scrollTop = this.elements.chatMessages.scrollHeight;
        }
    }

    /**
     * تعيين حالة التحميل
     */
    setLoadingState(loading) {
        if (this.elements.sendBtn) {
            this.elements.sendBtn.disabled = loading;
        }
        
        if (loading) {
            this.elements.loadingOverlay?.classList.remove('hidden');
        } else {
            this.elements.loadingOverlay?.classList.add('hidden');
        }
        
        this.state.isProcessing = loading;
    }

    /**
     * تحديث واجهة الصوت
     */
    updateVoiceUI(state) {
        const startBtn = this.elements.startRecordingBtn;
        const stopBtn = this.elements.stopRecordingBtn;

        switch (state) {
            case 'idle':
                startBtn?.classList.remove('hidden');
                stopBtn?.classList.add('hidden');
                this.updateVoiceStatus('اضغط للتحدث');
                break;

            case 'recording':
                startBtn?.classList.add('hidden');
                stopBtn?.classList.remove('hidden');
                this.updateVoiceStatus('جاري التسجيل... اضغط للإيقاف');
                break;

            case 'processing':
                startBtn?.classList.remove('hidden');
                stopBtn?.classList.add('hidden');
                this.updateVoiceStatus('جاري المعالجة...');
                break;

            case 'error':
                startBtn?.classList.remove('hidden');
                stopBtn?.classList.add('hidden');
                this.updateVoiceStatus('حدث خطأ. جرب مرة أخرى');
                break;
        }
    }

    /**
     * تحديث حالة الصوت
     */
    updateVoiceStatus(status) {
        if (this.elements.voiceStatus) {
            this.elements.voiceStatus.textContent = status;
        }
    }

    /**
     * تحديث النص المحول
     */
    updateTranscript(transcript) {
        if (this.elements.transcriptText) {
            this.elements.transcriptText.textContent = transcript;
        }
        this.state.currentTranscript = transcript;
    }

    /**
     * إظهار رد المساعد الصوتي
     */
    showAssistantResponse(data) {
        const responseDiv = this.elements.assistantResponse;
        if (!responseDiv) return;

        const responseText = responseDiv.querySelector('.response-text');
        if (responseText) {
            responseText.textContent = data.text;
        }

        responseDiv.classList.remove('hidden');

        // تحديث مستوى الثقة
        if (this.elements.confidenceLevel) {
            this.elements.confidenceLevel.textContent = Math.round(data.confidence * 100) + '%';
        }

        // العودة لحالة الخمول
        setTimeout(() => {
            this.updateVoiceUI('idle');
        }, 1000);
    }

    /**
     * إظهار رسالة خطأ
     */
    showError(message) {
        // يمكن تحسين هذا بإضافة نظام toast notifications
        alert(message);
    }

    /**
     * تحديث الإحصائيات
     */
    updateStats(responseTime, confidence) {
        // تحديث متوسط وقت الاستجابة
        const totalRequests = this.state.stats.totalMessages;
        const currentAvg = this.state.stats.avgResponseTime;
        this.state.stats.avgResponseTime = Math.round(
            (currentAvg * (totalRequests - 1) + responseTime) / totalRequests
        );

        // تحديث مستوى الثقة
        if (this.elements.confidenceLevel) {
            this.elements.confidenceLevel.textContent = Math.round(confidence * 100) + '%';
        }

        // تحديث وقت الاستجابة
        if (this.elements.responseTime) {
            this.elements.responseTime.textContent = responseTime + 'ms';
        }
    }

    /**
     * إظهار النافذة المنبثقة
     */
    showModal(type) {
        const modal = this.elements[type + 'Modal'];
        if (modal) {
            modal.classList.remove('hidden');
            
            // تحديث الإحصائيات إذا كانت نافذة الإحصائيات
            if (type === 'stats') {
                this.updateStatsModal();
            }
        }
    }

    /**
     * إخفاء النافذة المنبثقة
     */
    hideModal(type) {
        const modal = this.elements[type + 'Modal'];
        if (modal) {
            modal.classList.add('hidden');
        }
    }

    /**
     * تحديث نافذة الإحصائيات
     */
    updateStatsModal() {
        if (this.elements.totalMessages) {
            this.elements.totalMessages.textContent = this.state.stats.totalMessages;
        }
        
        if (this.elements.avgResponseTime) {
            this.elements.avgResponseTime.textContent = this.state.stats.avgResponseTime + 'ms';
        }
        
        if (this.elements.successRate) {
            this.elements.successRate.textContent = this.state.stats.successRate + '%';
        }
    }

    /**
     * حفظ الإعدادات
     */
    saveSettings() {
        // حفظ معرف المستخدم
        if (this.elements.userIdInput) {
            this.config.userId = this.elements.userIdInput.value || this.config.userId;
        }

        // حفظ الإعدادات في localStorage
        localStorage.setItem('assistantSettings', JSON.stringify({
            userId: this.config.userId,
            speechSpeed: this.elements.speechSpeedSlider?.value || 1,
            enableAudio: this.elements.enableAudioCheckbox?.checked || true
        }));

        this.hideModal('settings');
        console.log('✅ تم حفظ الإعدادات');
    }

    /**
     * تحميل الإعدادات المحفوظة
     */
    loadSettings() {
        try {
            const savedSettings = localStorage.getItem('assistantSettings');
            if (savedSettings) {
                const settings = JSON.parse(savedSettings);
                
                if (settings.userId) {
                    this.config.userId = settings.userId;
                    if (this.elements.userIdInput) {
                        this.elements.userIdInput.value = settings.userId;
                    }
                }
                
                if (settings.speechSpeed && this.elements.speechSpeedSlider) {
                    this.elements.speechSpeedSlider.value = settings.speechSpeed;
                    this.elements.speechSpeedValue.textContent = settings.speechSpeed + 'x';
                }
                
                if (typeof settings.enableAudio === 'boolean' && this.elements.enableAudioCheckbox) {
                    this.elements.enableAudioCheckbox.checked = settings.enableAudio;
                }
            }
        } catch (error) {
            console.error('❌ خطأ في تحميل الإعدادات:', error);
        }
    }

    /**
     * تنظيف الموارد
     */
    cleanup() {
        this.closeWebSocket();
        
        if (this.mediaRecorder && this.state.isRecording) {
            this.mediaRecorder.stop();
        }
        
        if (this.audioContext) {
            this.audioContext.close();
        }
        
        console.log('🧹 تم تنظيف الموارد');
    }
}

// تشغيل التطبيق عند تحميل الصفحة
document.addEventListener('DOMContentLoaded', () => {
    window.assistantApp = new AssistantApp();
});
