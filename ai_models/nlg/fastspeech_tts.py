
import torch
import soundfile as sf
from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer
import librosa
import numpy as np
from pydub import AudioSegment
from pydub.effects import normalize, compress_dynamic_range
import json
import os
from datetime import datetime

class UltraNaturalTTS:
    def __init__(self):
        print("[INFO] تحميل نظام TTS متقدم للصوت الطبيعي...")
        
        # نماذج متعددة للجودة العالية
        self.models = {
            'arabic': {
                'model': "tts_models/ar/cv/vits",
                'quality': 'high'
            },
            'english': {
                'model': "tts_models/en/ljspeech/tacotron2-DDC_ph",
                'quality': 'ultra_high'
            },
            'multilingual': {
                'model': "tts_models/multilingual/multi-dataset/xtts_v2",
                'quality': 'neural'
            }
        }
        
        self.manager = ModelManager()
        self.synthesizers = {}
        self.current_synthesizer = None
        
        # إعدادات الصوت الطبيعي المتقدمة
        self.voice_settings = {
            'speed_variance': 0.1,      # تنويع السرعة
            'pitch_variance': 0.05,     # تنويع النبرة  
            'pause_insertion': True,    # إدراج وقفات طبيعية
            'breathing_sounds': True,   # أصوات التنفس
            'emotion_modulation': True, # تعديل العاطفة
            'prosody_enhancement': True # تحسين الإيقاع
        }
        
        # خريطة المشاعر المتقدمة
        self.emotion_profiles = {
            'neutral': {
                'speed': 1.0, 'pitch': 1.0, 'energy': 1.0,
                'pauses': 'normal', 'breathing': 'subtle'
            },
            'happy': {
                'speed': 1.1, 'pitch': 1.2, 'energy': 1.3,
                'pauses': 'short', 'breathing': 'light'
            },
            'sad': {
                'speed': 0.8, 'pitch': 0.9, 'energy': 0.7,
                'pauses': 'long', 'breathing': 'heavy'
            },
            'excited': {
                'speed': 1.3, 'pitch': 1.4, 'energy': 1.5,
                'pauses': 'minimal', 'breathing': 'quick'
            },
            'calm': {
                'speed': 0.9, 'pitch': 1.0, 'energy': 0.8,
                'pauses': 'extended', 'breathing': 'deep'
            },
            'confident': {
                'speed': 1.0, 'pitch': 1.1, 'energy': 1.2,
                'pauses': 'strategic', 'breathing': 'controlled'
            }
        }
        
        self._load_default_synthesizer()
    
    def _load_default_synthesizer(self):
        """تحميل المحرك الافتراضي"""
        try:
            model_info = self.models['multilingual']
            model_path, config_path, _ = self.manager.download_model(model_info['model'])
            
            self.current_synthesizer = Synthesizer(
                tts_checkpoint=model_path,
                tts_config_path=config_path,
                use_cuda=torch.cuda.is_available()
            )
            print("✅ تم تحميل محرك TTS متعدد اللغات بنجاح")
        except Exception as e:
            print(f"❌ خطأ في تحميل المحرك: {e}")
            self._load_fallback_synthesizer()
    
    def _load_fallback_synthesizer(self):
        """تحميل محرك احتياطي"""
        try:
            model_info = self.models['english']
            model_path, config_path, _ = self.manager.download_model(model_info['model'])
            
            self.current_synthesizer = Synthesizer(
                tts_checkpoint=model_path,
                tts_config_path=config_path
            )
            print("✅ تم تحميل المحرك الاحتياطي")
        except Exception as e:
            print(f"❌ خطأ حرج في تحميل المحرك: {e}")
    
    def detect_language(self, text):
        """كشف لغة النص تلقائياً"""
        arabic_chars = sum(1 for char in text if '\u0600' <= char <= '\u06FF')
        total_chars = len([c for c in text if c.isalpha()])
        
        if total_chars == 0:
            return 'english'
        
        arabic_ratio = arabic_chars / total_chars
        
        if arabic_ratio > 0.3:
            return 'arabic'
        else:
            return 'english'
    
    def detect_emotion(self, text):
        """كشف العاطفة من النص"""
        text_lower = text.lower()
        
        emotion_keywords = {
            'happy': ['سعيد', 'فرح', 'ممتاز', 'رائع', 'جميل', 'happy', 'great', 'wonderful'],
            'sad': ['حزين', 'أسف', 'مؤسف', 'sad', 'sorry', 'disappointed'],
            'excited': ['متحمس', 'رائع', 'مذهل', 'excited', 'amazing', 'awesome'],
            'calm': ['هادئ', 'مريح', 'calm', 'peaceful', 'relax'],
            'confident': ['واثق', 'متأكد', 'confident', 'sure', 'certain']
        }
        
        for emotion, keywords in emotion_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return emotion
        
        return 'neutral'
    
    def preprocess_text(self, text, language='auto'):
        """معالجة النص لتحسين الطبيعية"""
        if language == 'auto':
            language = self.detect_language(text)
        
        # إضافة وقفات طبيعية
        if self.voice_settings['pause_insertion']:
            text = self._insert_natural_pauses(text)
        
        # تحسين النطق
        text = self._enhance_pronunciation(text, language)
        
        return text
    
    def _insert_natural_pauses(self, text):
        """إدراج وقفات طبيعية"""
        # وقفات قصيرة بعد الفواصل
        text = text.replace(',', ', <break time="0.3s"/>')
        text = text.replace('،', '، <break time="0.3s"/>')
        
        # وقفات متوسطة بعد النقاط
        text = text.replace('.', '. <break time="0.6s"/>')
        text = text.replace('؟', '؟ <break time="0.6s"/>')
        text = text.replace('!', '! <break time="0.6s"/>')
        
        # وقفات طويلة بعد الجمل المكتملة
        text = text.replace('. <break time="0.6s"/>', '. <break time="0.8s"/>')
        
        return text
    
    def _enhance_pronunciation(self, text, language):
        """تحسين النطق"""
        if language == 'arabic':
            # تحسينات النطق العربي
            pronunciation_map = {
                'الذكاء الاصطناعي': 'الذكاء الاص طناعي',
                'المساعد': 'المُساعد',
                'التقنية': 'التِقنية'
            }
        else:
            # تحسينات النطق الإنجليزي
            pronunciation_map = {
                'AI': 'A I',
                'API': 'A P I',
                'URL': 'U R L'
            }
        
        for original, improved in pronunciation_map.items():
            text = text.replace(original, improved)
        
        return text
    
    def synthesize_ultra_natural(self, text, emotion='auto', speaker_id=None, 
                                output_path="ultra_natural_output.wav"):
        """توليد صوت طبيعي للغاية"""
        
        # كشف العاطفة تلقائياً إذا لم تحدد
        if emotion == 'auto':
            emotion = self.detect_emotion(text)
        
        # معالجة النص
        processed_text = self.preprocess_text(text)
        
        print(f"🎙️ توليد صوت طبيعي: '{text[:50]}...'")
        print(f"🎭 العاطفة المكتشفة: {emotion}")
        
        try:
            # توليد الصوت الأساسي
            if speaker_id:
                wav = self.current_synthesizer.tts(
                    processed_text, 
                    speaker_name=speaker_id
                )
            else:
                wav = self.current_synthesizer.tts(processed_text)
            
            # تطبيق التحسينات المتقدمة
            enhanced_wav = self._apply_advanced_processing(wav, emotion)
            
            # حفظ الملف
            sf.write(output_path, enhanced_wav, 22050)
            
            # تطبيق المعالجة الصوتية المتقدمة
            self._post_process_audio(output_path, emotion)
            
            print(f"✅ تم حفظ الصوت الطبيعي في: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"❌ خطأ في توليد الصوت: {e}")
            return None
    
    def _apply_advanced_processing(self, wav, emotion):
        """تطبيق معالجة متقدمة للصوت"""
        enhanced_wav = np.copy(wav)
        profile = self.emotion_profiles.get(emotion, self.emotion_profiles['neutral'])
        
        # تعديل السرعة والنبرة
        if self.voice_settings['speed_variance']:
            # إضافة تنويع طفيف في السرعة
            speed_variation = np.random.normal(1.0, self.voice_settings['speed_variance'])
            speed_factor = profile['speed'] * speed_variation
            enhanced_wav = librosa.effects.time_stretch(enhanced_wav, rate=speed_factor)
        
        # تعديل النبرة
        if self.voice_settings['pitch_variance']:
            pitch_variation = np.random.normal(1.0, self.voice_settings['pitch_variance'])
            pitch_factor = profile['pitch'] * pitch_variation
            enhanced_wav = librosa.effects.pitch_shift(
                enhanced_wav, sr=22050, n_steps=pitch_factor
            )
        
        # إضافة أصوات التنفس الطبيعية
        if self.voice_settings['breathing_sounds']:
            enhanced_wav = self._add_breathing_sounds(enhanced_wav, profile)
        
        return enhanced_wav
    
    def _add_breathing_sounds(self, wav, profile):
        """إضافة أصوات تنفس طبيعية"""
        # إنشاء أصوات تنفس خفيفة
        breath_positions = self._find_breath_positions(wav)
        
        for pos in breath_positions:
            if pos < len(wav) - 1000:  # تأكد من وجود مساحة كافية
                breath_sound = self._generate_breath_sound(profile['breathing'])
                # دمج صوت التنفس
                end_pos = min(pos + len(breath_sound), len(wav))
                wav[pos:end_pos] += breath_sound[:end_pos-pos] * 0.1
        
        return wav
    
    def _find_breath_positions(self, wav):
        """العثور على مواضع مناسبة لأصوات التنفس"""
        # البحث عن فترات الصمت
        silence_threshold = 0.01
        silence_positions = []
        
        for i in range(0, len(wav) - 1000, 1000):
            segment = wav[i:i+1000]
            if np.max(np.abs(segment)) < silence_threshold:
                silence_positions.append(i)
        
        return silence_positions[::3]  # كل ثالث موضع صمت
    
    def _generate_breath_sound(self, breath_type):
        """توليد صوت تنفس"""
        duration = 0.2  # 200ms
        samples = int(22050 * duration)
        
        # ضوضاء بيضاء خفيفة تحاكي التنفس
        breath = np.random.normal(0, 0.01, samples)
        
        # تطبيق مرشح لتقليد صوت التنفس
        breath = librosa.effects.preemphasis(breath)
        
        # تشكيل الموجة
        envelope = np.exp(-np.linspace(0, 3, samples))
        breath *= envelope
        
        return breath
    
    def _post_process_audio(self, audio_path, emotion):
        """معالجة الصوت بعد التوليد"""
        try:
            # تحميل الملف الصوتي
            audio = AudioSegment.from_wav(audio_path)
            
            # تطبيق ضغط ديناميكي لتحسين الجودة
            audio = compress_dynamic_range(audio)
            
            # تطبيع مستوى الصوت
            audio = normalize(audio)
            
            # تحسينات خاصة بالعاطفة
            profile = self.emotion_profiles.get(emotion, self.emotion_profiles['neutral'])
            
            # تعديل مستوى الطاقة
            energy_factor = profile['energy']
            if energy_factor != 1.0:
                audio = audio + (20 * np.log10(energy_factor))
            
            # حفظ الملف المحسن
            audio.export(audio_path, format="wav")
            
        except Exception as e:
            print(f"⚠️ تحذير في معالجة الصوت: {e}")
    
    def create_voice_clone(self, reference_audio_path, text, output_path="cloned_voice.wav"):
        """استنساخ صوت من ملف مرجعي"""
        try:
            print("🎭 بدء استنساخ الصوت...")
            
            # تحميل الصوت المرجعي
            reference_wav, sr = librosa.load(reference_audio_path, sr=22050)
            
            # استخدام XTTS للاستنساخ
            if hasattr(self.current_synthesizer, 'tts_with_vc'):
                cloned_wav = self.current_synthesizer.tts_with_vc(
                    text=text,
                    speaker_wav=reference_wav
                )
            else:
                # إذا لم يدعم الاستنساخ، استخدم التوليد العادي
                cloned_wav = self.current_synthesizer.tts(text)
            
            sf.write(output_path, cloned_wav, 22050)
            print(f"✅ تم حفظ الصوت المستنسخ في: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"❌ خطأ في استنساخ الصوت: {e}")
            return None
    
    def analyze_voice_quality(self, audio_path):
        """تحليل جودة الصوت الناتج"""
        try:
            wav, sr = librosa.load(audio_path, sr=22050)
            
            analysis = {
                'duration': len(wav) / sr,
                'sample_rate': sr,
                'rms_energy': np.sqrt(np.mean(wav**2)),
                'zero_crossing_rate': np.mean(librosa.feature.zero_crossing_rate(wav)),
                'spectral_centroid': np.mean(librosa.feature.spectral_centroid(wav, sr=sr)),
                'naturalness_score': self._calculate_naturalness_score(wav, sr)
            }
            
            return analysis
            
        except Exception as e:
            print(f"❌ خطأ في تحليل الصوت: {e}")
            return None
    
    def _calculate_naturalness_score(self, wav, sr):
        """حساب نقاط الطبيعية"""
        # خوارزمية مبسطة لتقييم طبيعية الصوت
        
        # تحليل التردد الأساسي
        f0 = librosa.yin(wav, fmin=50, fmax=400)
        f0_variance = np.var(f0[f0 > 0])
        
        # تحليل الطيف
        spectral_rolloff = librosa.feature.spectral_rolloff(wav, sr=sr)
        spectral_variance = np.var(spectral_rolloff)
        
        # نقاط الطبيعية (0-1)
        naturalness = min(1.0, f0_variance / 1000 + spectral_variance / 10000)
        
        return naturalness

if __name__ == "__main__":
    tts = UltraNaturalTTS()
    
    # اختبارات متنوعة
    test_texts = [
        "مرحباً، أنا مساعدك الذكي الجديد!",
        "أشعر بسعادة كبيرة لمساعدتك اليوم",
        "للأسف، لم أتمكن من العثور على الملف المطلوب",
        "هذا رائع! لقد حققت نتائج ممتازة!"
    ]
    
    for i, text in enumerate(test_texts):
        output_file = f"test_ultra_natural_{i+1}.wav"
        tts.synthesize_ultra_natural(text, output_path=output_file)
        
        # تحليل الجودة
        quality = tts.analyze_voice_quality(output_file)
        if quality:
            print(f"📊 تحليل الجودة للنص {i+1}:")
            print(f"   المدة: {quality['duration']:.2f} ثانية")
            print(f"   نقاط الطبيعية: {quality['naturalness_score']:.2f}")
