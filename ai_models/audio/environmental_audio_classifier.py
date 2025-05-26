
import librosa
import numpy as np
import torch
import torch.nn as nn
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import json
from datetime import datetime
import sqlite3

class EnvironmentalAudioClassifier:
    """
    نظام تصنيف الأصوات البيئية والتمييز بينها
    يمكنه التعرف على:
    - أصوات الطبيعة (مطر، رياح، طيور)
    - أصوات المنزل (تلفزيون، موسيقى، أجهزة)
    - أصوات الشارع (سيارات، أجراس، إنشاءات)
    - أصوات بشرية (كلام، ضحك، بكاء)
    - أصوات الحيوانات
    """
    
    def __init__(self):
        print("🔊 تحميل نظام تصنيف الأصوات البيئية...")
        
        # فئات الأصوات
        self.sound_categories = {
            'human_voices': {
                'speech': 'كلام بشري',
                'laughter': 'ضحك',
                'crying': 'بكاء',
                'singing': 'غناء',
                'whispering': 'همس',
                'shouting': 'صراخ'
            },
            'nature_sounds': {
                'rain': 'مطر',
                'wind': 'رياح',
                'thunder': 'رعد',
                'birds': 'عصافير',
                'water_flow': 'تدفق الماء',
                'ocean_waves': 'أمواج البحر'
            },
            'household_sounds': {
                'tv': 'تلفزيون',
                'music': 'موسيقى',
                'washing_machine': 'غسالة',
                'microwave': 'ميكروويف',
                'vacuum_cleaner': 'مكنسة كهربائية',
                'door_bell': 'جرس الباب',
                'phone_ring': 'رنين هاتف'
            },
            'street_sounds': {
                'car_engine': 'محرك سيارة',
                'motorcycle': 'دراجة نارية',
                'horn': 'بوق سيارة',
                'construction': 'أعمال إنشاء',
                'siren': 'صفارة إنذار',
                'footsteps': 'خطوات أقدام'
            },
            'animal_sounds': {
                'dog_bark': 'نباح كلب',
                'cat_meow': 'مواء قطة',
                'bird_chirp': 'تغريد طائر',
                'horse_neigh': 'صهيل حصان',
                'cow_moo': 'خوار بقرة'
            },
            'mechanical_sounds': {
                'fan': 'مروحة',
                'air_conditioner': 'مكيف هواء',
                'computer_fan': 'مروحة كمبيوتر',
                'drill': 'مثقاب',
                'saw': 'منشار'
            }
        }
        
        # قاعدة بيانات للأصوات المكتشفة
        self.db_path = "detected_sounds.db"
        self._init_database()
        
        # إعدادات المعالجة
        self.sample_rate = 22050
        self.window_size = 2.0  # ثانيتان لكل نافذة
        self.overlap = 0.5  # تداخل 50%
        
        # تحميل نماذج التصنيف
        self._load_models()
        
        print("✅ نظام تصنيف الأصوات جاهز!")
    
    def _init_database(self):
        """إنشاء قاعدة البيانات"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detected_sounds (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                sound_category TEXT,
                sound_type TEXT,
                sound_name TEXT,
                confidence REAL,
                duration REAL,
                audio_features TEXT,
                context_info TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sound_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                start_time TEXT,
                end_time TEXT,
                total_sounds INTEGER,
                dominant_category TEXT,
                environment_type TEXT,
                summary TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _load_models(self):
        """تحميل النماذج"""
        try:
            # نموذج Wav2Vec2 للتصنيف العام
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                "facebook/wav2vec2-base-960h"
            )
            
            # نماذج مخصصة لكل فئة (سيتم استبدالها بنماذج مدربة)
            self.classifiers = {}
            
            print("✅ تم تحميل النماذج بنجاح")
            
        except Exception as e:
            print(f"⚠️ تحذير: {e}")
            print("📝 سيتم استخدام التصنيف المبني على الخصائص الصوتية")
    
    def extract_audio_features(self, audio_path, start_time=0, duration=None):
        """استخراج الخصائص الصوتية"""
        try:
            # تحميل الصوت
            y, sr = librosa.load(audio_path, sr=self.sample_rate, 
                               offset=start_time, duration=duration)
            
            if len(y) < sr * 0.1:  # أقل من 100ms
                return None
            
            features = {}
            
            # الخصائص الأساسية
            features['duration'] = len(y) / sr
            features['rms_energy'] = np.sqrt(np.mean(y**2))
            features['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(y)[0])
            
            # MFCC
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            features['mfcc_mean'] = np.mean(mfcc, axis=1)
            features['mfcc_std'] = np.std(mfcc, axis=1)
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            features['spectral_centroid'] = np.mean(spectral_centroids)
            features['spectral_centroid_std'] = np.std(spectral_centroids)
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            features['spectral_rolloff'] = np.mean(spectral_rolloff)
            
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            features['spectral_bandwidth'] = np.mean(spectral_bandwidth)
            
            # Chroma
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            features['chroma_mean'] = np.mean(chroma, axis=1)
            
            # Tempo
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            features['tempo'] = tempo
            
            # Pitch
            pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr)
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            if pitch_values:
                features['pitch_mean'] = np.mean(pitch_values)
                features['pitch_std'] = np.std(pitch_values)
            else:
                features['pitch_mean'] = 0
                features['pitch_std'] = 0
            
            # Harmonic and percussive components
            y_harmonic, y_percussive = librosa.effects.hpss(y)
            features['harmonic_ratio'] = np.sum(y_harmonic**2) / np.sum(y**2)
            features['percussive_ratio'] = np.sum(y_percussive**2) / np.sum(y**2)
            
            return features
            
        except Exception as e:
            print(f"❌ خطأ في استخراج الخصائص: {e}")
            return None
    
    def classify_sound_by_features(self, features):
        """تصنيف الصوت بناءً على الخصائص"""
        
        classifications = []
        
        # تصنيف الأصوات البشرية
        if self._is_human_voice(features):
            voice_type = self._classify_human_voice(features)
            classifications.append({
                'category': 'human_voices',
                'type': voice_type,
                'name': self.sound_categories['human_voices'].get(voice_type, voice_type),
                'confidence': 0.8
            })
        
        # تصنيف أصوات الطبيعة
        nature_type = self._classify_nature_sound(features)
        if nature_type:
            classifications.append({
                'category': 'nature_sounds',
                'type': nature_type,
                'name': self.sound_categories['nature_sounds'].get(nature_type, nature_type),
                'confidence': 0.7
            })
        
        # تصنيف الأصوات المنزلية
        household_type = self._classify_household_sound(features)
        if household_type:
            classifications.append({
                'category': 'household_sounds',
                'type': household_type,
                'name': self.sound_categories['household_sounds'].get(household_type, household_type),
                'confidence': 0.6
            })
        
        # تصنيف أصوات الشارع
        street_type = self._classify_street_sound(features)
        if street_type:
            classifications.append({
                'category': 'street_sounds',
                'type': street_type,
                'name': self.sound_categories['street_sounds'].get(street_type, street_type),
                'confidence': 0.6
            })
        
        # تصنيف أصوات الحيوانات
        animal_type = self._classify_animal_sound(features)
        if animal_type:
            classifications.append({
                'category': 'animal_sounds',
                'type': animal_type,
                'name': self.sound_categories['animal_sounds'].get(animal_type, animal_type),
                'confidence': 0.7
            })
        
        # ترتيب حسب الثقة
        classifications.sort(key=lambda x: x['confidence'], reverse=True)
        
        return classifications
    
    def _is_human_voice(self, features):
        """كشف الصوت البشري"""
        # معايير الصوت البشري
        pitch_in_range = 80 <= features['pitch_mean'] <= 500
        has_formants = features['spectral_centroid'] > 500
        harmonic_content = features['harmonic_ratio'] > 0.3
        
        return pitch_in_range and has_formants and harmonic_content
    
    def _classify_human_voice(self, features):
        """تصنيف نوع الصوت البشري"""
        
        energy = features['rms_energy']
        pitch_std = features['pitch_std']
        spectral_bandwidth = features['spectral_bandwidth']
        
        if energy > 0.1 and pitch_std > 50:
            return 'shouting'
        elif energy < 0.02:
            return 'whispering'
        elif spectral_bandwidth > 2000 and pitch_std > 30:
            return 'laughter'
        elif pitch_std > 40:
            return 'singing'
        else:
            return 'speech'
    
    def _classify_nature_sound(self, features):
        """تصنيف أصوات الطبيعة"""
        
        energy = features['rms_energy']
        zcr = features['zero_crossing_rate']
        spectral_centroid = features['spectral_centroid']
        
        if zcr > 0.15 and energy < 0.05:
            return 'wind'
        elif features['percussive_ratio'] > 0.7 and energy > 0.1:
            return 'thunder'
        elif spectral_centroid > 2000 and zcr > 0.1:
            return 'birds'
        elif zcr > 0.2 and spectral_centroid < 1000:
            return 'rain'
        elif features['harmonic_ratio'] > 0.5 and spectral_centroid < 500:
            return 'water_flow'
        
        return None
    
    def _classify_household_sound(self, features):
        """تصنيف الأصوات المنزلية"""
        
        tempo = features['tempo']
        energy = features['rms_energy']
        spectral_rolloff = features['spectral_rolloff']
        
        if features['harmonic_ratio'] > 0.6 and spectral_rolloff > 3000:
            return 'music'
        elif tempo > 0 and features['percussive_ratio'] > 0.5:
            return 'washing_machine'
        elif energy < 0.03 and spectral_rolloff < 2000:
            return 'tv'
        elif features['pitch_mean'] > 1000:
            return 'microwave'
        
        return None
    
    def _classify_street_sound(self, features):
        """تصنيف أصوات الشارع"""
        
        energy = features['rms_energy']
        spectral_centroid = features['spectral_centroid']
        
        if energy > 0.15 and spectral_centroid < 1000:
            return 'car_engine'
        elif energy > 0.2 and spectral_centroid > 2000:
            return 'motorcycle'
        elif features['pitch_mean'] > 500 and energy > 0.1:
            return 'horn'
        elif features['percussive_ratio'] > 0.8:
            return 'construction'
        
        return None
    
    def _classify_animal_sound(self, features):
        """تصنيف أصوات الحيوانات"""
        
        pitch_mean = features['pitch_mean']
        energy = features['rms_energy']
        
        if 200 <= pitch_mean <= 800 and energy > 0.05:
            return 'dog_bark'
        elif pitch_mean > 300 and features['harmonic_ratio'] > 0.4:
            return 'cat_meow'
        elif pitch_mean > 1000:
            return 'bird_chirp'
        
        return None
    
    def analyze_audio_file(self, audio_path):
        """تحليل ملف صوتي كامل"""
        
        print(f"🔍 تحليل الملف: {audio_path}")
        
        # الحصول على مدة الملف
        duration = librosa.get_duration(filename=audio_path)
        
        # تقسيم إلى نوافذ
        window_duration = self.window_size
        overlap_duration = window_duration * self.overlap
        step = window_duration - overlap_duration
        
        detected_sounds = []
        
        for start_time in np.arange(0, duration - window_duration, step):
            features = self.extract_audio_features(
                audio_path, start_time, window_duration
            )
            
            if features:
                classifications = self.classify_sound_by_features(features)
                
                for classification in classifications:
                    sound_info = {
                        'start_time': start_time,
                        'end_time': start_time + window_duration,
                        'duration': window_duration,
                        'timestamp': datetime.now().isoformat(),
                        **classification
                    }
                    detected_sounds.append(sound_info)
                    
                    # حفظ في قاعدة البيانات
                    self._save_detected_sound(sound_info, features)
        
        # تحليل النتائج
        analysis = self._analyze_session(detected_sounds)
        
        return {
            'total_duration': duration,
            'detected_sounds': detected_sounds,
            'analysis': analysis
        }
    
    def _save_detected_sound(self, sound_info, features):
        """حفظ الصوت المكتشف"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO detected_sounds 
            (timestamp, sound_category, sound_type, sound_name, confidence, 
             duration, audio_features, context_info)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            sound_info['timestamp'],
            sound_info['category'],
            sound_info['type'],
            sound_info['name'],
            sound_info['confidence'],
            sound_info['duration'],
            json.dumps(features),
            json.dumps({'start_time': sound_info['start_time']})
        ))
        
        conn.commit()
        conn.close()
    
    def _analyze_session(self, detected_sounds):
        """تحليل جلسة الأصوات"""
        
        if not detected_sounds:
            return {'summary': 'لم يتم اكتشاف أصوات'}
        
        # إحصائيات الفئات
        categories = {}
        for sound in detected_sounds:
            category = sound['category']
            categories[category] = categories.get(category, 0) + 1
        
        # الفئة المهيمنة
        dominant_category = max(categories.items(), key=lambda x: x[1])[0]
        
        # تحديد نوع البيئة
        environment_type = self._determine_environment(categories)
        
        # ملخص
        summary = f"تم اكتشاف {len(detected_sounds)} صوت. "
        summary += f"البيئة: {environment_type}. "
        summary += f"الفئة المهيمنة: {self._translate_category(dominant_category)}."
        
        return {
            'total_sounds': len(detected_sounds),
            'categories': categories,
            'dominant_category': dominant_category,
            'environment_type': environment_type,
            'summary': summary
        }
    
    def _determine_environment(self, categories):
        """تحديد نوع البيئة"""
        
        if 'nature_sounds' in categories and categories['nature_sounds'] > 2:
            return 'بيئة طبيعية'
        elif 'household_sounds' in categories and categories['household_sounds'] > 2:
            return 'بيئة منزلية'
        elif 'street_sounds' in categories and categories['street_sounds'] > 2:
            return 'بيئة حضرية'
        elif 'human_voices' in categories and categories['human_voices'] > 2:
            return 'بيئة اجتماعية'
        else:
            return 'بيئة مختلطة'
    
    def _translate_category(self, category):
        """ترجمة أسماء الفئات"""
        translations = {
            'human_voices': 'أصوات بشرية',
            'nature_sounds': 'أصوات طبيعة',
            'household_sounds': 'أصوات منزلية',
            'street_sounds': 'أصوات شارع',
            'animal_sounds': 'أصوات حيوانات',
            'mechanical_sounds': 'أصوات آلية'
        }
        return translations.get(category, category)
    
    def get_sound_statistics(self):
        """إحصائيات الأصوات"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # إحصائيات عامة
        cursor.execute('SELECT COUNT(*) FROM detected_sounds')
        total_sounds = cursor.fetchone()[0]
        
        # أكثر الفئات شيوعاً
        cursor.execute('''
            SELECT sound_category, COUNT(*) as count 
            FROM detected_sounds 
            GROUP BY sound_category 
            ORDER BY count DESC
        ''')
        category_stats = cursor.fetchall()
        
        # أحدث الأصوات
        cursor.execute('''
            SELECT sound_name, confidence, timestamp 
            FROM detected_sounds 
            ORDER BY timestamp DESC 
            LIMIT 10
        ''')
        recent_sounds = cursor.fetchall()
        
        conn.close()
        
        return {
            'total_detected': total_sounds,
            'category_distribution': category_stats,
            'recent_detections': recent_sounds
        }

if __name__ == "__main__":
    classifier = EnvironmentalAudioClassifier()
    
    print("🔊 نظام تصنيف الأصوات البيئية")
    print("=" * 50)
    
    # عرض الإحصائيات
    stats = classifier.get_sound_statistics()
    print(f"📊 الإحصائيات:")
    print(f"   - إجمالي الأصوات المكتشفة: {stats['total_detected']}")
    
    if stats['category_distribution']:
        print("   - توزيع الفئات:")
        for category, count in stats['category_distribution'][:5]:
            translated = classifier._translate_category(category)
            print(f"     • {translated}: {count}")
