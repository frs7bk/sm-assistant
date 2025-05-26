
import numpy as np
import librosa
import torch
import soundfile as sf
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cosine
import pickle
import os
import json
from datetime import datetime
import sqlite3
import hashlib

class VoiceBiometricsEngine:
    """
    نظام متقدم للتمييز بين الأصوات والبصمة الصوتية
    يمكنه:
    - التمييز بين المتحدثين المختلفين
    - إنشاء بصمات صوتية فريدة
    - كشف الأصوات المزيفة
    - تحليل خصائص الصوت الفريدة
    """
    
    def __init__(self, db_path="voice_profiles.db"):
        print("🎤 تحميل نظام التمييز الصوتي المتقدم...")
        
        self.db_path = db_path
        self.sample_rate = 16000
        self.n_mfcc = 13
        self.n_components = 32  # عدد مكونات GMM
        
        # إنشاء قاعدة البيانات
        self._init_database()
        
        # نماذج التعلم الآلي
        self.scaler = StandardScaler()
        self.voice_profiles = {}
        self.is_trained = False
        
        # إعدادات كشف الصوت المزيف
        self.deepfake_threshold = 0.7
        
        # خصائص صوتية متقدمة
        self.voice_features = [
            'mfcc', 'chroma', 'spectral_centroid', 'spectral_rolloff',
            'zero_crossing_rate', 'pitch', 'formants', 'jitter', 'shimmer'
        ]
        
        print("✅ نظام التمييز الصوتي جاهز!")
    
    def _init_database(self):
        """إنشاء قاعدة البيانات للملفات الصوتية"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS voice_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                speaker_id TEXT UNIQUE,
                speaker_name TEXT,
                voice_signature BLOB,
                features_json TEXT,
                recordings_count INTEGER DEFAULT 0,
                created_date TEXT,
                last_updated TEXT,
                confidence_score REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS voice_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                speaker_id TEXT,
                audio_path TEXT,
                recognition_confidence REAL,
                timestamp TEXT,
                features_extracted TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def extract_voice_features(self, audio_path):
        """استخراج الخصائص الصوتية المتقدمة"""
        try:
            # تحميل الملف الصوتي
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # إزالة الصمت
            y_trimmed, _ = librosa.effects.trim(y, top_db=20)
            
            if len(y_trimmed) < sr * 0.5:  # أقل من نصف ثانية
                raise ValueError("الملف الصوتي قصير جداً")
            
            features = {}
            
            # MFCC (الأهم للتمييز الصوتي)
            mfcc = librosa.feature.mfcc(y=y_trimmed, sr=sr, n_mfcc=self.n_mfcc)
            features['mfcc_mean'] = np.mean(mfcc, axis=1)
            features['mfcc_std'] = np.std(mfcc, axis=1)
            features['mfcc_delta'] = np.mean(librosa.feature.delta(mfcc), axis=1)
            
            # Chroma features
            chroma = librosa.feature.chroma_stft(y=y_trimmed, sr=sr)
            features['chroma_mean'] = np.mean(chroma, axis=1)
            features['chroma_std'] = np.std(chroma, axis=1)
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y_trimmed, sr=sr)[0]
            features['spectral_centroid_mean'] = np.mean(spectral_centroids)
            features['spectral_centroid_std'] = np.std(spectral_centroids)
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y_trimmed, sr=sr)[0]
            features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(y_trimmed)[0]
            features['zcr_mean'] = np.mean(zcr)
            features['zcr_std'] = np.std(zcr)
            
            # Pitch estimation
            pitches, magnitudes = librosa.core.piptrack(y=y_trimmed, sr=sr)
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            if pitch_values:
                features['pitch_mean'] = np.mean(pitch_values)
                features['pitch_std'] = np.std(pitch_values)
                features['pitch_min'] = np.min(pitch_values)
                features['pitch_max'] = np.max(pitch_values)
            else:
                features['pitch_mean'] = 0
                features['pitch_std'] = 0
                features['pitch_min'] = 0
                features['pitch_max'] = 0
            
            # Formants (تقدير تقريبي)
            formants = self._estimate_formants(y_trimmed, sr)
            features['formant_f1'] = formants[0] if len(formants) > 0 else 0
            features['formant_f2'] = formants[1] if len(formants) > 1 else 0
            
            # Jitter and Shimmer (تقدير مبسط)
            features['jitter'] = self._calculate_jitter(pitch_values)
            features['shimmer'] = self._calculate_shimmer(y_trimmed)
            
            # تحويل إلى مصفوفة واحدة
            feature_vector = np.concatenate([
                features['mfcc_mean'], features['mfcc_std'], features['mfcc_delta'],
                features['chroma_mean'], features['chroma_std'],
                [features['spectral_centroid_mean'], features['spectral_centroid_std']],
                [features['spectral_rolloff_mean']],
                [features['zcr_mean'], features['zcr_std']],
                [features['pitch_mean'], features['pitch_std'], 
                 features['pitch_min'], features['pitch_max']],
                [features['formant_f1'], features['formant_f2']],
                [features['jitter'], features['shimmer']]
            ])
            
            return feature_vector, features
            
        except Exception as e:
            print(f"❌ خطأ في استخراج الخصائص: {e}")
            return None, None
    
    def _estimate_formants(self, y, sr, n_formants=2):
        """تقدير التردد الأساسي (Formants)"""
        try:
            # حساب LPC
            from scipy.signal import lfilter
            
            # تطبيق نافذة هامينغ
            windowed = y * np.hamming(len(y))
            
            # LPC analysis
            lpc_order = int(2 + sr / 1000)
            lpc_coeffs = librosa.lpc(windowed, order=lpc_order)
            
            # العثور على الجذور
            roots = np.roots(lpc_coeffs)
            roots = roots[np.imag(roots) >= 0]
            
            # تحويل إلى ترددات
            formants = []
            for root in roots:
                if np.abs(root) < 1:
                    freq = np.angle(root) * sr / (2 * np.pi)
                    if 100 < freq < sr/2:  # نطاق معقول للـ formants
                        formants.append(freq)
            
            formants.sort()
            return formants[:n_formants]
            
        except Exception:
            return [0, 0]
    
    def _calculate_jitter(self, pitch_values):
        """حساب الـ Jitter (تذبذب الصوت)"""
        if len(pitch_values) < 2:
            return 0
        
        differences = np.diff(pitch_values)
        mean_diff = np.mean(np.abs(differences))
        mean_pitch = np.mean(pitch_values)
        
        return mean_diff / mean_pitch if mean_pitch > 0 else 0
    
    def _calculate_shimmer(self, y):
        """حساب الـ Shimmer (تذبذب الطاقة)"""
        # تقسيم الإشارة إلى إطارات
        frame_length = 1024
        hop_length = 512
        
        frames = librosa.util.frame(y, frame_length=frame_length, 
                                  hop_length=hop_length, axis=0)
        
        # حساب الطاقة لكل إطار
        energies = np.sum(frames**2, axis=1)
        
        if len(energies) < 2:
            return 0
        
        # حساب الفروق
        differences = np.diff(energies)
        mean_diff = np.mean(np.abs(differences))
        mean_energy = np.mean(energies)
        
        return mean_diff / mean_energy if mean_energy > 0 else 0
    
    def create_voice_profile(self, audio_files, speaker_name, speaker_id=None):
        """إنشاء ملف صوتي جديد"""
        
        if speaker_id is None:
            speaker_id = hashlib.md5(speaker_name.encode()).hexdigest()[:8]
        
        print(f"👤 إنشاء ملف صوتي لـ: {speaker_name}")
        
        all_features = []
        valid_files = 0
        
        for audio_file in audio_files:
            features, detailed_features = self.extract_voice_features(audio_file)
            if features is not None:
                all_features.append(features)
                valid_files += 1
        
        if valid_files < 1:
            print("❌ لا توجد ملفات صوتية صالحة")
            return False
        
        # تدريب نموذج GMM للمتحدث
        features_matrix = np.array(all_features)
        
        # تطبيع البيانات
        normalized_features = self.scaler.fit_transform(features_matrix)
        
        # تدريب GMM
        gmm_model = GaussianMixture(
            n_components=min(self.n_components, len(all_features)),
            covariance_type='diag',
            random_state=42
        )
        gmm_model.fit(normalized_features)
        
        # حساب متوسط الخصائص
        mean_features = np.mean(features_matrix, axis=0)
        
        # حفظ الملف الصوتي
        voice_signature = {
            'gmm_model': gmm_model,
            'scaler': self.scaler,
            'mean_features': mean_features,
            'feature_stats': {
                'mean': np.mean(features_matrix, axis=0),
                'std': np.std(features_matrix, axis=0),
                'min': np.min(features_matrix, axis=0),
                'max': np.max(features_matrix, axis=0)
            }
        }
        
        # حفظ في قاعدة البيانات
        self._save_voice_profile(speaker_id, speaker_name, voice_signature, valid_files)
        
        # حفظ في الذاكرة
        self.voice_profiles[speaker_id] = voice_signature
        
        print(f"✅ تم إنشاء الملف الصوتي بنجاح ({valid_files} ملفات)")
        return True
    
    def _save_voice_profile(self, speaker_id, speaker_name, voice_signature, recordings_count):
        """حفظ الملف الصوتي في قاعدة البيانات"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # تسلسل البيانات
        signature_blob = pickle.dumps(voice_signature)
        features_json = json.dumps({
            'mean_features': voice_signature['mean_features'].tolist(),
            'feature_stats': {
                key: value.tolist() for key, value in voice_signature['feature_stats'].items()
            }
        })
        
        current_time = datetime.now().isoformat()
        
        cursor.execute('''
            INSERT OR REPLACE INTO voice_profiles 
            (speaker_id, speaker_name, voice_signature, features_json, 
             recordings_count, created_date, last_updated, confidence_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (speaker_id, speaker_name, signature_blob, features_json,
              recordings_count, current_time, current_time, 0.95))
        
        conn.commit()
        conn.close()
    
    def identify_speaker(self, audio_path, threshold=0.7):
        """التعرف على المتحدث من الملف الصوتي"""
        
        # استخراج الخصائص
        features, detailed_features = self.extract_voice_features(audio_path)
        if features is None:
            return None, 0
        
        # تحميل الملفات الصوتية
        self._load_voice_profiles()
        
        if not self.voice_profiles:
            print("⚠️ لا توجد ملفات صوتية محفوظة")
            return None, 0
        
        best_match = None
        best_score = 0
        scores = {}
        
        print("🔍 البحث عن أفضل تطابق...")
        
        for speaker_id, profile in self.voice_profiles.items():
            try:
                # تطبيع الخصائص
                normalized_features = profile['scaler'].transform([features])
                
                # حساب احتمالية GMM
                log_likelihood = profile['gmm_model'].score(normalized_features)
                
                # حساب المسافة الكوسينية
                cosine_sim = 1 - cosine(features, profile['mean_features'])
                
                # الدرجة المركبة
                combined_score = (log_likelihood / 10) + cosine_sim
                scores[speaker_id] = combined_score
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_match = speaker_id
                    
            except Exception as e:
                print(f"⚠️ خطأ في مقارنة {speaker_id}: {e}")
                continue
        
        # تحويل النتيجة إلى نسبة مئوية
        confidence = min(max(best_score, 0), 1)
        
        if confidence >= threshold:
            speaker_name = self._get_speaker_name(best_match)
            print(f"✅ تم التعرف على: {speaker_name} (ثقة: {confidence:.2%})")
            
            # حفظ الجلسة
            self._save_session(best_match, audio_path, confidence, detailed_features)
            
            return {
                'speaker_id': best_match,
                'speaker_name': speaker_name,
                'confidence': confidence,
                'all_scores': scores
            }, confidence
        else:
            print(f"❌ لم يتم التعرف على المتحدث (أفضل نتيجة: {confidence:.2%})")
            return None, confidence
    
    def _load_voice_profiles(self):
        """تحميل الملفات الصوتية من قاعدة البيانات"""
        if self.voice_profiles:  # مُحمل مسبقاً
            return
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT speaker_id, voice_signature FROM voice_profiles')
        rows = cursor.fetchall()
        
        for speaker_id, signature_blob in rows:
            try:
                voice_signature = pickle.loads(signature_blob)
                self.voice_profiles[speaker_id] = voice_signature
            except Exception as e:
                print(f"⚠️ خطأ في تحميل ملف {speaker_id}: {e}")
        
        conn.close()
        print(f"📁 تم تحميل {len(self.voice_profiles)} ملف صوتي")
    
    def _get_speaker_name(self, speaker_id):
        """الحصول على اسم المتحدث"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT speaker_name FROM voice_profiles WHERE speaker_id = ?', 
                      (speaker_id,))
        result = cursor.fetchone()
        conn.close()
        
        return result[0] if result else "غير معروف"
    
    def _save_session(self, speaker_id, audio_path, confidence, features):
        """حفظ جلسة التعرف"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        session_id = hashlib.md5(f"{speaker_id}_{datetime.now()}".encode()).hexdigest()[:12]
        
        cursor.execute('''
            INSERT INTO voice_sessions 
            (session_id, speaker_id, audio_path, recognition_confidence, 
             timestamp, features_extracted)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (session_id, speaker_id, audio_path, confidence,
              datetime.now().isoformat(), json.dumps(features)))
        
        conn.commit()
        conn.close()
    
    def detect_voice_spoofing(self, audio_path):
        """كشف الأصوات المزيفة والمصطنعة"""
        
        features, detailed = self.extract_voice_features(audio_path)
        if features is None:
            return {'is_spoofed': True, 'confidence': 0, 'reason': 'فشل في تحليل الصوت'}
        
        # مؤشرات الصوت المزيف
        spoofing_indicators = {
            'pitch_variance_low': detailed['pitch_std'] < 10,
            'unnatural_formants': detailed['formant_f1'] == 0 or detailed['formant_f2'] == 0,
            'low_jitter': detailed['jitter'] < 0.001,
            'low_shimmer': detailed['shimmer'] < 0.001,
            'spectral_anomaly': detailed['spectral_centroid_std'] < 100
        }
        
        # حساب عدد المؤشرات الإيجابية
        positive_indicators = sum(spoofing_indicators.values())
        total_indicators = len(spoofing_indicators)
        
        spoofing_score = positive_indicators / total_indicators
        is_spoofed = spoofing_score >= self.deepfake_threshold
        
        return {
            'is_spoofed': is_spoofed,
            'confidence': spoofing_score,
            'indicators': spoofing_indicators,
            'recommendation': 'صوت مشكوك فيه' if is_spoofed else 'صوت طبيعي'
        }
    
    def get_voice_statistics(self):
        """إحصائيات النظام"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # عدد الملفات الصوتية
        cursor.execute('SELECT COUNT(*) FROM voice_profiles')
        profiles_count = cursor.fetchone()[0]
        
        # عدد الجلسات
        cursor.execute('SELECT COUNT(*) FROM voice_sessions')
        sessions_count = cursor.fetchone()[0]
        
        # أحدث نشاط
        cursor.execute('''
            SELECT speaker_name, recognition_confidence, timestamp 
            FROM voice_sessions 
            JOIN voice_profiles ON voice_sessions.speaker_id = voice_profiles.speaker_id 
            ORDER BY timestamp DESC LIMIT 5
        ''')
        recent_activity = cursor.fetchall()
        
        conn.close()
        
        return {
            'total_profiles': profiles_count,
            'total_sessions': sessions_count,
            'recent_activity': recent_activity,
            'system_status': 'نشط' if profiles_count > 0 else 'في انتظار البيانات'
        }
    
    def analyze_voice_characteristics(self, audio_path):
        """تحليل شامل لخصائص الصوت"""
        
        features, detailed = self.extract_voice_features(audio_path)
        if features is None:
            return None
        
        # تحليل النتائج
        analysis = {
            'basic_info': {
                'duration': f"{librosa.get_duration(filename=audio_path):.2f} ثانية",
                'sample_rate': f"{self.sample_rate} Hz"
            },
            'pitch_analysis': {
                'average_pitch': f"{detailed['pitch_mean']:.1f} Hz",
                'pitch_range': f"{detailed['pitch_max'] - detailed['pitch_min']:.1f} Hz",
                'pitch_stability': 'مستقر' if detailed['pitch_std'] < 20 else 'متذبذب'
            },
            'voice_quality': {
                'jitter': f"{detailed['jitter']:.4f}",
                'shimmer': f"{detailed['shimmer']:.4f}",
                'voice_quality': 'ممتاز' if detailed['jitter'] < 0.01 and detailed['shimmer'] < 0.1 else 'جيد'
            },
            'spectral_features': {
                'brightness': f"{detailed['spectral_centroid_mean']:.0f} Hz",
                'voice_richness': 'غني' if detailed['spectral_rolloff_mean'] > 3000 else 'عادي'
            },
            'speaker_characteristics': {
                'gender_estimate': 'أنثى' if detailed['pitch_mean'] > 180 else 'ذكر',
                'age_estimate': self._estimate_age(detailed),
                'voice_type': self._classify_voice_type(detailed)
            }
        }
        
        return analysis
    
    def _estimate_age(self, features):
        """تقدير العمر التقريبي"""
        # خوارزمية مبسطة لتقدير العمر
        pitch_mean = features['pitch_mean']
        jitter = features['jitter']
        shimmer = features['shimmer']
        
        if pitch_mean > 200 and jitter < 0.005:
            return "شاب (18-30)"
        elif pitch_mean > 150 and jitter < 0.01:
            return "متوسط العمر (30-50)"
        elif jitter > 0.02 or shimmer > 0.15:
            return "كبير السن (50+)"
        else:
            return "متوسط العمر"
    
    def _classify_voice_type(self, features):
        """تصنيف نوع الصوت"""
        pitch_mean = features['pitch_mean']
        
        if pitch_mean < 85:
            return "باص"
        elif pitch_mean < 165:
            return "باريتون"
        elif pitch_mean < 220:
            return "تينور"
        elif pitch_mean < 330:
            return "ألتو"
        else:
            return "سوبرانو"

if __name__ == "__main__":
    # اختبار النظام
    voice_engine = VoiceBiometricsEngine()
    
    print("🎤 نظام التمييز الصوتي المتقدم")
    print("=" * 50)
    
    # عرض الإحصائيات
    stats = voice_engine.get_voice_statistics()
    print(f"📊 الإحصائيات:")
    print(f"   - عدد الملفات الصوتية: {stats['total_profiles']}")
    print(f"   - عدد الجلسات: {stats['total_sessions']}")
    print(f"   - حالة النظام: {stats['system_status']}")
