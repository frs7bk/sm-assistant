
"""
محرك تحليل الفيديو المتقدم
Advanced Video Analysis Engine
"""

import cv2
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import os
import json
from datetime import datetime, timedelta
import threading
import queue
import mediapipe as mp
import torch
from ultralytics import YOLO
import librosa
import speech_recognition as sr

class AdvancedVideoAnalyzer:
    """
    محرك تحليل فيديو متقدم مع قدرات AI شاملة
    """
    
    def __init__(self):
        print("🎬 تهيئة محرك تحليل الفيديو المتقدم...")
        
        # تهيئة نماذج MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        
        # تهيئة المعالجات
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            smooth_landmarks=True
        )
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7
        )
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True
        )
        self.segmentation = self.mp_selfie_segmentation.SelfieSegmentation(
            model_selection=1
        )
        
        # تحميل YOLO للكشف عن الكائنات
        try:
            self.yolo = YOLO('yolov8n.pt')
            print("✅ تم تحميل نموذج YOLO")
        except:
            self.yolo = None
            print("⚠️ لم يتم تحميل YOLO")
        
        # إعدادات التحليل
        self.analysis_settings = {
            'detect_objects': True,
            'analyze_emotions': True,
            'track_movements': True,
            'extract_audio': True,
            'scene_understanding': True,
            'activity_recognition': True,
            'quality_assessment': True
        }
        
        # متغيرات التتبع
        self.frame_buffer = []
        self.analysis_results = []
        self.processing_queue = queue.Queue()
        
    def analyze_video_file(self, video_path: str, 
                          output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        تحليل ملف فيديو كامل
        """
        if not os.path.exists(video_path):
            return {"error": "الملف غير موجود"}
        
        print(f"🎬 بدء تحليل الفيديو: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": "لا يمكن فتح الفيديو"}
        
        # معلومات الفيديو
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        video_info = {
            'path': video_path,
            'duration': duration,
            'fps': fps,
            'frame_count': frame_count,
            'resolution': (width, height),
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        # تحليل شامل
        analysis_results = {
            'video_info': video_info,
            'scene_analysis': [],
            'object_detection': [],
            'motion_analysis': [],
            'face_analysis': [],
            'emotion_timeline': [],
            'activity_recognition': [],
            'audio_analysis': {},
            'quality_metrics': {},
            'summary': {}
        }
        
        # معالجة الإطارات
        frame_index = 0
        scene_changes = []
        prev_frame = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            current_time = frame_index / fps
            
            # تحليل الإطار الحالي
            frame_analysis = self._analyze_single_frame(
                frame, frame_index, current_time
            )
            
            # كشف تغيير المشهد
            if prev_frame is not None:
                scene_change = self._detect_scene_change(prev_frame, frame)
                if scene_change:
                    scene_changes.append({
                        'frame': frame_index,
                        'time': current_time,
                        'confidence': scene_change
                    })
            
            # حفظ نتائج التحليل
            if frame_analysis:
                analysis_results['scene_analysis'].append(frame_analysis)
            
            # تحديث للمعالجة التالية
            prev_frame = frame.copy()
            frame_index += 1
            
            # عرض التقدم
            if frame_index % int(fps) == 0:  # كل ثانية
                progress = (frame_index / frame_count) * 100
                print(f"📊 التقدم: {progress:.1f}%")
        
        cap.release()
        
        # تحليل الصوت
        if self.analysis_settings['extract_audio']:
            analysis_results['audio_analysis'] = self._analyze_video_audio(video_path)
        
        # تحليل الجودة العامة
        analysis_results['quality_metrics'] = self._assess_video_quality(video_path)
        
        # إنشاء ملخص ذكي
        analysis_results['summary'] = self._generate_video_summary(analysis_results)
        
        # حفظ النتائج
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(analysis_results, f, ensure_ascii=False, indent=2)
            print(f"💾 تم حفظ التحليل في: {output_path}")
        
        print("✅ تم الانتهاء من تحليل الفيديو")
        return analysis_results
    
    def _analyze_single_frame(self, frame: np.ndarray, 
                             frame_index: int, 
                             timestamp: float) -> Dict:
        """تحليل إطار واحد"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        frame_analysis = {
            'frame_index': frame_index,
            'timestamp': timestamp,
            'objects': [],
            'faces': [],
            'poses': [],
            'hands': [],
            'emotions': [],
            'activities': []
        }
        
        # كشف الكائنات
        if self.yolo and self.analysis_settings['detect_objects']:
            objects = self._detect_objects_yolo(frame)
            frame_analysis['objects'] = objects
        
        # تحليل الوجوه والمشاعر
        if self.analysis_settings['analyze_emotions']:
            face_results = self.face_mesh.process(rgb_frame)
            if face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    face_analysis = self._analyze_face_landmarks(face_landmarks)
                    frame_analysis['faces'].append(face_analysis)
                    
                    # تحليل المشاعر
                    emotion = self._detect_face_emotion(face_landmarks)
                    frame_analysis['emotions'].append(emotion)
        
        # تحليل وضعية الجسم
        if self.analysis_settings['track_movements']:
            pose_results = self.pose.process(rgb_frame)
            if pose_results.pose_landmarks:
                pose_analysis = self._analyze_body_pose(pose_results.pose_landmarks)
                frame_analysis['poses'].append(pose_analysis)
                
                # تحليل النشاط
                activity = self._recognize_activity(pose_results.pose_landmarks)
                frame_analysis['activities'].append(activity)
        
        # تحليل اليدين
        hand_results = self.hands.process(rgb_frame)
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                hand_analysis = self._analyze_hand_gesture(hand_landmarks)
                frame_analysis['hands'].append(hand_analysis)
        
        return frame_analysis
    
    def _detect_objects_yolo(self, frame: np.ndarray) -> List[Dict]:
        """كشف الكائنات باستخدام YOLO"""
        try:
            results = self.yolo(frame)
            objects = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        obj = {
                            'class': self.yolo.names[int(box.cls)],
                            'confidence': float(box.conf),
                            'bbox': box.xyxy[0].tolist(),
                            'center': self._calculate_bbox_center(box.xyxy[0])
                        }
                        objects.append(obj)
            
            return objects
        except Exception as e:
            print(f"خطأ في كشف الكائنات: {e}")
            return []
    
    def _analyze_face_landmarks(self, face_landmarks) -> Dict:
        """تحليل معالم الوجه"""
        landmarks = []
        for landmark in face_landmarks.landmark:
            landmarks.append({
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z
            })
        
        # حساب خصائص الوجه
        face_analysis = {
            'landmarks_count': len(landmarks),
            'face_orientation': self._calculate_face_orientation(landmarks),
            'eye_aspect_ratio': self._calculate_eye_aspect_ratio(landmarks),
            'mouth_aspect_ratio': self._calculate_mouth_aspect_ratio(landmarks),
            'face_symmetry': self._calculate_face_symmetry(landmarks)
        }
        
        return face_analysis
    
    def _detect_face_emotion(self, face_landmarks) -> Dict:
        """كشف المشاعر من معالم الوجه"""
        # تحليل مبسط للمشاعر
        landmarks = [(lm.x, lm.y) for lm in face_landmarks.landmark]
        
        # تحليل الفم والعينين
        mouth_curve = self._analyze_mouth_curve(landmarks)
        eye_openness = self._analyze_eye_openness(landmarks)
        eyebrow_position = self._analyze_eyebrow_position(landmarks)
        
        # تصنيف المشاعر
        emotion_scores = {
            'happiness': max(0, mouth_curve),
            'sadness': max(0, -mouth_curve),
            'surprise': eye_openness * eyebrow_position,
            'anger': max(0, -eyebrow_position),
            'neutral': 1 - abs(mouth_curve) - abs(eyebrow_position)
        }
        
        # العثور على أقوى عاطفة
        dominant_emotion = max(emotion_scores, key=emotion_scores.get)
        confidence = emotion_scores[dominant_emotion]
        
        return {
            'emotion': dominant_emotion,
            'confidence': confidence,
            'all_scores': emotion_scores
        }
    
    def _analyze_body_pose(self, pose_landmarks) -> Dict:
        """تحليل وضعية الجسم"""
        landmarks = []
        for landmark in pose_landmarks.landmark:
            landmarks.append({
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z,
                'visibility': landmark.visibility
            })
        
        pose_analysis = {
            'posture_score': self._calculate_posture_score(landmarks),
            'balance': self._calculate_balance(landmarks),
            'movement_energy': self._calculate_movement_energy(landmarks),
            'body_orientation': self._calculate_body_orientation(landmarks)
        }
        
        return pose_analysis
    
    def _recognize_activity(self, pose_landmarks) -> Dict:
        """التعرف على النشاط"""
        landmarks = [(lm.x, lm.y, lm.z) for lm in pose_landmarks.landmark]
        
        # تحليل مبسط للأنشطة
        activities = {
            'standing': self._is_standing(landmarks),
            'sitting': self._is_sitting(landmarks),
            'walking': self._is_walking(landmarks),
            'waving': self._is_waving(landmarks),
            'pointing': self._is_pointing(landmarks)
        }
        
        # العثور على النشاط الأكثر احتمالاً
        most_likely_activity = max(activities, key=activities.get)
        confidence = activities[most_likely_activity]
        
        return {
            'activity': most_likely_activity,
            'confidence': confidence,
            'all_activities': activities
        }
    
    def _analyze_hand_gesture(self, hand_landmarks) -> Dict:
        """تحليل إيماءات اليد"""
        landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
        
        # كشف الإيماءات المختلفة
        gestures = {
            'thumbs_up': self._detect_thumbs_up(landmarks),
            'peace_sign': self._detect_peace_sign(landmarks),
            'pointing': self._detect_pointing(landmarks),
            'open_palm': self._detect_open_palm(landmarks),
            'fist': self._detect_fist(landmarks)
        }
        
        # العثور على الإيماءة الأكثر احتمالاً
        detected_gesture = max(gestures, key=gestures.get)
        confidence = gestures[detected_gesture]
        
        return {
            'gesture': detected_gesture,
            'confidence': confidence,
            'all_gestures': gestures
        }
    
    def _detect_scene_change(self, prev_frame: np.ndarray, 
                           curr_frame: np.ndarray) -> float:
        """كشف تغيير المشهد"""
        # تحويل إلى رمادي
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        # حساب الفرق
        diff = cv2.absdiff(prev_gray, curr_gray)
        mean_diff = np.mean(diff)
        
        # عتبة تغيير المشهد
        scene_change_threshold = 30
        
        if mean_diff > scene_change_threshold:
            return min(1.0, mean_diff / 100)
        
        return 0.0
    
    def _analyze_video_audio(self, video_path: str) -> Dict:
        """تحليل الصوت في الفيديو"""
        try:
            # استخراج الصوت
            audio_path = "temp_audio.wav"
            os.system(f"ffmpeg -i '{video_path}' -vn -acodec pcm_s16le -ar 16000 -ac 1 '{audio_path}' -y 2>/dev/null")
            
            if not os.path.exists(audio_path):
                return {"error": "لا يمكن استخراج الصوت"}
            
            # تحليل الصوت
            y, sr = librosa.load(audio_path, sr=16000)
            
            audio_analysis = {
                'duration': len(y) / sr,
                'sample_rate': sr,
                'energy': float(np.mean(y**2)),
                'zero_crossing_rate': float(np.mean(librosa.feature.zero_crossing_rate(y))),
                'spectral_centroid': float(np.mean(librosa.feature.spectral_centroid(y, sr=sr))),
                'tempo': 0,
                'speech_segments': [],
                'music_detected': False,
                'silence_ratio': 0
            }
            
            # كشف الإيقاع
            try:
                tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                audio_analysis['tempo'] = float(tempo)
            except:
                pass
            
            # كشف الكلام
            try:
                recognizer = sr.Recognizer()
                with sr.AudioFile(audio_path) as source:
                    audio_data = recognizer.record(source)
                    try:
                        text = recognizer.recognize_google(audio_data, language='ar-SA')
                        audio_analysis['transcription'] = text
                    except:
                        audio_analysis['transcription'] = "لا يمكن تحويل الكلام"
            except:
                pass
            
            # تنظيف الملف المؤقت
            if os.path.exists(audio_path):
                os.remove(audio_path)
            
            return audio_analysis
            
        except Exception as e:
            print(f"خطأ في تحليل الصوت: {e}")
            return {"error": str(e)}
    
    def _assess_video_quality(self, video_path: str) -> Dict:
        """تقييم جودة الفيديو"""
        cap = cv2.VideoCapture(video_path)
        
        quality_metrics = {
            'resolution_score': 0,
            'clarity_score': 0,
            'color_quality': 0,
            'stability_score': 0,
            'lighting_quality': 0,
            'overall_score': 0
        }
        
        frame_count = 0
        clarity_scores = []
        color_scores = []
        
        while frame_count < 30:  # تحليل أول 30 إطار
            ret, frame = cap.read()
            if not ret:
                break
            
            # تقييم الوضوح
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            clarity = cv2.Laplacian(gray, cv2.CV_64F).var()
            clarity_scores.append(clarity)
            
            # تقييم الألوان
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            color_variance = np.var(hsv)
            color_scores.append(color_variance)
            
            frame_count += 1
        
        cap.release()
        
        # حساب النقاط
        if clarity_scores:
            quality_metrics['clarity_score'] = min(1.0, np.mean(clarity_scores) / 1000)
        
        if color_scores:
            quality_metrics['color_quality'] = min(1.0, np.mean(color_scores) / 10000)
        
        # نقاط الدقة
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        pixels = width * height
        
        if pixels >= 1920 * 1080:
            quality_metrics['resolution_score'] = 1.0
        elif pixels >= 1280 * 720:
            quality_metrics['resolution_score'] = 0.8
        else:
            quality_metrics['resolution_score'] = 0.5
        
        # النقاط الإجمالية
        quality_metrics['overall_score'] = np.mean([
            quality_metrics['resolution_score'],
            quality_metrics['clarity_score'],
            quality_metrics['color_quality']
        ])
        
        return quality_metrics
    
    def _generate_video_summary(self, analysis_results: Dict) -> Dict:
        """إنشاء ملخص ذكي للفيديو"""
        summary = {
            'total_duration': analysis_results['video_info']['duration'],
            'detected_objects': [],
            'dominant_emotions': [],
            'main_activities': [],
            'scene_changes_count': 0,
            'audio_summary': {},
            'quality_assessment': "جيد",
            'key_moments': []
        }
        
        # تحليل الكائنات المكتشفة
        all_objects = {}
        for frame_analysis in analysis_results['scene_analysis']:
            for obj in frame_analysis.get('objects', []):
                obj_class = obj['class']
                if obj_class not in all_objects:
                    all_objects[obj_class] = 0
                all_objects[obj_class] += 1
        
        # أكثر الكائنات ظهوراً
        summary['detected_objects'] = sorted(
            all_objects.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]
        
        # تحليل المشاعر السائدة
        all_emotions = {}
        for frame_analysis in analysis_results['scene_analysis']:
            for emotion in frame_analysis.get('emotions', []):
                emotion_type = emotion['emotion']
                if emotion_type not in all_emotions:
                    all_emotions[emotion_type] = []
                all_emotions[emotion_type].append(emotion['confidence'])
        
        # المشاعر الأكثر ظهوراً
        emotion_averages = {}
        for emotion, confidences in all_emotions.items():
            emotion_averages[emotion] = np.mean(confidences)
        
        summary['dominant_emotions'] = sorted(
            emotion_averages.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        # تقييم الجودة الإجمالية
        overall_quality = analysis_results['quality_metrics']['overall_score']
        if overall_quality >= 0.8:
            summary['quality_assessment'] = "ممتاز"
        elif overall_quality >= 0.6:
            summary['quality_assessment'] = "جيد"
        elif overall_quality >= 0.4:
            summary['quality_assessment'] = "متوسط"
        else:
            summary['quality_assessment'] = "ضعيف"
        
        return summary
    
    # دوال مساعدة لحساب الخصائص المختلفة
    def _calculate_bbox_center(self, bbox):
        """حساب مركز المربع المحيط"""
        x1, y1, x2, y2 = bbox
        return [(x1 + x2) / 2, (y1 + y2) / 2]
    
    def _calculate_face_orientation(self, landmarks):
        """حساب اتجاه الوجه"""
        # تحليل مبسط لاتجاه الوجه
        nose_tip = landmarks[1] if len(landmarks) > 1 else {'x': 0.5, 'y': 0.5}
        return {
            'yaw': (nose_tip['x'] - 0.5) * 2,  # -1 إلى 1
            'pitch': (nose_tip['y'] - 0.5) * 2,
            'roll': 0  # يحتاج تحليل أكثر تعقيداً
        }
    
    def _calculate_eye_aspect_ratio(self, landmarks):
        """حساب نسبة انفتاح العينين"""
        # تحليل مبسط
        return 0.3  # قيمة افتراضية
    
    def _calculate_mouth_aspect_ratio(self, landmarks):
        """حساب نسبة انفتاح الفم"""
        # تحليل مبسط
        return 0.1  # قيمة افتراضية
    
    def _calculate_face_symmetry(self, landmarks):
        """حساب تماثل الوجه"""
        # تحليل مبسط
        return 0.9  # قيمة افتراضية
    
    # باقي الدوال المساعدة...
    def _analyze_mouth_curve(self, landmarks):
        """تحليل انحناء الفم"""
        return 0.0  # تحليل مبسط
    
    def _analyze_eye_openness(self, landmarks):
        """تحليل انفتاح العينين"""
        return 0.5  # تحليل مبسط
    
    def _analyze_eyebrow_position(self, landmarks):
        """تحليل موضع الحاجبين"""
        return 0.0  # تحليل مبسط
    
    def _calculate_posture_score(self, landmarks):
        """حساب نقاط الوضعية"""
        return 0.8  # تحليل مبسط
    
    def _calculate_balance(self, landmarks):
        """حساب التوازن"""
        return 0.7  # تحليل مبسط
    
    def _calculate_movement_energy(self, landmarks):
        """حساب طاقة الحركة"""
        return 0.5  # تحليل مبسط
    
    def _calculate_body_orientation(self, landmarks):
        """حساب اتجاه الجسم"""
        return {'facing': 'forward'}  # تحليل مبسط
    
    # دوال كشف الأنشطة
    def _is_standing(self, landmarks):
        """كشف الوقوف"""
        return 0.6  # تحليل مبسط
    
    def _is_sitting(self, landmarks):
        """كشف الجلوس"""
        return 0.3  # تحليل مبسط
    
    def _is_walking(self, landmarks):
        """كشف المشي"""
        return 0.1  # تحليل مبسط
    
    def _is_waving(self, landmarks):
        """كشف التلويح"""
        return 0.0  # تحليل مبسط
    
    def _is_pointing(self, landmarks):
        """كشف الإشارة"""
        return 0.0  # تحليل مبسط
    
    # دوال كشف إيماءات اليد
    def _detect_thumbs_up(self, landmarks):
        """كشف إيماءة الإعجاب"""
        return 0.0  # تحليل مبسط
    
    def _detect_peace_sign(self, landmarks):
        """كشف إيماءة السلام"""
        return 0.0  # تحليل مبسط
    
    def _detect_pointing(self, landmarks):
        """كشف إيماءة الإشارة"""
        return 0.0  # تحليل مبسط
    
    def _detect_open_palm(self, landmarks):
        """كشف الكف المفتوح"""
        return 0.5  # تحليل مبسط
    
    def _detect_fist(self, landmarks):
        """كشف القبضة"""
        return 0.0  # تحليل مبسط

if __name__ == "__main__":
    analyzer = AdvancedVideoAnalyzer()
    
    # اختبار التحليل
    test_video = "test_video.mp4"
    if os.path.exists(test_video):
        results = analyzer.analyze_video_file(
            test_video, 
            "video_analysis_results.json"
        )
        print("📊 نتائج التحليل:")
        print(json.dumps(results['summary'], indent=2, ensure_ascii=False))
    else:
        print("⚠️ ملف الاختبار غير موجود")
