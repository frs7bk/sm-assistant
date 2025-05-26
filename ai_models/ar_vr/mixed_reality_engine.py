
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
محرك الواقع المختلط المتقدم
Advanced Mixed Reality Engine
"""

import asyncio
import logging
import numpy as np
import cv2
import json
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
import threading
import queue
import math

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False

@dataclass
class SpatialObject:
    """كائن مكاني في الواقع المختلط"""
    object_id: str
    position: Tuple[float, float, float]
    rotation: Tuple[float, float, float]
    scale: Tuple[float, float, float]
    object_type: str
    properties: Dict[str, Any]
    is_virtual: bool
    confidence: float = 1.0
    last_updated: datetime = None
    
    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.now()

@dataclass
class SpatialAnchor:
    """مرساة مكانية للواقع المختلط"""
    anchor_id: str
    world_position: Tuple[float, float, float]
    local_position: Tuple[float, float, float]
    orientation: Tuple[float, float, float, float]  # quaternion
    anchor_type: str
    confidence: float
    created_at: datetime
    attached_objects: List[str]

@dataclass
class GestureCommand:
    """أمر الإيماءة"""
    gesture_type: str
    confidence: float
    parameters: Dict[str, Any]
    timestamp: datetime
    hand_landmarks: Optional[List[Tuple[float, float, float]]] = None

class SpatialMappingEngine:
    """محرك رسم الخرائط المكانية"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # خريطة العالم المكانية
        self.spatial_map: Dict[str, SpatialObject] = {}
        self.spatial_anchors: Dict[str, SpatialAnchor] = {}
        
        # كاميرا وأجهزة الاستشعار
        self.camera = None
        self.depth_camera = None
        
        # نماذج الكشف
        if MEDIAPIPE_AVAILABLE:
            self.mp_hands = mp.solutions.hands
            self.mp_pose = mp.solutions.pose
            self.mp_face = mp.solutions.face_detection
            self.hands_detector = None
            self.pose_detector = None
            self.face_detector = None
        
        # معايرة الكاميرا
        self.camera_matrix = None
        self.distortion_coeffs = None
        
        # حالة النظام
        self.is_tracking = False
        self.tracking_quality = 0.0
        
        # إعدادات النظام
        self.config = {
            "max_tracking_distance": 5.0,  # متر
            "min_confidence": 0.7,
            "anchor_persistence_time": 300,  # ثانية
            "gesture_timeout": 2.0,
            "spatial_resolution": 0.01,  # متر
            "max_objects": 1000
        }

    async def initialize(self):
        """تهيئة محرك رسم الخرائط المكانية"""
        
        try:
            self.logger.info("🗺️ تهيئة محرك رسم الخرائط المكانية...")
            
            # تهيئة الكاميرا
            await self._initialize_camera()
            
            # تهيئة كاشفات MediaPipe
            if MEDIAPIPE_AVAILABLE:
                await self._initialize_detectors()
            
            # تحميل معايرة الكاميرا
            await self._load_camera_calibration()
            
            self.logger.info("✅ تم تهيئة محرك رسم الخرائط المكانية")
            
        except Exception as e:
            self.logger.error(f"❌ فشل تهيئة محرك رسم الخرائط المكانية: {e}")

    async def _initialize_camera(self):
        """تهيئة الكاميرا"""
        
        try:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                raise Exception("فشل في فتح الكاميرا")
            
            # تعيين دقة عالية
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            
        except Exception as e:
            self.logger.warning(f"خطأ في تهيئة الكاميرا: {e}")

    async def _initialize_detectors(self):
        """تهيئة كاشفات MediaPipe"""
        
        try:
            self.hands_detector = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
            )
            
            self.pose_detector = self.mp_pose.Pose(
                static_image_mode=False,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
            )
            
            self.face_detector = self.mp_face.FaceDetection(
                min_detection_confidence=0.7
            )
            
        except Exception as e:
            self.logger.warning(f"خطأ في تهيئة كاشفات MediaPipe: {e}")

    async def _load_camera_calibration(self):
        """تحميل معايرة الكاميرا"""
        
        try:
            calibration_file = Path("data/camera_calibration.json")
            
            if calibration_file.exists():
                with open(calibration_file, 'r') as f:
                    calibration_data = json.load(f)
                
                self.camera_matrix = np.array(calibration_data["camera_matrix"])
                self.distortion_coeffs = np.array(calibration_data["distortion_coeffs"])
            else:
                # معايرة افتراضية
                self.camera_matrix = np.array([
                    [800, 0, 320],
                    [0, 800, 240],
                    [0, 0, 1]
                ], dtype=np.float32)
                self.distortion_coeffs = np.zeros((4, 1))
                
        except Exception as e:
            self.logger.warning(f"خطأ في تحميل معايرة الكاميرا: {e}")

    async def start_spatial_tracking(self) -> bool:
        """بدء تتبع الخريطة المكانية"""
        
        try:
            if not self.camera or not self.camera.isOpened():
                await self._initialize_camera()
            
            self.is_tracking = True
            
            # بدء خيط التتبع
            tracking_thread = threading.Thread(
                target=self._tracking_loop,
                daemon=True
            )
            tracking_thread.start()
            
            self.logger.info("🎯 بدء التتبع المكاني")
            return True
            
        except Exception as e:
            self.logger.error(f"خطأ في بدء التتبع المكاني: {e}")
            return False

    def _tracking_loop(self):
        """حلقة التتبع الرئيسية"""
        
        while self.is_tracking:
            try:
                if not self.camera or not self.camera.isOpened():
                    break
                
                ret, frame = self.camera.read()
                if not ret:
                    continue
                
                # معالجة الإطار
                processed_frame = self._process_frame(frame)
                
                # تحديث الخريطة المكانية
                self._update_spatial_map(processed_frame)
                
                # تنظيف الكائنات القديمة
                self._cleanup_old_objects()
                
            except Exception as e:
                self.logger.error(f"خطأ في حلقة التتبع: {e}")
                break

    def _process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """معالجة إطار الكاميرا"""
        
        results = {
            "frame": frame,
            "hands": [],
            "pose": None,
            "faces": [],
            "objects": [],
            "timestamp": datetime.now()
        }
        
        try:
            # تحويل إلى RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if MEDIAPIPE_AVAILABLE and self.hands_detector:
                # كشف اليدين
                hands_results = self.hands_detector.process(rgb_frame)
                if hands_results.multi_hand_landmarks:
                    for hand_landmarks in hands_results.multi_hand_landmarks:
                        results["hands"].append(self._extract_hand_landmarks(hand_landmarks))
                
                # كشف الوضعية
                if self.pose_detector:
                    pose_results = self.pose_detector.process(rgb_frame)
                    if pose_results.pose_landmarks:
                        results["pose"] = self._extract_pose_landmarks(pose_results.pose_landmarks)
                
                # كشف الوجوه
                if self.face_detector:
                    face_results = self.face_detector.process(rgb_frame)
                    if face_results.detections:
                        for detection in face_results.detections:
                            results["faces"].append(self._extract_face_detection(detection))
            
            # كشف الكائنات الأخرى
            objects = self._detect_objects(frame)
            results["objects"] = objects
            
        except Exception as e:
            self.logger.warning(f"خطأ في معالجة الإطار: {e}")
        
        return results

    def _extract_hand_landmarks(self, hand_landmarks) -> List[Tuple[float, float, float]]:
        """استخراج نقاط اليد"""
        
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.append((landmark.x, landmark.y, landmark.z))
        return landmarks

    def _extract_pose_landmarks(self, pose_landmarks) -> List[Tuple[float, float, float]]:
        """استخراج نقاط الوضعية"""
        
        landmarks = []
        for landmark in pose_landmarks.landmark:
            landmarks.append((landmark.x, landmark.y, landmark.z))
        return landmarks

    def _extract_face_detection(self, detection) -> Dict[str, Any]:
        """استخراج كشف الوجه"""
        
        bbox = detection.location_data.relative_bounding_box
        return {
            "bbox": [bbox.xmin, bbox.ymin, bbox.width, bbox.height],
            "confidence": detection.score[0]
        }

    def _detect_objects(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """كشف الكائنات في الإطار"""
        
        objects = []
        
        try:
            # كشف الحواف والأشكال
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # البحث عن الكونتورات
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area > 500:  # تصفية الكونتورات الصغيرة
                    # حساب المستطيل المحيط
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # تقدير الموقع ثلاثي الأبعاد
                    center_x = x + w // 2
                    center_y = y + h // 2
                    estimated_depth = self._estimate_depth(w, h)
                    
                    world_pos = self._screen_to_world(center_x, center_y, estimated_depth)
                    
                    objects.append({
                        "id": f"object_{i}",
                        "type": "unknown",
                        "bbox": [x, y, w, h],
                        "area": area,
                        "world_position": world_pos,
                        "confidence": 0.7
                    })
            
        except Exception as e:
            self.logger.warning(f"خطأ في كشف الكائنات: {e}")
        
        return objects

    def _estimate_depth(self, width: int, height: int) -> float:
        """تقدير العمق بناءً على حجم الكائن"""
        
        # تقدير بسيط بناءً على حجم الكائن
        # يمكن تحسينه باستخدام كاميرا العمق أو التعلم الآلي
        
        object_size = math.sqrt(width * height)
        if object_size > 200:
            return 0.5  # قريب
        elif object_size > 100:
            return 1.0  # متوسط
        else:
            return 2.0  # بعيد

    def _screen_to_world(self, x: int, y: int, depth: float) -> Tuple[float, float, float]:
        """تحويل إحداثيات الشاشة إلى إحداثيات العالم"""
        
        if self.camera_matrix is None:
            return (0.0, 0.0, depth)
        
        try:
            # تحويل إلى إحداثيات الكاميرا المعيارية
            fx = self.camera_matrix[0, 0]
            fy = self.camera_matrix[1, 1]
            cx = self.camera_matrix[0, 2]
            cy = self.camera_matrix[1, 2]
            
            # حساب الموقع ثلاثي الأبعاد
            world_x = (x - cx) * depth / fx
            world_y = (y - cy) * depth / fy
            world_z = depth
            
            return (world_x, world_y, world_z)
            
        except Exception as e:
            self.logger.warning(f"خطأ في تحويل الإحداثيات: {e}")
            return (0.0, 0.0, depth)

    def _update_spatial_map(self, processed_frame: Dict[str, Any]):
        """تحديث الخريطة المكانية"""
        
        try:
            timestamp = processed_frame["timestamp"]
            
            # تحديث كائنات اليدين
            for i, hand in enumerate(processed_frame["hands"]):
                hand_id = f"hand_{i}"
                if len(hand) > 8:  # نقطة المعصم
                    wrist_pos = hand[0]  # نقطة المعصم
                    world_pos = self._screen_to_world(
                        int(wrist_pos[0] * 1280),
                        int(wrist_pos[1] * 720),
                        1.0
                    )
                    
                    self._update_spatial_object(
                        hand_id,
                        world_pos,
                        "hand",
                        {"landmarks": hand},
                        is_virtual=False,
                        confidence=0.9
                    )
            
            # تحديث كائنات الوجوه
            for i, face in enumerate(processed_frame["faces"]):
                face_id = f"face_{i}"
                bbox = face["bbox"]
                center_x = int((bbox[0] + bbox[2] / 2) * 1280)
                center_y = int((bbox[1] + bbox[3] / 2) * 720)
                
                world_pos = self._screen_to_world(center_x, center_y, 1.5)
                
                self._update_spatial_object(
                    face_id,
                    world_pos,
                    "face",
                    {"bbox": bbox},
                    is_virtual=False,
                    confidence=face["confidence"]
                )
            
            # تحديث الكائنات المكتشفة
            for obj in processed_frame["objects"]:
                self._update_spatial_object(
                    obj["id"],
                    obj["world_position"],
                    obj["type"],
                    {"bbox": obj["bbox"], "area": obj["area"]},
                    is_virtual=False,
                    confidence=obj["confidence"]
                )
            
        except Exception as e:
            self.logger.warning(f"خطأ في تحديث الخريطة المكانية: {e}")

    def _update_spatial_object(
        self,
        object_id: str,
        position: Tuple[float, float, float],
        object_type: str,
        properties: Dict[str, Any],
        is_virtual: bool,
        confidence: float
    ):
        """تحديث كائن مكاني"""
        
        if object_id in self.spatial_map:
            # تحديث كائن موجود
            obj = self.spatial_map[object_id]
            obj.position = position
            obj.properties.update(properties)
            obj.confidence = confidence
            obj.last_updated = datetime.now()
        else:
            # إنشاء كائن جديد
            if len(self.spatial_map) < self.config["max_objects"]:
                self.spatial_map[object_id] = SpatialObject(
                    object_id=object_id,
                    position=position,
                    rotation=(0.0, 0.0, 0.0),
                    scale=(1.0, 1.0, 1.0),
                    object_type=object_type,
                    properties=properties,
                    is_virtual=is_virtual,
                    confidence=confidence,
                    last_updated=datetime.now()
                )

    def _cleanup_old_objects(self):
        """تنظيف الكائنات القديمة"""
        
        current_time = datetime.now()
        objects_to_remove = []
        
        for object_id, obj in self.spatial_map.items():
            time_diff = (current_time - obj.last_updated).total_seconds()
            
            # إزالة الكائنات القديمة (غير الافتراضية)
            if not obj.is_virtual and time_diff > 5.0:
                objects_to_remove.append(object_id)
        
        for object_id in objects_to_remove:
            del self.spatial_map[object_id]

class MixedRealityEngine:
    """محرك الواقع المختلط الرئيسي"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # المكونات الأساسية
        self.spatial_mapping = SpatialMappingEngine()
        
        # الكائنات الافتراضية
        self.virtual_objects: Dict[str, SpatialObject] = {}
        
        # واجهة المستخدم المكانية
        self.spatial_ui_elements: Dict[str, Dict[str, Any]] = {}
        
        # نظام التفاعل
        self.interaction_zones: Dict[str, Dict[str, Any]] = {}
        self.active_gestures: Dict[str, GestureCommand] = {}
        
        # حالة النظام
        self.is_active = False
        self.current_mode = "tracking"  # tracking, interaction, creation
        
        # إعدادات العرض
        self.display_config = {
            "show_spatial_map": True,
            "show_virtual_objects": True,
            "show_ui_elements": True,
            "show_debug_info": False,
            "transparency": 0.7
        }

    async def initialize(self):
        """تهيئة محرك الواقع المختلط"""
        
        try:
            self.logger.info("🥽 تهيئة محرك الواقع المختلط...")
            
            # تهيئة رسم الخرائط المكانية
            await self.spatial_mapping.initialize()
            
            # إنشاء عناصر واجهة المستخدم الافتراضية
            await self._create_default_ui()
            
            # تهيئة مناطق التفاعل
            await self._setup_interaction_zones()
            
            self.is_active = True
            self.logger.info("✅ تم تهيئة محرك الواقع المختلط")
            
        except Exception as e:
            self.logger.error(f"❌ فشل تهيئة محرك الواقع المختلط: {e}")

    async def _create_default_ui(self):
        """إنشاء واجهة المستخدم الافتراضية"""
        
        try:
            # قائمة رئيسية افتراضية
            self.spatial_ui_elements["main_menu"] = {
                "type": "menu",
                "position": (0.0, 0.5, -1.0),
                "items": [
                    {"id": "create_object", "label": "إنشاء كائن", "action": "create_virtual_object"},
                    {"id": "settings", "label": "الإعدادات", "action": "open_settings"},
                    {"id": "help", "label": "المساعدة", "action": "show_help"}
                ],
                "visible": False,
                "size": (0.3, 0.4)
            }
            
            # لوحة معلومات
            self.spatial_ui_elements["info_panel"] = {
                "type": "panel",
                "position": (0.5, 0.8, -0.8),
                "content": "مرحباً بك في الواقع المختلط",
                "visible": True,
                "size": (0.4, 0.1),
                "background_color": (0, 0, 0, 0.7)
            }
            
        except Exception as e:
            self.logger.warning(f"خطأ في إنشاء واجهة المستخدم: {e}")

    async def _setup_interaction_zones(self):
        """إعداد مناطق التفاعل"""
        
        try:
            # منطقة تفاعل اليد اليمنى
            self.interaction_zones["right_hand"] = {
                "type": "hand_interaction",
                "radius": 0.2,
                "actions": ["select", "grab", "point"],
                "sensitivity": 0.8
            }
            
            # منطقة تفاعل الصوت
            self.interaction_zones["voice"] = {
                "type": "voice_interaction",
                "radius": 2.0,
                "actions": ["command", "dictation", "navigation"],
                "sensitivity": 0.7
            }
            
            # منطقة تفاعل النظر
            self.interaction_zones["gaze"] = {
                "type": "gaze_interaction",
                "radius": 0.1,
                "actions": ["focus", "select", "scroll"],
                "sensitivity": 0.9
            }
            
        except Exception as e:
            self.logger.warning(f"خطأ في إعداد مناطق التفاعل: {e}")

    async def start_mixed_reality_session(self) -> bool:
        """بدء جلسة الواقع المختلط"""
        
        try:
            if not self.is_active:
                await self.initialize()
            
            # بدء التتبع المكاني
            success = await self.spatial_mapping.start_spatial_tracking()
            
            if success:
                # بدء حلقة العرض
                display_thread = threading.Thread(
                    target=self._display_loop,
                    daemon=True
                )
                display_thread.start()
                
                self.logger.info("🎮 بدء جلسة الواقع المختلط")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"خطأ في بدء جلسة الواقع المختلط: {e}")
            return False

    def _display_loop(self):
        """حلقة العرض الرئيسية"""
        
        while self.is_active:
            try:
                if not self.spatial_mapping.camera or not self.spatial_mapping.camera.isOpened():
                    break
                
                ret, frame = self.spatial_mapping.camera.read()
                if not ret:
                    continue
                
                # إنشاء العرض المختلط
                mixed_frame = self._create_mixed_display(frame)
                
                # عرض الإطار
                cv2.imshow("Mixed Reality", mixed_frame)
                
                # معالجة أحداث لوحة المفاتيح
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('m'):
                    await self._toggle_main_menu()
                elif key == ord('i'):
                    self.display_config["show_debug_info"] = not self.display_config["show_debug_info"]
                
            except Exception as e:
                self.logger.error(f"خطأ في حلقة العرض: {e}")
                break

    def _create_mixed_display(self, frame: np.ndarray) -> np.ndarray:
        """إنشاء العرض المختلط"""
        
        mixed_frame = frame.copy()
        
        try:
            # رسم الخريطة المكانية
            if self.display_config["show_spatial_map"]:
                mixed_frame = self._draw_spatial_map(mixed_frame)
            
            # رسم الكائنات الافتراضية
            if self.display_config["show_virtual_objects"]:
                mixed_frame = self._draw_virtual_objects(mixed_frame)
            
            # رسم واجهة المستخدم
            if self.display_config["show_ui_elements"]:
                mixed_frame = self._draw_ui_elements(mixed_frame)
            
            # رسم معلومات التصحيح
            if self.display_config["show_debug_info"]:
                mixed_frame = self._draw_debug_info(mixed_frame)
            
        except Exception as e:
            self.logger.warning(f"خطأ في إنشاء العرض المختلط: {e}")
        
        return mixed_frame

    def _draw_spatial_map(self, frame: np.ndarray) -> np.ndarray:
        """رسم الخريطة المكانية"""
        
        try:
            for obj_id, obj in self.spatial_mapping.spatial_map.items():
                if not obj.is_virtual:
                    # تحويل الموقع العالمي إلى إحداثيات الشاشة
                    screen_pos = self._world_to_screen(obj.position)
                    
                    if screen_pos:
                        x, y = screen_pos
                        
                        # رسم نقطة للكائن
                        color = self._get_object_color(obj.object_type)
                        cv2.circle(frame, (int(x), int(y)), 5, color, -1)
                        
                        # رسم تسمية
                        cv2.putText(
                            frame,
                            f"{obj.object_type}_{obj_id[-2:]}",
                            (int(x) + 10, int(y) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color,
                            1
                        )
            
        except Exception as e:
            self.logger.warning(f"خطأ في رسم الخريطة المكانية: {e}")
        
        return frame

    def _world_to_screen(self, world_pos: Tuple[float, float, float]) -> Optional[Tuple[int, int]]:
        """تحويل الموقع العالمي إلى إحداثيات الشاشة"""
        
        try:
            if self.spatial_mapping.camera_matrix is None:
                return None
            
            x, y, z = world_pos
            
            if z <= 0:
                return None
            
            # الإسقاط على الشاشة
            fx = self.spatial_mapping.camera_matrix[0, 0]
            fy = self.spatial_mapping.camera_matrix[1, 1]
            cx = self.spatial_mapping.camera_matrix[0, 2]
            cy = self.spatial_mapping.camera_matrix[1, 2]
            
            screen_x = int((x * fx / z) + cx)
            screen_y = int((y * fy / z) + cy)
            
            # التحقق من الحدود
            if 0 <= screen_x < 1280 and 0 <= screen_y < 720:
                return (screen_x, screen_y)
            
            return None
            
        except Exception as e:
            self.logger.warning(f"خطأ في تحويل الإحداثيات: {e}")
            return None

    def _get_object_color(self, object_type: str) -> Tuple[int, int, int]:
        """الحصول على لون الكائن حسب النوع"""
        
        colors = {
            "hand": (0, 255, 0),      # أخضر
            "face": (255, 0, 0),      # أحمر
            "unknown": (0, 0, 255),   # أزرق
            "virtual": (255, 255, 0), # أصفر
            "ui": (255, 0, 255)       # أرجواني
        }
        
        return colors.get(object_type, (128, 128, 128))

    def _draw_virtual_objects(self, frame: np.ndarray) -> np.ndarray:
        """رسم الكائنات الافتراضية"""
        
        try:
            for obj_id, obj in self.virtual_objects.items():
                screen_pos = self._world_to_screen(obj.position)
                
                if screen_pos:
                    x, y = screen_pos
                    
                    # رسم الكائن الافتراضي
                    if obj.object_type == "cube":
                        self._draw_virtual_cube(frame, (x, y), obj.scale[0] * 50)
                    elif obj.object_type == "sphere":
                        self._draw_virtual_sphere(frame, (x, y), obj.scale[0] * 30)
                    elif obj.object_type == "text":
                        self._draw_virtual_text(frame, (x, y), obj.properties.get("text", ""))
            
        except Exception as e:
            self.logger.warning(f"خطأ في رسم الكائنات الافتراضية: {e}")
        
        return frame

    def _draw_virtual_cube(self, frame: np.ndarray, center: Tuple[int, int], size: int):
        """رسم مكعب افتراضي"""
        
        x, y = center
        half_size = size // 2
        
        # رسم مستطيل بسيط لتمثيل المكعب
        cv2.rectangle(
            frame,
            (x - half_size, y - half_size),
            (x + half_size, y + half_size),
            (255, 255, 0),
            2
        )
        
        # إضافة تأثير ثلاثي الأبعاد
        offset = size // 4
        cv2.line(frame, (x - half_size, y - half_size), (x - half_size + offset, y - half_size - offset), (255, 255, 0), 2)
        cv2.line(frame, (x + half_size, y - half_size), (x + half_size + offset, y - half_size - offset), (255, 255, 0), 2)
        cv2.line(frame, (x + half_size, y + half_size), (x + half_size + offset, y + half_size - offset), (255, 255, 0), 2)
        cv2.line(frame, (x - half_size, y + half_size), (x - half_size + offset, y + half_size - offset), (255, 255, 0), 2)

    def _draw_virtual_sphere(self, frame: np.ndarray, center: Tuple[int, int], radius: int):
        """رسم كرة افتراضية"""
        
        cv2.circle(frame, center, radius, (0, 255, 255), 2)
        cv2.circle(frame, center, radius // 2, (0, 255, 255), 1)

    def _draw_virtual_text(self, frame: np.ndarray, position: Tuple[int, int], text: str):
        """رسم نص افتراضي"""
        
        cv2.putText(
            frame,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2
        )

    def _draw_ui_elements(self, frame: np.ndarray) -> np.ndarray:
        """رسم عناصر واجهة المستخدم"""
        
        try:
            for element_id, element in self.spatial_ui_elements.items():
                if not element.get("visible", True):
                    continue
                
                screen_pos = self._world_to_screen(element["position"])
                
                if screen_pos:
                    if element["type"] == "menu":
                        self._draw_menu(frame, screen_pos, element)
                    elif element["type"] == "panel":
                        self._draw_panel(frame, screen_pos, element)
            
        except Exception as e:
            self.logger.warning(f"خطأ في رسم واجهة المستخدم: {e}")
        
        return frame

    def _draw_menu(self, frame: np.ndarray, position: Tuple[int, int], menu: Dict[str, Any]):
        """رسم قائمة"""
        
        x, y = position
        item_height = 40
        
        for i, item in enumerate(menu.get("items", [])):
            item_y = y + i * item_height
            
            # رسم خلفية العنصر
            cv2.rectangle(
                frame,
                (x, item_y),
                (x + 200, item_y + item_height - 5),
                (50, 50, 50),
                -1
            )
            
            # رسم النص
            cv2.putText(
                frame,
                item["label"],
                (x + 10, item_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

    def _draw_panel(self, frame: np.ndarray, position: Tuple[int, int], panel: Dict[str, Any]):
        """رسم لوحة"""
        
        x, y = position
        width, height = panel.get("size", (200, 50))
        width = int(width * 500)
        height = int(height * 500)
        
        # رسم خلفية اللوحة
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + width, y + height), (0, 0, 0), -1)
        
        alpha = panel.get("background_color", [0, 0, 0, 0.7])[3]
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # رسم النص
        cv2.putText(
            frame,
            panel.get("content", ""),
            (x + 10, y + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

    def _draw_debug_info(self, frame: np.ndarray) -> np.ndarray:
        """رسم معلومات التصحيح"""
        
        try:
            debug_info = [
                f"Objects: {len(self.spatial_mapping.spatial_map)}",
                f"Virtual: {len(self.virtual_objects)}",
                f"Mode: {self.current_mode}",
                f"Tracking: {self.spatial_mapping.is_tracking}",
                f"Quality: {self.spatial_mapping.tracking_quality:.1f}"
            ]
            
            for i, info in enumerate(debug_info):
                cv2.putText(
                    frame,
                    info,
                    (10, 30 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    1
                )
            
        except Exception as e:
            self.logger.warning(f"خطأ في رسم معلومات التصحيح: {e}")
        
        return frame

    async def create_virtual_object(
        self,
        object_type: str,
        position: Tuple[float, float, float],
        properties: Dict[str, Any] = None
    ) -> str:
        """إنشاء كائن افتراضي"""
        
        try:
            object_id = f"virtual_{len(self.virtual_objects)}"
            
            virtual_object = SpatialObject(
                object_id=object_id,
                position=position,
                rotation=(0.0, 0.0, 0.0),
                scale=(1.0, 1.0, 1.0),
                object_type=object_type,
                properties=properties or {},
                is_virtual=True,
                confidence=1.0
            )
            
            self.virtual_objects[object_id] = virtual_object
            
            self.logger.info(f"تم إنشاء كائن افتراضي: {object_id}")
            return object_id
            
        except Exception as e:
            self.logger.error(f"خطأ في إنشاء كائن افتراضي: {e}")
            return ""

    async def detect_gesture(self, hand_landmarks: List[Tuple[float, float, float]]) -> Optional[GestureCommand]:
        """كشف الإيماءة"""
        
        try:
            if len(hand_landmarks) < 21:  # يد MediaPipe لها 21 نقطة
                return None
            
            # تحليل الإيماءات البسيطة
            gesture = self._analyze_hand_gesture(hand_landmarks)
            
            if gesture:
                return GestureCommand(
                    gesture_type=gesture["type"],
                    confidence=gesture["confidence"],
                    parameters=gesture.get("parameters", {}),
                    timestamp=datetime.now(),
                    hand_landmarks=hand_landmarks
                )
            
            return None
            
        except Exception as e:
            self.logger.warning(f"خطأ في كشف الإيماءة: {e}")
            return None

    def _analyze_hand_gesture(self, landmarks: List[Tuple[float, float, float]]) -> Optional[Dict[str, Any]]:
        """تحليل إيماءة اليد"""
        
        try:
            # نقاط الأصابع (أطراف الأصابع)
            finger_tips = [4, 8, 12, 16, 20]  # الإبهام، السبابة، الوسطى، البنصر، الخنصر
            finger_pips = [3, 6, 10, 14, 18]  # المفاصل الوسطى
            
            # حساب حالة كل إصبع (مفتوح/مغلق)
            fingers_up = []
            
            # الإبهام (مقارنة x)
            if landmarks[finger_tips[0]][0] > landmarks[finger_pips[0]][0]:
                fingers_up.append(1)
            else:
                fingers_up.append(0)
            
            # باقي الأصابع (مقارنة y)
            for i in range(1, 5):
                if landmarks[finger_tips[i]][1] < landmarks[finger_pips[i]][1]:
                    fingers_up.append(1)
                else:
                    fingers_up.append(0)
            
            # تحديد نوع الإيماءة
            total_fingers = sum(fingers_up)
            
            if total_fingers == 0:
                return {"type": "fist", "confidence": 0.9}
            elif total_fingers == 1 and fingers_up[1] == 1:  # السبابة فقط
                return {"type": "point", "confidence": 0.8}
            elif total_fingers == 2 and fingers_up[1] == 1 and fingers_up[2] == 1:  # سبابة ووسطى
                return {"type": "peace", "confidence": 0.8}
            elif total_fingers == 5:
                return {"type": "open_hand", "confidence": 0.9}
            elif total_fingers == 1 and fingers_up[0] == 1:  # الإبهام فقط
                return {"type": "thumbs_up", "confidence": 0.8}
            
            return {"type": "unknown", "confidence": 0.5}
            
        except Exception as e:
            self.logger.warning(f"خطأ في تحليل إيماءة اليد: {e}")
            return None

    async def process_voice_command(self, command: str) -> Dict[str, Any]:
        """معالجة أمر صوتي في الواقع المختلط"""
        
        try:
            command_lower = command.lower().strip()
            
            if "أنشئ" in command_lower or "create" in command_lower:
                # أمر إنشاء كائن
                if "مكعب" in command_lower or "cube" in command_lower:
                    object_id = await self.create_virtual_object("cube", (0.0, 0.0, -1.0))
                    return {"success": True, "action": "create_cube", "object_id": object_id}
                
                elif "كرة" in command_lower or "sphere" in command_lower:
                    object_id = await self.create_virtual_object("sphere", (0.0, 0.0, -1.0))
                    return {"success": True, "action": "create_sphere", "object_id": object_id}
                
                elif "نص" in command_lower or "text" in command_lower:
                    object_id = await self.create_virtual_object(
                        "text", 
                        (0.0, 0.0, -1.0),
                        {"text": "مرحباً بك!"}
                    )
                    return {"success": True, "action": "create_text", "object_id": object_id}
            
            elif "اظهر القائمة" in command_lower or "show menu" in command_lower:
                await self._toggle_main_menu()
                return {"success": True, "action": "toggle_menu"}
            
            elif "امسح" in command_lower or "clear" in command_lower:
                self.virtual_objects.clear()
                return {"success": True, "action": "clear_objects"}
            
            elif "معلومات" in command_lower or "info" in command_lower:
                self.display_config["show_debug_info"] = not self.display_config["show_debug_info"]
                return {"success": True, "action": "toggle_debug"}
            
            return {"success": False, "error": "أمر غير مفهوم"}
            
        except Exception as e:
            self.logger.error(f"خطأ في معالجة الأمر الصوتي: {e}")
            return {"success": False, "error": str(e)}

    async def _toggle_main_menu(self):
        """تبديل القائمة الرئيسية"""
        
        if "main_menu" in self.spatial_ui_elements:
            current_state = self.spatial_ui_elements["main_menu"]["visible"]
            self.spatial_ui_elements["main_menu"]["visible"] = not current_state

    async def get_spatial_analytics(self) -> Dict[str, Any]:
        """الحصول على تحليلات الخريطة المكانية"""
        
        try:
            # إحصائيات الكائنات
            object_stats = {}
            for obj in self.spatial_mapping.spatial_map.values():
                obj_type = obj.object_type
                if obj_type not in object_stats:
                    object_stats[obj_type] = {"count": 0, "avg_confidence": 0.0}
                
                object_stats[obj_type]["count"] += 1
                object_stats[obj_type]["avg_confidence"] += obj.confidence
            
            # حساب متوسط الثقة
            for stats in object_stats.values():
                stats["avg_confidence"] /= stats["count"]
            
            # تحليل التوزيع المكاني
            positions = [obj.position for obj in self.spatial_mapping.spatial_map.values()]
            spatial_analysis = self._analyze_spatial_distribution(positions)
            
            return {
                "total_objects": len(self.spatial_mapping.spatial_map),
                "virtual_objects": len(self.virtual_objects),
                "object_statistics": object_stats,
                "spatial_distribution": spatial_analysis,
                "tracking_quality": self.spatial_mapping.tracking_quality,
                "system_status": {
                    "is_active": self.is_active,
                    "current_mode": self.current_mode,
                    "camera_status": self.spatial_mapping.camera is not None
                }
            }
            
        except Exception as e:
            self.logger.error(f"خطأ في الحصول على التحليلات المكانية: {e}")
            return {"error": str(e)}

    def _analyze_spatial_distribution(self, positions: List[Tuple[float, float, float]]) -> Dict[str, Any]:
        """تحليل التوزيع المكاني"""
        
        if not positions:
            return {"center": (0, 0, 0), "spread": 0.0, "density": 0.0}
        
        try:
            # حساب المركز
            center_x = sum(pos[0] for pos in positions) / len(positions)
            center_y = sum(pos[1] for pos in positions) / len(positions)
            center_z = sum(pos[2] for pos in positions) / len(positions)
            center = (center_x, center_y, center_z)
            
            # حساب الانتشار
            distances = [
                math.sqrt((pos[0] - center_x)**2 + (pos[1] - center_y)**2 + (pos[2] - center_z)**2)
                for pos in positions
            ]
            spread = max(distances) if distances else 0.0
            
            # حساب الكثافة
            volume = (4/3) * math.pi * (spread**3) if spread > 0 else 1.0
            density = len(positions) / volume
            
            return {
                "center": center,
                "spread": spread,
                "density": density,
                "object_count": len(positions)
            }
            
        except Exception as e:
            self.logger.warning(f"خطأ في تحليل التوزيع المكاني: {e}")
            return {"center": (0, 0, 0), "spread": 0.0, "density": 0.0}

    async def stop_mixed_reality_session(self):
        """إيقاف جلسة الواقع المختلط"""
        
        try:
            self.is_active = False
            self.spatial_mapping.is_tracking = False
            
            if self.spatial_mapping.camera:
                self.spatial_mapping.camera.release()
            
            cv2.destroyAllWindows()
            
            self.logger.info("🔴 تم إيقاف جلسة الواقع المختلط")
            
        except Exception as e:
            self.logger.error(f"خطأ في إيقاف جلسة الواقع المختلط: {e}")

# إنشاء مثيل عام
mixed_reality_engine = MixedRealityEngine()

async def get_mixed_reality_engine() -> MixedRealityEngine:
    """الحصول على محرك الواقع المختلط"""
    return mixed_reality_engine

if __name__ == "__main__":
    async def test_mixed_reality():
        """اختبار محرك الواقع المختلط"""
        print("🥽 اختبار محرك الواقع المختلط")
        print("=" * 50)
        
        engine = await get_mixed_reality_engine()
        
        # بدء الجلسة
        success = await engine.start_mixed_reality_session()
        
        if success:
            print("✅ تم بدء جلسة الواقع المختلط")
            print("🎮 الأوامر المتاحة:")
            print("   • 'q' - خروج")
            print("   • 'm' - تبديل القائمة")
            print("   • 'i' - معلومات التصحيح")
            
            # إنشاء بعض الكائنات التجريبية
            await engine.create_virtual_object("cube", (0.0, 0.0, -1.0))
            await engine.create_virtual_object("sphere", (0.5, 0.0, -1.5))
            await engine.create_virtual_object("text", (-0.5, 0.0, -1.2), {"text": "اختبار النص"})
            
            # انتظار المستخدم
            print("⏳ اضغط 'q' في نافذة العرض للخروج...")
            
            try:
                while engine.is_active:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                pass
            
            await engine.stop_mixed_reality_session()
        else:
            print("❌ فشل في بدء جلسة الواقع المختلط")
    
    asyncio.run(test_mixed_reality())
