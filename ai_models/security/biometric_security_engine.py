
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
محرك الأمان البيومتري المتقدم
حماية متعددة الطبقات بالبصمات والتعرف على الوجه والصوت
"""

import asyncio
import logging
import cv2
import numpy as np
import hashlib
import json
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import sqlite3
import face_recognition
import librosa
import base64
import threading
import queue
from dataclasses import dataclass, asdict
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

@dataclass
class BiometricTemplate:
    """قالب بيومتري"""
    template_id: str
    user_id: str
    modality: str  # face, voice, fingerprint, iris
    template_data: bytes
    confidence_threshold: float
    created_at: datetime
    last_used: datetime
    usage_count: int
    is_active: bool

@dataclass
class SecurityEvent:
    """حدث أمني"""
    event_id: str
    event_type: str
    user_id: str
    timestamp: datetime
    success: bool
    confidence_score: float
    additional_data: Dict[str, Any]

class BiometricSecurityEngine:
    """محرك الأمان البيومتري المتقدم"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # قاعدة البيانات الأمنية
        self.security_db_path = Path("data/security/biometric.db")
        self.security_db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_security_database()
        
        # مفتاح التشفير
        self.encryption_key = self._generate_encryption_key()
        self.cipher_suite = Fernet(self.encryption_key)
        
        # قوالب البيومترية المحفوظة
        self.biometric_templates = {}
        
        # إعدادات الأمان
        self.security_settings = {
            "face_confidence_threshold": 0.6,
            "voice_confidence_threshold": 0.7,
            "fingerprint_confidence_threshold": 0.8,
            "multi_factor_required": True,
            "max_failed_attempts": 3,
            "lockout_duration_minutes": 15,
            "template_update_frequency_days": 30
        }
        
        # إحصائيات الأمان
        self.security_stats = {
            "total_authentications": 0,
            "successful_authentications": 0,
            "failed_attempts": 0,
            "security_breaches": 0,
            "average_confidence_score": 0.0
        }
        
        # قائمة المستخدمين المحظورين مؤقتاً
        self.locked_users = {}
        
        # تحميل القوالب الموجودة
        self._load_biometric_templates()
        
        # بدء مراقب الأمان
        self._start_security_monitor()
    
    def _init_security_database(self):
        """تهيئة قاعدة البيانات الأمنية"""
        conn = sqlite3.connect(self.security_db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS biometric_templates (
                template_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                modality TEXT NOT NULL,
                template_data BLOB,
                confidence_threshold REAL,
                created_at REAL,
                last_used REAL,
                usage_count INTEGER DEFAULT 0,
                is_active BOOLEAN DEFAULT 1
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS security_events (
                event_id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                user_id TEXT,
                timestamp REAL,
                success BOOLEAN,
                confidence_score REAL,
                additional_data TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS failed_attempts (
                user_id TEXT,
                timestamp REAL,
                attempt_type TEXT,
                ip_address TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _generate_encryption_key(self) -> bytes:
        """توليد مفتاح التشفير"""
        # في التطبيق الحقيقي، يجب تخزين هذا بشكل آمن
        password = b"biometric_security_2024"
        salt = b"salt_for_biometric_engine"
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password))
        return key
    
    async def enroll_face(self, user_id: str, image_data: Union[np.ndarray, bytes]) -> Dict[str, Any]:
        """تسجيل بصمة الوجه"""
        try:
            # تحويل البيانات إلى صورة
            if isinstance(image_data, bytes):
                nparr = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                image = image_data
            
            # تحويل من BGR إلى RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # العثور على الوجوه
            face_locations = face_recognition.face_locations(rgb_image)
            
            if not face_locations:
                return {
                    "success": False,
                    "message": "لم يتم العثور على وجه في الصورة"
                }
            
            if len(face_locations) > 1:
                return {
                    "success": False,
                    "message": "تم العثور على أكثر من وجه. يرجى استخدام صورة تحتوي على وجه واحد فقط"
                }
            
            # استخراج ميزات الوجه
            face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
            
            if not face_encodings:
                return {
                    "success": False,
                    "message": "فشل في استخراج ميزات الوجه"
                }
            
            face_encoding = face_encodings[0]
            
            # تشفير وحفظ القالب
            template_id = f"face_{user_id}_{datetime.now().timestamp()}"
            encrypted_template = self.cipher_suite.encrypt(pickle.dumps(face_encoding))
            
            template = BiometricTemplate(
                template_id=template_id,
                user_id=user_id,
                modality="face",
                template_data=encrypted_template,
                confidence_threshold=self.security_settings["face_confidence_threshold"],
                created_at=datetime.now(),
                last_used=datetime.now(),
                usage_count=0,
                is_active=True
            )
            
            # حفظ في الذاكرة وقاعدة البيانات
            self.biometric_templates[template_id] = template
            await self._save_template_to_database(template)
            
            # تسجيل الحدث الأمني
            await self._log_security_event("face_enrollment", user_id, True, 1.0, 
                                         {"template_id": template_id})
            
            return {
                "success": True,
                "message": "تم تسجيل بصمة الوجه بنجاح",
                "template_id": template_id
            }
        
        except Exception as e:
            self.logger.error(f"خطأ في تسجيل بصمة الوجه: {e}")
            await self._log_security_event("face_enrollment", user_id, False, 0.0, 
                                         {"error": str(e)})
            return {
                "success": False,
                "message": f"خطأ في تسجيل بصمة الوجه: {str(e)}"
            }
    
    async def verify_face(self, user_id: str, image_data: Union[np.ndarray, bytes]) -> Dict[str, Any]:
        """التحقق من بصمة الوجه"""
        try:
            # فحص الحظر المؤقت
            if await self._is_user_locked(user_id):
                return {
                    "success": False,
                    "message": "المستخدم محظور مؤقتاً بسبب المحاولات الفاشلة",
                    "locked_until": self.locked_users[user_id].isoformat()
                }
            
            # تحويل البيانات إلى صورة
            if isinstance(image_data, bytes):
                nparr = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                image = image_data
            
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # العثور على الوجوه واستخراج الميزات
            face_locations = face_recognition.face_locations(rgb_image)
            if not face_locations:
                await self._record_failed_attempt(user_id, "face_verification")
                return {
                    "success": False,
                    "message": "لم يتم العثور على وجه في الصورة"
                }
            
            face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
            if not face_encodings:
                await self._record_failed_attempt(user_id, "face_verification")
                return {
                    "success": False,
                    "message": "فشل في استخراج ميزات الوجه"
                }
            
            unknown_face_encoding = face_encodings[0]
            
            # البحث عن قوالب الوجه للمستخدم
            user_face_templates = [
                template for template in self.biometric_templates.values()
                if template.user_id == user_id and template.modality == "face" and template.is_active
            ]
            
            if not user_face_templates:
                await self._record_failed_attempt(user_id, "face_verification")
                return {
                    "success": False,
                    "message": "لا توجد قوالب وجه مسجلة لهذا المستخدم"
                }
            
            # مقارنة مع كل القوالب المسجلة
            best_match_score = 0.0
            best_template = None
            
            for template in user_face_templates:
                try:
                    # فك تشفير القالب
                    decrypted_data = self.cipher_suite.decrypt(template.template_data)
                    known_face_encoding = pickle.loads(decrypted_data)
                    
                    # حساب المسافة (التشابه)
                    face_distances = face_recognition.face_distance([known_face_encoding], unknown_face_encoding)
                    confidence_score = 1 - face_distances[0]  # تحويل المسافة إلى نقاط ثقة
                    
                    if confidence_score > best_match_score:
                        best_match_score = confidence_score
                        best_template = template
                
                except Exception as e:
                    self.logger.error(f"خطأ في فك تشفير قالب الوجه: {e}")
                    continue
            
            # تقييم النتيجة
            is_verified = (best_match_score >= self.security_settings["face_confidence_threshold"])
            
            if is_verified and best_template:
                # تحديث استخدام القالب
                best_template.last_used = datetime.now()
                best_template.usage_count += 1
                await self._update_template_usage(best_template)
                
                await self._log_security_event("face_verification", user_id, True, best_match_score,
                                             {"template_id": best_template.template_id})
                
                self.security_stats["successful_authentications"] += 1
                
                return {
                    "success": True,
                    "message": "تم التحقق من بصمة الوجه بنجاح",
                    "confidence_score": best_match_score,
                    "template_id": best_template.template_id
                }
            else:
                await self._record_failed_attempt(user_id, "face_verification")
                await self._log_security_event("face_verification", user_id, False, best_match_score,
                                             {"reason": "confidence_below_threshold"})
                
                return {
                    "success": False,
                    "message": "فشل في التحقق من بصمة الوجه",
                    "confidence_score": best_match_score
                }
        
        except Exception as e:
            self.logger.error(f"خطأ في التحقق من بصمة الوجه: {e}")
            await self._record_failed_attempt(user_id, "face_verification")
            await self._log_security_event("face_verification", user_id, False, 0.0,
                                         {"error": str(e)})
            return {
                "success": False,
                "message": f"خطأ في التحقق من بصمة الوجه: {str(e)}"
            }
    
    async def enroll_voice(self, user_id: str, audio_data: bytes) -> Dict[str, Any]:
        """تسجيل بصمة الصوت"""
        try:
            # تحويل البيانات الصوتية
            audio_array = np.frombuffer(audio_data, dtype=np.float32)
            
            # استخراج ميزات الصوت باستخدام MFCC
            mfcc_features = librosa.feature.mfcc(y=audio_array, sr=22050, n_mfcc=13)
            
            # حساب الإحصائيات للميزات
            voice_features = {
                "mfcc_mean": np.mean(mfcc_features, axis=1),
                "mfcc_std": np.std(mfcc_features, axis=1),
                "spectral_centroid": np.mean(librosa.feature.spectral_centroid(y=audio_array, sr=22050)),
                "zero_crossing_rate": np.mean(librosa.feature.zero_crossing_rate(audio_array))
            }
            
            # تشفير وحفظ القالب
            template_id = f"voice_{user_id}_{datetime.now().timestamp()}"
            encrypted_template = self.cipher_suite.encrypt(pickle.dumps(voice_features))
            
            template = BiometricTemplate(
                template_id=template_id,
                user_id=user_id,
                modality="voice",
                template_data=encrypted_template,
                confidence_threshold=self.security_settings["voice_confidence_threshold"],
                created_at=datetime.now(),
                last_used=datetime.now(),
                usage_count=0,
                is_active=True
            )
            
            # حفظ في الذاكرة وقاعدة البيانات
            self.biometric_templates[template_id] = template
            await self._save_template_to_database(template)
            
            await self._log_security_event("voice_enrollment", user_id, True, 1.0,
                                         {"template_id": template_id})
            
            return {
                "success": True,
                "message": "تم تسجيل بصمة الصوت بنجاح",
                "template_id": template_id
            }
        
        except Exception as e:
            self.logger.error(f"خطأ في تسجيل بصمة الصوت: {e}")
            await self._log_security_event("voice_enrollment", user_id, False, 0.0,
                                         {"error": str(e)})
            return {
                "success": False,
                "message": f"خطأ في تسجيل بصمة الصوت: {str(e)}"
            }
    
    async def verify_voice(self, user_id: str, audio_data: bytes) -> Dict[str, Any]:
        """التحقق من بصمة الصوت"""
        try:
            if await self._is_user_locked(user_id):
                return {
                    "success": False,
                    "message": "المستخدم محظور مؤقتاً",
                    "locked_until": self.locked_users[user_id].isoformat()
                }
            
            # استخراج ميزات الصوت
            audio_array = np.frombuffer(audio_data, dtype=np.float32)
            mfcc_features = librosa.feature.mfcc(y=audio_array, sr=22050, n_mfcc=13)
            
            test_features = {
                "mfcc_mean": np.mean(mfcc_features, axis=1),
                "mfcc_std": np.std(mfcc_features, axis=1),
                "spectral_centroid": np.mean(librosa.feature.spectral_centroid(y=audio_array, sr=22050)),
                "zero_crossing_rate": np.mean(librosa.feature.zero_crossing_rate(audio_array))
            }
            
            # البحث عن قوالب الصوت للمستخدم
            user_voice_templates = [
                template for template in self.biometric_templates.values()
                if template.user_id == user_id and template.modality == "voice" and template.is_active
            ]
            
            if not user_voice_templates:
                await self._record_failed_attempt(user_id, "voice_verification")
                return {
                    "success": False,
                    "message": "لا توجد قوالب صوت مسجلة لهذا المستخدم"
                }
            
            # مقارنة مع كل القوالب المسجلة
            best_match_score = 0.0
            best_template = None
            
            for template in user_voice_templates:
                try:
                    # فك تشفير القالب
                    decrypted_data = self.cipher_suite.decrypt(template.template_data)
                    stored_features = pickle.loads(decrypted_data)
                    
                    # حساب التشابه باستخدام المسافة الكوسينية
                    confidence_score = self._calculate_voice_similarity(test_features, stored_features)
                    
                    if confidence_score > best_match_score:
                        best_match_score = confidence_score
                        best_template = template
                
                except Exception as e:
                    self.logger.error(f"خطأ في فك تشفير قالب الصوت: {e}")
                    continue
            
            # تقييم النتيجة
            is_verified = (best_match_score >= self.security_settings["voice_confidence_threshold"])
            
            if is_verified and best_template:
                best_template.last_used = datetime.now()
                best_template.usage_count += 1
                await self._update_template_usage(best_template)
                
                await self._log_security_event("voice_verification", user_id, True, best_match_score,
                                             {"template_id": best_template.template_id})
                
                self.security_stats["successful_authentications"] += 1
                
                return {
                    "success": True,
                    "message": "تم التحقق من بصمة الصوت بنجاح",
                    "confidence_score": best_match_score,
                    "template_id": best_template.template_id
                }
            else:
                await self._record_failed_attempt(user_id, "voice_verification")
                await self._log_security_event("voice_verification", user_id, False, best_match_score,
                                             {"reason": "confidence_below_threshold"})
                
                return {
                    "success": False,
                    "message": "فشل في التحقق من بصمة الصوت",
                    "confidence_score": best_match_score
                }
        
        except Exception as e:
            self.logger.error(f"خطأ في التحقق من بصمة الصوت: {e}")
            await self._record_failed_attempt(user_id, "voice_verification")
            return {
                "success": False,
                "message": f"خطأ في التحقق من بصمة الصوت: {str(e)}"
            }
    
    def _calculate_voice_similarity(self, features1: Dict[str, Any], features2: Dict[str, Any]) -> float:
        """حساب التشابه بين ميزات الصوت"""
        try:
            similarities = []
            
            # مقارنة MFCC
            mfcc_sim = np.corrcoef(features1["mfcc_mean"], features2["mfcc_mean"])[0, 1]
            if not np.isnan(mfcc_sim):
                similarities.append(abs(mfcc_sim))
            
            # مقارنة الخصائص الأخرى
            centroid_diff = abs(features1["spectral_centroid"] - features2["spectral_centroid"])
            centroid_sim = 1 - min(centroid_diff / 1000.0, 1.0)  # تطبيع
            similarities.append(centroid_sim)
            
            zcr_diff = abs(features1["zero_crossing_rate"] - features2["zero_crossing_rate"])
            zcr_sim = 1 - min(zcr_diff, 1.0)
            similarities.append(zcr_sim)
            
            return np.mean(similarities) if similarities else 0.0
        
        except Exception as e:
            self.logger.error(f"خطأ في حساب تشابه الصوت: {e}")
            return 0.0
    
    async def multi_factor_authentication(self, user_id: str, 
                                        face_data: Optional[Union[np.ndarray, bytes]] = None,
                                        voice_data: Optional[bytes] = None) -> Dict[str, Any]:
        """مصادقة متعددة العوامل"""
        try:
            results = {}
            total_confidence = 0.0
            verification_count = 0
            
            # التحقق من بصمة الوجه
            if face_data is not None:
                face_result = await self.verify_face(user_id, face_data)
                results["face"] = face_result
                if face_result["success"]:
                    total_confidence += face_result["confidence_score"]
                    verification_count += 1
            
            # التحقق من بصمة الصوت
            if voice_data is not None:
                voice_result = await self.verify_voice(user_id, voice_data)
                results["voice"] = voice_result
                if voice_result["success"]:
                    total_confidence += voice_result["confidence_score"]
                    verification_count += 1
            
            # تقييم النتيجة الإجمالية
            if verification_count == 0:
                return {
                    "success": False,
                    "message": "لم يتم تقديم أي بيانات بيومترية للتحقق",
                    "results": results
                }
            
            average_confidence = total_confidence / verification_count
            
            # متطلبات المصادقة متعددة العوامل
            if self.security_settings["multi_factor_required"]:
                required_factors = 2
                if verification_count < required_factors:
                    return {
                        "success": False,
                        "message": f"يتطلب التحقق من {required_factors} عوامل بيومترية على الأقل",
                        "verified_factors": verification_count,
                        "results": results
                    }
            
            # نجح التحقق إذا كان متوسط الثقة مرتفع بما فيه الكفاية
            min_mfa_confidence = 0.7
            is_authenticated = (average_confidence >= min_mfa_confidence)
            
            if is_authenticated:
                await self._log_security_event("multi_factor_auth", user_id, True, average_confidence,
                                             {"verified_factors": verification_count, "results": results})
                
                return {
                    "success": True,
                    "message": "تم التحقق بنجاح من خلال المصادقة متعددة العوامل",
                    "average_confidence": average_confidence,
                    "verified_factors": verification_count,
                    "results": results
                }
            else:
                await self._log_security_event("multi_factor_auth", user_id, False, average_confidence,
                                             {"verified_factors": verification_count, "results": results})
                
                return {
                    "success": False,
                    "message": "فشل في المصادقة متعددة العوامل",
                    "average_confidence": average_confidence,
                    "verified_factors": verification_count,
                    "results": results
                }
        
        except Exception as e:
            self.logger.error(f"خطأ في المصادقة متعددة العوامل: {e}")
            return {
                "success": False,
                "message": f"خطأ في المصادقة متعددة العوامل: {str(e)}"
            }
    
    async def _is_user_locked(self, user_id: str) -> bool:
        """فحص ما إذا كان المستخدم محظور مؤقتاً"""
        if user_id not in self.locked_users:
            return False
        
        lockout_end = self.locked_users[user_id]
        if datetime.now() >= lockout_end:
            del self.locked_users[user_id]
            return False
        
        return True
    
    async def _record_failed_attempt(self, user_id: str, attempt_type: str):
        """تسجيل محاولة فاشلة"""
        try:
            conn = sqlite3.connect(self.security_db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO failed_attempts (user_id, timestamp, attempt_type, ip_address)
                VALUES (?, ?, ?, ?)
            """, (user_id, datetime.now().timestamp(), attempt_type, "127.0.0.1"))
            
            # فحص عدد المحاولات الفاشلة في آخر ساعة
            one_hour_ago = (datetime.now() - timedelta(hours=1)).timestamp()
            cursor.execute("""
                SELECT COUNT(*) FROM failed_attempts 
                WHERE user_id = ? AND timestamp > ?
            """, (user_id, one_hour_ago))
            
            failed_count = cursor.fetchone()[0]
            
            conn.commit()
            conn.close()
            
            self.security_stats["failed_attempts"] += 1
            
            # حظر المستخدم إذا تجاوز الحد المسموح
            if failed_count >= self.security_settings["max_failed_attempts"]:
                lockout_duration = timedelta(minutes=self.security_settings["lockout_duration_minutes"])
                self.locked_users[user_id] = datetime.now() + lockout_duration
                
                await self._log_security_event("user_locked", user_id, False, 0.0,
                                             {"failed_attempts": failed_count,
                                              "lockout_until": self.locked_users[user_id].isoformat()})
        
        except Exception as e:
            self.logger.error(f"خطأ في تسجيل المحاولة الفاشلة: {e}")
    
    async def _save_template_to_database(self, template: BiometricTemplate):
        """حفظ القالب في قاعدة البيانات"""
        try:
            conn = sqlite3.connect(self.security_db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO biometric_templates
                (template_id, user_id, modality, template_data, confidence_threshold,
                 created_at, last_used, usage_count, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                template.template_id,
                template.user_id,
                template.modality,
                template.template_data,
                template.confidence_threshold,
                template.created_at.timestamp(),
                template.last_used.timestamp(),
                template.usage_count,
                template.is_active
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"خطأ في حفظ القالب: {e}")
    
    async def _update_template_usage(self, template: BiometricTemplate):
        """تحديث استخدام القالب"""
        try:
            conn = sqlite3.connect(self.security_db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE biometric_templates 
                SET last_used = ?, usage_count = ?
                WHERE template_id = ?
            """, (template.last_used.timestamp(), template.usage_count, template.template_id))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"خطأ في تحديث استخدام القالب: {e}")
    
    async def _log_security_event(self, event_type: str, user_id: str, success: bool,
                                confidence_score: float, additional_data: Dict[str, Any]):
        """تسجيل حدث أمني"""
        try:
            event_id = f"sec_{datetime.now().timestamp()}_{event_type}"
            
            conn = sqlite3.connect(self.security_db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO security_events
                (event_id, event_type, user_id, timestamp, success, confidence_score, additional_data)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                event_id,
                event_type,
                user_id,
                datetime.now().timestamp(),
                success,
                confidence_score,
                json.dumps(additional_data, ensure_ascii=False)
            ))
            
            conn.commit()
            conn.close()
            
            self.security_stats["total_authentications"] += 1
            
            # تحديث متوسط نقاط الثقة
            current_avg = self.security_stats["average_confidence_score"]
            total_auths = self.security_stats["total_authentications"]
            new_avg = (current_avg * (total_auths - 1) + confidence_score) / total_auths
            self.security_stats["average_confidence_score"] = new_avg
            
        except Exception as e:
            self.logger.error(f"خطأ في تسجيل الحدث الأمني: {e}")
    
    def _load_biometric_templates(self):
        """تحميل القوالب البيومترية من قاعدة البيانات"""
        try:
            conn = sqlite3.connect(self.security_db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM biometric_templates WHERE is_active = 1")
            rows = cursor.fetchall()
            
            for row in rows:
                template = BiometricTemplate(
                    template_id=row[0],
                    user_id=row[1],
                    modality=row[2],
                    template_data=row[3],
                    confidence_threshold=row[4],
                    created_at=datetime.fromtimestamp(row[5]),
                    last_used=datetime.fromtimestamp(row[6]),
                    usage_count=row[7],
                    is_active=bool(row[8])
                )
                self.biometric_templates[template.template_id] = template
            
            conn.close()
            self.logger.info(f"تم تحميل {len(self.biometric_templates)} قالب بيومتري")
            
        except Exception as e:
            self.logger.error(f"خطأ في تحميل القوالب البيومترية: {e}")
    
    def _start_security_monitor(self):
        """بدء مراقب الأمان"""
        def security_monitor():
            while True:
                try:
                    # تنظيف المحاولات الفاشلة القديمة
                    asyncio.run(self._cleanup_old_failed_attempts())
                    
                    # فحص القوالب المنتهية الصلاحية
                    asyncio.run(self._check_template_expiry())
                    
                    # تحديث الإحصائيات
                    asyncio.run(self._update_security_statistics())
                    
                    # انتظار 10 دقائق
                    threading.Event().wait(600)
                    
                except Exception as e:
                    self.logger.error(f"خطأ في مراقب الأمان: {e}")
        
        monitor_thread = threading.Thread(target=security_monitor, daemon=True)
        monitor_thread.start()
    
    async def _cleanup_old_failed_attempts(self):
        """تنظيف المحاولات الفاشلة القديمة"""
        try:
            one_day_ago = (datetime.now() - timedelta(days=1)).timestamp()
            
            conn = sqlite3.connect(self.security_db_path)
            cursor = conn.cursor()
            
            cursor.execute("DELETE FROM failed_attempts WHERE timestamp < ?", (one_day_ago,))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"خطأ في تنظيف المحاولات الفاشلة: {e}")
    
    async def _check_template_expiry(self):
        """فحص انتهاء صلاحية القوالب"""
        try:
            expiry_days = self.security_settings["template_update_frequency_days"]
            expiry_date = datetime.now() - timedelta(days=expiry_days)
            
            expired_templates = [
                template for template in self.biometric_templates.values()
                if template.created_at < expiry_date
            ]
            
            for template in expired_templates:
                self.logger.warning(f"القالب {template.template_id} انتهت صلاحيته")
                # يمكن إضافة منطق للتنبيه أو إزالة القوالب المنتهية
                
        except Exception as e:
            self.logger.error(f"خطأ في فحص انتهاء صلاحية القوالب: {e}")
    
    async def _update_security_statistics(self):
        """تحديث الإحصائيات الأمنية"""
        try:
            conn = sqlite3.connect(self.security_db_path)
            cursor = conn.cursor()
            
            # إحصائيات اليوم
            today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
            
            cursor.execute("""
                SELECT COUNT(*) FROM security_events 
                WHERE timestamp > ? AND success = 1
            """, (today_start,))
            today_successful = cursor.fetchone()[0]
            
            cursor.execute("""
                SELECT COUNT(*) FROM security_events 
                WHERE timestamp > ?
            """, (today_start,))
            today_total = cursor.fetchone()[0]
            
            # تحديث الإحصائيات
            self.security_stats["today_successful_authentications"] = today_successful
            self.security_stats["today_total_authentications"] = today_total
            self.security_stats["today_success_rate"] = (
                today_successful / max(today_total, 1)
            )
            
            conn.close()
            
        except Exception as e:
            self.logger.error(f"خطأ في تحديث الإحصائيات الأمنية: {e}")
    
    async def get_security_status(self, user_id: str) -> Dict[str, Any]:
        """الحصول على حالة الأمان للمستخدم"""
        try:
            user_templates = [
                template for template in self.biometric_templates.values()
                if template.user_id == user_id and template.is_active
            ]
            
            template_summary = {}
            for template in user_templates:
                modality = template.modality
                if modality not in template_summary:
                    template_summary[modality] = []
                
                template_summary[modality].append({
                    "template_id": template.template_id,
                    "created_at": template.created_at.isoformat(),
                    "last_used": template.last_used.isoformat(),
                    "usage_count": template.usage_count
                })
            
            # فحص الحظر
            is_locked = await self._is_user_locked(user_id)
            locked_until = None
            if is_locked:
                locked_until = self.locked_users[user_id].isoformat()
            
            return {
                "user_id": user_id,
                "enrolled_modalities": list(template_summary.keys()),
                "template_count": len(user_templates),
                "templates": template_summary,
                "is_locked": is_locked,
                "locked_until": locked_until,
                "multi_factor_enabled": self.security_settings["multi_factor_required"],
                "security_level": self._calculate_security_level(user_templates)
            }
        
        except Exception as e:
            self.logger.error(f"خطأ في الحصول على حالة الأمان: {e}")
            return {"error": str(e)}
    
    def _calculate_security_level(self, templates: List[BiometricTemplate]) -> str:
        """حساب مستوى الأمان"""
        modality_count = len(set(template.modality for template in templates))
        
        if modality_count >= 3:
            return "عالي جداً"
        elif modality_count >= 2:
            return "عالي"
        elif modality_count >= 1:
            return "متوسط"
        else:
            return "منخفض"
    
    def get_security_statistics(self) -> Dict[str, Any]:
        """الحصول على الإحصائيات الأمنية"""
        return {
            "global_stats": self.security_stats,
            "total_templates": len(self.biometric_templates),
            "templates_by_modality": {
                modality: len([t for t in self.biometric_templates.values() if t.modality == modality])
                for modality in ["face", "voice", "fingerprint", "iris"]
            },
            "active_locked_users": len(self.locked_users),
            "security_settings": self.security_settings
        }

# إنشاء مثيل عام
biometric_security_engine = BiometricSecurityEngine()

def get_biometric_security_engine() -> BiometricSecurityEngine:
    """الحصول على محرك الأمان البيومتري"""
    return biometric_security_engine

if __name__ == "__main__":
    # اختبار النظام
    async def test_biometric_system():
        engine = get_biometric_security_engine()
        
        print("🔐 اختبار محرك الأمان البيومتري...")
        
        # محاكاة تسجيل بصمة الوجه
        # في التطبيق الحقيقي، ستأتي هذه البيانات من الكاميرا
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        enrollment_result = await engine.enroll_face("test_user", test_image)
        print(f"تسجيل بصمة الوجه: {enrollment_result}")
        
        # الحصول على حالة الأمان
        security_status = await engine.get_security_status("test_user")
        print(f"حالة الأمان: {security_status}")
        
        # الحصول على الإحصائيات
        stats = engine.get_security_statistics()
        print(f"الإحصائيات: {stats}")
    
    asyncio.run(test_biometric_system())
