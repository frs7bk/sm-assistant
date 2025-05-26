
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
نظام الأتمتة الذكية والتحكم في التطبيقات
يمكنه فتح التطبيقات، إدارة المهام، والتحكم في النظام صوتياً
"""

import asyncio
import logging
import subprocess
import platform
import psutil
import os
import time
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Tuple
from pathlib import Path
import threading
import queue
from dataclasses import dataclass
import pyautogui
import cv2
import numpy as np
from PIL import Image, ImageGrab
import requests
import webbrowser

@dataclass
class AutomationTask:
    """مهمة الأتمتة"""
    id: str
    name: str
    command: str
    parameters: Dict[str, Any]
    schedule: Optional[str]
    priority: int
    status: str
    created_at: datetime
    last_executed: Optional[datetime]
    success_count: int
    failure_count: int
    description: str

class ApplicationController:
    """تحكم في التطبيقات"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.system_type = platform.system().lower()
        
        # قاموس التطبيقات الشائعة
        self.applications = {
            "windows": {
                "فوتوشوب": ["photoshop.exe", "Photoshop.exe"],
                "إليستريتور": ["illustrator.exe", "Ai.exe"],
                "أفتر إفكتس": ["AfterFX.exe", "afterfx.exe"],
                "بريمير": ["Premiere Pro.exe", "premiere.exe"],
                "كروم": ["chrome.exe", "Google Chrome"],
                "فايرفوكس": ["firefox.exe", "Firefox"],
                "سبوتيفاي": ["Spotify.exe", "spotify.exe"],
                "ديسكورد": ["Discord.exe", "discord.exe"],
                "سلاك": ["slack.exe", "Slack.exe"],
                "زوم": ["Zoom.exe", "zoom.exe"],
                "تيمز": ["Teams.exe", "msteams.exe"],
                "أوتلوك": ["OUTLOOK.EXE", "outlook.exe"],
                "وورد": ["WINWORD.EXE", "winword.exe"],
                "إكسل": ["EXCEL.EXE", "excel.exe"],
                "باوربوينت": ["POWERPNT.EXE", "powerpnt.exe"],
                "نوتباد": ["notepad.exe", "Notepad"],
                "كود": ["Code.exe", "code.exe"],
                "بايتشارم": ["pycharm64.exe", "pycharm.exe"]
            },
            "darwin": {  # macOS
                "فوتوشوب": ["Adobe Photoshop 2024", "Photoshop"],
                "إليستريتور": ["Adobe Illustrator 2024", "Illustrator"],
                "أفتر إفكتس": ["Adobe After Effects 2024", "After Effects"],
                "بريمير": ["Adobe Premiere Pro 2024", "Premiere Pro"],
                "كروم": ["Google Chrome", "Chrome"],
                "فايرفوكس": ["Firefox", "Firefox"],
                "سبوتيفاي": ["Spotify", "Spotify"],
                "ديسكورد": ["Discord", "Discord"],
                "سلاك": ["Slack", "Slack"],
                "زوم": ["zoom.us", "Zoom"],
                "تيمز": ["Microsoft Teams", "Teams"],
                "أوتلوك": ["Microsoft Outlook", "Outlook"],
                "وورد": ["Microsoft Word", "Word"],
                "إكسل": ["Microsoft Excel", "Excel"],
                "باوربوينت": ["Microsoft PowerPoint", "PowerPoint"],
                "كود": ["Visual Studio Code", "Code"],
                "بايتشارم": ["PyCharm", "PyCharm"]
            },
            "linux": {
                "فوتوشوب": ["gimp", "krita"],  # بدائل على لينكس
                "إليستريتور": ["inkscape", "Inkscape"],
                "كروم": ["google-chrome", "chromium-browser"],
                "فايرفوكس": ["firefox", "Firefox"],
                "سبوتيفاي": ["spotify", "Spotify"],
                "ديسكورد": ["discord", "Discord"],
                "سلاك": ["slack", "Slack"],
                "زوم": ["zoom", "Zoom"],
                "كود": ["code", "Code"],
                "بايتشارم": ["pycharm", "PyCharm"]
            }
        }
        
        # الحصول على قائمة التطبيقات للنظام الحالي
        self.system_apps = self.applications.get(self.system_type, {})
    
    async def open_application(self, app_name: str) -> Dict[str, Any]:
        """فتح تطبيق"""
        try:
            app_name_lower = app_name.lower()
            app_executables = None
            
            # البحث عن التطبيق
            for key, executables in self.system_apps.items():
                if key in app_name_lower or app_name_lower in key:
                    app_executables = executables
                    break
            
            if not app_executables:
                return {
                    "success": False,
                    "message": f"التطبيق '{app_name}' غير موجود في القائمة",
                    "available_apps": list(self.system_apps.keys())
                }
            
            # محاولة فتح التطبيق
            for executable in app_executables:
                try:
                    if self.system_type == "windows":
                        # محاولة تشغيل التطبيق مباشرة
                        subprocess.Popen(executable, shell=True)
                        await asyncio.sleep(2)  # انتظار لتشغيل التطبيق
                        
                        # التحقق من تشغيل التطبيق
                        if self._is_process_running(executable):
                            return {
                                "success": True,
                                "message": f"تم فتح {app_name} بنجاح",
                                "executable": executable
                            }
                    
                    elif self.system_type == "darwin":  # macOS
                        subprocess.run(["open", "-a", executable], check=True)
                        return {
                            "success": True,
                            "message": f"تم فتح {app_name} بنجاح",
                            "executable": executable
                        }
                    
                    elif self.system_type == "linux":
                        subprocess.Popen([executable], stdout=subprocess.DEVNULL, 
                                       stderr=subprocess.DEVNULL)
                        return {
                            "success": True,
                            "message": f"تم فتح {app_name} بنجاح",
                            "executable": executable
                        }
                
                except Exception as e:
                    self.logger.debug(f"فشل في تشغيل {executable}: {e}")
                    continue
            
            return {
                "success": False,
                "message": f"فشل في فتح {app_name}. قد يكون غير مثبت.",
                "tried_executables": app_executables
            }
        
        except Exception as e:
            self.logger.error(f"خطأ في فتح التطبيق {app_name}: {e}")
            return {
                "success": False,
                "message": f"خطأ في فتح التطبيق: {str(e)}"
            }
    
    async def close_application(self, app_name: str) -> Dict[str, Any]:
        """إغلاق تطبيق"""
        try:
            closed_processes = []
            app_name_lower = app_name.lower()
            
            # البحث عن العمليات الجارية
            for proc in psutil.process_iter(['pid', 'name', 'exe']):
                try:
                    proc_name = proc.info['name'].lower() if proc.info['name'] else ""
                    
                    # البحث عن تطابق في اسم العملية
                    if (app_name_lower in proc_name or 
                        any(app_name_lower in exe.lower() for exe in self.system_apps.get(app_name_lower, []))):
                        
                        proc.terminate()
                        closed_processes.append(proc.info['name'])
                        
                        # انتظار الإغلاق السلمي
                        try:
                            proc.wait(timeout=5)
                        except psutil.TimeoutExpired:
                            proc.kill()  # إجبار الإغلاق
                
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            if closed_processes:
                return {
                    "success": True,
                    "message": f"تم إغلاق التطبيقات: {', '.join(closed_processes)}",
                    "closed_processes": closed_processes
                }
            else:
                return {
                    "success": False,
                    "message": f"لم يتم العثور على {app_name} في العمليات الجارية"
                }
        
        except Exception as e:
            self.logger.error(f"خطأ في إغلاق التطبيق {app_name}: {e}")
            return {
                "success": False,
                "message": f"خطأ في إغلاق التطبيق: {str(e)}"
            }
    
    def _is_process_running(self, process_name: str) -> bool:
        """التحقق من تشغيل عملية"""
        try:
            for proc in psutil.process_iter(['name']):
                if proc.info['name'] and process_name.lower() in proc.info['name'].lower():
                    return True
            return False
        except Exception:
            return False
    
    def get_running_applications(self) -> List[Dict[str, Any]]:
        """الحصول على قائمة التطبيقات الجارية"""
        running_apps = []
        
        try:
            for proc in psutil.process_iter(['pid', 'name', 'memory_info', 'cpu_percent']):
                try:
                    proc_info = proc.info
                    if proc_info['name'] and not proc_info['name'].startswith('System'):
                        running_apps.append({
                            "pid": proc_info['pid'],
                            "name": proc_info['name'],
                            "memory_mb": proc_info['memory_info'].rss / (1024 * 1024),
                            "cpu_percent": proc_info['cpu_percent']
                        })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        
        except Exception as e:
            self.logger.error(f"خطأ في الحصول على التطبيقات الجارية: {e}")
        
        return sorted(running_apps, key=lambda x: x['memory_mb'], reverse=True)

class SystemController:
    """تحكم في النظام"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.system_type = platform.system().lower()
    
    async def adjust_volume(self, action: str, amount: int = 10) -> Dict[str, Any]:
        """تعديل مستوى الصوت"""
        try:
            if self.system_type == "windows":
                if action == "up":
                    subprocess.run(["powershell", "-c", 
                                  f"[audio]::Volume += {amount/100}"], check=True)
                elif action == "down":
                    subprocess.run(["powershell", "-c", 
                                  f"[audio]::Volume -= {amount/100}"], check=True)
                elif action == "mute":
                    subprocess.run(["powershell", "-c", 
                                  "[audio]::Mute = !$([audio]::Mute)"], check=True)
            
            elif self.system_type == "darwin":  # macOS
                if action == "up":
                    subprocess.run(["osascript", "-e", 
                                  f"set volume output volume (output volume of (get volume settings) + {amount})"])
                elif action == "down":
                    subprocess.run(["osascript", "-e", 
                                  f"set volume output volume (output volume of (get volume settings) - {amount})"])
                elif action == "mute":
                    subprocess.run(["osascript", "-e", "set volume with output muted"])
            
            elif self.system_type == "linux":
                if action == "up":
                    subprocess.run(["amixer", "sset", "Master", f"{amount}%+"])
                elif action == "down":
                    subprocess.run(["amixer", "sset", "Master", f"{amount}%-"])
                elif action == "mute":
                    subprocess.run(["amixer", "sset", "Master", "toggle"])
            
            return {
                "success": True,
                "message": f"تم {action} الصوت بمقدار {amount}%"
            }
        
        except Exception as e:
            self.logger.error(f"خطأ في تعديل الصوت: {e}")
            return {
                "success": False,
                "message": f"خطأ في تعديل الصوت: {str(e)}"
            }
    
    async def take_screenshot(self, save_path: Optional[str] = None) -> Dict[str, Any]:
        """أخذ لقطة شاشة"""
        try:
            # تحديد مسار الحفظ
            if not save_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                screenshots_dir = Path("data/screenshots")
                screenshots_dir.mkdir(parents=True, exist_ok=True)
                save_path = screenshots_dir / f"screenshot_{timestamp}.png"
            
            # أخذ لقطة الشاشة
            screenshot = ImageGrab.grab()
            screenshot.save(save_path)
            
            return {
                "success": True,
                "message": "تم أخذ لقطة الشاشة بنجاح",
                "path": str(save_path),
                "size": screenshot.size
            }
        
        except Exception as e:
            self.logger.error(f"خطأ في أخذ لقطة الشاشة: {e}")
            return {
                "success": False,
                "message": f"خطأ في أخذ لقطة الشاشة: {str(e)}"
            }
    
    async def open_folder(self, folder_path: str) -> Dict[str, Any]:
        """فتح مجلد"""
        try:
            path = Path(folder_path)
            
            # التحقق من وجود المجلد
            if not path.exists():
                return {
                    "success": False,
                    "message": f"المجلد غير موجود: {folder_path}"
                }
            
            if self.system_type == "windows":
                subprocess.run(["explorer", str(path)], check=True)
            elif self.system_type == "darwin":  # macOS
                subprocess.run(["open", str(path)], check=True)
            elif self.system_type == "linux":
                subprocess.run(["xdg-open", str(path)], check=True)
            
            return {
                "success": True,
                "message": f"تم فتح المجلد: {folder_path}"
            }
        
        except Exception as e:
            self.logger.error(f"خطأ في فتح المجلد: {e}")
            return {
                "success": False,
                "message": f"خطأ في فتح المجلد: {str(e)}"
            }
    
    async def get_system_info(self) -> Dict[str, Any]:
        """الحصول على معلومات النظام"""
        try:
            return {
                "platform": platform.platform(),
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "cpu_count": psutil.cpu_count(),
                "memory_total": psutil.virtual_memory().total,
                "memory_available": psutil.virtual_memory().available,
                "disk_usage": {
                    "total": psutil.disk_usage('/').total if self.system_type != "windows" else psutil.disk_usage('C:').total,
                    "used": psutil.disk_usage('/').used if self.system_type != "windows" else psutil.disk_usage('C:').used,
                    "free": psutil.disk_usage('/').free if self.system_type != "windows" else psutil.disk_usage('C:').free
                }
            }
        
        except Exception as e:
            self.logger.error(f"خطأ في الحصول على معلومات النظام: {e}")
            return {"error": str(e)}

class WebController:
    """تحكم في المتصفح والويب"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def open_url(self, url: str) -> Dict[str, Any]:
        """فتح رابط في المتصفح"""
        try:
            # التأكد من وجود البروتوكول
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            webbrowser.open(url)
            
            return {
                "success": True,
                "message": f"تم فتح الرابط: {url}"
            }
        
        except Exception as e:
            self.logger.error(f"خطأ في فتح الرابط: {e}")
            return {
                "success": False,
                "message": f"خطأ في فتح الرابط: {str(e)}"
            }
    
    async def search_google(self, query: str) -> Dict[str, Any]:
        """البحث في جوجل"""
        try:
            search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
            webbrowser.open(search_url)
            
            return {
                "success": True,
                "message": f"تم البحث عن: {query}",
                "url": search_url
            }
        
        except Exception as e:
            self.logger.error(f"خطأ في البحث: {e}")
            return {
                "success": False,
                "message": f"خطأ في البحث: {str(e)}"
            }

class SmartTaskAutomation:
    """نظام الأتمتة الذكية للمهام"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # المتحكمات الفرعية
        self.app_controller = ApplicationController()
        self.system_controller = SystemController()
        self.web_controller = WebController()
        
        # قاعدة بيانات المهام
        self.tasks_db_path = Path("data/automation/tasks.db")
        self.tasks_db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_tasks_database()
        
        # قائمة انتظار المهام
        self.task_queue = queue.Queue()
        self.automation_worker = None
        
        # الأوامر المدعومة
        self.supported_commands = {
            # تحكم في التطبيقات
            "open_app": self.app_controller.open_application,
            "close_app": self.app_controller.close_application,
            
            # تحكم في النظام
            "volume_up": lambda: self.system_controller.adjust_volume("up"),
            "volume_down": lambda: self.system_controller.adjust_volume("down"),
            "mute": lambda: self.system_controller.adjust_volume("mute"),
            "screenshot": self.system_controller.take_screenshot,
            "open_folder": self.system_controller.open_folder,
            
            # تحكم في الويب
            "open_url": self.web_controller.open_url,
            "search_google": self.web_controller.search_google,
            
            # أوامر مركبة
            "open_work_apps": self._open_work_applications,
            "close_all_browsers": self._close_all_browsers,
            "system_cleanup": self._perform_system_cleanup
        }
        
        # إحصائيات الأتمتة
        self.automation_stats = {
            "total_tasks_executed": 0,
            "successful_tasks": 0,
            "failed_tasks": 0,
            "most_used_commands": {},
            "average_execution_time": 0.0
        }
        
        self._start_automation_worker()
    
    def _init_tasks_database(self):
        """تهيئة قاعدة بيانات المهام"""
        conn = sqlite3.connect(self.tasks_db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS automation_tasks (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                command TEXT NOT NULL,
                parameters TEXT,
                schedule TEXT,
                priority INTEGER DEFAULT 5,
                status TEXT DEFAULT 'pending',
                created_at REAL,
                last_executed REAL,
                success_count INTEGER DEFAULT 0,
                failure_count INTEGER DEFAULT 0,
                description TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    async def execute_command(self, command: str, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """تنفيذ أمر"""
        start_time = time.time()
        
        try:
            if command not in self.supported_commands:
                return {
                    "success": False,
                    "message": f"الأمر '{command}' غير مدعوم",
                    "supported_commands": list(self.supported_commands.keys())
                }
            
            # تنفيذ الأمر
            command_func = self.supported_commands[command]
            
            if parameters:
                result = await command_func(**parameters)
            else:
                result = await command_func()
            
            # تحديث الإحصائيات
            execution_time = time.time() - start_time
            self._update_stats(command, True, execution_time)
            
            result["execution_time"] = execution_time
            return result
        
        except Exception as e:
            execution_time = time.time() - start_time
            self._update_stats(command, False, execution_time)
            
            self.logger.error(f"خطأ في تنفيذ الأمر {command}: {e}")
            return {
                "success": False,
                "message": f"خطأ في تنفيذ الأمر: {str(e)}",
                "execution_time": execution_time
            }
    
    async def process_voice_command(self, voice_input: str) -> Dict[str, Any]:
        """معالجة الأوامر الصوتية"""
        try:
            # تحليل الأمر الصوتي
            parsed_command = self._parse_voice_command(voice_input)
            
            if not parsed_command:
                return {
                    "success": False,
                    "message": "لم أستطع فهم الأمر",
                    "suggestions": [
                        "فتح فوتوشوب",
                        "أرفع الصوت",
                        "خذ لقطة شاشة",
                        "ابحث في جوجل عن الطقس"
                    ]
                }
            
            # تنفيذ الأمر
            result = await self.execute_command(
                parsed_command["command"],
                parsed_command["parameters"]
            )
            
            return result
        
        except Exception as e:
            self.logger.error(f"خطأ في معالجة الأمر الصوتي: {e}")
            return {
                "success": False,
                "message": f"خطأ في معالجة الأمر الصوتي: {str(e)}"
            }
    
    def _parse_voice_command(self, voice_input: str) -> Optional[Dict[str, Any]]:
        """تحليل الأمر الصوتي"""
        voice_lower = voice_input.lower()
        
        # أوامر فتح التطبيقات
        if "فتح" in voice_lower or "شغل" in voice_lower:
            app_keywords = {
                "فوتوشوب": "فوتوشوب",
                "إليستريتور": "إليستريتور",
                "أفتر إفكتس": "أفتر إفكتس",
                "بريمير": "بريمير",
                "كروم": "كروم",
                "فايرفوكس": "فايرفوكس",
                "سبوتيفاي": "سبوتيفاي",
                "ديسكورد": "ديسكورد",
                "كود": "كود"
            }
            
            for keyword, app_name in app_keywords.items():
                if keyword in voice_lower:
                    return {
                        "command": "open_app",
                        "parameters": {"app_name": app_name}
                    }
        
        # أوامر إغلاق التطبيقات
        elif "أغلق" in voice_lower or "أقفل" in voice_lower:
            app_keywords = {
                "فوتوشوب": "فوتوشوب",
                "إليستريتور": "إليستريتور",
                "كروم": "كروم",
                "المتصفح": "كروم"
            }
            
            for keyword, app_name in app_keywords.items():
                if keyword in voice_lower:
                    return {
                        "command": "close_app",
                        "parameters": {"app_name": app_name}
                    }
        
        # أوامر الصوت
        elif "أرفع الصوت" in voice_lower or "زود الصوت" in voice_lower:
            return {"command": "volume_up", "parameters": {}}
        
        elif "أخفض الصوت" in voice_lower or "قلل الصوت" in voice_lower:
            return {"command": "volume_down", "parameters": {}}
        
        elif "اكتم الصوت" in voice_lower or "اقفل الصوت" in voice_lower:
            return {"command": "mute", "parameters": {}}
        
        # أوامر لقطة الشاشة
        elif "خذ لقطة" in voice_lower or "صور الشاشة" in voice_lower:
            return {"command": "screenshot", "parameters": {}}
        
        # أوامر البحث
        elif "ابحث" in voice_lower and "جوجل" in voice_lower:
            # استخراج محتوى البحث
            search_terms = ["ابحث في جوجل عن", "ابحث عن", "بحث عن"]
            search_query = voice_input
            
            for term in search_terms:
                if term in voice_input:
                    search_query = voice_input.split(term, 1)[1].strip()
                    break
            
            return {
                "command": "search_google",
                "parameters": {"query": search_query}
            }
        
        # أوامر فتح المواقع
        elif "افتح موقع" in voice_lower or "اذهب إلى" in voice_lower:
            url_indicators = ["افتح موقع", "اذهب إلى", "فتح"]
            url = voice_input
            
            for indicator in url_indicators:
                if indicator in voice_input:
                    url = voice_input.split(indicator, 1)[1].strip()
                    break
            
            return {
                "command": "open_url",
                "parameters": {"url": url}
            }
        
        return None
    
    async def _open_work_applications(self) -> Dict[str, Any]:
        """فتح تطبيقات العمل"""
        work_apps = ["كروم", "كود", "سلاك", "ديسكورد"]
        results = []
        
        for app in work_apps:
            result = await self.app_controller.open_application(app)
            results.append(f"{app}: {'نجح' if result['success'] else 'فشل'}")
            await asyncio.sleep(1)  # تأخير بين التطبيقات
        
        return {
            "success": True,
            "message": "تم فتح تطبيقات العمل",
            "details": results
        }
    
    async def _close_all_browsers(self) -> Dict[str, Any]:
        """إغلاق جميع المتصفحات"""
        browsers = ["كروم", "فايرفوكس", "إيدج"]
        results = []
        
        for browser in browsers:
            result = await self.app_controller.close_application(browser)
            if result["success"]:
                results.append(f"تم إغلاق {browser}")
        
        return {
            "success": True,
            "message": "تم إغلاق المتصفحات",
            "details": results
        }
    
    async def _perform_system_cleanup(self) -> Dict[str, Any]:
        """تنظيف النظام"""
        try:
            # أخذ لقطة شاشة قبل التنظيف
            screenshot_result = await self.system_controller.take_screenshot()
            
            # الحصول على معلومات النظام
            system_info = await self.system_controller.get_system_info()
            
            # تنظيف الملفات المؤقتة (مبسط)
            temp_dirs = []
            if platform.system() == "Windows":
                temp_dirs = [os.environ.get("TEMP", ""), "C:/Windows/Temp"]
            
            cleaned_files = 0
            for temp_dir in temp_dirs:
                if os.path.exists(temp_dir):
                    for file in os.listdir(temp_dir):
                        try:
                            file_path = os.path.join(temp_dir, file)
                            if os.path.isfile(file_path):
                                os.remove(file_path)
                                cleaned_files += 1
                        except Exception:
                            continue
            
            return {
                "success": True,
                "message": f"تم تنظيف النظام - حذف {cleaned_files} ملف مؤقت",
                "screenshot_taken": screenshot_result["success"],
                "system_info": system_info
            }
        
        except Exception as e:
            return {
                "success": False,
                "message": f"خطأ في تنظيف النظام: {str(e)}"
            }
    
    def _update_stats(self, command: str, success: bool, execution_time: float):
        """تحديث إحصائيات الأتمتة"""
        self.automation_stats["total_tasks_executed"] += 1
        
        if success:
            self.automation_stats["successful_tasks"] += 1
        else:
            self.automation_stats["failed_tasks"] += 1
        
        # تحديث الأوامر الأكثر استخداماً
        if command not in self.automation_stats["most_used_commands"]:
            self.automation_stats["most_used_commands"][command] = 0
        self.automation_stats["most_used_commands"][command] += 1
        
        # تحديث متوسط وقت التنفيذ
        total_tasks = self.automation_stats["total_tasks_executed"]
        current_avg = self.automation_stats["average_execution_time"]
        new_avg = (current_avg * (total_tasks - 1) + execution_time) / total_tasks
        self.automation_stats["average_execution_time"] = new_avg
    
    def _start_automation_worker(self):
        """بدء عامل الأتمتة في الخلفية"""
        def automation_worker():
            while True:
                try:
                    task = self.task_queue.get(timeout=1)
                    if task is None:
                        break
                    
                    # تنفيذ المهمة
                    asyncio.run(self._execute_scheduled_task(task))
                    self.task_queue.task_done()
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    self.logger.error(f"خطأ في عامل الأتمتة: {e}")
        
        self.automation_worker = threading.Thread(target=automation_worker, daemon=True)
        self.automation_worker.start()
    
    async def _execute_scheduled_task(self, task: Dict[str, Any]):
        """تنفيذ مهمة مجدولة"""
        try:
            command = task["command"]
            parameters = json.loads(task.get("parameters", "{}"))
            
            result = await self.execute_command(command, parameters)
            
            # تحديث إحصائيات المهمة في قاعدة البيانات
            conn = sqlite3.connect(self.tasks_db_path)
            cursor = conn.cursor()
            
            if result["success"]:
                cursor.execute("""
                    UPDATE automation_tasks 
                    SET success_count = success_count + 1, last_executed = ?
                    WHERE id = ?
                """, (datetime.now().timestamp(), task["id"]))
            else:
                cursor.execute("""
                    UPDATE automation_tasks 
                    SET failure_count = failure_count + 1, last_executed = ?
                    WHERE id = ?
                """, (datetime.now().timestamp(), task["id"]))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"خطأ في تنفيذ المهمة المجدولة: {e}")
    
    def get_automation_stats(self) -> Dict[str, Any]:
        """الحصول على إحصائيات الأتمتة"""
        return self.automation_stats.copy()
    
    def get_running_applications(self) -> List[Dict[str, Any]]:
        """الحصول على التطبيقات الجارية"""
        return self.app_controller.get_running_applications()

# إنشاء مثيل عام
smart_automation = SmartTaskAutomation()

def get_smart_automation() -> SmartTaskAutomation:
    """الحصول على نظام الأتمتة الذكية"""
    return smart_automation

if __name__ == "__main__":
    # اختبار النظام
    async def test_automation():
        automation = get_smart_automation()
        
        # اختبار فتح تطبيق
        result = await automation.execute_command("open_app", {"app_name": "كروم"})
        print(f"فتح كروم: {result}")
        
        # اختبار أمر صوتي
        voice_result = await automation.process_voice_command("فتح فوتوشوب")
        print(f"أمر صوتي: {voice_result}")
        
        # اختبار لقطة شاشة
        screenshot_result = await automation.execute_command("screenshot")
        print(f"لقطة شاشة: {screenshot_result}")
        
        # عرض الإحصائيات
        stats = automation.get_automation_stats()
        print(f"الإحصائيات: {stats}")
    
    asyncio.run(test_automation())
