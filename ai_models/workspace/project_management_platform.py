
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
منصة إدارة المشاريع الذكية
Smart Project Management Platform
"""

import asyncio
import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
import sqlite3
from enum import Enum
import uuid

class TaskStatus(Enum):
    """حالات المهام"""
    TODO = "todo"
    IN_PROGRESS = "in_progress"
    REVIEW = "review"
    DONE = "done"
    BLOCKED = "blocked"

class Priority(Enum):
    """مستويات الأولوية"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5

class ProjectStatus(Enum):
    """حالات المشاريع"""
    PLANNING = "planning"
    ACTIVE = "active"
    ON_HOLD = "on_hold"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

@dataclass
class Task:
    """مهمة في المشروع"""
    task_id: str
    title: str
    description: str
    status: TaskStatus
    priority: Priority
    assigned_to: Optional[str]
    project_id: str
    created_at: datetime
    due_date: Optional[datetime]
    estimated_hours: float
    actual_hours: float = 0.0
    progress: float = 0.0
    dependencies: List[str] = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.tags is None:
            self.tags = []

@dataclass
class Project:
    """مشروع"""
    project_id: str
    name: str
    description: str
    status: ProjectStatus
    owner: str
    team_members: List[str]
    created_at: datetime
    start_date: Optional[datetime]
    end_date: Optional[datetime]
    budget: float = 0.0
    progress: float = 0.0
    milestones: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.milestones is None:
            self.milestones = []

@dataclass
class TeamMember:
    """عضو الفريق"""
    member_id: str
    name: str
    email: str
    role: str
    skills: List[str]
    availability: Dict[str, float]  # اليوم -> ساعات متاحة
    workload: float = 0.0

class ProjectManagementPlatform:
    """منصة إدارة المشاريع الذكية"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.db_path = Path("data/project_management.db")
        self.db_path.parent.mkdir(exist_ok=True)
        
        # البيانات
        self.projects: Dict[str, Project] = {}
        self.tasks: Dict[str, Task] = {}
        self.team_members: Dict[str, TeamMember] = {}
        
        # محركات الذكاء الاصطناعي
        self.ai_engines = {
            "task_optimizer": self._optimize_task_allocation,
            "timeline_predictor": self._predict_project_timeline,
            "resource_balancer": self._balance_team_workload,
            "risk_analyzer": self._analyze_project_risks,
            "productivity_tracker": self._track_team_productivity
        }

    async def initialize(self):
        """تهيئة منصة إدارة المشاريع"""
        try:
            self.logger.info("📋 تهيئة منصة إدارة المشاريع الذكية...")
            
            await self._initialize_database()
            await self._load_existing_data()
            await self._setup_ai_engines()
            
            self.logger.info("✅ تم تهيئة منصة إدارة المشاريع")
            
        except Exception as e:
            self.logger.error(f"❌ فشل تهيئة منصة إدارة المشاريع: {e}")

    async def _initialize_database(self):
        """تهيئة قاعدة البيانات"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # جدول المشاريع
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS projects (
                project_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                status TEXT NOT NULL,
                owner TEXT NOT NULL,
                team_members TEXT,
                created_at TEXT NOT NULL,
                start_date TEXT,
                end_date TEXT,
                budget REAL DEFAULT 0,
                progress REAL DEFAULT 0,
                milestones TEXT
            )
        """)
        
        # جدول المهام
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                description TEXT,
                status TEXT NOT NULL,
                priority INTEGER NOT NULL,
                assigned_to TEXT,
                project_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                due_date TEXT,
                estimated_hours REAL,
                actual_hours REAL DEFAULT 0,
                progress REAL DEFAULT 0,
                dependencies TEXT,
                tags TEXT,
                FOREIGN KEY (project_id) REFERENCES projects (project_id)
            )
        """)
        
        # جدول أعضاء الفريق
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS team_members (
                member_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                role TEXT,
                skills TEXT,
                availability TEXT,
                workload REAL DEFAULT 0
            )
        """)
        
        # جدول تتبع الوقت
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS time_tracking (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL,
                member_id TEXT NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT,
                hours_worked REAL,
                notes TEXT,
                FOREIGN KEY (task_id) REFERENCES tasks (task_id),
                FOREIGN KEY (member_id) REFERENCES team_members (member_id)
            )
        """)
        
        # جدول التقارير
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS reports (
                report_id TEXT PRIMARY KEY,
                report_type TEXT NOT NULL,
                project_id TEXT,
                generated_at TEXT NOT NULL,
                data TEXT NOT NULL,
                insights TEXT
            )
        """)
        
        conn.commit()
        conn.close()

    async def _load_existing_data(self):
        """تحميل البيانات الموجودة"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # تحميل المشاريع
        cursor.execute("SELECT * FROM projects")
        for row in cursor.fetchall():
            project = Project(
                project_id=row[0],
                name=row[1],
                description=row[2],
                status=ProjectStatus(row[3]),
                owner=row[4],
                team_members=json.loads(row[5]) if row[5] else [],
                created_at=datetime.fromisoformat(row[6]),
                start_date=datetime.fromisoformat(row[7]) if row[7] else None,
                end_date=datetime.fromisoformat(row[8]) if row[8] else None,
                budget=row[9],
                progress=row[10],
                milestones=json.loads(row[11]) if row[11] else []
            )
            self.projects[project.project_id] = project
        
        # تحميل المهام
        cursor.execute("SELECT * FROM tasks")
        for row in cursor.fetchall():
            task = Task(
                task_id=row[0],
                title=row[1],
                description=row[2],
                status=TaskStatus(row[3]),
                priority=Priority(row[4]),
                assigned_to=row[5],
                project_id=row[6],
                created_at=datetime.fromisoformat(row[7]),
                due_date=datetime.fromisoformat(row[8]) if row[8] else None,
                estimated_hours=row[9],
                actual_hours=row[10],
                progress=row[11],
                dependencies=json.loads(row[12]) if row[12] else [],
                tags=json.loads(row[13]) if row[13] else []
            )
            self.tasks[task.task_id] = task
        
        # تحميل أعضاء الفريق
        cursor.execute("SELECT * FROM team_members")
        for row in cursor.fetchall():
            member = TeamMember(
                member_id=row[0],
                name=row[1],
                email=row[2],
                role=row[3],
                skills=json.loads(row[4]) if row[4] else [],
                availability=json.loads(row[5]) if row[5] else {},
                workload=row[6]
            )
            self.team_members[member.member_id] = member
        
        conn.close()

    async def _setup_ai_engines(self):
        """إعداد محركات الذكاء الاصطناعي"""
        # إعداد المحركات المختلفة
        self.logger.info("🤖 إعداد محركات الذكاء الاصطناعي للمشاريع")

    async def create_project(
        self,
        name: str,
        description: str,
        owner: str,
        team_members: List[str] = None,
        budget: float = 0.0,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> str:
        """إنشاء مشروع جديد"""
        try:
            project_id = str(uuid.uuid4())
            
            project = Project(
                project_id=project_id,
                name=name,
                description=description,
                status=ProjectStatus.PLANNING,
                owner=owner,
                team_members=team_members or [],
                created_at=datetime.now(),
                start_date=start_date,
                end_date=end_date,
                budget=budget
            )
            
            self.projects[project_id] = project
            
            # حفظ في قاعدة البيانات
            await self._save_project(project)
            
            self.logger.info(f"📋 تم إنشاء مشروع جديد: {name}")
            return project_id
            
        except Exception as e:
            self.logger.error(f"خطأ في إنشاء المشروع: {e}")
            raise

    async def create_task(
        self,
        title: str,
        description: str,
        project_id: str,
        priority: Priority = Priority.MEDIUM,
        assigned_to: Optional[str] = None,
        due_date: Optional[datetime] = None,
        estimated_hours: float = 1.0,
        dependencies: List[str] = None,
        tags: List[str] = None
    ) -> str:
        """إنشاء مهمة جديدة"""
        try:
            if project_id not in self.projects:
                raise ValueError(f"المشروع {project_id} غير موجود")
            
            task_id = str(uuid.uuid4())
            
            task = Task(
                task_id=task_id,
                title=title,
                description=description,
                status=TaskStatus.TODO,
                priority=priority,
                assigned_to=assigned_to,
                project_id=project_id,
                created_at=datetime.now(),
                due_date=due_date,
                estimated_hours=estimated_hours,
                dependencies=dependencies or [],
                tags=tags or []
            )
            
            self.tasks[task_id] = task
            
            # حفظ في قاعدة البيانات
            await self._save_task(task)
            
            # تحديث الأعباء إذا تم تعيين المهمة
            if assigned_to:
                await self._update_member_workload(assigned_to)
            
            self.logger.info(f"✅ تم إنشاء مهمة جديدة: {title}")
            return task_id
            
        except Exception as e:
            self.logger.error(f"خطأ في إنشاء المهمة: {e}")
            raise

    async def add_team_member(
        self,
        name: str,
        email: str,
        role: str,
        skills: List[str] = None
    ) -> str:
        """إضافة عضو فريق جديد"""
        try:
            member_id = str(uuid.uuid4())
            
            member = TeamMember(
                member_id=member_id,
                name=name,
                email=email,
                role=role,
                skills=skills or [],
                availability={
                    "monday": 8.0,
                    "tuesday": 8.0,
                    "wednesday": 8.0,
                    "thursday": 8.0,
                    "friday": 8.0,
                    "saturday": 0.0,
                    "sunday": 0.0
                }
            )
            
            self.team_members[member_id] = member
            
            # حفظ في قاعدة البيانات
            await self._save_team_member(member)
            
            self.logger.info(f"👤 تم إضافة عضو فريق جديد: {name}")
            return member_id
            
        except Exception as e:
            self.logger.error(f"خطأ في إضافة عضو الفريق: {e}")
            raise

    async def update_task_status(self, task_id: str, new_status: TaskStatus) -> bool:
        """تحديث حالة المهمة"""
        try:
            if task_id not in self.tasks:
                return False
            
            task = self.tasks[task_id]
            old_status = task.status
            task.status = new_status
            
            # تحديث التقدم بناءً على الحالة
            if new_status == TaskStatus.DONE:
                task.progress = 100.0
            elif new_status == TaskStatus.IN_PROGRESS:
                if task.progress == 0.0:
                    task.progress = 10.0
            
            # حفظ التحديث
            await self._save_task(task)
            
            # تحديث تقدم المشروع
            await self._update_project_progress(task.project_id)
            
            self.logger.info(f"🔄 تم تحديث حالة المهمة {task.title} من {old_status.value} إلى {new_status.value}")
            return True
            
        except Exception as e:
            self.logger.error(f"خطأ في تحديث حالة المهمة: {e}")
            return False

    async def assign_task(self, task_id: str, member_id: str) -> bool:
        """تعيين مهمة لعضو في الفريق"""
        try:
            if task_id not in self.tasks or member_id not in self.team_members:
                return False
            
            task = self.tasks[task_id]
            old_assignee = task.assigned_to
            task.assigned_to = member_id
            
            # حفظ التحديث
            await self._save_task(task)
            
            # تحديث أعباء العمل
            if old_assignee:
                await self._update_member_workload(old_assignee)
            await self._update_member_workload(member_id)
            
            member_name = self.team_members[member_id].name
            self.logger.info(f"📋 تم تعيين المهمة {task.title} إلى {member_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"خطأ في تعيين المهمة: {e}")
            return False

    async def _optimize_task_allocation(self) -> Dict[str, Any]:
        """تحسين توزيع المهام باستخدام الذكاء الاصطناعي"""
        recommendations = []
        
        # البحث عن المهام غير المعينة
        unassigned_tasks = [task for task in self.tasks.values() if not task.assigned_to]
        
        for task in unassigned_tasks:
            # العثور على أفضل عضو فريق للمهمة
            best_member = await self._find_best_assignee(task)
            
            if best_member:
                recommendations.append({
                    "task_id": task.task_id,
                    "task_title": task.title,
                    "recommended_assignee": best_member["member_id"],
                    "assignee_name": best_member["name"],
                    "confidence": best_member["confidence"],
                    "reason": best_member["reason"]
                })
        
        return {
            "optimization_type": "task_allocation",
            "recommendations": recommendations,
            "total_unassigned": len(unassigned_tasks)
        }

    async def _find_best_assignee(self, task: Task) -> Optional[Dict[str, Any]]:
        """العثور على أفضل عضو فريق لمهمة معينة"""
        # الحصول على أعضاء الفريق في المشروع
        project = self.projects[task.project_id]
        available_members = [
            member for member_id, member in self.team_members.items()
            if member_id in project.team_members
        ]
        
        if not available_members:
            return None
        
        best_score = 0
        best_member = None
        
        for member in available_members:
            score = 0
            reasons = []
            
            # تقييم المهارات
            task_tags = task.tags
            matching_skills = len(set(member.skills) & set(task_tags))
            if matching_skills > 0:
                score += matching_skills * 20
                reasons.append(f"يمتلك {matching_skills} مهارات متطابقة")
            
            # تقييم العبء الحالي
            if member.workload < 0.8:  # أقل من 80% محمل
                score += (0.8 - member.workload) * 30
                reasons.append("لديه سعة إضافية في العمل")
            
            # تقييم الأولوية
            if task.priority == Priority.HIGH or task.priority == Priority.URGENT:
                if member.workload < 0.6:
                    score += 15
                    reasons.append("متاح للمهام عالية الأولوية")
            
            if score > best_score:
                best_score = score
                best_member = {
                    "member_id": member.member_id,
                    "name": member.name,
                    "confidence": min(score / 100, 1.0),
                    "reason": "; ".join(reasons) if reasons else "تطابق عام جيد"
                }
        
        return best_member

    async def _predict_project_timeline(self) -> Dict[str, Any]:
        """التنبؤ بالجدول الزمني للمشاريع"""
        predictions = []
        
        for project in self.projects.values():
            if project.status in [ProjectStatus.PLANNING, ProjectStatus.ACTIVE]:
                prediction = await self._calculate_project_timeline(project)
                predictions.append(prediction)
        
        return {
            "prediction_type": "project_timeline",
            "predictions": predictions
        }

    async def _calculate_project_timeline(self, project: Project) -> Dict[str, Any]:
        """حساب الجدول الزمني المتوقع لمشروع"""
        project_tasks = [task for task in self.tasks.values() if task.project_id == project.project_id]
        
        if not project_tasks:
            return {
                "project_id": project.project_id,
                "project_name": project.name,
                "predicted_completion": None,
                "confidence": 0.0,
                "risk_factors": ["لا توجد مهام محددة"]
            }
        
        # حساب إجمالي الساعات المطلوبة
        total_estimated = sum(task.estimated_hours for task in project_tasks)
        completed_hours = sum(task.actual_hours for task in project_tasks if task.status == TaskStatus.DONE)
        remaining_hours = total_estimated - completed_hours
        
        # حساب متوسط الإنتاجية اليومية
        team_capacity = self._calculate_team_daily_capacity(project.team_members)
        
        if team_capacity > 0:
            days_remaining = remaining_hours / team_capacity
            predicted_completion = datetime.now() + timedelta(days=days_remaining)
        else:
            predicted_completion = None
            days_remaining = float('inf')
        
        # تحليل عوامل المخاطر
        risk_factors = []
        confidence = 0.8
        
        if days_remaining > 30:
            risk_factors.append("مدة طويلة متبقية")
            confidence -= 0.1
        
        blocked_tasks = len([task for task in project_tasks if task.status == TaskStatus.BLOCKED])
        if blocked_tasks > 0:
            risk_factors.append(f"{blocked_tasks} مهام محجوبة")
            confidence -= 0.2
        
        overdue_tasks = len([
            task for task in project_tasks 
            if task.due_date and task.due_date < datetime.now() and task.status != TaskStatus.DONE
        ])
        if overdue_tasks > 0:
            risk_factors.append(f"{overdue_tasks} مهام متأخرة")
            confidence -= 0.15
        
        return {
            "project_id": project.project_id,
            "project_name": project.name,
            "predicted_completion": predicted_completion.isoformat() if predicted_completion else None,
            "days_remaining": days_remaining if days_remaining != float('inf') else None,
            "confidence": max(0.1, confidence),
            "risk_factors": risk_factors,
            "progress_percentage": project.progress
        }

    def _calculate_team_daily_capacity(self, team_member_ids: List[str]) -> float:
        """حساب السعة اليومية للفريق"""
        total_capacity = 0.0
        
        for member_id in team_member_ids:
            if member_id in self.team_members:
                member = self.team_members[member_id]
                # حساب متوسط الساعات اليومية
                daily_hours = sum(member.availability.values()) / 7
                # تطبيق عامل الكفاءة (85%)
                effective_hours = daily_hours * 0.85 * (1 - member.workload)
                total_capacity += effective_hours
        
        return total_capacity

    async def _balance_team_workload(self) -> Dict[str, Any]:
        """توازن أعباء العمل في الفريق"""
        workload_analysis = []
        recommendations = []
        
        for member in self.team_members.values():
            analysis = {
                "member_id": member.member_id,
                "name": member.name,
                "current_workload": member.workload,
                "status": self._get_workload_status(member.workload)
            }
            workload_analysis.append(analysis)
            
            # توصيات للتوازن
            if member.workload > 0.9:
                recommendations.append({
                    "type": "redistribute",
                    "member_id": member.member_id,
                    "member_name": member.name,
                    "recommendation": "إعادة توزيع بعض المهام لتخفيف العبء",
                    "priority": "high"
                })
            elif member.workload < 0.3:
                recommendations.append({
                    "type": "assign_more",
                    "member_id": member.member_id,
                    "member_name": member.name,
                    "recommendation": "يمكن تعيين مهام إضافية",
                    "priority": "medium"
                })
        
        return {
            "analysis_type": "workload_balancing",
            "team_analysis": workload_analysis,
            "recommendations": recommendations
        }

    def _get_workload_status(self, workload: float) -> str:
        """تحديد حالة العبء"""
        if workload >= 0.9:
            return "محمل بشدة"
        elif workload >= 0.7:
            return "محمل جيداً"
        elif workload >= 0.4:
            return "عبء متوسط"
        else:
            return "عبء خفيف"

    async def _analyze_project_risks(self) -> Dict[str, Any]:
        """تحليل مخاطر المشاريع"""
        risk_analysis = []
        
        for project in self.projects.values():
            if project.status in [ProjectStatus.ACTIVE, ProjectStatus.PLANNING]:
                risks = await self._identify_project_risks(project)
                risk_analysis.append(risks)
        
        return {
            "analysis_type": "risk_analysis",
            "project_risks": risk_analysis
        }

    async def _identify_project_risks(self, project: Project) -> Dict[str, Any]:
        """تحديد مخاطر مشروع معين"""
        risks = []
        risk_score = 0
        
        project_tasks = [task for task in self.tasks.values() if task.project_id == project.project_id]
        
        # مخاطر التأخير
        overdue_count = len([
            task for task in project_tasks 
            if task.due_date and task.due_date < datetime.now() and task.status != TaskStatus.DONE
        ])
        
        if overdue_count > 0:
            risk_score += overdue_count * 15
            risks.append({
                "type": "timeline_risk",
                "severity": "high" if overdue_count > 3 else "medium",
                "description": f"{overdue_count} مهام متأخرة",
                "impact": "تأخير في الجدول الزمني"
            })
        
        # مخاطر الموارد
        unassigned_count = len([task for task in project_tasks if not task.assigned_to])
        if unassigned_count > 0:
            risk_score += unassigned_count * 10
            risks.append({
                "type": "resource_risk",
                "severity": "medium",
                "description": f"{unassigned_count} مهام غير معينة",
                "impact": "نقص في توزيع الموارد"
            })
        
        # مخاطر التبعيات
        blocked_count = len([task for task in project_tasks if task.status == TaskStatus.BLOCKED])
        if blocked_count > 0:
            risk_score += blocked_count * 20
            risks.append({
                "type": "dependency_risk",
                "severity": "high",
                "description": f"{blocked_count} مهام محجوبة",
                "impact": "توقف في التدفق"
            })
        
        # تقييم المخاطر العام
        if risk_score >= 100:
            overall_risk = "high"
        elif risk_score >= 50:
            overall_risk = "medium"
        else:
            overall_risk = "low"
        
        return {
            "project_id": project.project_id,
            "project_name": project.name,
            "overall_risk": overall_risk,
            "risk_score": risk_score,
            "identified_risks": risks
        }

    async def _track_team_productivity(self) -> Dict[str, Any]:
        """تتبع إنتاجية الفريق"""
        productivity_data = []
        
        for member in self.team_members.values():
            productivity = await self._calculate_member_productivity(member)
            productivity_data.append(productivity)
        
        return {
            "analysis_type": "productivity_tracking",
            "team_productivity": productivity_data
        }

    async def _calculate_member_productivity(self, member: TeamMember) -> Dict[str, Any]:
        """حساب إنتاجية عضو الفريق"""
        member_tasks = [task for task in self.tasks.values() if task.assigned_to == member.member_id]
        
        if not member_tasks:
            return {
                "member_id": member.member_id,
                "name": member.name,
                "productivity_score": 0,
                "completed_tasks": 0,
                "efficiency": 0,
                "notes": "لا توجد مهام معينة"
            }
        
        completed_tasks = [task for task in member_tasks if task.status == TaskStatus.DONE]
        
        # حساب الكفاءة
        total_estimated = sum(task.estimated_hours for task in completed_tasks)
        total_actual = sum(task.actual_hours for task in completed_tasks)
        
        efficiency = (total_estimated / total_actual) if total_actual > 0 else 0
        
        # حساب نقاط الإنتاجية
        productivity_score = len(completed_tasks) * 10
        if efficiency > 1.0:  # أنجز بوقت أقل من المتوقع
            productivity_score += (efficiency - 1.0) * 20
        
        return {
            "member_id": member.member_id,
            "name": member.name,
            "productivity_score": round(productivity_score, 2),
            "completed_tasks": len(completed_tasks),
            "efficiency": round(efficiency, 2),
            "total_estimated_hours": total_estimated,
            "total_actual_hours": total_actual
        }

    async def _update_member_workload(self, member_id: str):
        """تحديث عبء العمل لعضو الفريق"""
        if member_id not in self.team_members:
            return
        
        member_tasks = [
            task for task in self.tasks.values() 
            if task.assigned_to == member_id and task.status != TaskStatus.DONE
        ]
        
        total_hours = sum(task.estimated_hours for task in member_tasks)
        member = self.team_members[member_id]
        
        # حساب السعة الأسبوعية
        weekly_capacity = sum(member.availability.values())
        workload = min(total_hours / weekly_capacity, 1.0) if weekly_capacity > 0 else 0
        
        member.workload = workload
        await self._save_team_member(member)

    async def _update_project_progress(self, project_id: str):
        """تحديث تقدم المشروع"""
        if project_id not in self.projects:
            return
        
        project_tasks = [task for task in self.tasks.values() if task.project_id == project_id]
        
        if not project_tasks:
            return
        
        total_progress = sum(task.progress for task in project_tasks)
        average_progress = total_progress / len(project_tasks)
        
        project = self.projects[project_id]
        project.progress = round(average_progress, 2)
        
        await self._save_project(project)

    async def _save_project(self, project: Project):
        """حفظ المشروع في قاعدة البيانات"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO projects 
            (project_id, name, description, status, owner, team_members, 
             created_at, start_date, end_date, budget, progress, milestones)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            project.project_id,
            project.name,
            project.description,
            project.status.value,
            project.owner,
            json.dumps(project.team_members),
            project.created_at.isoformat(),
            project.start_date.isoformat() if project.start_date else None,
            project.end_date.isoformat() if project.end_date else None,
            project.budget,
            project.progress,
            json.dumps(project.milestones)
        ))
        
        conn.commit()
        conn.close()

    async def _save_task(self, task: Task):
        """حفظ المهمة في قاعدة البيانات"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO tasks 
            (task_id, title, description, status, priority, assigned_to, project_id,
             created_at, due_date, estimated_hours, actual_hours, progress, dependencies, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            task.task_id,
            task.title,
            task.description,
            task.status.value,
            task.priority.value,
            task.assigned_to,
            task.project_id,
            task.created_at.isoformat(),
            task.due_date.isoformat() if task.due_date else None,
            task.estimated_hours,
            task.actual_hours,
            task.progress,
            json.dumps(task.dependencies),
            json.dumps(task.tags)
        ))
        
        conn.commit()
        conn.close()

    async def _save_team_member(self, member: TeamMember):
        """حفظ عضو الفريق في قاعدة البيانات"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO team_members 
            (member_id, name, email, role, skills, availability, workload)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            member.member_id,
            member.name,
            member.email,
            member.role,
            json.dumps(member.skills),
            json.dumps(member.availability),
            member.workload
        ))
        
        conn.commit()
        conn.close()

    async def get_ai_insights(self) -> Dict[str, Any]:
        """الحصول على رؤى الذكاء الاصطناعي"""
        insights = {}
        
        for engine_name, engine_func in self.ai_engines.items():
            try:
                result = await engine_func()
                insights[engine_name] = result
            except Exception as e:
                self.logger.error(f"خطأ في محرك {engine_name}: {e}")
                insights[engine_name] = {"error": str(e)}
        
        return {
            "generated_at": datetime.now().isoformat(),
            "ai_insights": insights
        }

    async def get_project_dashboard(self, project_id: str) -> Dict[str, Any]:
        """الحصول على لوحة تحكم المشروع"""
        if project_id not in self.projects:
            return {"error": "المشروع غير موجود"}
        
        project = self.projects[project_id]
        project_tasks = [task for task in self.tasks.values() if task.project_id == project_id]
        
        # إحصائيات المهام
        task_stats = {
            "total": len(project_tasks),
            "todo": len([t for t in project_tasks if t.status == TaskStatus.TODO]),
            "in_progress": len([t for t in project_tasks if t.status == TaskStatus.IN_PROGRESS]),
            "review": len([t for t in project_tasks if t.status == TaskStatus.REVIEW]),
            "done": len([t for t in project_tasks if t.status == TaskStatus.DONE]),
            "blocked": len([t for t in project_tasks if t.status == TaskStatus.BLOCKED])
        }
        
        # إحصائيات الأولوية
        priority_stats = {}
        for priority in Priority:
            priority_stats[priority.name.lower()] = len([
                t for t in project_tasks if t.priority == priority
            ])
        
        # أعضاء الفريق وأعباؤهم
        team_stats = []
        for member_id in project.team_members:
            if member_id in self.team_members:
                member = self.team_members[member_id]
                member_task_count = len([t for t in project_tasks if t.assigned_to == member_id])
                team_stats.append({
                    "name": member.name,
                    "role": member.role,
                    "assigned_tasks": member_task_count,
                    "workload": member.workload
                })
        
        return {
            "project_info": {
                "name": project.name,
                "status": project.status.value,
                "progress": project.progress,
                "budget": project.budget
            },
            "task_statistics": task_stats,
            "priority_distribution": priority_stats,
            "team_overview": team_stats,
            "timeline": {
                "start_date": project.start_date.isoformat() if project.start_date else None,
                "end_date": project.end_date.isoformat() if project.end_date else None,
                "created_at": project.created_at.isoformat()
            }
        }

# إنشاء مثيل عام
project_platform = ProjectManagementPlatform()

async def get_project_platform() -> ProjectManagementPlatform:
    """الحصول على منصة إدارة المشاريع"""
    return project_platform

if __name__ == "__main__":
    async def test_project_platform():
        """اختبار منصة إدارة المشاريع"""
        print("📋 اختبار منصة إدارة المشاريع الذكية")
        print("=" * 50)
        
        platform = await get_project_platform()
        await platform.initialize()
        
        # إنشاء عضو فريق
        member_id = await platform.add_team_member(
            name="أحمد محمد",
            email="ahmed@example.com",
            role="مطور",
            skills=["Python", "AI", "Web Development"]
        )
        print(f"👤 تم إنشاء عضو فريق: {member_id}")
        
        # إنشاء مشروع
        project_id = await platform.create_project(
            name="مشروع المساعد الذكي",
            description="تطوير مساعد ذكي متقدم",
            owner="مدير المشروع",
            team_members=[member_id],
            budget=50000.0
        )
        print(f"📋 تم إنشاء مشروع: {project_id}")
        
        # إنشاء مهام
        task1_id = await platform.create_task(
            title="تصميم واجهة المستخدم",
            description="تصميم واجهة تفاعلية للمساعد",
            project_id=project_id,
            priority=Priority.HIGH,
            estimated_hours=16.0,
            tags=["UI", "Design"]
        )
        
        task2_id = await platform.create_task(
            title="تطوير نماذج الذكاء الاصطناعي",
            description="تطوير وتدريب نماذج AI",
            project_id=project_id,
            priority=Priority.CRITICAL,
            estimated_hours=40.0,
            tags=["AI", "Machine Learning"]
        )
        
        print(f"✅ تم إنشاء مهمتين")
        
        # تعيين المهام
        await platform.assign_task(task1_id, member_id)
        await platform.assign_task(task2_id, member_id)
        print("📋 تم تعيين المهام")
        
        # تحديث حالة مهمة
        await platform.update_task_status(task1_id, TaskStatus.IN_PROGRESS)
        print("🔄 تم تحديث حالة المهمة")
        
        # الحصول على رؤى الذكاء الاصطناعي
        print("\n🤖 رؤى الذكاء الاصطناعي:")
        insights = await platform.get_ai_insights()
        
        for engine_name, result in insights["ai_insights"].items():
            if "error" not in result:
                print(f"✅ {engine_name}: تم التحليل بنجاح")
            else:
                print(f"❌ {engine_name}: {result['error']}")
        
        # عرض لوحة تحكم المشروع
        print("\n📊 لوحة تحكم المشروع:")
        dashboard = await platform.get_project_dashboard(project_id)
        print(f"📈 تقدم المشروع: {dashboard['project_info']['progress']}%")
        print(f"📋 إجمالي المهام: {dashboard['task_statistics']['total']}")
        print(f"👥 أعضاء الفريق: {len(dashboard['team_overview'])}")
    
    asyncio.run(test_project_platform())
