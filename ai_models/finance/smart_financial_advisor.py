
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
المستشار المالي الذكي المتقدم
Advanced Smart Financial Advisor
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import json
import sqlite3
from enum import Enum
import yfinance as yf
import requests

class InvestmentType(Enum):
    """أنواع الاستثمارات"""
    STOCKS = "stocks"
    BONDS = "bonds"
    CRYPTO = "cryptocurrency"
    REAL_ESTATE = "real_estate"
    COMMODITIES = "commodities"
    MUTUAL_FUNDS = "mutual_funds"
    ETF = "etf"
    SAVINGS = "savings"

class RiskTolerance(Enum):
    """مستويات تحمل المخاطر"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    VERY_AGGRESSIVE = "very_aggressive"

@dataclass
class FinancialGoal:
    """هدف مالي"""
    goal_id: str
    name: str
    target_amount: float
    current_amount: float
    target_date: datetime
    priority: int
    goal_type: str  # retirement, house, education, etc.
    monthly_contribution: float
    risk_tolerance: RiskTolerance
    created_at: datetime
    
    @property
    def progress_percentage(self) -> float:
        return min((self.current_amount / self.target_amount) * 100, 100)
    
    @property
    def months_remaining(self) -> int:
        return max(0, (self.target_date - datetime.now()).days // 30)

@dataclass
class Investment:
    """استثمار"""
    investment_id: str
    name: str
    symbol: str
    investment_type: InvestmentType
    current_value: float
    purchase_price: float
    quantity: float
    purchase_date: datetime
    last_updated: datetime
    
    @property
    def total_return(self) -> float:
        return (self.current_value - self.purchase_price) * self.quantity
    
    @property
    def return_percentage(self) -> float:
        if self.purchase_price > 0:
            return ((self.current_value - self.purchase_price) / self.purchase_price) * 100
        return 0.0

@dataclass
class Transaction:
    """معاملة مالية"""
    transaction_id: str
    date: datetime
    amount: float
    category: str
    description: str
    is_income: bool
    tags: List[str]
    recurring: bool = False
    recurring_frequency: Optional[str] = None

class MarketDataProvider:
    """موفر بيانات السوق"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.cache = {}
        self.cache_expiry = {}
    
    async def get_stock_price(self, symbol: str) -> Optional[Dict[str, Any]]:
        """الحصول على سعر السهم"""
        
        try:
            # فحص التخزين المؤقت
            if self._is_cache_valid(symbol):
                return self.cache[symbol]
            
            # جلب البيانات من Yahoo Finance
            stock = yf.Ticker(symbol)
            info = stock.info
            hist = stock.history(period="1d")
            
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                previous_close = info.get('previousClose', current_price)
                
                data = {
                    "symbol": symbol,
                    "current_price": float(current_price),
                    "previous_close": float(previous_close),
                    "change": float(current_price - previous_close),
                    "change_percent": float(((current_price - previous_close) / previous_close) * 100),
                    "volume": int(hist['Volume'].iloc[-1]),
                    "market_cap": info.get('marketCap'),
                    "pe_ratio": info.get('forwardPE'),
                    "dividend_yield": info.get('dividendYield'),
                    "last_updated": datetime.now().isoformat()
                }
                
                # حفظ في التخزين المؤقت
                self.cache[symbol] = data
                self.cache_expiry[symbol] = datetime.now() + timedelta(minutes=15)
                
                return data
            
            return None
            
        except Exception as e:
            self.logger.error(f"خطأ في الحصول على سعر السهم {symbol}: {e}")
            return None
    
    async def get_crypto_price(self, symbol: str) -> Optional[Dict[str, Any]]:
        """الحصول على سعر العملة المشفرة"""
        
        try:
            # استخدام CoinGecko API
            url = f"https://api.coingecko.com/api/v3/simple/price"
            params = {
                "ids": symbol.lower(),
                "vs_currencies": "usd",
                "include_24hr_change": "true",
                "include_market_cap": "true"
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if symbol.lower() in data:
                crypto_data = data[symbol.lower()]
                
                return {
                    "symbol": symbol.upper(),
                    "current_price": crypto_data["usd"],
                    "change_24h": crypto_data.get("usd_24h_change", 0),
                    "market_cap": crypto_data.get("usd_market_cap"),
                    "last_updated": datetime.now().isoformat()
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"خطأ في الحصول على سعر العملة المشفرة {symbol}: {e}")
            return None
    
    def _is_cache_valid(self, symbol: str) -> bool:
        """فحص صحة التخزين المؤقت"""
        
        return (
            symbol in self.cache and
            symbol in self.cache_expiry and
            datetime.now() < self.cache_expiry[symbol]
        )

class PortfolioAnalyzer:
    """محلل المحفظة الاستثمارية"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze_portfolio(self, investments: List[Investment]) -> Dict[str, Any]:
        """تحليل المحفظة الاستثمارية"""
        
        try:
            if not investments:
                return {"error": "لا توجد استثمارات للتحليل"}
            
            # حساب القيم الأساسية
            total_value = sum(inv.current_value * inv.quantity for inv in investments)
            total_cost = sum(inv.purchase_price * inv.quantity for inv in investments)
            total_return = total_value - total_cost
            total_return_percent = (total_return / total_cost * 100) if total_cost > 0 else 0
            
            # تحليل التنويع
            diversification = self._analyze_diversification(investments)
            
            # تحليل المخاطر
            risk_analysis = self._analyze_risk(investments)
            
            # أداء الاستثمارات الفردية
            performance = []
            for inv in investments:
                performance.append({
                    "name": inv.name,
                    "symbol": inv.symbol,
                    "type": inv.investment_type.value,
                    "value": inv.current_value * inv.quantity,
                    "return_amount": inv.total_return,
                    "return_percent": inv.return_percentage,
                    "weight": (inv.current_value * inv.quantity / total_value) * 100
                })
            
            # ترتيب حسب الأداء
            performance.sort(key=lambda x: x["return_percent"], reverse=True)
            
            return {
                "summary": {
                    "total_value": round(total_value, 2),
                    "total_cost": round(total_cost, 2),
                    "total_return": round(total_return, 2),
                    "return_percentage": round(total_return_percent, 2),
                    "num_holdings": len(investments)
                },
                "diversification": diversification,
                "risk_analysis": risk_analysis,
                "performance": performance,
                "recommendations": self._generate_portfolio_recommendations(investments, diversification, risk_analysis)
            }
            
        except Exception as e:
            self.logger.error(f"خطأ في تحليل المحفظة: {e}")
            return {"error": str(e)}
    
    def _analyze_diversification(self, investments: List[Investment]) -> Dict[str, Any]:
        """تحليل التنويع"""
        
        # توزيع حسب نوع الاستثمار
        type_distribution = {}
        total_value = sum(inv.current_value * inv.quantity for inv in investments)
        
        for inv in investments:
            inv_type = inv.investment_type.value
            inv_value = inv.current_value * inv.quantity
            
            if inv_type not in type_distribution:
                type_distribution[inv_type] = 0
            type_distribution[inv_type] += inv_value
        
        # تحويل إلى نسب مئوية
        for inv_type in type_distribution:
            type_distribution[inv_type] = (type_distribution[inv_type] / total_value) * 100
        
        # حساب مؤشر التنويع
        diversification_score = self._calculate_diversification_score(type_distribution)
        
        return {
            "by_type": type_distribution,
            "diversification_score": diversification_score,
            "concentration_risk": max(type_distribution.values()) if type_distribution else 0
        }
    
    def _calculate_diversification_score(self, distribution: Dict[str, float]) -> float:
        """حساب مؤشر التنويع (0-100)"""
        
        if not distribution:
            return 0
        
        # استخدام مؤشر Herfindahl-Hirschman
        hhi = sum((percentage / 100) ** 2 for percentage in distribution.values())
        
        # تحويل إلى مؤشر التنويع (أعلى = أفضل تنويع)
        max_hhi = 1.0  # عندما يكون كل شيء في استثمار واحد
        min_hhi = 1.0 / len(distribution)  # عندما يكون التوزيع متساوي
        
        if max_hhi == min_hhi:
            return 100
        
        diversification_score = (1 - (hhi - min_hhi) / (max_hhi - min_hhi)) * 100
        return max(0, min(100, diversification_score))
    
    def _analyze_risk(self, investments: List[Investment]) -> Dict[str, Any]:
        """تحليل المخاطر"""
        
        # تقدير المخاطر بناءً على نوع الاستثمار
        risk_weights = {
            InvestmentType.STOCKS: 0.7,
            InvestmentType.CRYPTO: 0.9,
            InvestmentType.BONDS: 0.3,
            InvestmentType.REAL_ESTATE: 0.5,
            InvestmentType.COMMODITIES: 0.6,
            InvestmentType.MUTUAL_FUNDS: 0.4,
            InvestmentType.ETF: 0.4,
            InvestmentType.SAVINGS: 0.1
        }
        
        total_value = sum(inv.current_value * inv.quantity for inv in investments)
        weighted_risk = 0
        
        for inv in investments:
            weight = (inv.current_value * inv.quantity) / total_value
            risk = risk_weights.get(inv.investment_type, 0.5)
            weighted_risk += weight * risk
        
        # تصنيف المخاطر
        if weighted_risk < 0.3:
            risk_level = "منخفض"
        elif weighted_risk < 0.5:
            risk_level = "متوسط"
        elif weighted_risk < 0.7:
            risk_level = "عالي"
        else:
            risk_level = "عالي جداً"
        
        return {
            "overall_risk_score": round(weighted_risk * 100, 1),
            "risk_level": risk_level,
            "volatility_estimate": self._estimate_portfolio_volatility(investments)
        }
    
    def _estimate_portfolio_volatility(self, investments: List[Investment]) -> float:
        """تقدير تقلبات المحفظة"""
        
        # تقدير بسيط للتقلبات بناءً على الأداء التاريخي
        returns = [inv.return_percentage for inv in investments if inv.return_percentage != 0]
        
        if len(returns) > 1:
            return float(np.std(returns))
        else:
            return 15.0  # تقدير افتراضي
    
    def _generate_portfolio_recommendations(
        self,
        investments: List[Investment],
        diversification: Dict[str, Any],
        risk_analysis: Dict[str, Any]
    ) -> List[str]:
        """توليد توصيات المحفظة"""
        
        recommendations = []
        
        # توصيات التنويع
        if diversification["diversification_score"] < 50:
            recommendations.append("ينصح بزيادة التنويع في محفظتك لتقليل المخاطر")
        
        if diversification["concentration_risk"] > 50:
            recommendations.append("تجنب التركيز الزائد في نوع واحد من الاستثمارات")
        
        # توصيات المخاطر
        if risk_analysis["overall_risk_score"] > 70:
            recommendations.append("مستوى المخاطر مرتفع، فكر في إضافة استثمارات أكثر أماناً")
        
        if risk_analysis["overall_risk_score"] < 30:
            recommendations.append("مستوى المخاطر منخفض جداً، يمكن إضافة استثمارات بعائد أعلى")
        
        # توصيات الأداء
        poor_performers = [inv for inv in investments if inv.return_percentage < -10]
        if poor_performers:
            recommendations.append(f"مراجعة {len(poor_performers)} استثمار بأداء ضعيف")
        
        return recommendations

class FinancialPlanningEngine:
    """محرك التخطيط المالي"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def create_investment_plan(
        self,
        goal: FinancialGoal,
        current_age: int,
        retirement_age: int = 65,
        inflation_rate: float = 0.03
    ) -> Dict[str, Any]:
        """إنشاء خطة استثمار"""
        
        try:
            months_to_goal = goal.months_remaining
            amount_needed = goal.target_amount - goal.current_amount
            
            if months_to_goal <= 0:
                return {"error": "تاريخ الهدف قد انتهى"}
            
            # حساب معدل العائد المطلوب
            if goal.monthly_contribution > 0:
                required_return = self._calculate_required_return(
                    goal.current_amount,
                    goal.target_amount,
                    goal.monthly_contribution,
                    months_to_goal
                )
            else:
                required_return = self._calculate_simple_return(
                    goal.current_amount,
                    goal.target_amount,
                    months_to_goal
                )
            
            # اقتراح تخصيص الأصول
            asset_allocation = self._suggest_asset_allocation(
                goal.risk_tolerance,
                months_to_goal,
                current_age
            )
            
            # حساب التوقعات
            projections = self._calculate_projections(
                goal.current_amount,
                goal.monthly_contribution,
                required_return,
                months_to_goal,
                inflation_rate
            )
            
            return {
                "goal_analysis": {
                    "amount_needed": round(amount_needed, 2),
                    "months_to_goal": months_to_goal,
                    "required_monthly_return": round(required_return * 100, 2),
                    "required_annual_return": round(required_return * 12 * 100, 2)
                },
                "asset_allocation": asset_allocation,
                "projections": projections,
                "recommendations": self._generate_planning_recommendations(goal, required_return, current_age)
            }
            
        except Exception as e:
            self.logger.error(f"خطأ في إنشاء خطة الاستثمار: {e}")
            return {"error": str(e)}
    
    def _calculate_required_return(
        self,
        current_amount: float,
        target_amount: float,
        monthly_contribution: float,
        months: int
    ) -> float:
        """حساب معدل العائد المطلوب"""
        
        if monthly_contribution == 0:
            return ((target_amount / current_amount) ** (1/months)) - 1
        
        # استخدام الصيغة المالية للقيمة المستقبلية للدفعات المنتظمة
        # تقريب بسيط
        total_contributions = monthly_contribution * months
        growth_needed = target_amount - current_amount - total_contributions
        
        if current_amount + total_contributions > 0:
            return growth_needed / (current_amount + total_contributions/2) / months
        else:
            return 0.01  # 1% شهرياً كافتراضي
    
    def _calculate_simple_return(
        self,
        current_amount: float,
        target_amount: float,
        months: int
    ) -> float:
        """حساب العائد البسيط المطلوب"""
        
        if current_amount > 0 and months > 0:
            return ((target_amount / current_amount) ** (1/months)) - 1
        return 0
    
    def _suggest_asset_allocation(
        self,
        risk_tolerance: RiskTolerance,
        months_to_goal: int,
        current_age: int
    ) -> Dict[str, float]:
        """اقتراح تخصيص الأصول"""
        
        # قاعدة أساسية: عمر المستثمر = نسبة السندات
        base_bonds_percent = min(current_age, 80)
        base_stocks_percent = 100 - base_bonds_percent
        
        # تعديل حسب تحمل المخاطر
        risk_adjustments = {
            RiskTolerance.CONSERVATIVE: {"stocks": -20, "bonds": +15, "savings": +5},
            RiskTolerance.MODERATE: {"stocks": 0, "bonds": 0, "savings": 0},
            RiskTolerance.AGGRESSIVE: {"stocks": +15, "bonds": -10, "alternatives": +5},
            RiskTolerance.VERY_AGGRESSIVE: {"stocks": +25, "bonds": -15, "alternatives": +10}
        }
        
        # تعديل حسب الأفق الزمني
        if months_to_goal < 24:  # أقل من سنتين
            time_adjustment = {"stocks": -15, "bonds": +10, "savings": +5}
        elif months_to_goal > 120:  # أكثر من 10 سنوات
            time_adjustment = {"stocks": +10, "bonds": -5, "alternatives": +5}
        else:
            time_adjustment = {"stocks": 0, "bonds": 0, "savings": 0}
        
        # حساب التخصيص النهائي
        allocation = {
            "stocks": base_stocks_percent,
            "bonds": base_bonds_percent,
            "alternatives": 0,
            "savings": 0
        }
        
        # تطبيق التعديلات
        adjustments = risk_adjustments.get(risk_tolerance, {})
        for asset, adjustment in adjustments.items():
            if asset in allocation:
                allocation[asset] += adjustment
        
        for asset, adjustment in time_adjustment.items():
            if asset in allocation:
                allocation[asset] += adjustment
        
        # التأكد من أن المجموع = 100%
        total = sum(allocation.values())
        if total != 100:
            for asset in allocation:
                allocation[asset] = allocation[asset] / total * 100
        
        # إزالة القيم السلبية
        for asset in allocation:
            allocation[asset] = max(0, allocation[asset])
        
        return {k: round(v, 1) for k, v in allocation.items() if v > 0}
    
    def _calculate_projections(
        self,
        current_amount: float,
        monthly_contribution: float,
        monthly_return: float,
        months: int,
        inflation_rate: float
    ) -> Dict[str, Any]:
        """حساب التوقعات المالية"""
        
        projections = []
        balance = current_amount
        monthly_inflation = inflation_rate / 12
        
        for month in range(1, months + 1):
            # إضافة المساهمة الشهرية
            balance += monthly_contribution
            
            # تطبيق العائد
            balance *= (1 + monthly_return)
            
            # حساب القوة الشرائية (مع التضخم)
            real_value = balance / ((1 + monthly_inflation) ** month)
            
            if month % 12 == 0 or month == months:  # كل سنة أو الشهر الأخير
                projections.append({
                    "month": month,
                    "year": month // 12 + (1 if month % 12 > 0 else 0),
                    "nominal_value": round(balance, 2),
                    "real_value": round(real_value, 2),
                    "total_contributions": round(monthly_contribution * month, 2)
                })
        
        return {
            "monthly_projections": projections,
            "final_amount": round(balance, 2),
            "total_contributions": round(monthly_contribution * months, 2),
            "investment_growth": round(balance - current_amount - monthly_contribution * months, 2)
        }
    
    def _generate_planning_recommendations(
        self,
        goal: FinancialGoal,
        required_return: float,
        current_age: int
    ) -> List[str]:
        """توليد توصيات التخطيط"""
        
        recommendations = []
        annual_return = required_return * 12 * 100
        
        if annual_return > 15:
            recommendations.append("العائد المطلوب مرتفع جداً، فكر في زيادة المساهمة الشهرية أو تمديد الفترة الزمنية")
        
        if annual_return < 3:
            recommendations.append("العائد المطلوب منخفض، يمكن الاستثمار في خيارات أكثر أماناً")
        
        if goal.months_remaining < 24:
            recommendations.append("الأفق الزمني قصير، ركز على الاستثمارات قليلة المخاطر")
        
        if current_age > 50 and goal.risk_tolerance == RiskTolerance.VERY_AGGRESSIVE:
            recommendations.append("قد يكون من الأفضل تقليل المخاطر مع اقتراب سن التقاعد")
        
        if goal.monthly_contribution < (goal.target_amount - goal.current_amount) / goal.months_remaining * 0.8:
            recommendations.append("المساهمة الشهرية قد تكون غير كافية لتحقيق الهدف")
        
        return recommendations

class SmartFinancialAdvisor:
    """المستشار المالي الذكي الرئيسي"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # قاعدة البيانات
        self.db_path = Path("data/financial_advisor.db")
        self.db_path.parent.mkdir(exist_ok=True)
        
        # المكونات
        self.market_data = MarketDataProvider()
        self.portfolio_analyzer = PortfolioAnalyzer()
        self.planning_engine = FinancialPlanningEngine()
        
        # البيانات المالية
        self.financial_goals: Dict[str, FinancialGoal] = {}
        self.investments: Dict[str, Investment] = {}
        self.transactions: List[Transaction] = []
        
        # ملف تعريف المخاطر
        self.user_profile = {
            "age": 30,
            "income": 0,
            "risk_tolerance": RiskTolerance.MODERATE,
            "investment_experience": "beginner",
            "financial_goals": [],
            "time_horizon": "medium"
        }

    async def initialize(self):
        """تهيئة المستشار المالي"""
        
        try:
            self.logger.info("💰 تهيئة المستشار المالي الذكي...")
            
            # إنشاء قاعدة البيانات
            await self._initialize_database()
            
            # تحميل البيانات
            await self._load_user_data()
            
            # تحديث أسعار السوق
            await self._update_market_data()
            
            self.logger.info("✅ تم تهيئة المستشار المالي")
            
        except Exception as e:
            self.logger.error(f"❌ فشل تهيئة المستشار المالي: {e}")

    async def _initialize_database(self):
        """تهيئة قاعدة البيانات"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # جدول الأهداف المالية
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS financial_goals (
                goal_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                target_amount REAL NOT NULL,
                current_amount REAL NOT NULL,
                target_date TEXT NOT NULL,
                priority INTEGER,
                goal_type TEXT,
                monthly_contribution REAL,
                risk_tolerance TEXT,
                created_at TEXT NOT NULL
            )
        """)
        
        # جدول الاستثمارات
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS investments (
                investment_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                symbol TEXT NOT NULL,
                investment_type TEXT NOT NULL,
                current_value REAL NOT NULL,
                purchase_price REAL NOT NULL,
                quantity REAL NOT NULL,
                purchase_date TEXT NOT NULL,
                last_updated TEXT NOT NULL
            )
        """)
        
        # جدول المعاملات
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                transaction_id TEXT PRIMARY KEY,
                date TEXT NOT NULL,
                amount REAL NOT NULL,
                category TEXT NOT NULL,
                description TEXT,
                is_income INTEGER NOT NULL,
                tags TEXT,
                recurring INTEGER DEFAULT 0,
                recurring_frequency TEXT
            )
        """)
        
        # جدول ملف المستخدم
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_profile (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
        
        conn.commit()
        conn.close()

    async def _load_user_data(self):
        """تحميل بيانات المستخدم"""
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # تحميل الأهداف المالية
            cursor.execute("SELECT * FROM financial_goals")
            for row in cursor.fetchall():
                goal = FinancialGoal(
                    goal_id=row[0],
                    name=row[1],
                    target_amount=row[2],
                    current_amount=row[3],
                    target_date=datetime.fromisoformat(row[4]),
                    priority=row[5],
                    goal_type=row[6],
                    monthly_contribution=row[7],
                    risk_tolerance=RiskTolerance(row[8]),
                    created_at=datetime.fromisoformat(row[9])
                )
                self.financial_goals[goal.goal_id] = goal
            
            # تحميل الاستثمارات
            cursor.execute("SELECT * FROM investments")
            for row in cursor.fetchall():
                investment = Investment(
                    investment_id=row[0],
                    name=row[1],
                    symbol=row[2],
                    investment_type=InvestmentType(row[3]),
                    current_value=row[4],
                    purchase_price=row[5],
                    quantity=row[6],
                    purchase_date=datetime.fromisoformat(row[7]),
                    last_updated=datetime.fromisoformat(row[8])
                )
                self.investments[investment.investment_id] = investment
            
            conn.close()
            
        except Exception as e:
            self.logger.error(f"خطأ في تحميل بيانات المستخدم: {e}")

    async def _update_market_data(self):
        """تحديث بيانات السوق"""
        
        try:
            for investment in self.investments.values():
                if investment.investment_type in [InvestmentType.STOCKS, InvestmentType.ETF]:
                    market_data = await self.market_data.get_stock_price(investment.symbol)
                    if market_data:
                        investment.current_value = market_data["current_price"]
                        investment.last_updated = datetime.now()
                
                elif investment.investment_type == InvestmentType.CRYPTO:
                    crypto_data = await self.market_data.get_crypto_price(investment.symbol)
                    if crypto_data:
                        investment.current_value = crypto_data["current_price"]
                        investment.last_updated = datetime.now()
            
        except Exception as e:
            self.logger.warning(f"خطأ في تحديث بيانات السوق: {e}")

    async def add_financial_goal(
        self,
        name: str,
        target_amount: float,
        target_date: datetime,
        goal_type: str = "general",
        monthly_contribution: float = 0,
        risk_tolerance: RiskTolerance = RiskTolerance.MODERATE,
        priority: int = 5
    ) -> Dict[str, Any]:
        """إضافة هدف مالي جديد"""
        
        try:
            goal_id = f"goal_{datetime.now().timestamp()}"
            
            goal = FinancialGoal(
                goal_id=goal_id,
                name=name,
                target_amount=target_amount,
                current_amount=0.0,
                target_date=target_date,
                priority=priority,
                goal_type=goal_type,
                monthly_contribution=monthly_contribution,
                risk_tolerance=risk_tolerance,
                created_at=datetime.now()
            )
            
            self.financial_goals[goal_id] = goal
            
            # حفظ في قاعدة البيانات
            await self._save_goal_to_db(goal)
            
            # إنشاء خطة استثمار
            investment_plan = self.planning_engine.create_investment_plan(
                goal,
                self.user_profile["age"]
            )
            
            return {
                "success": True,
                "goal_id": goal_id,
                "goal": asdict(goal),
                "investment_plan": investment_plan,
                "message": f"تم إضافة الهدف المالي: {name}"
            }
            
        except Exception as e:
            self.logger.error(f"خطأ في إضافة الهدف المالي: {e}")
            return {"success": False, "error": str(e)}

    async def _save_goal_to_db(self, goal: FinancialGoal):
        """حفظ الهدف في قاعدة البيانات"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO financial_goals
            (goal_id, name, target_amount, current_amount, target_date, priority,
             goal_type, monthly_contribution, risk_tolerance, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            goal.goal_id, goal.name, goal.target_amount, goal.current_amount,
            goal.target_date.isoformat(), goal.priority, goal.goal_type,
            goal.monthly_contribution, goal.risk_tolerance.value,
            goal.created_at.isoformat()
        ))
        
        conn.commit()
        conn.close()

    async def add_investment(
        self,
        name: str,
        symbol: str,
        investment_type: InvestmentType,
        quantity: float,
        purchase_price: float,
        purchase_date: datetime = None
    ) -> Dict[str, Any]:
        """إضافة استثمار جديد"""
        
        try:
            investment_id = f"inv_{datetime.now().timestamp()}"
            
            if purchase_date is None:
                purchase_date = datetime.now()
            
            # الحصول على السعر الحالي
            current_value = purchase_price  # افتراضي
            
            if investment_type in [InvestmentType.STOCKS, InvestmentType.ETF]:
                market_data = await self.market_data.get_stock_price(symbol)
                if market_data:
                    current_value = market_data["current_price"]
            
            elif investment_type == InvestmentType.CRYPTO:
                crypto_data = await self.market_data.get_crypto_price(symbol)
                if crypto_data:
                    current_value = crypto_data["current_price"]
            
            investment = Investment(
                investment_id=investment_id,
                name=name,
                symbol=symbol,
                investment_type=investment_type,
                current_value=current_value,
                purchase_price=purchase_price,
                quantity=quantity,
                purchase_date=purchase_date,
                last_updated=datetime.now()
            )
            
            self.investments[investment_id] = investment
            
            # حفظ في قاعدة البيانات
            await self._save_investment_to_db(investment)
            
            return {
                "success": True,
                "investment_id": investment_id,
                "investment": asdict(investment),
                "current_return": investment.total_return,
                "return_percentage": investment.return_percentage,
                "message": f"تم إضافة الاستثمار: {name}"
            }
            
        except Exception as e:
            self.logger.error(f"خطأ في إضافة الاستثمار: {e}")
            return {"success": False, "error": str(e)}

    async def _save_investment_to_db(self, investment: Investment):
        """حفظ الاستثمار في قاعدة البيانات"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO investments
            (investment_id, name, symbol, investment_type, current_value,
             purchase_price, quantity, purchase_date, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            investment.investment_id, investment.name, investment.symbol,
            investment.investment_type.value, investment.current_value,
            investment.purchase_price, investment.quantity,
            investment.purchase_date.isoformat(), investment.last_updated.isoformat()
        ))
        
        conn.commit()
        conn.close()

    async def get_portfolio_analysis(self) -> Dict[str, Any]:
        """الحصول على تحليل المحفظة"""
        
        try:
            # تحديث الأسعار
            await self._update_market_data()
            
            # تحليل المحفظة
            investments_list = list(self.investments.values())
            analysis = self.portfolio_analyzer.analyze_portfolio(investments_list)
            
            # إضافة تحليل الأهداف
            goals_analysis = await self._analyze_goals_progress()
            
            return {
                "portfolio_analysis": analysis,
                "goals_analysis": goals_analysis,
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"خطأ في تحليل المحفظة: {e}")
            return {"error": str(e)}

    async def _analyze_goals_progress(self) -> Dict[str, Any]:
        """تحليل تقدم الأهداف"""
        
        goals_summary = []
        
        for goal in self.financial_goals.values():
            # حساب التقدم المتوقع
            months_elapsed = max(0, (datetime.now() - goal.created_at).days // 30)
            expected_progress = min(
                goal.monthly_contribution * months_elapsed,
                goal.target_amount
            )
            
            goals_summary.append({
                "name": goal.name,
                "progress_percentage": goal.progress_percentage,
                "amount_needed": goal.target_amount - goal.current_amount,
                "months_remaining": goal.months_remaining,
                "on_track": goal.current_amount >= expected_progress * 0.9,
                "priority": goal.priority
            })
        
        return {
            "total_goals": len(self.financial_goals),
            "goals_on_track": len([g for g in goals_summary if g["on_track"]]),
            "goals_summary": goals_summary
        }

    async def get_personalized_advice(self, query: str) -> Dict[str, Any]:
        """الحصول على نصائح مالية مخصصة"""
        
        try:
            query_lower = query.lower().strip()
            advice = {"recommendations": [], "analysis": {}}
            
            # تحليل الاستفسار وتقديم النصيحة
            if any(word in query_lower for word in ["استثمار", "investment", "portfolio"]):
                advice = await self._get_investment_advice()
            
            elif any(word in query_lower for word in ["retirement", "تقاعد", "pension"]):
                advice = await self._get_retirement_advice()
            
            elif any(word in query_lower for word in ["budget", "ميزانية", "spending"]):
                advice = await self._get_budgeting_advice()
            
            elif any(word in query_lower for word in ["debt", "loan", "ديون", "قرض"]):
                advice = await self._get_debt_advice()
            
            elif any(word in query_lower for word in ["emergency", "طوارئ", "savings"]):
                advice = await self._get_emergency_fund_advice()
            
            else:
                advice = await self._get_general_financial_advice()
            
            return {
                "query": query,
                "advice": advice,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"خطأ في تقديم النصيحة المالية: {e}")
            return {"error": str(e)}

    async def _get_investment_advice(self) -> Dict[str, Any]:
        """نصائح الاستثمار"""
        
        portfolio_analysis = await self.get_portfolio_analysis()
        
        recommendations = []
        
        if not self.investments:
            recommendations.append("ابدأ بالاستثمار في صناديق المؤشرات المتنوعة")
            recommendations.append("خصص طوارئ 3-6 أشهر من النفقات قبل الاستثمار")
        else:
            analysis = portfolio_analysis.get("portfolio_analysis", {})
            
            if analysis.get("diversification", {}).get("diversification_score", 0) < 50:
                recommendations.append("زد التنويع في محفظتك الاستثمارية")
            
            if analysis.get("risk_analysis", {}).get("overall_risk_score", 0) > 70:
                recommendations.append("فكر في تقليل المخاطر بإضافة استثمارات أكثر أماناً")
        
        return {
            "recommendations": recommendations,
            "analysis": portfolio_analysis.get("portfolio_analysis", {}),
            "suggested_allocation": self.planning_engine._suggest_asset_allocation(
                self.user_profile["risk_tolerance"],
                120,  # 10 سنوات افتراضي
                self.user_profile["age"]
            )
        }

    async def _get_retirement_advice(self) -> Dict[str, Any]:
        """نصائح التقاعد"""
        
        current_age = self.user_profile["age"]
        retirement_age = 65
        years_to_retirement = retirement_age - current_age
        
        recommendations = []
        
        if years_to_retirement > 30:
            recommendations.append("ابدأ الادخار للتقاعد مبكراً للاستفادة من قوة التراكب")
            recommendations.append("استثمر بجرأة أكبر لأن لديك وقت طويل")
        elif years_to_retirement > 10:
            recommendations.append("زد مساهماتك في خطة التقاعد")
            recommendations.append("راجع تخصيص أصولك لتوازن النمو والأمان")
        else:
            recommendations.append("ركز على الاستثمارات الآمنة")
            recommendations.append("فكر في تأخير التقاعد لزيادة المدخرات")
        
        # حساب المبلغ المقترح للتقاعد
        estimated_annual_expenses = 50000  # افتراضي
        retirement_goal = estimated_annual_expenses * 25  # قاعدة 4%
        
        return {
            "recommendations": recommendations,
            "analysis": {
                "years_to_retirement": years_to_retirement,
                "estimated_retirement_goal": retirement_goal,
                "recommended_monthly_savings": retirement_goal / (years_to_retirement * 12)
            }
        }

    async def _get_budgeting_advice(self) -> Dict[str, Any]:
        """نصائح الميزانية"""
        
        recommendations = [
            "طبق قاعدة 50/30/20: 50% للضروريات، 30% للرغبات، 20% للادخار",
            "راقب نفقاتك لمدة شهر لفهم أنماط الإنفاق",
            "قلل النفقات المتكررة غير الضرورية",
            "استخدم تطبيقات الميزانية لتتبع نفقاتك"
        ]
        
        return {
            "recommendations": recommendations,
            "suggested_categories": {
                "housing": 25,
                "food": 15,
                "transportation": 10,
                "utilities": 5,
                "entertainment": 10,
                "savings": 20,
                "other": 15
            }
        }

    async def _get_debt_advice(self) -> Dict[str, Any]:
        """نصائح إدارة الديون"""
        
        recommendations = [
            "سدد الديون عالية الفائدة أولاً (طريقة الانهيار الجليدي)",
            "فكر في توحيد الديون لتقليل الفوائد",
            "تجنب اقتراض أموال جديدة أثناء سداد الديون",
            "ضع خطة واضحة لسداد كل دين"
        ]
        
        return {
            "recommendations": recommendations,
            "debt_strategies": [
                "طريقة كرة الثلج: ابدأ بأصغر دين",
                "طريقة الانهيار الجليدي: ابدأ بأعلى فائدة",
                "التوحيد: ادمج الديون في قرض واحد بفائدة أقل"
            ]
        }

    async def _get_emergency_fund_advice(self) -> Dict[str, Any]:
        """نصائح صندوق الطوارئ"""
        
        recommendations = [
            "ابدأ بهدف صغير: وفر 1000 دولار كصندوق طوارئ أولي",
            "اهدف لتوفير 3-6 أشهر من نفقاتك المعيشية",
            "احتفظ بصندوق الطوارئ في حساب ادخار عالي العائد",
            "لا تستخدم صندوق الطوارئ إلا في الحالات الطارئة الحقيقية"
        ]
        
        estimated_monthly_expenses = 3000  # افتراضي
        emergency_fund_target = estimated_monthly_expenses * 6
        
        return {
            "recommendations": recommendations,
            "analysis": {
                "recommended_amount": emergency_fund_target,
                "monthly_target": emergency_fund_target / 12,
                "importance_score": 10
            }
        }

    async def _get_general_financial_advice(self) -> Dict[str, Any]:
        """نصائح مالية عامة"""
        
        recommendations = [
            "ضع أهدافاً مالية واضحة وقابلة للقياس",
            "ثقف نفسك مالياً بقراءة الكتب والمقالات المالية",
            "راجع وضعك المالي بانتظام",
            "لا تستثمر في شيء لا تفهمه",
            "ابدأ مبكراً واستفد من قوة التراكب"
        ]
        
        return {
            "recommendations": recommendations,
            "financial_principles": [
                "اعيش بأقل من دخلي",
                "ادخر أولاً ثم أنفق",
                "نوع استثماراتي",
                "تجنب الديون عالية الفائدة",
                "استثمر في تعليمي المالي"
            ]
        }

# إنشاء مثيل عام
financial_advisor = SmartFinancialAdvisor()

async def get_financial_advisor() -> SmartFinancialAdvisor:
    """الحصول على المستشار المالي"""
    return financial_advisor

if __name__ == "__main__":
    async def test_financial_advisor():
        """اختبار المستشار المالي"""
        print("💰 اختبار المستشار المالي الذكي")
        print("=" * 50)
        
        advisor = await get_financial_advisor()
        await advisor.initialize()
        
        # إضافة هدف مالي
        print("\n🎯 إضافة هدف مالي للتقاعد")
        goal_result = await advisor.add_financial_goal(
            name="التقاعد المريح",
            target_amount=1000000,
            target_date=datetime.now() + timedelta(days=365*25),
            goal_type="retirement",
            monthly_contribution=2000,
            risk_tolerance=RiskTolerance.MODERATE
        )
        print(f"✅ {goal_result.get('message', 'تم إضافة الهدف')}")
        
        # إضافة استثمار
        print("\n📈 إضافة استثمار في الأسهم")
        investment_result = await advisor.add_investment(
            name="أسهم أبل",
            symbol="AAPL",
            investment_type=InvestmentType.STOCKS,
            quantity=10,
            purchase_price=150.0
        )
        print(f"✅ {investment_result.get('message', 'تم إضافة الاستثمار')}")
        
        # تحليل المحفظة
        print("\n📊 تحليل المحفظة")
        portfolio_analysis = await advisor.get_portfolio_analysis()
        if "portfolio_analysis" in portfolio_analysis:
            summary = portfolio_analysis["portfolio_analysis"]["summary"]
            print(f"💼 إجمالي قيمة المحفظة: ${summary['total_value']:,.2f}")
            print(f"📈 إجمالي العائد: ${summary['total_return']:,.2f} ({summary['return_percentage']:.1f}%)")
        
        # نصائح مخصصة
        print("\n💡 نصائح مخصصة")
        advice_queries = [
            "نصائح الاستثمار للمبتدئين",
            "كيف أخطط للتقاعد؟",
            "ما هو أفضل صندوق طوارئ؟"
        ]
        
        for query in advice_queries:
            advice = await advisor.get_personalized_advice(query)
            recommendations = advice.get("advice", {}).get("recommendations", [])
            print(f"\n🤔 '{query}':")
            for i, rec in enumerate(recommendations[:2], 1):
                print(f"   {i}. {rec}")
    
    asyncio.run(test_financial_advisor())
