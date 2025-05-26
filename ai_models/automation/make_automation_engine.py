
"""
محرك أتمتة متقدم لموقع Make.com
Advanced Make.com Automation Engine
"""

import time
import json
from typing import Dict, List, Any
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
import requests

class MakeAutomationEngine:
    """
    محرك أتمتة متقدم لإنشاء workflows في Make.com
    """
    
    def __init__(self):
        self.driver = None
        self.wait = None
        self.logged_in = False
        self.current_scenario = None
        
    def setup_browser(self):
        """إعداد المتصفح"""
        options = webdriver.ChromeOptions()
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        self.driver = webdriver.Chrome(options=options)
        self.wait = WebDriverWait(self.driver, 20)
        
    def auto_login_make(self, email: str, password: str):
        """تسجيل دخول تلقائي إلى Make.com"""
        try:
            self.driver.get("https://make.com/login")
            
            # إدخال البريد الإلكتروني
            email_field = self.wait.until(
                EC.presence_of_element_located((By.NAME, "email"))
            )
            email_field.send_keys(email)
            
            # إدخال كلمة المرور
            password_field = self.driver.find_element(By.NAME, "password")
            password_field.send_keys(password)
            
            # النقر على زر تسجيل الدخول
            login_button = self.driver.find_element(By.CSS_SELECTOR, "button[type='submit']")
            login_button.click()
            
            # انتظار تحميل لوحة التحكم
            self.wait.until(
                EC.presence_of_element_located((By.CLASS_NAME, "dashboard"))
            )
            
            self.logged_in = True
            return True
            
        except Exception as e:
            print(f"خطأ في تسجيل الدخول: {e}")
            return False
    
    def create_new_scenario(self, scenario_name: str, description: str = ""):
        """إنشاء سيناريو جديد"""
        try:
            # النقر على إنشاء سيناريو جديد
            create_button = self.wait.until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Create')]"))
            )
            create_button.click()
            
            # اختيار "Scenario" من القائمة
            scenario_option = self.wait.until(
                EC.element_to_be_clickable((By.XPATH, "//span[contains(text(), 'Scenario')]"))
            )
            scenario_option.click()
            
            # إدخال اسم السيناريو
            name_field = self.wait.until(
                EC.presence_of_element_located((By.NAME, "name"))
            )
            name_field.clear()
            name_field.send_keys(scenario_name)
            
            # إدخال الوصف إذا كان موجوداً
            if description:
                desc_field = self.driver.find_element(By.NAME, "description")
                desc_field.send_keys(description)
            
            # النقر على إنشاء
            create_final_button = self.driver.find_element(
                By.XPATH, "//button[contains(text(), 'Create scenario')]"
            )
            create_final_button.click()
            
            # انتظار تحميل محرر السيناريو
            self.wait.until(
                EC.presence_of_element_located((By.CLASS_NAME, "scenario-editor"))
            )
            
            self.current_scenario = scenario_name
            return True
            
        except Exception as e:
            print(f"خطأ في إنشاء السيناريو: {e}")
            return False
    
    def add_trigger_module(self, app_name: str, trigger_type: str, config: Dict):
        """إضافة وحدة مشغل (Trigger)"""
        try:
            # النقر على أيقونة إضافة وحدة
            add_module_button = self.wait.until(
                EC.element_to_be_clickable((By.CLASS_NAME, "add-module-btn"))
            )
            add_module_button.click()
            
            # البحث عن التطبيق
            search_field = self.wait.until(
                EC.presence_of_element_located((By.PLACEHOLDER, "Search for an app"))
            )
            search_field.send_keys(app_name)
            
            # اختيار التطبيق من النتائج
            app_result = self.wait.until(
                EC.element_to_be_clickable((By.XPATH, f"//div[contains(text(), '{app_name}')]"))
            )
            app_result.click()
            
            # اختيار نوع المشغل
            trigger_option = self.wait.until(
                EC.element_to_be_clickable((By.XPATH, f"//span[contains(text(), '{trigger_type}')]"))
            )
            trigger_option.click()
            
            # ملء إعدادات الوحدة
            self._fill_module_config(config)
            
            # حفظ الوحدة
            save_button = self.driver.find_element(By.XPATH, "//button[contains(text(), 'OK')]")
            save_button.click()
            
            return True
            
        except Exception as e:
            print(f"خطأ في إضافة وحدة المشغل: {e}")
            return False
    
    def add_action_module(self, app_name: str, action_type: str, config: Dict):
        """إضافة وحدة إجراء (Action)"""
        try:
            # النقر على أيقونة إضافة وحدة
            add_module_button = self.wait.until(
                EC.element_to_be_clickable((By.CLASS_NAME, "add-module-btn"))
            )
            add_module_button.click()
            
            # البحث والاختيار مثل المشغل
            search_field = self.wait.until(
                EC.presence_of_element_located((By.PLACEHOLDER, "Search for an app"))
            )
            search_field.send_keys(app_name)
            
            app_result = self.wait.until(
                EC.element_to_be_clickable((By.XPATH, f"//div[contains(text(), '{app_name}')]"))
            )
            app_result.click()
            
            # اختيار نوع الإجراء
            action_option = self.wait.until(
                EC.element_to_be_clickable((By.XPATH, f"//span[contains(text(), '{action_type}')]"))
            )
            action_option.click()
            
            # ملء الإعدادات
            self._fill_module_config(config)
            
            # حفظ
            save_button = self.driver.find_element(By.XPATH, "//button[contains(text(), 'OK')]")
            save_button.click()
            
            return True
            
        except Exception as e:
            print(f"خطأ في إضافة وحدة الإجراء: {e}")
            return False
    
    def _fill_module_config(self, config: Dict):
        """ملء إعدادات الوحدة"""
        for field_name, value in config.items():
            try:
                # البحث عن الحقل
                field = self.driver.find_element(
                    By.XPATH, f"//input[@name='{field_name}'] | //textarea[@name='{field_name}']"
                )
                field.clear()
                field.send_keys(str(value))
                
            except:
                # محاولة بطرق أخرى إذا فشلت الطريقة الأولى
                try:
                    field = self.driver.find_element(
                        By.XPATH, f"//label[contains(text(), '{field_name}')]/following-sibling::*//input"
                    )
                    field.clear()
                    field.send_keys(str(value))
                except:
                    print(f"لم يتم العثور على الحقل: {field_name}")
    
    def setup_api_connections(self, connections: List[Dict]):
        """إعداد اتصالات API"""
        for connection in connections:
            try:
                app_name = connection['app']
                api_key = connection['api_key']
                
                # الذهاب لإعدادات الاتصالات
                self.driver.get("https://make.com/connections")
                
                # إضافة اتصال جديد
                add_connection_btn = self.wait.until(
                    EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Add')]"))
                )
                add_connection_btn.click()
                
                # البحث عن التطبيق
                search_app = self.wait.until(
                    EC.presence_of_element_located((By.PLACEHOLDER, "Search"))
                )
                search_app.send_keys(app_name)
                
                # اختيار التطبيق
                app_option = self.wait.until(
                    EC.element_to_be_clickable((By.XPATH, f"//div[contains(text(), '{app_name}')]"))
                )
                app_option.click()
                
                # إدخال مفتاح API
                api_field = self.wait.until(
                    EC.presence_of_element_located((By.NAME, "api_key"))
                )
                api_field.send_keys(api_key)
                
                # حفظ الاتصال
                save_connection = self.driver.find_element(
                    By.XPATH, "//button[contains(text(), 'Save')]"
                )
                save_connection.click()
                
                time.sleep(2)  # انتظار قصير
                
            except Exception as e:
                print(f"خطأ في إعداد اتصال {connection['app']}: {e}")
    
    def create_complete_workflow(self, workflow_config: Dict):
        """إنشاء workflow كامل بناءً على الإعدادات"""
        try:
            # إنشاء السيناريو
            success = self.create_new_scenario(
                workflow_config['name'], 
                workflow_config.get('description', '')
            )
            
            if not success:
                return False
            
            # إضافة المشغل
            trigger = workflow_config['trigger']
            success = self.add_trigger_module(
                trigger['app'],
                trigger['type'],
                trigger['config']
            )
            
            if not success:
                return False
            
            # إضافة الإجراءات
            for action in workflow_config['actions']:
                success = self.add_action_module(
                    action['app'],
                    action['type'],
                    action['config']
                )
                
                if not success:
                    print(f"فشل في إضافة إجراء: {action['app']}")
                    return False
            
            # تشغيل اختبار
            self.test_scenario()
            
            # تشغيل السيناريو
            self.activate_scenario()
            
            return True
            
        except Exception as e:
            print(f"خطأ في إنشاء workflow: {e}")
            return False
    
    def test_scenario(self):
        """اختبار السيناريو"""
        try:
            test_button = self.wait.until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Run once')]"))
            )
            test_button.click()
            
            # انتظار النتائج
            time.sleep(10)
            
            return True
            
        except Exception as e:
            print(f"خطأ في اختبار السيناريو: {e}")
            return False
    
    def activate_scenario(self):
        """تفعيل السيناريو"""
        try:
            activate_button = self.wait.until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Turn on')]"))
            )
            activate_button.click()
            
            return True
            
        except Exception as e:
            print(f"خطأ في تفعيل السيناريو: {e}")
            return False
    
    def close(self):
        """إغلاق المتصفح"""
        if self.driver:
            self.driver.quit()

# مثال على الاستخدام
def create_sample_workflow():
    """مثال على إنشاء workflow معقد"""
    
    automation = MakeAutomationEngine()
    automation.setup_browser()
    
    # تسجيل الدخول
    automation.auto_login_make("your_email@gmail.com", "your_password")
    
    # إعداد الاتصالات
    connections = [
        {
            'app': 'Gmail',
            'api_key': 'your_gmail_api_key'
        },
        {
            'app': 'Slack',
            'api_key': 'your_slack_api_key'
        },
        {
            'app': 'Trello',
            'api_key': 'your_trello_api_key'
        }
    ]
    
    automation.setup_api_connections(connections)
    
    # إعداد workflow
    workflow_config = {
        'name': 'معالج البريد الإلكتروني التلقائي',
        'description': 'معالجة الرسائل الواردة وإرسال إشعارات',
        'trigger': {
            'app': 'Gmail',
            'type': 'Watch emails',
            'config': {
                'folder': 'INBOX',
                'label': 'Important'
            }
        },
        'actions': [
            {
                'app': 'OpenAI',
                'type': 'Create completion',
                'config': {
                    'prompt': 'لخص هذا البريد الإلكتروني: {{Gmail.subject}}',
                    'max_tokens': 100
                }
            },
            {
                'app': 'Slack',
                'type': 'Send message',
                'config': {
                    'channel': '#emails',
                    'message': 'رسالة جديدة: {{OpenAI.summary}}'
                }
            },
            {
                'app': 'Trello',
                'type': 'Create card',
                'config': {
                    'board': 'المهام',
                    'list': 'الرسائل الجديدة',
                    'title': '{{Gmail.subject}}',
                    'description': '{{OpenAI.summary}}'
                }
            }
        ]
    }
    
    # إنشاء workflow
    success = automation.create_complete_workflow(workflow_config)
    
    if success:
        print("✅ تم إنشاء workflow بنجاح!")
    else:
        print("❌ فشل في إنشاء workflow")
    
    # إغلاق المتصفح
    automation.close()

if __name__ == "__main__":
    create_sample_workflow()
