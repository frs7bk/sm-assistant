
"""
خبير التصميم والفيديو إديت الذكي مع فهم برامج أدوبي
Intelligent Design and Video Editing Expert with Adobe Software Understanding
"""

import cv2
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import json
import os
from datetime import datetime
import colorsys
from PIL import Image, ImageEnhance, ImageFilter
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

class CreativeDesignExpert:
    """
    خبير ذكي في التصميم والفيديو إديت مع فهم عميق لبرامج أدوبي
    """
    
    def __init__(self):
        print("🎨 تهيئة خبير التصميم والفيديو إديت الذكي...")
        
        # قاعدة بيانات برامج أدوبي والميزات
        self.adobe_software = {
            'photoshop': {
                'features': [
                    'layers', 'masks', 'blending_modes', 'filters', 'adjustments',
                    'brushes', 'selection_tools', 'retouching', 'compositing'
                ],
                'shortcuts': {
                    'ctrl+j': 'نسخ الطبقة',
                    'ctrl+shift+alt+e': 'دمج جميع الطبقات المرئية',
                    'b': 'أداة الفرشاة',
                    'v': 'أداة التحديد',
                    'ctrl+t': 'التحويل الحر'
                },
                'workflows': {
                    'photo_retouching': [
                        'إنشاء طبقة جديدة للتعديلات',
                        'استخدام أداة Healing Brush للعيوب',
                        'تطبيق مرشحات التنعيم',
                        'ضبط الألوان والتباين',
                        'حفظ النسخة النهائية'
                    ],
                    'photo_manipulation': [
                        'فصل العناصر باستخدام الأقنعة',
                        'تطبيق تأثيرات الإضاءة',
                        'دمج عدة صور',
                        'إضافة التأثيرات الخاصة',
                        'التشطيب النهائي'
                    ]
                }
            },
            'illustrator': {
                'features': [
                    'vector_graphics', 'pen_tool', 'typography', 'gradients',
                    'pathfinder', 'appearance_panel', 'symbols', 'brushes'
                ],
                'shortcuts': {
                    'p': 'أداة القلم',
                    'a': 'أداة التحديد المباشر',
                    'v': 'أداة التحديد',
                    't': 'أداة النص',
                    'ctrl+g': 'تجميع العناصر'
                },
                'workflows': {
                    'logo_design': [
                        'إنشاء مفهوم التصميم والرسم الأولي',
                        'استخدام أداة القلم لرسم الأشكال',
                        'تطبيق الألوان والتدرجات',
                        'إضافة النصوص والخطوط',
                        'تحسين التفاصيل والتشطيب'
                    ],
                    'icon_design': [
                        'رسم الشكل الأساسي',
                        'تطبيق الألوان المناسبة',
                        'إضافة الظلال والإضاءة',
                        'تحسين الوضوح للأحجام المختلفة',
                        'إنشاء متغيرات مختلفة'
                    ]
                }
            },
            'after_effects': {
                'features': [
                    'keyframes', 'expressions', 'effects', 'compositions',
                    'masks', 'tracking', '3d_layers', 'particle_systems'
                ],
                'shortcuts': {
                    'ctrl+k': 'إعدادات التركيب',
                    'u': 'إظهار الخصائص المتحركة',
                    'p': 'إظهار خاصية الموضع',
                    's': 'إظهار خاصية المقياس',
                    'ctrl+d': 'نسخ الطبقة'
                },
                'workflows': {
                    'motion_graphics': [
                        'إنشاء التركيب وإعداد المخطط الزمني',
                        'إضافة العناصر النصية والجرافيكية',
                        'تطبيق الحركات والانتقالات',
                        'إضافة التأثيرات البصرية',
                        'التصدير بالجودة المطلوبة'
                    ],
                    'visual_effects': [
                        'تحليل اللقطة والتخطيط للتأثير',
                        'إنشاء الماسكات والتتبع',
                        'تطبيق التأثيرات والتركيب',
                        'ضبط الألوان والإضاءة',
                        'التصدير النهائي'
                    ]
                }
            },
            'premiere_pro': {
                'features': [
                    'timeline_editing', 'color_correction', 'audio_mixing',
                    'transitions', 'effects', 'multicam', 'proxy_workflows'
                ],
                'shortcuts': {
                    'c': 'أداة القطع',
                    'v': 'أداة التحديد',
                    'ctrl+m': 'التصدير',
                    'ctrl+k': 'قطع اللقطة',
                    'ctrl+shift+;': 'إضافة علامة'
                },
                'workflows': {
                    'video_editing': [
                        'استيراد المواد وتنظيمها',
                        'إنشاء التسلسل الزمني',
                        'قطع وترتيب اللقطات',
                        'إضافة الانتقالات والتأثيرات',
                        'تصحيح الألوان والصوت',
                        'التصدير النهائي'
                    ]
                }
            }
        }
        
        # قواعد التصميم والمبادئ
        self.design_principles = {
            'color_theory': {
                'complementary': 'الألوان المتقابلة في دائرة الألوان',
                'analogous': 'الألوان المتجاورة في دائرة الألوان',
                'triadic': 'ثلاثة ألوان متباعدة بالتساوي',
                'monochromatic': 'درجات مختلفة من نفس اللون'
            },
            'composition': {
                'rule_of_thirds': 'قانون الأثلاث للتركيب المتوازن',
                'golden_ratio': 'النسبة الذهبية للجمال الطبيعي',
                'leading_lines': 'الخطوط الموجهة لتوجيه النظر',
                'symmetry': 'التماثل للتوازن البصري'
            },
            'typography': {
                'hierarchy': 'التدرج الهرمي للنصوص',
                'contrast': 'التباين للوضوح',
                'alignment': 'المحاذاة للتنظيم',
                'proximity': 'القرب للربط المنطقي'
            }
        }
        
        # إعدادات التحليل
        self.analysis_capabilities = {
            'color_analysis': True,
            'composition_analysis': True,
            'style_recognition': True,
            'quality_assessment': True,
            'trend_analysis': True
        }
        
    def analyze_design_project(self, image_path: str) -> Dict[str, Any]:
        """تحليل شامل لمشروع تصميم"""
        if not os.path.exists(image_path):
            return {"error": "الملف غير موجود"}
        
        print(f"🎨 تحليل التصميم: {image_path}")
        
        # تحميل الصورة
        image = cv2.imread(image_path)
        if image is None:
            return {"error": "لا يمكن قراءة الصورة"}
        
        analysis_result = {
            'file_info': {
                'path': image_path,
                'dimensions': (image.shape[1], image.shape[0]),
                'analysis_time': datetime.now().isoformat()
            },
            'color_analysis': self._analyze_colors(image),
            'composition_analysis': self._analyze_composition(image),
            'style_analysis': self._analyze_style(image),
            'quality_assessment': self._assess_design_quality(image),
            'improvement_suggestions': [],
            'software_recommendations': {},
            'workflow_suggestions': []
        }
        
        # إنتاج الاقتراحات
        analysis_result['improvement_suggestions'] = self._generate_improvement_suggestions(analysis_result)
        analysis_result['software_recommendations'] = self._recommend_adobe_software(analysis_result)
        analysis_result['workflow_suggestions'] = self._suggest_workflows(analysis_result)
        
        return analysis_result
    
    def _analyze_colors(self, image: np.ndarray) -> Dict[str, Any]:
        """تحليل الألوان المتقدم"""
        # تحويل إلى RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # استخراج الألوان الرئيسية
        pixels = rgb_image.reshape(-1, 3)
        kmeans = KMeans(n_clusters=5, random_state=42)
        kmeans.fit(pixels)
        
        dominant_colors = kmeans.cluster_centers_.astype(int)
        color_percentages = np.bincount(kmeans.labels_) / len(kmeans.labels_)
        
        # تحليل الألوان
        color_analysis = {
            'dominant_colors': [],
            'color_harmony': self._analyze_color_harmony(dominant_colors),
            'color_temperature': self._analyze_color_temperature(dominant_colors),
            'color_contrast': self._analyze_color_contrast(image),
            'color_scheme_type': self._classify_color_scheme(dominant_colors)
        }
        
        # تفاصيل الألوان المهيمنة
        for i, (color, percentage) in enumerate(zip(dominant_colors, color_percentages)):
            color_info = {
                'rgb': color.tolist(),
                'hex': f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}",
                'percentage': float(percentage),
                'hsv': self._rgb_to_hsv(color),
                'color_name': self._get_color_name(color)
            }
            color_analysis['dominant_colors'].append(color_info)
        
        return color_analysis
    
    def _analyze_composition(self, image: np.ndarray) -> Dict[str, Any]:
        """تحليل التركيب والتخطيط"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        composition_analysis = {
            'rule_of_thirds_score': self._analyze_rule_of_thirds(gray),
            'balance_score': self._analyze_visual_balance(gray),
            'focal_points': self._detect_focal_points(gray),
            'symmetry_score': self._analyze_symmetry(gray),
            'leading_lines': self._detect_leading_lines(gray),
            'negative_space': self._analyze_negative_space(gray)
        }
        
        return composition_analysis
    
    def _analyze_style(self, image: np.ndarray) -> Dict[str, Any]:
        """تحليل الطراز والأسلوب"""
        style_analysis = {
            'design_style': self._classify_design_style(image),
            'complexity_level': self._measure_complexity(image),
            'modernism_score': self._assess_modernism(image),
            'minimalism_score': self._assess_minimalism(image),
            'artistic_influence': self._detect_artistic_influence(image)
        }
        
        return style_analysis
    
    def _assess_design_quality(self, image: np.ndarray) -> Dict[str, Any]:
        """تقييم جودة التصميم"""
        quality_assessment = {
            'overall_score': 0.0,
            'technical_quality': self._assess_technical_quality(image),
            'aesthetic_appeal': self._assess_aesthetic_appeal(image),
            'usability_score': self._assess_usability(image),
            'innovation_score': self._assess_innovation(image),
            'professional_level': 'متوسط'
        }
        
        # حساب النقاط الإجمالية
        scores = [
            quality_assessment['technical_quality'],
            quality_assessment['aesthetic_appeal'],
            quality_assessment['usability_score'],
            quality_assessment['innovation_score']
        ]
        
        quality_assessment['overall_score'] = np.mean(scores)
        
        # تحديد المستوى المهني
        if quality_assessment['overall_score'] >= 0.9:
            quality_assessment['professional_level'] = 'خبير'
        elif quality_assessment['overall_score'] >= 0.7:
            quality_assessment['professional_level'] = 'متقدم'
        elif quality_assessment['overall_score'] >= 0.5:
            quality_assessment['professional_level'] = 'متوسط'
        else:
            quality_assessment['professional_level'] = 'مبتدئ'
        
        return quality_assessment
    
    def provide_adobe_guidance(self, software: str, task: str) -> Dict[str, Any]:
        """تقديم إرشادات لاستخدام برامج أدوبي"""
        if software not in self.adobe_software:
            return {"error": f"البرنامج {software} غير مدعوم"}
        
        software_info = self.adobe_software[software]
        
        guidance = {
            'software': software,
            'task': task,
            'recommended_workflow': [],
            'essential_shortcuts': {},
            'tips_and_tricks': [],
            'common_mistakes': [],
            'advanced_techniques': []
        }
        
        # البحث عن سير العمل المناسب
        for workflow_name, steps in software_info.get('workflows', {}).items():
            if task.lower() in workflow_name or workflow_name in task.lower():
                guidance['recommended_workflow'] = steps
                break
        
        # إضافة اختصارات مفيدة
        guidance['essential_shortcuts'] = dict(list(software_info['shortcuts'].items())[:5])
        
        # نصائح وحيل
        guidance['tips_and_tricks'] = self._get_software_tips(software, task)
        
        # أخطاء شائعة
        guidance['common_mistakes'] = self._get_common_mistakes(software)
        
        # تقنيات متقدمة
        guidance['advanced_techniques'] = self._get_advanced_techniques(software, task)
        
        return guidance
    
    def create_color_palette(self, base_color: str, palette_type: str = 'complementary') -> Dict[str, Any]:
        """إنشاء لوحة ألوان متناسقة"""
        try:
            # تحويل اللون الأساسي إلى RGB
            if base_color.startswith('#'):
                base_rgb = tuple(int(base_color[i:i+2], 16) for i in (1, 3, 5))
            else:
                return {"error": "صيغة اللون غير صحيحة"}
            
            # تحويل إلى HSV للتعامل مع الألوان
            base_hsv = colorsys.rgb_to_hsv(base_rgb[0]/255, base_rgb[1]/255, base_rgb[2]/255)
            
            palette = {
                'base_color': {
                    'rgb': base_rgb,
                    'hex': base_color,
                    'hsv': base_hsv
                },
                'palette_type': palette_type,
                'colors': [],
                'usage_suggestions': {}
            }
            
            # إنتاج الألوان حسب النوع
            if palette_type == 'complementary':
                colors = self._generate_complementary_palette(base_hsv)
            elif palette_type == 'analogous':
                colors = self._generate_analogous_palette(base_hsv)
            elif palette_type == 'triadic':
                colors = self._generate_triadic_palette(base_hsv)
            elif palette_type == 'monochromatic':
                colors = self._generate_monochromatic_palette(base_hsv)
            else:
                colors = self._generate_complementary_palette(base_hsv)
            
            # تحويل الألوان إلى صيغ مختلفة
            for i, hsv_color in enumerate(colors):
                rgb = colorsys.hsv_to_rgb(*hsv_color)
                rgb_int = tuple(int(c * 255) for c in rgb)
                hex_color = f"#{rgb_int[0]:02x}{rgb_int[1]:02x}{rgb_int[2]:02x}"
                
                color_info = {
                    'index': i,
                    'rgb': rgb_int,
                    'hex': hex_color,
                    'hsv': hsv_color,
                    'name': self._get_color_name(rgb_int)
                }
                palette['colors'].append(color_info)
            
            # اقتراحات الاستخدام
            palette['usage_suggestions'] = self._generate_color_usage_suggestions(palette_type)
            
            return palette
            
        except Exception as e:
            return {"error": f"خطأ في إنشاء اللوحة: {str(e)}"}
    
    def analyze_video_project(self, video_path: str) -> Dict[str, Any]:
        """تحليل مشروع فيديو"""
        if not os.path.exists(video_path):
            return {"error": "ملف الفيديو غير موجود"}
        
        print(f"🎬 تحليل مشروع الفيديو: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": "لا يمكن فتح ملف الفيديو"}
        
        # معلومات الفيديو
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        video_analysis = {
            'video_info': {
                'duration': duration,
                'fps': fps,
                'resolution': (width, height),
                'frame_count': frame_count
            },
            'visual_analysis': {},
            'technical_assessment': {},
            'editing_suggestions': [],
            'software_recommendations': {},
            'workflow_optimization': []
        }
        
        # تحليل عينة من الإطارات
        sample_frames = []
        frame_step = max(1, frame_count // 10)  # 10 إطارات عينة
        
        for i in range(0, frame_count, frame_step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                sample_frames.append(frame)
            if len(sample_frames) >= 10:
                break
        
        cap.release()
        
        if sample_frames:
            # تحليل بصري
            video_analysis['visual_analysis'] = self._analyze_video_visuals(sample_frames)
            
            # تقييم تقني
            video_analysis['technical_assessment'] = self._assess_video_technical_quality(
                sample_frames, fps, (width, height)
            )
            
            # اقتراحات التحرير
            video_analysis['editing_suggestions'] = self._generate_video_editing_suggestions(
                video_analysis
            )
            
            # توصيات البرامج
            video_analysis['software_recommendations'] = self._recommend_video_software(
                video_analysis
            )
        
        return video_analysis
    
    # دوال مساعدة للتحليل المتقدم
    def _analyze_color_harmony(self, colors: np.ndarray) -> float:
        """تحليل انسجام الألوان"""
        if len(colors) < 2:
            return 1.0
        
        # حساب المسافات بين الألوان في مساحة HSV
        harmony_score = 0.0
        color_count = 0
        
        for i in range(len(colors)):
            for j in range(i + 1, len(colors)):
                hsv1 = self._rgb_to_hsv(colors[i])
                hsv2 = self._rgb_to_hsv(colors[j])
                
                # حساب الفرق في درجة اللون
                hue_diff = abs(hsv1[0] - hsv2[0])
                hue_diff = min(hue_diff, 360 - hue_diff)  # أقصر مسافة في الدائرة
                
                # تقييم الانسجام
                if hue_diff < 30 or hue_diff > 150:  # متشابه أو متقابل
                    harmony_score += 1.0
                elif 60 <= hue_diff <= 120:  # متناسق
                    harmony_score += 0.8
                else:
                    harmony_score += 0.3
                
                color_count += 1
        
        return harmony_score / color_count if color_count > 0 else 1.0
    
    def _analyze_color_temperature(self, colors: np.ndarray) -> str:
        """تحليل درجة حرارة الألوان"""
        warm_count = 0
        cool_count = 0
        
        for color in colors:
            hsv = self._rgb_to_hsv(color)
            hue = hsv[0]
            
            # الألوان الدافئة: أحمر، برتقالي، أصفر (0-60, 300-360)
            if (0 <= hue <= 60) or (300 <= hue <= 360):
                warm_count += 1
            # الألوان الباردة: أزرق، أخضر، بنفسجي (120-300)
            elif 120 <= hue <= 300:
                cool_count += 1
        
        if warm_count > cool_count:
            return 'دافئة'
        elif cool_count > warm_count:
            return 'باردة'
        else:
            return 'متوازنة'
    
    def _rgb_to_hsv(self, rgb: np.ndarray) -> Tuple[float, float, float]:
        """تحويل RGB إلى HSV"""
        r, g, b = rgb / 255.0
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        return (h * 360, s, v)
    
    def _get_color_name(self, rgb: Tuple[int, int, int]) -> str:
        """الحصول على اسم اللون"""
        r, g, b = rgb
        
        # أسماء ألوان أساسية باللغة العربية
        if r > 200 and g < 100 and b < 100:
            return "أحمر"
        elif r < 100 and g > 200 and b < 100:
            return "أخضر"
        elif r < 100 and g < 100 and b > 200:
            return "أزرق"
        elif r > 200 and g > 200 and b < 100:
            return "أصفر"
        elif r > 200 and g < 100 and b > 200:
            return "بنفسجي"
        elif r < 100 and g > 200 and b > 200:
            return "سماوي"
        elif r > 200 and g > 200 and b > 200:
            return "أبيض"
        elif r < 100 and g < 100 and b < 100:
            return "أسود"
        else:
            return "ملون"
    
    # باقي الدوال المساعدة سيتم تطويرها حسب الحاجة...
    def _analyze_rule_of_thirds(self, gray: np.ndarray) -> float:
        """تحليل قانون الأثلاث"""
        height, width = gray.shape
        
        # خطوط الأثلاث
        third_h = height // 3
        third_w = width // 3
        
        # نقاط التقاطع
        intersections = [
            (third_w, third_h), (2 * third_w, third_h),
            (third_w, 2 * third_h), (2 * third_w, 2 * third_h)
        ]
        
        # تحليل كثافة النقاط حول التقاطعات
        score = 0.0
        for x, y in intersections:
            region = gray[max(0, y-20):min(height, y+20), 
                         max(0, x-20):min(width, x+20)]
            if region.size > 0:
                intensity = np.std(region)  # التنوع في المنطقة
                score += min(1.0, intensity / 50.0)
        
        return score / len(intersections)
    
    def _generate_improvement_suggestions(self, analysis: Dict) -> List[str]:
        """توليد اقتراحات التحسين"""
        suggestions = []
        
        # اقتراحات الألوان
        color_harmony = analysis['color_analysis'].get('color_harmony', 0)
        if color_harmony < 0.6:
            suggestions.append("🎨 حسن انسجام الألوان باستخدام دائرة الألوان")
        
        # اقتراحات التركيب
        composition_score = analysis['composition_analysis'].get('rule_of_thirds_score', 0)
        if composition_score < 0.5:
            suggestions.append("📐 طبق قانون الأثلاث لتحسين التركيب")
        
        # اقتراحات الجودة
        overall_score = analysis['quality_assessment'].get('overall_score', 0)
        if overall_score < 0.7:
            suggestions.append("⚡ حسن الجودة التقنية والدقة")
        
        # اقتراحات عامة
        suggestions.extend([
            "🖼️ أضف المزيد من التباين لتحسين الوضوح",
            "✨ استخدم مساحات بيضاء أكثر للتوازن",
            "🔄 جرب تخطيطات مختلفة لتحسين التأثير"
        ])
        
        return suggestions[:5]
    
    def _recommend_adobe_software(self, analysis: Dict) -> Dict[str, str]:
        """توصية برامج أدوبي المناسبة"""
        recommendations = {}
        
        # بناءً على نوع التصميم
        style = analysis['style_analysis'].get('design_style', 'general')
        
        if 'photo' in style.lower():
            recommendations['primary'] = 'photoshop'
            recommendations['reason'] = 'للتعديل على الصور والريتوش'
        elif 'vector' in style.lower() or 'logo' in style.lower():
            recommendations['primary'] = 'illustrator'
            recommendations['reason'] = 'للتصميم المتجه والشعارات'
        elif 'motion' in style.lower():
            recommendations['primary'] = 'after_effects'
            recommendations['reason'] = 'للموشن جرافيك والتأثيرات'
        else:
            recommendations['primary'] = 'photoshop'
            recommendations['reason'] = 'للتصميم العام والتعديل'
        
        # برامج مساعدة
        recommendations['secondary'] = ['illustrator', 'after_effects']
        
        return recommendations
    
    # دوال إضافية للتحليل المتقدم...
    def _classify_design_style(self, image: np.ndarray) -> str:
        """تصنيف طراز التصميم"""
        return "حديث"  # تحليل مبسط
    
    def _measure_complexity(self, image: np.ndarray) -> float:
        """قياس مستوى التعقيد"""
        return 0.6  # تحليل مبسط
    
    def _assess_technical_quality(self, image: np.ndarray) -> float:
        """تقييم الجودة التقنية"""
        return 0.8  # تحليل مبسط
    
    def _assess_aesthetic_appeal(self, image: np.ndarray) -> float:
        """تقييم الجاذبية الجمالية"""
        return 0.7  # تحليل مبسط
    
    def _assess_usability(self, image: np.ndarray) -> float:
        """تقييم سهولة الاستخدام"""
        return 0.8  # تحليل مبسط
    
    def _assess_innovation(self, image: np.ndarray) -> float:
        """تقييم الإبداع والابتكار"""
        return 0.6  # تحليل مبسط

# مثال على الاستخدام
if __name__ == "__main__":
    expert = CreativeDesignExpert()
    
    # اختبار إنشاء لوحة ألوان
    palette = expert.create_color_palette("#3498db", "complementary")
    print("🎨 لوحة الألوان:")
    for color in palette.get('colors', [])[:3]:
        print(f"  {color['hex']} - {color['name']}")
    
    # اختبار إرشادات أدوبي
    guidance = expert.provide_adobe_guidance("photoshop", "photo retouching")
    print(f"\n📚 إرشادات Photoshop:")
    print(f"سير العمل: {guidance['recommended_workflow'][:3]}")
    
    print(f"\n⌨️ اختصارات مفيدة:")
    for shortcut, description in list(guidance['essential_shortcuts'].items())[:3]:
        print(f"  {shortcut}: {description}")
