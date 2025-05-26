
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
نماذج التعلم العميق للتنبؤات المتقدمة
شبكات عصبية عميقة للبيانات الضخمة
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import logging
from typing import Dict, List, Any, Optional, Tuple
import asyncio
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib

class AdvancedNeuralNetwork(nn.Module):
    """شبكة عصبية متقدمة للتنبؤ"""
    
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int, dropout_rate: float = 0.3):
        super(AdvancedNeuralNetwork, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # طبقات مخفية
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # طبقة الإخراج
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
        
        # تهيئة الأوزان
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """تهيئة الأوزان"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)
    
    def forward(self, x):
        return self.network(x)

class LSTMPredictor(nn.Module):
    """نموذج LSTM للسلاسل الزمنية"""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int, dropout: float = 0.2):
        super(LSTMPredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])  # آخر إخراج
        out = self.fc(out)
        return out

class TransformerPredictor(nn.Module):
    """نموذج Transformer للتنبؤ المتقدم"""
    
    def __init__(self, input_size: int, d_model: int, nhead: int, num_layers: int, output_size: int):
        super(TransformerPredictor, self).__init__()
        
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1000, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.output_projection = nn.Linear(d_model, output_size)
        
    def forward(self, x):
        seq_len = x.size(1)
        x = self.input_projection(x)
        x += self.pos_encoding[:seq_len, :].unsqueeze(0)
        
        x = self.transformer(x)
        x = x.mean(dim=1)  # تجميع التسلسل
        x = self.output_projection(x)
        return x

class DeepLearningPredictor:
    """معالج التعلم العميق المتقدم"""
    
    def __init__(self, device: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        
        self.logger.info(f"تم تهيئة معالج التعلم العميق على: {self.device}")
    
    async def prepare_data(self, data: pd.DataFrame, target_column: str, 
                          test_size: float = 0.2) -> Dict[str, Any]:
        """تحضير البيانات للتدريب"""
        try:
            # فصل الميزات والهدف
            X = data.drop(columns=[target_column])
            y = data[target_column]
            
            # معالجة الأعمدة الفئوية
            categorical_columns = X.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                    X[col] = self.encoders[col].fit_transform(X[col].astype(str))
                else:
                    X[col] = self.encoders[col].transform(X[col].astype(str))
            
            # تطبيع البيانات
            if 'X_scaler' not in self.scalers:
                self.scalers['X_scaler'] = StandardScaler()
                X_scaled = self.scalers['X_scaler'].fit_transform(X)
            else:
                X_scaled = self.scalers['X_scaler'].transform(X)
            
            # معالجة الهدف
            if y.dtype == 'object':
                if 'y_encoder' not in self.encoders:
                    self.encoders['y_encoder'] = LabelEncoder()
                    y_encoded = self.encoders['y_encoder'].fit_transform(y.astype(str))
                else:
                    y_encoded = self.encoders['y_encoder'].transform(y.astype(str))
                task_type = 'classification'
                output_size = len(np.unique(y_encoded))
            else:
                if 'y_scaler' not in self.scalers:
                    self.scalers['y_scaler'] = StandardScaler()
                    y_encoded = self.scalers['y_scaler'].fit_transform(y.values.reshape(-1, 1)).flatten()
                else:
                    y_encoded = self.scalers['y_scaler'].transform(y.values.reshape(-1, 1)).flatten()
                task_type = 'regression'
                output_size = 1
            
            # تقسيم البيانات
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_encoded, test_size=test_size, random_state=42
            )
            
            # تحويل إلى tensors
            X_train_tensor = torch.FloatTensor(X_train).to(self.device)
            X_test_tensor = torch.FloatTensor(X_test).to(self.device)
            y_train_tensor = torch.FloatTensor(y_train).to(self.device)
            y_test_tensor = torch.FloatTensor(y_test).to(self.device)
            
            if task_type == 'classification':
                y_train_tensor = y_train_tensor.long()
                y_test_tensor = y_test_tensor.long()
            
            return {
                'X_train': X_train_tensor,
                'X_test': X_test_tensor,
                'y_train': y_train_tensor,
                'y_test': y_test_tensor,
                'input_size': X_train.shape[1],
                'output_size': output_size,
                'task_type': task_type,
                'feature_names': list(X.columns)
            }
            
        except Exception as e:
            self.logger.error(f"خطأ في تحضير البيانات: {e}")
            return {}
    
    async def train_neural_network(self, data_info: Dict[str, Any], 
                                 model_name: str = "neural_net",
                                 hidden_sizes: List[int] = [128, 64, 32],
                                 epochs: int = 100,
                                 learning_rate: float = 0.001) -> Dict[str, Any]:
        """تدريب الشبكة العصبية"""
        try:
            # إنشاء النموذج
            model = AdvancedNeuralNetwork(
                input_size=data_info['input_size'],
                hidden_sizes=hidden_sizes,
                output_size=data_info['output_size']
            ).to(self.device)
            
            # تحديد دالة الخسارة والمحسن
            if data_info['task_type'] == 'classification':
                criterion = nn.CrossEntropyLoss()
            else:
                criterion = nn.MSELoss()
            
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
            
            # إعداد DataLoader
            train_dataset = TensorDataset(data_info['X_train'], data_info['y_train'])
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            
            # تدريب النموذج
            train_losses = []
            val_losses = []
            
            model.train()
            for epoch in range(epochs):
                epoch_loss = 0.0
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    
                    if data_info['task_type'] == 'classification':
                        loss = criterion(outputs, batch_y)
                    else:
                        loss = criterion(outputs.squeeze(), batch_y)
                    
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                
                # تقييم على بيانات الاختبار
                model.eval()
                with torch.no_grad():
                    test_outputs = model(data_info['X_test'])
                    if data_info['task_type'] == 'classification':
                        val_loss = criterion(test_outputs, data_info['y_test'])
                    else:
                        val_loss = criterion(test_outputs.squeeze(), data_info['y_test'])
                
                train_losses.append(epoch_loss / len(train_loader))
                val_losses.append(val_loss.item())
                
                scheduler.step(val_loss)
                model.train()
                
                if (epoch + 1) % 20 == 0:
                    self.logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")
            
            # حفظ النموذج
            self.models[model_name] = model
            
            # تقييم أداء النموذج
            model.eval()
            with torch.no_grad():
                train_pred = model(data_info['X_train'])
                test_pred = model(data_info['X_test'])
                
                if data_info['task_type'] == 'classification':
                    train_acc = accuracy_score(
                        data_info['y_train'].cpu().numpy(),
                        torch.argmax(train_pred, dim=1).cpu().numpy()
                    )
                    test_acc = accuracy_score(
                        data_info['y_test'].cpu().numpy(),
                        torch.argmax(test_pred, dim=1).cpu().numpy()
                    )
                    
                    results = {
                        'model_name': model_name,
                        'task_type': data_info['task_type'],
                        'train_accuracy': float(train_acc),
                        'test_accuracy': float(test_acc),
                        'train_losses': train_losses,
                        'val_losses': val_losses,
                        'epochs_trained': epochs
                    }
                else:
                    # تحويل التنبؤات إلى القيم الأصلية
                    if 'y_scaler' in self.scalers:
                        train_pred_original = self.scalers['y_scaler'].inverse_transform(
                            train_pred.squeeze().cpu().numpy().reshape(-1, 1)
                        ).flatten()
                        test_pred_original = self.scalers['y_scaler'].inverse_transform(
                            test_pred.squeeze().cpu().numpy().reshape(-1, 1)
                        ).flatten()
                        train_true_original = self.scalers['y_scaler'].inverse_transform(
                            data_info['y_train'].cpu().numpy().reshape(-1, 1)
                        ).flatten()
                        test_true_original = self.scalers['y_scaler'].inverse_transform(
                            data_info['y_test'].cpu().numpy().reshape(-1, 1)
                        ).flatten()
                    else:
                        train_pred_original = train_pred.squeeze().cpu().numpy()
                        test_pred_original = test_pred.squeeze().cpu().numpy()
                        train_true_original = data_info['y_train'].cpu().numpy()
                        test_true_original = data_info['y_test'].cpu().numpy()
                    
                    train_rmse = np.sqrt(mean_squared_error(train_true_original, train_pred_original))
                    test_rmse = np.sqrt(mean_squared_error(test_true_original, test_pred_original))
                    train_r2 = r2_score(train_true_original, train_pred_original)
                    test_r2 = r2_score(test_true_original, test_pred_original)
                    
                    results = {
                        'model_name': model_name,
                        'task_type': data_info['task_type'],
                        'train_rmse': float(train_rmse),
                        'test_rmse': float(test_rmse),
                        'train_r2': float(train_r2),
                        'test_r2': float(test_r2),
                        'train_losses': train_losses,
                        'val_losses': val_losses,
                        'epochs_trained': epochs
                    }
            
            return results
            
        except Exception as e:
            self.logger.error(f"خطأ في تدريب الشبكة العصبية: {e}")
            return {}
    
    async def train_lstm_model(self, time_series_data: np.ndarray, 
                              sequence_length: int = 10,
                              model_name: str = "lstm_model",
                              epochs: int = 50) -> Dict[str, Any]:
        """تدريب نموذج LSTM للسلاسل الزمنية"""
        try:
            # تحضير بيانات السلاسل الزمنية
            def create_sequences(data, seq_length):
                X, y = [], []
                for i in range(len(data) - seq_length):
                    X.append(data[i:(i + seq_length)])
                    y.append(data[i + seq_length])
                return np.array(X), np.array(y)
            
            # تطبيع البيانات
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(time_series_data.reshape(-1, 1)).flatten()
            self.scalers[f'{model_name}_scaler'] = scaler
            
            # إنشاء التسلسلات
            X, y = create_sequences(scaled_data, sequence_length)
            
            # تقسيم البيانات
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # تحويل إلى tensors
            X_train = torch.FloatTensor(X_train).unsqueeze(-1).to(self.device)
            X_test = torch.FloatTensor(X_test).unsqueeze(-1).to(self.device)
            y_train = torch.FloatTensor(y_train).to(self.device)
            y_test = torch.FloatTensor(y_test).to(self.device)
            
            # إنشاء النموذج
            model = LSTMPredictor(
                input_size=1,
                hidden_size=64,
                num_layers=2,
                output_size=1
            ).to(self.device)
            
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # تدريب النموذج
            train_losses = []
            val_losses = []
            
            model.train()
            for epoch in range(epochs):
                optimizer.zero_grad()
                outputs = model(X_train)
                loss = criterion(outputs.squeeze(), y_train)
                loss.backward()
                optimizer.step()
                
                # تقييم
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_test)
                    val_loss = criterion(val_outputs.squeeze(), y_test)
                
                train_losses.append(loss.item())
                val_losses.append(val_loss.item())
                model.train()
                
                if (epoch + 1) % 10 == 0:
                    self.logger.info(f"LSTM Epoch {epoch+1}/{epochs}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")
            
            # حفظ النموذج
            self.models[model_name] = model
            
            # تقييم الأداء
            model.eval()
            with torch.no_grad():
                train_pred = model(X_train).squeeze().cpu().numpy()
                test_pred = model(X_test).squeeze().cpu().numpy()
                
                # تحويل إلى القيم الأصلية
                train_pred_original = scaler.inverse_transform(train_pred.reshape(-1, 1)).flatten()
                test_pred_original = scaler.inverse_transform(test_pred.reshape(-1, 1)).flatten()
                train_true_original = scaler.inverse_transform(y_train.cpu().numpy().reshape(-1, 1)).flatten()
                test_true_original = scaler.inverse_transform(y_test.cpu().numpy().reshape(-1, 1)).flatten()
                
                train_rmse = np.sqrt(mean_squared_error(train_true_original, train_pred_original))
                test_rmse = np.sqrt(mean_squared_error(test_true_original, test_pred_original))
                train_r2 = r2_score(train_true_original, train_pred_original)
                test_r2 = r2_score(test_true_original, test_pred_original)
            
            return {
                'model_name': model_name,
                'model_type': 'LSTM',
                'sequence_length': sequence_length,
                'train_rmse': float(train_rmse),
                'test_rmse': float(test_rmse),
                'train_r2': float(train_r2),
                'test_r2': float(test_r2),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'epochs_trained': epochs
            }
            
        except Exception as e:
            self.logger.error(f"خطأ في تدريب LSTM: {e}")
            return {}
    
    async def predict(self, model_name: str, new_data: np.ndarray) -> np.ndarray:
        """التنبؤ باستخدام النموذج المدرب"""
        try:
            if model_name not in self.models:
                raise ValueError(f"النموذج {model_name} غير موجود")
            
            model = self.models[model_name]
            model.eval()
            
            # تحضير البيانات
            if f'{model_name}_scaler' in self.scalers:
                scaled_data = self.scalers[f'{model_name}_scaler'].transform(new_data.reshape(-1, 1)).flatten()
                input_tensor = torch.FloatTensor(scaled_data).unsqueeze(0).to(self.device)
            else:
                input_tensor = torch.FloatTensor(new_data).to(self.device)
            
            # التنبؤ
            with torch.no_grad():
                predictions = model(input_tensor).cpu().numpy()
            
            # تحويل إلى القيم الأصلية إذا لزم الأمر
            if f'{model_name}_scaler' in self.scalers:
                predictions = self.scalers[f'{model_name}_scaler'].inverse_transform(
                    predictions.reshape(-1, 1)
                ).flatten()
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"خطأ في التنبؤ: {e}")
            return np.array([])
    
    def save_models(self, save_path: str = "models/"):
        """حفظ النماذج"""
        try:
            save_path = Path(save_path)
            save_path.mkdir(exist_ok=True)
            
            for model_name, model in self.models.items():
                torch.save(model.state_dict(), save_path / f"{model_name}.pth")
            
            # حفظ المعايرات والمرمزات
            joblib.dump(self.scalers, save_path / "scalers.pkl")
            joblib.dump(self.encoders, save_path / "encoders.pkl")
            
            self.logger.info(f"تم حفظ {len(self.models)} نموذج في {save_path}")
            
        except Exception as e:
            self.logger.error(f"خطأ في حفظ النماذج: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """معلومات النماذج المدربة"""
        return {
            'total_models': len(self.models),
            'model_names': list(self.models.keys()),
            'device': self.device,
            'available_scalers': list(self.scalers.keys()),
            'available_encoders': list(self.encoders.keys())
        }

# مثال للاستخدام
async def main():
    """مثال شامل لاستخدام التعلم العميق"""
    predictor = DeepLearningPredictor()
    
    try:
        print("🧠 بدء التعلم العميق المتقدم...")
        
        # إنشاء بيانات تجريبية
        np.random.seed(42)
        n_samples = 1000
        
        # بيانات للتصنيف
        X_class = np.random.randn(n_samples, 10)
        y_class = (X_class[:, 0] + X_class[:, 1] > 0).astype(int)
        
        classification_data = pd.DataFrame(X_class, columns=[f'feature_{i}' for i in range(10)])
        classification_data['target'] = y_class
        
        # تحضير البيانات للتصنيف
        class_data_info = await predictor.prepare_data(classification_data, 'target')
        
        # تدريب نموذج التصنيف
        class_results = await predictor.train_neural_network(
            class_data_info, 
            model_name="classification_model",
            epochs=50
        )
        print(f"🎯 نتائج التصنيف: {class_results}")
        
        # بيانات للانحدار
        X_reg = np.random.randn(n_samples, 5)
        y_reg = X_reg[:, 0] * 2 + X_reg[:, 1] * -1 + np.random.randn(n_samples) * 0.1
        
        regression_data = pd.DataFrame(X_reg, columns=[f'feature_{i}' for i in range(5)])
        regression_data['target'] = y_reg
        
        # تحضير البيانات للانحدار
        reg_data_info = await predictor.prepare_data(regression_data, 'target')
        
        # تدريب نموذج الانحدار
        reg_results = await predictor.train_neural_network(
            reg_data_info,
            model_name="regression_model",
            epochs=50
        )
        print(f"📈 نتائج الانحدار: {reg_results}")
        
        # بيانات السلاسل الزمنية
        time_series = np.sin(np.linspace(0, 100, 1000)) + np.random.randn(1000) * 0.1
        
        # تدريب نموذج LSTM
        lstm_results = await predictor.train_lstm_model(
            time_series,
            sequence_length=20,
            model_name="time_series_lstm",
            epochs=30
        )
        print(f"⏰ نتائج LSTM: {lstm_results}")
        
        # معلومات النماذج
        model_info = predictor.get_model_info()
        print(f"📊 معلومات النماذج: {model_info}")
        
        # حفظ النماذج
        predictor.save_models()
        
    except Exception as e:
        print(f"❌ خطأ: {e}")

if __name__ == "__main__":
    asyncio.run(main())
