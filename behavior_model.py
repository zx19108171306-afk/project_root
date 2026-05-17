import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import os

# ================= 配置区域 =================
DATA_PATH = "/data/zhangzixin/Project_root/data/DSL-StrongPasswordData.csv"
MODEL_SAVE_PATH = "/data/zhangzixin/Project_root/models/personality_ae.pth"
USER_ID = "s002"  # 选择一个特定用户作为“合法用户”进行基准建模
EPOCHS = 200
BATCH_SIZE = 16
# ===========================================

if not os.path.exists("/data/zhangzixin/Project_root/models"):
    os.makedirs("/data/zhangzixin/Project_root/models")

# 1. 数据加载与特征工程 [cite: 62, 68]
def prepare_data(path, user_id):
    df = pd.read_csv(path)
    
    # 过滤出目标用户的数据 (每个用户通常有400条记录)
    user_data = df[df['subject'] == user_id]
    
    # 提取时间特征列：H (Hold time), DD (Down-Down), UD (Up-Down) 
    # 该数据集从第4列开始是击键时间数据
    features = user_data.iloc[:, 3:].values
    
    # 归一化处理，确保模型收敛
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)
    
    # 转换为 PyTorch 张量
    return torch.tensor(scaled_features, dtype=torch.float32), scaler

# 2. 定义自编码器模型（数字人格模型） [cite: 72]
class PersonalityAE(nn.Module):
    def __init__(self, input_dim):
        super(PersonalityAE, self).__init__()
        # 编码器：压缩特征，提取肌肉记忆模式 
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8) 
        )
        # 解码器：尝试重构原始输入
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# 3. 训练函数 [cite: 73, 75]
def train_model(train_data):
    input_dim = train_data.shape[1]
    model = PersonalityAE(input_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    print(f"开始为用户 {USER_ID} 训练数字人格模型...")
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        
        output = model(train_data)
        loss = criterion(output, train_data)
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.6f}")
            
    return model

# --- 执行实验 ---
if __name__ == "__main__":
    # 加载合法用户的行为样本
    data_tensor, scaler = prepare_data(DATA_PATH, USER_ID)
    
    # 使用前 200 条数据作为训练集（模拟注册/初始阶段的基准采集） [cite: 73]
    train_set = data_tensor[:200]
    
    # 训练模型
    personality_model = train_model(train_set)
    
    # 保存模型权重
    torch.save({
        'model_state_dict': personality_model.state_dict(),
        'input_dim': data_tensor.shape[1],
        'user_id': USER_ID
    }, MODEL_SAVE_PATH)
    
    print(f"\n实验完成！用户 {USER_ID} 的数字人格模型已保存至: {MODEL_SAVE_PATH}")