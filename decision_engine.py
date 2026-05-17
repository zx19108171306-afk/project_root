import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import json

# --- 1. 定义与加载模型结构 ---
class PersonalityAE(nn.Module):
    def __init__(self, input_dim):
        super(PersonalityAE, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 16), nn.ReLU(), nn.Linear(16, 8))
        self.decoder = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, input_dim))
    def forward(self, x):
        return self.decoder(self.encoder(x))

# --- 2. 核心决策引擎类 ---
class DecisionEngine:
    def __init__(self, w1=0.4, w2=0.6):
        self.w1 = w1 # 语义风险权重
        self.w2 = w2 # 行为偏离权重 

    def get_action(self, s_score, d_score):
        """
        s_score: DeepSeek 评分 (0-100, 高为安全)
        d_score: 行为偏离评分 (重构误差转换, 高为异动)
        """
        # 计算综合风险 R
        total_risk = self.w1 * (100 - s_score) + self.w2 * d_score
        
        # 梯度挑战逻辑 
        if total_risk < 25:
            return "静默通行 (风险极低)", total_risk
        elif total_risk < 55:
            return "低强度增强 (二次行为检测)", total_risk
        elif total_risk < 85:
            return "高强度加固 (触发 DeepSeek MFA 建议)", total_risk
        else:
            return "即时阻断 (检测到自动化攻击)", total_risk

# --- 3. 执行测试流程 ---
def run_integrated_test():
    # A. 加载之前训练的模型
    checkpoint = torch.load("/data/zhangzixin/Project_root/models/personality_ae.pth")
    model = PersonalityAE(checkpoint['input_dim'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # B. 加载数据集用于模拟攻击行为 (使用 s003 模拟攻击者)
    full_data = pd.read_csv("/data/zhangzixin/Project_root/data/DSL-StrongPasswordData.csv")
    attacker_data = full_data[full_data['subject'] == 's003'].iloc[0:1, 3:].values
    scaler = MinMaxScaler()
    attacker_tensor = torch.tensor(scaler.fit_transform(attacker_data), dtype=torch.float32)

    # C. 模拟三个典型场景 [cite: 124]
    scenarios = [
        {"name": "场景 1: 合法用户 (s002) + 安全密码", "s": 90, "data": "baseline"},
        {"name": "场景 2: 合法用户 (s002) + 弱语义密码", "s": 35, "data": "baseline"},
        {"name": "场景 3: 模拟攻击者 (s003) + 窃取正确密码", "s": 85, "data": "attacker"}
    ]

    engine = DecisionEngine()
    print(f"--- 论文第 6.2 节：防御效能对比实验 ---\n")

    for sc in scenarios:
        # 计算行为偏离度 D (基于重构误差)
        if sc["data"] == "baseline":
            # 模拟极低偏差
            d_score = 15.0 
        else:
            # 计算攻击者数据输入模型后的误差 [cite: 77, 79]
            with torch.no_grad():
                reconstructed = model(attacker_tensor)
                d_score = torch.mean((reconstructed - attacker_tensor)**2).item() * 1000

        action, risk = engine.get_action(sc["s"], d_score)
        print(f"[{sc['name']}]")
        print(f" > 综合风险 R: {risk:.2f} | 建议决策: {action}\n")

if __name__ == "__main__":
    run_integrated_test()