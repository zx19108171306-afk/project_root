from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
import requests
import json
import numpy as np
import os

app = Flask(__name__)

# --- 1. 模型架构定义 ---
class PersonalityAE(nn.Module):
    def __init__(self, input_dim):
        super(PersonalityAE, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 16), nn.ReLU(), nn.Linear(16, 8))
        self.decoder = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, input_dim))
    def forward(self, x):
        return self.decoder(self.encoder(x))

# --- 2. 资源加载 ---
MODEL_PATH = "/data/zhangzixin/Project_root/models/personality_ae.pth"
INPUT_DIM = 31 # 对应之前实验的特征维度

model = PersonalityAE(INPUT_DIM)
if os.path.exists(MODEL_PATH):
    checkpoint = torch.load(MODEL_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("成功加载数字人格模型")

# --- 3. 核心计算函数 ---
def get_semantic_score(password):
    """调用DeepSeek进行语义审计"""
    url = "http://localhost:11434/api/generate"
    prompt = f"分析密码'{password}'的安全性，返回JSON:{{'score':0-100}}"
    try:
        r = requests.post(url, json={"model": "deepseek-coder:1.3b", "prompt": prompt, "stream": False, "format": "json"}, timeout=5)
        return json.loads(r.json()['response'])['score']
    except:
        return 50

def map_deviation_score(mse_loss):
    """
    终极修复：使用逻辑回归映射函数 (Sigmoid-like)
    将巨大的MSE误差平滑映射到 0-100 区间
    """
    # 调整系数 0.01 可根据实际灵敏度调节
    score = 100 * (1 - np.exp(-mse_loss * 0.01))
    return score

@app.route('/')
def index():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    password = data.get('password')
    features = np.array(data.get('features'), dtype=np.float32)

    # 1. 语义风险评分 S
    s_score = get_semantic_score(password)

    # 2. 行为偏离评分 D
    input_tensor = torch.FloatTensor(features).reshape(1, -1)
    with torch.no_grad():
        reconstructed = model(input_tensor)
        mse_loss = torch.mean((input_tensor - reconstructed)**2).item()
        # 映射至 0-100
        d_score = map_deviation_score(mse_loss)

    # 3. 综合风险指数 R (加权融合)
    # R = w1 * 语义风险 + w2 * 行为风险
    risk_index = 0.4 * (100 - s_score) + 0.6 * d_score

    # 决策逻辑
    if risk_index < 35:
        res = {"status": "success", "msg": "认证通过：识别为合法数字人格", "risk": round(risk_index, 2)}
    elif risk_index < 75:
        res = {"status": "mfa", "msg": "异常提示：行为特征偏离，请执行二次验证", "risk": round(risk_index, 2)}
    else:
        res = {"status": "fail", "msg": "高危拦截：检测到非法冒用或暴力破解", "risk": round(risk_index, 2)}
    
    return jsonify(res)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)