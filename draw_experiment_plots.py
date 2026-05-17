import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import os
import matplotlib.font_manager as fm
import urllib.request

# ================= 字体在线加载逻辑 =================
def load_online_font():
    # 字体保存路径
    font_dir = "/data/zhangzixin/Project_root/fonts"
    if not os.path.exists(font_dir):
        os.makedirs(font_dir)
    
    font_path = os.path.join(font_dir, "SimHei.ttf")
    
    # 如果本地没有字体文件，则从 GitHub 下载一个精简版中文字体
    if not os.path.exists(font_path):
        print("正在下载中文字体文件，请稍候...")
        font_url = "https://github.com/adobe-fonts/source-han-sans/raw/release/OTF/SimplifiedChinese/SourceHanSansSC-Regular.otf"
        try:
            urllib.request.urlretrieve(font_url, font_path)
            print("字体下载完成。")
        except Exception as e:
            print(f"下载失败: {e}，请手动上传一个 .ttf 字体到 {font_dir}")
            return None

    # 手动注册字体
    fe = fm.FontEntry(fname=font_path, name='CustomFont')
    fm.fontManager.ttflist.insert(0, fe)
    plt.rcParams['font.sans-serif'] = ['CustomFont']
    plt.rcParams['axes.unicode_minus'] = False
    return 'CustomFont'

# 执行字体加载
custom_font_name = load_online_font()
if custom_font_name:
    sns.set_theme(style="whitegrid", font=custom_font_name)
else:
    sns.set_theme(style="whitegrid")

# ================= 路径配置与模型定义 =================
BASE_DIR = "/data/zhangzixin/Project_root"
SEMANTIC_CSV = f"{BASE_DIR}/results/semantic_analysis_report.csv"
BEHAVIOR_CSV = f"{BASE_DIR}/data/DSL-StrongPasswordData.csv"
MODEL_PATH = f"{BASE_DIR}/models/personality_ae.pth"
PLOT_DIR = f"{BASE_DIR}/results/plots"

if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)

class PersonalityAE(nn.Module):
    def __init__(self, input_dim):
        super(PersonalityAE, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 16), nn.ReLU(), nn.Linear(16, 8))
        self.decoder = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, input_dim))
    def forward(self, x):
        return self.decoder(self.encoder(x))

def draw_plots():
    # --- 图 1：语义风险分布 ---
    if os.path.exists(SEMANTIC_CSV):
        print("生成图 1: 语义风险分布...")
        df_sem = pd.read_csv(SEMANTIC_CSV)
        plt.figure(figsize=(8, 7))
        # 统计分布 [cite: 50]
        counts = df_sem['risk_level'].value_counts()
        plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140)
        plt.title('密码语义风险等级分布分析', fontsize=14)
        plt.savefig(f"{PLOT_DIR}/1_semantic_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()

    # --- 图 2：重构误差分布 (D) ---
    if os.path.exists(MODEL_PATH) and os.path.exists(BEHAVIOR_CSV):
        print("生成图 2: 行为偏离度对比...")
        checkpoint = torch.load(MODEL_PATH)
        model = PersonalityAE(checkpoint['input_dim'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # 加载击键动力学数据 [cite: 35, 62]
        df_beh = pd.read_csv(BEHAVIOR_CSV)
        data_v = df_beh[df_beh['subject'] == 's002'].iloc[200:300, 3:].values
        data_a = df_beh[df_beh['subject'] == 's003'].iloc[0:100, 3:].values
        
        scaler = MinMaxScaler()
        v_tensor = torch.tensor(scaler.fit_transform(data_v), dtype=torch.float32)
        a_tensor = torch.tensor(scaler.fit_transform(data_a), dtype=torch.float32)

        with torch.no_grad():
            # 计算重构误差 [cite: 77, 83]
            err_v = torch.mean((model(v_tensor) - v_tensor)**2, dim=1).numpy() * 100
            err_a = torch.mean((model(a_tensor) - a_tensor)**2, dim=1).numpy() * 100

        plt.figure(figsize=(10, 6))
        sns.kdeplot(err_v, fill=True, label='合法用户 (s002)', color='teal')
        sns.kdeplot(err_a, fill=True, label='模拟攻击者 (s003)', color='salmon')
        plt.title('用户“数字人格”重构误差密度分布对比', fontsize=14)
        plt.xlabel('重构误差评分 (MSE x 100)', fontsize=12)
        plt.ylabel('密度', fontsize=12)
        plt.legend()
        plt.savefig(f"{PLOT_DIR}/2_behavior_deviation.png", dpi=300, bbox_inches='tight')
        plt.close()

    # --- 图 3：防御效能对比 (基于表 6.2) ---
    print("生成图 3: 防御效能对比...")
    # 基于论文 130 节数据 
    methods = ['传统静态密码', '固定 MFA', '本文方案 (AI赋能)']
    accuracy = [5.0, 72.0, 98.6] 
    resilience = [12.0, 68.0, 99.1]

    x = range(len(methods))
    plt.figure(figsize=(10, 6))
    plt.bar(x, accuracy, width=0.3, label='撞库攻击防御率', color='skyblue')
    plt.bar([i + 0.3 for i in x], resilience, width=0.3, label='暴力破解抗性', color='coral')
    
    plt.xticks([i + 0.15 for i in x], methods)
    plt.ylabel('成功拦截率 (%)', fontsize=12)
    plt.title('不同身份认证方案防御效能对比', fontsize=14)
    plt.ylim(0, 115)
    
    # 添加数字标注，确保数字可见
    for i, v in enumerate(accuracy): plt.text(i-0.05, v+2, str(v))
    for i, v in enumerate(resilience): plt.text(i+0.25, v+2, str(v))
    
    plt.legend()
    plt.savefig(f"{PLOT_DIR}/3_defense_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    draw_plots()