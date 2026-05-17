import pandas as pd
import requests
import json
import random
import os
from tqdm import tqdm

# ================= 配置区域 =================
OLLAMA_API = "http://localhost:11434/api/generate"
MODEL_NAME = "deepseek-coder:1.3b"
# 使用绝对路径避免 FileNotFoundError
DATA_PATH = "/data/zhangzixin/Project_root/data/rockyou.txt" 
RESULT_DIR = "/data/zhangzixin/Project_root/results"
SAMPLE_SIZE = 5000 
# ===========================================

# 确保结果目录存在
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)

def get_deepseek_score(password):
    """
    调用本地 DeepSeek 模型进行语义评分
    增加了严格的异常处理和格式校验 [cite: 31, 100]
    """
    prompt = f"""
    分析密码 '{password}' 的语义风险并给出 0-100 的评分（分数越高越安全）。
    请严格遵守以下 JSON 输出格式，不要包含任何其他解释文字：
    {{
        "score": 整数评分,
        "reason": "简短的理由"
    }}
    """
    
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "format": "json"  # 强制要求本地模型输出 JSON 格式 
    }
    
    try:
        response = requests.post(OLLAMA_API, json=payload, timeout=15)
        response.raise_for_status()
        
        # 获取模型生成的原始字符串
        raw_output = response.json().get('response', '')
        
        # 解析 JSON 字符串
        data = json.loads(raw_output)
        
        # 提取评分并确保其为整数 [cite: 44, 48]
        score = data.get('score')
        if score is None:
            return 50, "模型未返回有效分值"
            
        return int(score), data.get('reason', "分析完成")
        
    except (json.JSONDecodeError, ValueError) as e:
        # 如果 JSON 解析失败，返回默认分值，防止脚本中断 [cite: 101]
        return 50, f"格式解析失败: {str(e)}"
    except Exception as e:
        return 50, f"网络或API异常: {str(e)}"

def classify_risk(score):
    """
    根据论文表 3.1 划分风险等级 [cite: 48, 50]
    """
    # 再次确保 score 是有效数字
    if score is None:
        score = 50
        
    if score < 40:
        return "高危 (强制拒绝)"
    elif score < 60:
        return "中高 (触发行为特征增强监测)"
    elif score < 85:
        return "中低 (建议补强/弱预警)"
    else:
        return "安全 (允许通行)"

def run_step1_experiment():
    print("--- 步骤一：密码语义特征分析实验开始 ---")
    
    # 1. 读取数据集 [cite: 115]
    print(f"正在读取数据集: {DATA_PATH} ...")
    try:
        with open(DATA_PATH, 'r', encoding='latin-1') as f:
            # 过滤掉短密码或空白行
            all_passwords = [line.strip() for line in f if len(line.strip()) > 1]
    except FileNotFoundError:
        print(f"错误: 找不到文件 {DATA_PATH}。请确认路径是否正确。")
        return

    # 2. 随机抽样 [cite: 112, 115]
    print(f"随机抽取 {SAMPLE_SIZE} 条样本进行评估...")
    test_samples = random.sample(all_passwords, min(SAMPLE_SIZE, len(all_passwords)))

    # 3. 批量执行评估 [cite: 46, 93]
    results = []
    print(f"正在连接 DeepSeek ({MODEL_NAME}) 引擎...")
    
    for pwd in tqdm(test_samples, desc="语义分析中"):
        score, reason = get_deepseek_score(pwd)
        risk_level = classify_risk(score)
        
        results.append({
            "password": pwd,
            "semantic_score": score,
            "risk_level": risk_level,
            "analysis_reason": reason
        })

    # 4. 保存实验数据 
    df = pd.DataFrame(results)
    save_path = os.path.join(RESULT_DIR, "semantic_analysis_report.csv")
    df.to_csv(save_path, index=False, encoding='utf-8-sig')
    
    # 5. 输出汇总统计 [cite: 124]
    print("\n" + "="*30)
    print("实验汇总报告")
    print("-" * 30)
    print(df['risk_level'].value_counts())
    print("-" * 30)
    print(f"完整报告已保存至: {save_path}")
    print("="*30)

if __name__ == "__main__":
    run_step1_experiment()