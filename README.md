# AI-Powered Password Security System
> 基于用户行为特征与语义分析的动态密码安全实验项目

---

## 📌 项目简介
本项目是面向学术研究的**密码安全增强实验代码**，结合用户行为特征（击键动力学、设备指纹）与DeepSeek语义分析，实现动态密码加固、风险评估与自适应防御机制。

项目核心目标：
- 研究行为特征在身份认证中的有效性
- 探索AI语义分析在密码安全场景下的应用
- 提供可复现的实验代码与数据集


---

## 🛠️ 环境依赖
- Python 3.10+
- PyTorch 1.13+
- 其他依赖：`numpy`, `pandas`, `matplotlib`, `scikit-learn`

一键安装依赖：
```bash
pip install -r requirements.txt
# 1. 数据预处理（可选）
python generate_thesis_data.py

# 2. 启动主程序
python app.py

# 3. 生成实验结果可视化
python draw_experiment_plots.py

---

---


## ⚠️ 项目声明
- **用途**：仅用于学术研究与实验验证。
- **禁止商用**：未经许可，不得用于商业场景。
- **免责声明**：实验性质，不保证生产安全。
