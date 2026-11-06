# PEMFC 剩余寿命预测终端 · 精简部署版（dc-zhongduan）

质子交换膜燃料电池（PEMFC）剩余寿命预测终端的**可直接部署精简副本**：只保留运行所需的核心代码
（已去除注释与说明性文档字符串），并随包附带模型权重与一份小型演示数据，适合部署到
Streamlit Community Cloud 或任意可运行 Python 的服务器。

## 目录内容

| 路径 | 说明 |
| --- | --- |
| `app.py` | 主程序（Streamlit 界面与交互逻辑） |
| `live_canvas.py` | 交互画布：预测曲线 + RUL 概率分布（Canvas 渲染、缩略图预加载、内存红线） |
| `rul_pipeline.py` | 数值核心：数据规范化、指数衰减律拟合、分位数标定、蒙特卡洛 RUL 估计 |
| `models_terminal.py` | GRU / GRU-Transformer 模型结构 |
| `data_memory.py` | 历史数据记忆库（跨次预测的经验池） |
| `nn_library.py` | 神经网络训练数据仓库（按内容指纹去重） |
| `resource_guard.py` | 资源监控与加载遮罩 |
| `model_warm.py` | 模型预热（缩短首次预测等待） |
| `模型库/` | 训练好的模型权重 `pemfc_forecast_v4.pt` 与其元信息 |
| `数据/demo_fc1.csv` | 内置演示数据（FC1 测试集，0.1 h 采样，约 0.5 MB） |
| `.streamlit/config.toml` | 运行配置（关闭文件监视、上传上限 500 MB） |
| `requirements.txt` | 依赖清单（PyTorch 使用 CPU 轮子） |
| `packages.txt` | 云端 apt 依赖（中文字体 `fonts-noto-cjk`） |

运行时目录（`输出/`、`数据记忆库/`、`神经网络库/`）会在首次预测时自动创建，无需预先准备。

## 本地运行

```bash
pip install -r requirements.txt
streamlit run app.py
```

浏览器打开 `http://localhost:8501`，在左侧选择「内置演示数据」→ 点「开始预测」即可看到结果。

## 部署到 Streamlit Community Cloud

1. 把本目录所在仓库推到 GitHub（本仓库已包含本目录）。
2. 打开 <https://share.streamlit.io>，用 GitHub 账号登录并授权。
3. New app：Repository 选本仓库、Branch 选 `main`、**Main file path 填 `dc-zhongduan/app.py`**。
4. 点 Deploy，构建完成后得到公网地址（形如 `https://<应用名>.streamlit.app`）。

注意事项：

- 云端为 Linux 环境，`requirements.txt` 已改用 CPU 版 PyTorch（体积小、无需显卡）；
- 中文图表字体由 `packages.txt` 中的 `fonts-noto-cjk` 提供；
- 免费实例重启后 `输出/`、`数据记忆库/` 等运行产物不会保留（属正常现象），数据可通过上传重新分析。
