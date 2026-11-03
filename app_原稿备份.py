import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

# ================= 1. 网页基础设置 =================
st.set_page_config(page_title="燃料电池寿命预测系统", layout="wide")


import torch
# 你的模型定义类必须也放在这里，或者从别的文件 import 进来
# class YourGRUModel(torch.nn.Module): ...

# 使用缓存，只加载一次模型
@st.cache_resource
def load_my_model():
    # 这里填你模型保存的路径
    model = torch.load('电池v1', weights_only=False)
    model.eval() # 切换到测试模式
    return model

model = load_my_model()


st.title("🔋 质子交换膜燃料电池 (PEMFC) 预测系统")
st.markdown("### 基于 GRU + 物理约束 的剩余寿命 (RUL) 概率预测")
st.divider()  # 画一条分割线

# ================= 2. 侧边栏设置 =================
st.sidebar.header("🕹️ 参数控制面板")
# 用滑动条让用户选失效阈值 (FT)
ft_threshold = st.sidebar.slider("选择失效阈值 (FT %)", min_value=3.5, max_value=6.0, value=4.5, step=0.5)
st.sidebar.write(f"当前对应的失效电压阈值: {3.22 - (ft_threshold / 100) * 3.22:.4f} V")

# 文件上传组件
uploaded_file = st.sidebar.file_uploader("上传历史电压数据 (CSV)", type=["csv"])

st.sidebar.divider()
st.sidebar.info("💡 提示：如果没有数据，可直接点击下方按钮使用演示数据。")

# ================= 3. 网页主布局 (分两列) =================
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📊 历史数据监控")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.dataframe(data.tail(10))  # 只展示最后10行
        st.line_chart(data)  # 简单画个折线图
    else:
        st.warning("等待上传数据... 或使用演示模式。")

with col2:
    st.subheader("🔮 AI 模型预测结果")

    # 核心触发按钮
    if st.button("🚀 开始蒙特卡洛预测", type="primary"):
        if uploaded_file is None:
            st.error("请先在左侧上传数据文件！(演示模式下请忽略)")

        with st.spinner('模型正在加载物理约束并进行 50000 条路径推理...'):
            # 1. 处理上传的数据 (真实业务逻辑)
            # 假设你读取了上传的 data，并处理成了模型需要的 tensor
            # input_tensor = process_data(data)

            # 2. 调用你的模型进行蒙特卡洛推理
            # 这一步需要你自己写，假设返回了 4000 条 RUL 预测结果，存为一个 numpy 数组 rul_predictions
            # rul_predictions = run_monte_carlo(model, input_tensor)

            # --- 下面是为了演示写的数据，你替换成上面真实算出来的 ---
            rul_predictions = np.random.normal(300, 30, 4000)  # 替换这行！

            st.success("✅ 预测完成！")

            # 3. 动态计算真实指标
            p50 = np.percentile(rul_predictions, 50)
            p5 = np.percentile(rul_predictions, 5)
            p95 = np.percentile(rul_predictions, 95)

            metric_col1, metric_col2, metric_col3 = st.columns(3)
            metric_col1.metric(label="剩余寿命 (中位数)", value=f"{p50:.0f} 小时")
            metric_col2.metric(label="P5 (悲观)", value=f"{p5:.0f} 小时")
            metric_col3.metric(label="P95 (乐观)", value=f"{p95:.0f} 小时")

            # 4. 用真实数据画直方图
            st.markdown("#### 概率分布与置信区间")
            fig, ax = plt.subplots(figsize=(8, 4))
            # 注意这里换成了 rul_predictions
            ax.hist(rul_predictions, bins=40, color='skyblue', edgecolor='black', alpha=0.7)
            ax.axvline(p50, color='red', linestyle='dashed', linewidth=2, label=f'中位数 P50={p50:.0f}h')
            ax.axvline(p5, color='purple', linestyle='dashed', linewidth=1.5, label=f'P5={p5:.0f}h')
            ax.axvline(p95, color='purple', linestyle='dashed', linewidth=1.5, label=f'P95={p95:.0f}h')
            ax.set_xlabel('剩余寿命 RUL (h)')
            ax.set_ylabel('频数 (路径数)')
            ax.legend()

            st.pyplot(fig)

            st.info("🎉 完美！下一步，把你 .ipynb 里面真实的 RUL 直方图和生存曲线图用 st.pyplot() 替换掉上面的假图吧！")