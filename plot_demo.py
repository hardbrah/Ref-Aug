import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def generate_heatmap_data(shape=(3, 5), similarity=1.0, seed=42):
    """
    生成一个可复现的热力图数据，支持控制相似性。

    参数：
        shape: tuple，热力图的行列数，例如 (3, 5)
        similarity: float ∈ [0, 1]，相似性参数
                    - 1.0 → 所有值几乎一样（极高相似度）
                    - 0.0 → 完全随机（低相似度）
        seed: int，随机数种子，保证可复现
    返回：
        np.ndarray，热力图数据
    """
    np.random.seed(seed)
    # 基础均值
    base_value = np.random.rand()
    # 随机噪声
    noise = np.random.rand(*shape)
    # 融合相似性：相似性越高，噪声影响越小
    data = similarity * base_value + (1 - similarity) * noise
    return data


def plot_dot_heatmap(data, cmap="viridis", dot_size=1200, relative=False):
    rows, cols = data.shape
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    x = x.flatten()
    y = y.flatten()
    colors = data.flatten()

    fig, ax = plt.subplots(figsize=(cols, rows))
    scatter = ax.scatter(x, y, c=colors, s=dot_size, cmap=cmap, edgecolors="black")
    ax.set_xticks(np.arange(cols))
    ax.set_yticks(np.arange(rows))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.invert_yaxis()
    ax.set_aspect("equal")

    # 添加 colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    if relative:
        # 隐藏刻度
        cbar.set_ticks([])
        # 添加 Low/High
        cbar.ax.text(
            0.5, -0.05, "Low", ha="center", va="top", transform=cbar.ax.transAxes
        )
        cbar.ax.text(
            0.5, 1.05, "High", ha="center", va="bottom", transform=cbar.ax.transAxes
        )
    else:
        cbar.set_label("Activation Frequency", rotation=270, labelpad=15)

    plt.show()


# 示例：生成相似度高和低的两张热力图
if __name__ == "__main__":
    shape = (3, 5)

    # 高相似度
    data_high = generate_heatmap_data(shape, similarity=0.9, seed=42)
    plot_dot_heatmap(data_high, cmap="viridis", relative=True)

    # 低相似度
    data_low = generate_heatmap_data(shape, similarity=0.2, seed=42)
    plot_dot_heatmap(data_low, cmap="viridis", relative=True)
