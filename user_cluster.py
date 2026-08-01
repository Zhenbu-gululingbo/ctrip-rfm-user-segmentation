import os
import warnings
import platform

os.environ["LOKY_MAX_CPU_COUNT"] = "6"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

if platform.system() == "Windows":
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
elif platform.system() == "Darwin":
    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS"]
else:
    plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei"]
plt.rcParams["axes.unicode_minus"] = False

cur_path = os.path.dirname(__file__)
csv_file = os.path.join(cur_path, "..", "data", "UserBehavior.csv")
df = pd.read_csv(csv_file, header=None)
df.columns = ["user_id", "item_id", "category_id", "behavior_type", "timestamp"]

df = df.dropna()
df = df.drop_duplicates()
start_ts = 1511539200
end_ts = 1512316798
df = df[(df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)]
df["time"] = pd.to_datetime(df["timestamp"], unit="s").dt.tz_localize(None)
print(f"清洗&过滤时间后有效数据量：{len(df)}")

max_date = df["time"].max()
df["is_buy"] = (df["behavior_type"] == "buy")
rfm = df.groupby("user_id").agg(
    Recency=("time", lambda x: (max_date - x.max()).days),
    Frequency=("time", "count"),
    Monetary=("is_buy", "sum")
).reset_index()

print("RFM特征构建完成，用户总数：", len(rfm))
print("R值校验 | 最小值：", rfm["Recency"].min(), "最大值：", rfm["Recency"].max())

rfm_features = rfm[["Recency", "Frequency", "Monetary"]].copy()
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_features)

output_dir = os.path.join(cur_path, "..", "output")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

inertia = []
k_range = range(2, 10)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    km.fit(rfm_scaled)
    inertia.append(km.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(k_range, inertia, marker='o')
plt.title("K-Means 肘部法则曲线")
plt.xlabel("聚类数量K")
plt.ylabel("SSE 簇内误差平方和")
plt.grid(alpha=0.3)
plt.savefig(os.path.join(output_dir, "kmeans_elbow.png"), dpi=300)
plt.close()

kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init="auto",
    algorithm="lloyd"
)
rfm["cluster"] = kmeans.fit_predict(rfm_scaled)

cluster_mapping = {
    0: "普通活跃用户",
    1: "高频潜力用户",
    2: "轻度沉寂用户",
    3: "高价值忠实付费用户",
    4: "深度流失沉睡用户"
}
rfm["user_type"] = rfm["cluster"].map(cluster_mapping)

print("\n=====各人群聚类统计=====")
cluster_summary = rfm.groupby("cluster").agg(
    用户数量=("user_id", "count"),
    平均最近间隔R=("Recency", "mean"),
    平均行为频次F=("Frequency", "mean"),
    平均购买次数M=("Monetary", "mean")
).round(2)
print(cluster_summary)

output_file = os.path.join(output_dir, "user_rfm_cluster.csv")
rfm.to_csv(output_file, index=False, encoding="utf-8-sig")
print(f"\n✅ 用户RFM聚类结果已保存至：{output_file}")

plt.figure(figsize=(10, 5))
user_count = rfm["user_type"].value_counts()
user_count.plot(kind="bar", color="#4472C4")
plt.title("各类用户数量分布")
plt.ylabel("用户数量")
plt.xlabel("用户类型")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "user_distribute.png"), dpi=300)
plt.close()

sample_rfm = rfm.sample(n=50000, random_state=42)
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection="3d")
scatter = ax.scatter(
    sample_rfm["Recency"],
    sample_rfm["Frequency"],
    sample_rfm["Monetary"],
    c=sample_rfm["cluster"],
    cmap="viridis",
    alpha=0.6
)
ax.set_xlabel("Recency 最近间隔天数")
ax.set_ylabel("Frequency 行为频次")
ax.set_zlabel("Monetary 购买次数")
ax.set_title("RFM三维用户聚类散点图(采样5万条)")
plt.colorbar(scatter, label="Cluster编号")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "rfm_3d_cluster.png"), dpi=300)
plt.close()

summary_df = rfm.groupby("user_type")[["Recency", "Frequency", "Monetary"]].mean()
summary_df.plot(kind="bar", figsize=(12, 6))
plt.title("各人群RFM指标均值对比")
plt.ylabel("指标平均值")
plt.xlabel("用户类型")
plt.xticks(rotation=30)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "rfm_group_compare.png"), dpi=300)
plt.close()

print("✅ 可视化图片全部保存至output文件夹！")