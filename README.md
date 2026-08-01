# 携程业务场景 RFM 用户分层建模
##  项目背景
模拟旅游平台用户行为数据，搭建RFM价值模型，结合K-Means聚类实现用户分群；
完成数据清洗、异常值过滤、特征工程、聚类建模、结果可视化，用于精细化运营与合规数据分析。

##  技术栈
Python | Pandas | NumPy | Scikit-learn | Matplotlib

##  功能模块
1. 数据预处理：缺失值填充、重复数据剔除、IQR法则筛选异常数据
2. R/F/M指标计算，用户价值特征构建
3. K-Means聚类，划分5类用户群体
4. 聚类结果可视化，输出用户分层清单
5. 数据合规性统计分析

##  运行方式
1. 安装依赖
```bash
pip install -r requirements.txt

## 项目流程
1. 数据预处理：使用fillna填充缺失数据，drop_duplicates清除重复记录；借助percentile计算四分位数，通过IQR规则过滤异常数据。
2. 特征构建：提取R(Recency)、F(Frequency)、M(Monetary)三类用户价值特征。
3. 聚类建模：基于RFM特征使用K-Means算法实现用户分群。
4. 结果可视化：输出用户聚类分布图，直观展示不同群体特征。

## 聚类可视化结果
