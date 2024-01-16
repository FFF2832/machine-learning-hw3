import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from scipy.cluster import hierarchy



import math
# 讀取CSV檔案
data = pd.read_csv("./HW3_Credit Card Dataset.csv")
# print(data.shape)
# print(data.head())
# print(data.info())

# --------------------------------------------------------------資料清理與視覺化圖表 (Part 1)--------------------------------------------------------------


# # 查看資料前幾行
# print(data.head())

# # 檢查缺失值
# print(data.isnull().sum())

# 丟掉不必要的欄位
data.drop('CUST_ID', axis=1, inplace=True)

# 刪除缺失值（使用平均值）
data = data.dropna()  # 刪除有缺失值的整列資料







# 計算平均值和標準差
mean = data.mean()
std = data.std()

# 正規化資料
normed_data = (data - mean) / std
print(type(normed_data))
print(normed_data.columns)

# 使用StandardScaler進行標準化
scaler = preprocessing.StandardScaler().fit(data)
data_scaled = scaler.transform(data)


# print(data_scaled[:5])
# print(np.array(normed_data.head()))


# 畫盒鬚圖
plt.figure()
sns.boxplot(x="value", y="variable", orient='h', data=pd.melt(normed_data)); # pd.melt()把多個欄位合併成一個
plt.savefig('box_pre.png')
# 

# 丟掉不必要的欄位
newdata=normed_data.drop(['TENURE'], axis=1)

# print(normed_data.head())
# print(newdata.shape)
indices=[]
features = newdata.columns
for i, feature in enumerate(features):
    q1, q3 = np.percentile(newdata[feature], [25, 75])
    above = q3 + 1.5 * (q3 - q1)
    below = q1 - 1.5 * (q3 - q1)
    drop_index=newdata[ (newdata[feature]<below) | (newdata[feature]>above) ].index
    indices.append(np.array(drop_index))
#     print(len(indices[-1]),indices[-1][-5:])
# print(np.unique(np.concatenate(indices)).shape)
newdata=newdata.drop(np.concatenate(indices))
# print(newdata.shape)
# plt.figure()
# sns.boxplot(x="value", y="variable", orient='h', data=pd.melt(newdata)); # pd.melt()把多個欄位合併成一個
# plt.savefig('box_post.png')
# 







# --------------------------------------------------------------敘述性統計分析 (Part 2)--------------------------------------------------------------

# data.mean()
# data.var()
# data.std()
# print(data.describe())
#正規化(normalization)
mean = data.mean()
std = data.std()
normed_data = (data - mean) / std  # (data - data_stats['mean']) / data_stats['std']
# print(data.describe())

# --------------------------------------------------------------特徵相關性分析(Part 3)--------------------------------------------------------------
correlation_matrix = newdata.corr().abs()
plt.figure(figsize=(12, 10))
sns.heatmap(data=correlation_matrix, annot=True)
plt.figure()
plt.savefig('correlation_heatmap.png')
# plt.show()
# # 丟掉相關性相對不高的
# newdata=normed_data.drop(['BALANCE_FREQUENCY','PAYMENTS'], axis=1)

# --------------------------------------------------------------PCA降維處理與分析(Part 4)--------------------------------------------------------------
pca = PCA(n_components=2)
num_pc = 10
pca = PCA(n_components=num_pc)
pca_data = pca.fit_transform(newdata)

# print(pca_data.shape)
# print(pca_data[:5])

print(pca.explained_variance_) # 特徵值
print(pca.explained_variance_ratio_) # 解釋變異比例
print(pca.explained_variance_ratio_.sum())

pca_df = pd.DataFrame(data=pca_data[:,:2], columns=['PC1', 'PC2'])

# 可視化PCA降維結果

plt.figure()
plt.scatter(x=pca_df['PC1'], y=pca_df['PC2'])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.savefig('pca_scatter.png')



var = np.array(pca.explained_variance_ratio_)
cum_var = np.cumsum(var)
plt.figure()
plt.bar(range(1, len(var)+1), var, alpha=0.7, align='center',
        label='Individual explained variance')
plt.step(range(1, len(cum_var)+1), cum_var, where='mid', color='k',
         label='Cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principle components')
plt.legend()


# # --------------------------------------------------------------資料分割與建置3個分群模型(Part 5)--------------------------------------------------------------

# # 切割資料集
# train_x,test_x = train_test_split(data, test_size=0.2, random_state=42)
# 切割資料集
print(pca_data.shape)
print(pca_data[:,:2].shape)
# train_x,test_x = train_test_split(pca_data[:,:2], test_size=0.2, random_state=42)
train_x = pca_data[:,:2]
# print(pca_data.shape)
# train_x,test_x = train_test_split(pca_data, test_size=0.2, random_state=42)


# # -------------------------------------------------------------- K-means--------------------------------------------------------------

colors = ['red', 'orange', 'yellow', 'green', 'cyan',
          'blue', 'purple', 'brown', 'grey', 'black']
# 產生的資料組數 (10)
clusters = 10
# K 值的範圍 (2~10)，可能的群數，從k_range 中找出最適合的群數
k_range = range(3, clusters + 1)



# print(train_x[:10])


# distortions = []
scores = []
# 記錄每種 K 值建出的 KMeans 模型的成效
for i in k_range:
    kmeans = KMeans(n_clusters=i).fit(train_x)
#     distortions.append(kmeans.inertia_) # 誤差平方和 (SSE)，這個是kmeans 方法用到的
    scores.append(silhouette_score(train_x, kmeans.predict(train_x))) # 側影係數(輪廓係數)

# 找出最大的側影係數來決定 K 值，selected_K 是最後決定出來的最佳群數
#scores=[s1, s2, s3 ..., s9]
selected_K = scores.index(max(scores)) + 3 #最高分數之元素的索引值加2 才是最佳群數
print(selected_K)
print(scores[selected_K - 3], scores)



kmeans = KMeans(n_clusters=selected_K).fit(train_x)
train_y = kmeans.predict(train_x)
# print('train_y:',train_y)
print(silhouette_score(train_x, train_y))
# print(silhouette_score(test_x, kmeans.predict(test_x)))

# kmeans = KMeans(n_clusters=selected_K).fit(test_x)
# test_y = kmeans.predict(test_x)
# print('test_y:',test_y)
# print(silhouette_score(test_x, test_y))

# # 對訓練集重新建立 KMeans 模型並預測目標值
# kmeans = KMeans(n_clusters=selected_K).fit(train_x)
# train_y = kmeans.predict(train_x)

# #  對測試集重新建立 KMeans 模型並預測目標值
# kmeans = KMeans(n_clusters=selected_K).fit(test_x)
# test_y = kmeans.predict(test_x)
# print('new_dy:',train_y) #為每個資料點指派新的群索引值(由原本10 群變成selected_K 群)


# 新分組的資料中心點
centers = kmeans.cluster_centers_
plt.rcParams['font.size'] = 12
plt.figure(figsize=(12, 7))
# # 原始資料分組
# plt.subplot(221)
# plt.title(f'Original data ({clusters} groups)')
# plt.scatter(test_x.T[0], test_x.T[1], cmap=plt.cm.Set1)

# 新資料分組
plt.subplot(121)
plt.title(f'KMeans={selected_K} groups')
plt.scatter(train_x.T[0], train_x.T[1], c=train_y, cmap=plt.cm.Set3)
plt.scatter(centers.T[0], centers.T[1], marker='^', color='orange')
for i in range(centers.shape[0]): # 標上各分組中心點
    plt.text(centers.T[0][i], centers.T[1][i], str(i + 1),
             fontdict={'color': 'red', 'weight': 'bold', 'size': 24})

#  # 新資料分組
# plt.subplot(121)
# plt.title(f'KMeans={selected_K} groups')
# plt.scatter(test_x.T[0], test_x.T[1], c=test_y, cmap=plt.cm.Set3)
# plt.scatter(centers.T[0], centers.T[1], marker='^', color='orange')
# for i in range(centers.shape[0]): # 標上各分組中心點
#     plt.text(centers.T[0][i], centers.T[1][i], str(i + 1),
#              fontdict={'color': 'red', 'weight': 'bold', 'size': 24})

# # 繪製誤差平方和圖 (手肘法)
# plt.figure()
# plt.subplot(223)
# plt.title('SSE (elbow method)')
# plt.plot(k_range, distortions)
# plt.plot(selected_K, distortions[selected_K - 2], 'go') # 最佳解

# 繪製係數圖

plt.subplot(122)
plt.title('Silhouette score')
plt.plot(k_range, scores)
plt.plot(selected_K, scores[selected_K - 3], 'go') # 最佳解
plt.tight_layout()


plt.savefig('knncluster.png')


# # -------------------------------------------------------------- Hierarchical Clustering--------------------------------------------------------------

# from sklearn.cluster import AgglomerativeClustering

# # 將 train_x 轉換為 DataFrame
# train_x_df = pd.DataFrame(train_x)

# # 執行階層聚類
# cls = AgglomerativeClustering(n_clusters=4)
# cls.fit(train_x_df)
# labels = cls.labels_



# # 使用集群顏色繪製熱度圖
# lut = dict(zip(train_x_df['Type1'].unique(), "rbg"))
# row_colors = train_x_df['Type1'].map(lut)
# g = sns.clustermap(train_x_df, cmap='YlGnBu', row_colors=row_colors)
# plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)


scores=[]
min_k=3
k_range=range(min_k,11)
for n in k_range:
    ml=AgglomerativeClustering(n_clusters=n, affinity='euclidean', linkage='ward')
    s=silhouette_score(train_x, ml.fit_predict(train_x))
    scores.append(s)
  
print("\n")
print(scores)
selected_K = scores.index(max(scores)) + min_k #最高分數之元素的索引值加2 才是最佳群數
print(selected_K)
print(max(scores))

ml=AgglomerativeClustering(n_clusters=selected_K, affinity='euclidean', linkage='ward')
pred_y=ml.fit_predict(train_x)

import random
plt.rcParams['font.size'] = 12

for i in range(selected_K):
    plt.subplot(121)
    R=int(random.random() * 255)
    G=int(random.random() * 255)
    B=int(random.random() * 255)
    color = '#%02x%02x%02x' % (R, G, B)
    plt.scatter(train_x[pred_y==i,0],train_x[pred_y==i,1],c=color) #,marker='x')

# 繪製係數圖
plt.figure()
plt.subplot(122)
plt.title('Silhouette score')
plt.plot(k_range, scores)
plt.plot(selected_K, scores[selected_K - min_k], 'go') # 最佳解
plt.tight_layout()

# plt.show()
# 生成樹狀圖的 linkage 模型
model = hierarchy.linkage(train_x, 'ward')

# 繪製樹狀圖
plt.figure()
hierarchy.dendrogram(model, orientation="top", labels=pred_y)
plt.xlabel("Sample index")
plt.ylabel('Cluster distance')

# # -------------------------------------------------------------- DBSCAN--------------------------------------------------------------


# train_x = pca_data[:,:2]
# # # 取出前 100 隻寶可夢進行分群
# # X = df.loc[df.index < 100, 'HP':'Speed']
# clf = DBSCAN(eps=0.2, min_samples=100).fit(train_x)

# pred_y=clf.labels_
# print(pred_y.shape)
# print(pred_y[:5])
# print(np.unique(pred_y))

# s=silhouette_score(train_x, pred_y)
# print(s)
# # pd.Series(clf.labels_).value_counts()


# plt.figure()
# plt.rcParams['font.size'] = 12
# for i in range(selected_K):

#     R=int(random.random() * 255)
#     G=int(random.random() * 255)
#     B=int(random.random() * 255)
#     color = '#%02x%02x%02x' % (R, G, B)
#     plt.scatter(train_x[pred_y==i,0],train_x[pred_y==i,1],c=color) #,marker='x')

train_x = pca_data[:, :2]

clf = DBSCAN(eps=0.3, min_samples=75).fit(train_x)

pred_y = clf.labels_

# print(pred_y.shape)
# print(pred_y[:10])
# print(pred_y != -1)
# print(pred_y [pred_y != -1][:10])
print(len(np.where(pred_y==-1)[0]),np.unique(pred_y))


# 去除噪音點
non_noise_points = train_x[pred_y != -1]
non_noise_labels = pred_y[pred_y != -1]


print(non_noise_points.shape)
print(non_noise_labels.shape)
print(non_noise_labels[:5])
print(np.unique(non_noise_labels))

s = silhouette_score(non_noise_points, non_noise_labels)
print(s)

plt.figure()
plt.rcParams['font.size'] = 12
for i in range(len(np.unique(non_noise_labels))):
    R = int(random.random() * 255)
    G = int(random.random() * 255)
    B = int(random.random() * 255)
    color = '#%02x%02x%02x' % (R, G, B)
    plt.scatter(non_noise_points[non_noise_labels == i, 0], non_noise_points[non_noise_labels == i, 1], c=color)

# plt.show()

plt.show()

# # --------------------------------------------------------------綜合比較3個模型的分群結果與分析討論(Part 6)--------------------------------------------------------------


# 輸出各模型的評估指標

