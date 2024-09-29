from gensim.models import Word2Vec
from sklearn.cluster import KMeans
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from gensim.utils import simple_preprocess

# 示例文本数据
sentences = [
    "Barack Obama was born in Honolulu.",
    "Donald Trump was born in New York.",
    "Elon Musk founded SpaceX.",
    "Bill Gates founded Microsoft.",
    "Steve Jobs was the CEO of Apple.",
    "Sundar Pichai is the CEO of Google.",
    "Tim Cook is the CEO of Apple."
]

# 对文本进行预处理
sentences = [simple_preprocess(sentence) for sentence in sentences]

# 使用 Word2Vec 训练词嵌入模型
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1)

# 提取实体对及其词向量表示
entity_pairs = [
    ("Obama", "Honolulu"),
    ("Trump", "New York"),
    ("Musk", "SpaceX"),
    ("Gates", "Microsoft"),
    ("Jobs", "Apple"),
    ("Pichai", "Google"),
    ("Cook", "Apple")
]

# 获取实体对的向量表示
entity_vectors = []
for entity1, entity2 in entity_pairs:
    vector1 = model.wv[entity1.lower()] if entity1.lower() in model.wv else np.zeros(100)
    vector2 = model.wv[entity2.lower()] if entity2.lower() in model.wv else np.zeros(100)
    # 将两个实体的向量连接在一起，作为关系的特征表示
    entity_vectors.append(np.concatenate((vector1, vector2)))

# 将实体向量转化为 NumPy 数组
entity_vectors = np.array(entity_vectors)

# 使用 KMeans 进行聚类
num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
labels = kmeans.fit_predict(entity_vectors)

# 输出每个实体对所属的聚类
for i, (entity1, entity2) in enumerate(entity_pairs):
    print(f"Entity Pair: ({entity1}, {entity2}) - Cluster: {labels[i]}")

# 使用 PCA 降维并可视化结果
pca = PCA(n_components=2)
reduced_vectors = pca.fit_transform(entity_vectors)

plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], c=labels, cmap='viridis')
for i, (entity1, entity2) in enumerate(entity_pairs):
    plt.text(reduced_vectors[i, 0], reduced_vectors[i, 1], f"{entity1}-{entity2}")

plt.title('Relation Clustering using Word2Vec and KMeans')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()
