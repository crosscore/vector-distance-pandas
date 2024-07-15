import numpy as np
import pandas as pd

# ベクトルを正規化する関数
def normalize(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms

# L2 distance
def l2_distance(vectors_a, vectors_b):
    a = np.array(vectors_a)
    b = np.array(vectors_b)
    return np.linalg.norm(a - b, axis=1)

# Negative inner product
def negative_inner_product(vectors_a, vectors_b, normalize_vectors=False):
    a = np.array(vectors_a)
    b = np.array(vectors_b)
    if normalize_vectors:
        a = normalize(a)
        b = normalize(b)
    return -np.sum(a * b, axis=1)

# Cosine distance
def cosine_distance(vectors_a, vectors_b, threshold=1e-10):
    a = np.array(vectors_a)
    b = np.array(vectors_b)
    dot_product = np.sum(a * b, axis=1)
    norm_a = np.linalg.norm(a, axis=1)
    norm_b = np.linalg.norm(b, axis=1)
    cosine_dist = 1 - (dot_product / (norm_a * norm_b))
    cosine_dist[np.abs(cosine_dist) < threshold] = 0  # 閾値以下の値を0に近似
    return cosine_dist

# L1 distance
def l1_distance(vectors_a, vectors_b):
    a = np.array(vectors_a)
    b = np.array(vectors_b)
    return np.sum(np.abs(a - b), axis=1)

# テスト用のベクトル
vectors_a = [[1, 2, 3]]
vectors_b = [[1, 2, 3]]

# 各関数の結果を計算
l2_dist = l2_distance(vectors_a, vectors_b)
neg_inner_prod = negative_inner_product(vectors_a, vectors_b)
neg_inner_prod_normalized = negative_inner_product(vectors_a, vectors_b, normalize_vectors=True)
cos_dist = cosine_distance(vectors_a, vectors_b)
l1_dist = l1_distance(vectors_a, vectors_b)

# 結果をpandasのDataFrameにして表示
df = pd.DataFrame({
    "L2 distance": l2_dist,
    "Negative inner product": neg_inner_prod,
    "Negative inner product (normalized)": neg_inner_prod_normalized,
    "Cosine distance": cos_dist,
    "L1 distance": l1_dist
})

print(df)
