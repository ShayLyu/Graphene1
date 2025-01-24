import json
import os
import hdbscan
from sentence_transformers import SentenceTransformer


def hdbscan_clustering(input_file: str, min_cluster_size=3):
    """基于 HDBSCAN 进行自动聚类，对小簇不强行分类"""
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found!")
        return

    # 1. 读取 JSON 文件并提取所有键
    with open(input_file, 'r', encoding='utf-8') as f:
        rankings_data = json.load(f)

    product_categories = list(rankings_data.keys())
    print(f"共有 {len(product_categories)} 个类别(键)\n")

    # 2. 嵌入/向量化
    model = SentenceTransformer('GanymedeNil/text2vec-large-chinese')
    embeddings = model.encode(product_categories)

    # 3. 初次聚类
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean')
    labels = clusterer.fit_predict(embeddings)

    # 4. 收集初次聚类结果
    cluster_dict = {}
    for label, category in zip(labels, product_categories):
        if label == -1:
            cluster_name = "Noise"
        else:
            cluster_name = f"Cluster-{label}"
        cluster_dict.setdefault(cluster_name, []).append(category)

    print("初次 HDBSCAN 聚类结果：")
    for cluster_name, categories in cluster_dict.items():
        print(f"{cluster_name} 类别数量：{len(categories)}")
        sample_preview = ", ".join(categories[:10])  # 只预览前10个，避免打印过多
        print(f"  示例：{sample_preview}...\n")

    # 5. 找到特别大的簇，进行二次聚类(示例：对Cluster-1进行二次聚类)
    # 假设我们认为 1000 个以上就算“大”
    THRESHOLD = 1000
    for c_name in list(cluster_dict.keys()):
        categories_in_this_cluster = cluster_dict[c_name]
        if c_name != "Noise" and len(categories_in_this_cluster) >= THRESHOLD:
            print(f"对 {c_name} 进行二次聚类，原簇大小={len(categories_in_this_cluster)}")

            # 5.1 二次聚类：先对这些类别重新做 embedding
            sub_embeddings = model.encode(categories_in_this_cluster)
            # 这里可以用新的 min_cluster_size 做更细的拆分，比如调得小一些
            sub_clusterer = hdbscan.HDBSCAN(min_cluster_size=5, metric='euclidean')
            sub_labels = sub_clusterer.fit_predict(sub_embeddings)

            # 5.2 生成新的子簇
            sub_cluster_dict = {}
            for lbl, cat in zip(sub_labels, categories_in_this_cluster):
                if lbl == -1:
                    sub_c_name = f"{c_name}-Noise"
                else:
                    # 例如: 'Cluster-1' -> 'Cluster-1-0', 'Cluster-1-1', ...
                    sub_c_name = f"{c_name}-{lbl}"
                sub_cluster_dict.setdefault(sub_c_name, []).append(cat)

            # 5.3 把原来的 cluster_dict[c_name] 替换为新的子簇
            #    并在 cluster_dict 中删除原簇 c_name
            del cluster_dict[c_name]
            # 将新的子簇合并到 cluster_dict
            for new_c_name, sub_cats in sub_cluster_dict.items():
                cluster_dict[new_c_name] = sub_cats

    # 6. 最终结果输出
    print("\n二次聚类后的结果：")
    for cluster_name, categories in cluster_dict.items():
        print(f"{cluster_name} 类别数量：{len(categories)}")
        sample_preview = ", ".join(categories[:10])
        print(f"  示例：{sample_preview}...\n")


# 示例调用
if __name__ == "__main__":
    hdbscan_clustering(
        input_file='./data/jsonl/company_rankings_by_product.json',
        min_cluster_size=3
    )
