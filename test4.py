import json
import os
def count_categories_in_rankings(input_file: str):
    """统计每个类别的数量"""
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found!")
        return

    # 读取排名数据
    with open(input_file, 'r', encoding='utf-8') as f:
        rankings_data = json.load(f)

    # 统计每个产品类别的数量
    category_counts = {category: len(rankings) for category, rankings in rankings_data.items()}

    # 打印结果
    print(f"每个产品类别的数量：")
    for category, count in category_counts.items():
        print(f"{category}: {count} 个")

# 示例调用
count_categories_in_rankings(input_file='./data/jsonl/company_rankings_by_product.json')
