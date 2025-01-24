import os
import json
import pandas as pd


def classify_and_rank_products(jsonl_dir: str, output_dir: str):
    """对“产品-AI”列进行分类并排序"""
    companies_file = os.path.join(jsonl_dir, '1.01石墨烯相关企业标签v3.2_20241022.jsonl')

    if not os.path.exists(companies_file):
        print(f"Error: {companies_file} not found!")
        return

    # 读取和处理数据
    companies_data = []
    with open(companies_file, 'r', encoding='utf-8') as f:
        for line in f:
            companies_data.append(json.loads(line))

    df = pd.DataFrame(companies_data)

    # 确保必要列存在
    required_columns = ['企业名称', '总分', '产品-AI']
    for col in required_columns:
        if col not in df.columns:
            print(f"Error: Missing required column '{col}' in data!")
            return

    # 确保分数是数值类型
    df['总分'] = pd.to_numeric(df['总分'], errors='coerce')
    df = df.dropna(subset=['总分'])

    # 按“产品-AI”列进行拆分
    df['产品-AI'] = df['产品-AI'].str.split('//')
    expanded_df = df.explode('产品-AI')
    expanded_df['产品-AI'] = expanded_df['产品-AI'].str.strip()

    # 按“产品-AI”分类和排序
    rankings_by_product = {}

    for product, group in expanded_df.groupby('产品-AI'):
        # 对每个产品类别的企业按分数排序
        ranked_companies = group.sort_values(by='总分', ascending=False)

        # 生成该产品类别的排名数据
        product_rankings = []
        for idx, row in ranked_companies.iterrows():
            company_info = {
                'rank': len(product_rankings) + 1,
                'name': row['企业名称'],
                'province': row.get('所属省份', ''),
                'city': row.get('所属城市', ''),
                'county': row.get('所属区县', ''),
                'tag': row.get('企业标签', ''),
                'patent_count': row.get('石墨烯相关发明专利数量', 0),
                'product': row['产品-AI'],
                'score': float(row['总分']),
                'type': row.get('企业类型', '')
            }
            product_rankings.append(company_info)

        rankings_by_product[product] = product_rankings

    # 保存文件
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'company_rankings_by_product.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(rankings_by_product, f, ensure_ascii=False, indent=2)

    print(f"Company rankings by product category saved to {output_file}")


# 示例调用
classify_and_rank_products(jsonl_dir='./data/jsonl', output_dir='./data/jsonl')
