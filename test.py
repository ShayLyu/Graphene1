import os
import json
import pandas as pd

def generate_company_rankings(jsonl_dir: str, output_dir: str):
    """生成各省企业排名文件，按“小类”进一步细化分组"""
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

    # 确保分数是数值类型
    df['总分'] = pd.to_numeric(df['总分'])

    # 对“小类”列按逗号或其他分隔符进行进一步细分
    expanded_rows = []
    for idx, row in df.iterrows():
        subcategories = [subcategory.strip() for subcategory in str(row['小类']).split('、')]
        for subcategory in subcategories:
            new_row = row.copy()
            new_row['小类'] = subcategory
            expanded_rows.append(new_row)

    expanded_df = pd.DataFrame(expanded_rows)

    # 按“小类”分组，每个小类内部按分数排序
    rankings_by_subcategory = {}

    for subcategory, group in expanded_df.groupby('小类'):
        # 对每个小类内的企业按分数排序
        ranked_companies = group.sort_values(by='总分', ascending=False)

        # 生成该小类的排名数据
        subcategory_rankings = []
        for idx, row in ranked_companies.iterrows():
            company_info = {
                'rank': len(subcategory_rankings) + 1,
                'name': row['企业名称'],
                'province': row['所属省份'],
                'city': row['所属城市'],
                'county': row['所属区县'],
                'tag': row['企业标签'],
                'patent_count': row['石墨烯相关发明专利数量'],
                'product': row['产品-AI'],
                'score': float(row['总分']),
                'type': row.get('企业类型', '')  # 如果有企业类型的话
            }
            subcategory_rankings.append(company_info)

        rankings_by_subcategory[subcategory] = subcategory_rankings

    # 保存文件
    output_file = os.path.join(output_dir, 'company_rankings_by_detailed_subcategory.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(rankings_by_subcategory, f, ensure_ascii=False, indent=2)

    print(f"Company rankings by detailed subcategory saved to {output_file}")

# 示例调用
generate_company_rankings(jsonl_dir='./data/jsonl', output_dir='./data/jsonl')
