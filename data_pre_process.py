import pandas as pd
import json
import os
from pathlib import Path

def excel_to_jsonl(input_dir: str, output_dir: str):
    """
    将指定目录下的所有 xlsx 文件转换为 jsonl 格式
    
    Args:
        input_dir: 输入目录路径，包含 xlsx 文件
        output_dir: 输出目录路径，用于保存 jsonl 文件
    """
    # 创建输出目录（如果不存在）
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 获取所有 xlsx 文件
    xlsx_files = [f for f in os.listdir(input_dir) if f.endswith(('.xlsx', '.xls'))]
    
    for xlsx_file in xlsx_files:
        input_path = os.path.join(input_dir, xlsx_file)
        output_path = os.path.join(output_dir, f"{os.path.splitext(xlsx_file)[0]}.jsonl")
        
        print(f"Processing {xlsx_file}...")
        
        try:
            # 读取 Excel 文件
            df = pd.read_excel(input_path)
            
            # 删除所有空行
            df = df.dropna(how='all')
            
            # 删除完全重复的行
            df = df.drop_duplicates()
            
            # 如果是专利数据文件，进行特殊处理
            if '专利共同发明人-处理后' in df.columns:
                # 重命名列
                df = df.rename(columns={'专利共同发明人-处理后': '专利发明人'})
                # 删除原始的发明人列
                if '发明人-处理后' in df.columns:
                    df = df.drop(columns=['发明人-处理后'])
            
            # 将每一行转换为 JSON 并写入文件
            with open(output_path, 'w', encoding='utf-8') as f:
                for _, row in df.iterrows():
                    # 将行数据转换为字典，并处理 NaN 值
                    row_dict = row.to_dict()
                    cleaned_dict = {
                        k: ('' if pd.isna(v) else str(v).strip())
                        for k, v in row_dict.items()
                    }
                    
                    # 写入 JSONL 文件
                    f.write(json.dumps(cleaned_dict, ensure_ascii=False) + '\n')
            
            print(f"Successfully converted {xlsx_file} to {os.path.basename(output_path)}")
            
        except Exception as e:
            print(f"Error processing {xlsx_file}: {str(e)}")

def generate_expert_rankings(jsonl_dir: str, output_dir: str, top_n: int = 250):
    """从 experts.jsonl 生成简化版的专家排名文件"""
    experts_file = os.path.join(jsonl_dir, 'experts.jsonl')
    
    if not os.path.exists(experts_file):
        print(f"Error: {experts_file} not found!")
        return
    
    # 读取和处理数据
    experts_data = []
    with open(experts_file, 'r', encoding='utf-8') as f:
        for line in f:
            experts_data.append(json.loads(line))
    
    df = pd.DataFrame(experts_data)
    df['专利计数项'] = pd.to_numeric(df['专利计数项'])
    
    # 筛选并排序
    domestic_experts = df[df['学者来源'] == '国内']
    ranked_experts = domestic_experts.sort_values(by='专利计数项', ascending=False).head(top_n)
    
    # 简化输出数据
    output_data = []
    for idx, row in ranked_experts.iterrows():
        expert_info = {
            'rank': len(output_data) + 1,
            'name': row['发明人'],
            'patents': int(row['专利计数项']),
            'title': row['人物职称'],
            'province': row['所属省份'],
            'research_field': row['研究领域']
        }
        output_data.append(expert_info)
    
    # 保存文件
    output_file = os.path.join(output_dir, 'expert_rankings.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"Expert rankings saved to {output_file}")

def generate_company_rankings(jsonl_dir: str, output_dir: str):
    """生成各省企业排名文件"""
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
    
    # 按省份分组，每个省份内部按分数排序
    rankings_by_province = {}
    
    for province, group in df.groupby('所属省份'):
        # 对每个省份内的企业按分数排序
        ranked_companies = group.sort_values(by='总分', ascending=False)
        
        # 生成该省的排名数据
        province_rankings = []
        for idx, row in ranked_companies.iterrows():
            company_info = {
                'rank': len(province_rankings) + 1,
                'name': row['企业名称'],
                'city': row['所属城市'],
                'county': row['所属区县'],
                'tag': row['企业标签'],
                'patent_count': row['石墨烯相关发明专利数量'],
                'product': row['产品-AI'],
                'score': float(row['总分']),
                'type': row.get('企业类型', '')  # 如果有企业类型的话
            }
            province_rankings.append(company_info)
        
        rankings_by_province[province] = province_rankings
    
    # 保存文件
    output_file = os.path.join(output_dir, 'company_rankings.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(rankings_by_province, f, ensure_ascii=False, indent=2)
    
    print(f"Company rankings saved to {output_file}")

def main():
    # 设置输入和输出目录
    input_dir = "./data"
    output_dir = "./data/jsonl"
    
    # 创建输出目录（如果不存在）
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 执行 Excel 到 JSONL 的转换
    excel_to_jsonl(input_dir, output_dir)
    
    # 生成专家排名
    generate_expert_rankings(output_dir, output_dir)
    
    # 生成企业排名
    generate_company_rankings(output_dir, output_dir)
    
    print("\nAll processing completed!")

if __name__ == "__main__":
    main() 