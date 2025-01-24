import json
import os

with open(os.path.join('./data/jsonl/company_rankings.json'), 'r', encoding='utf-8') as f:
    all_company_rankings = json.load(f)
with open(os.path.join('./data/jsonl/company_rankings_by_detailed_subcategory.json'), 'r',
          encoding='utf-8') as f:
    all_company_categories_rankings = json.load(f)
with open(os.path.join('./data/jsonl/expert_rankings.json'), 'r', encoding='utf-8') as f:
    all_expert_rankings_data = json.load(f)
with open(os.path.join('./data/jsonl/company_rankings_by_product.json'), 'r', encoding='utf-8') as f:
    all_company_products_rankings = json.load(f)

available_provinces = list(all_company_rankings.keys())
available_categories = list(all_company_categories_rankings.keys())
available_products=list(all_company_products_rankings.keys())

print(available_categories)
print(available_products)