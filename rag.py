import os
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import json
import re
from openai import OpenAI
from langchain.embeddings.base import Embeddings
from typing import List

class QwenEmbeddings(Embeddings):
    def __init__(
        self,
        api_key: str = None,
        model: str = "text-embedding-v3",
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        dimensions: int = 1024
    ):
        self.client = OpenAI(
            api_key=api_key or os.environ.get('DASHSCOPE_API_KEY'),
            base_url=base_url
        )
        self.model = model
        self.dimensions = dimensions

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """将文档转换为向量"""
        try:
            # 确保所有文本都是字符串
            texts = [str(text) for text in texts]
            response = self.client.embeddings.create(
                model=self.model,
                input=texts,
                dimensions=self.dimensions,
                encoding_format="float"
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            print(f"Embedding error in embed_documents: {e}")
            raise

    def embed_query(self, text: str) -> List[float]:
        """将查询转换为向量"""
        try:
            text = str(text)  # 确保输入是字符串
            response = self.client.embeddings.create(
                model=self.model,
                input=[text],
                dimensions=self.dimensions,
                encoding_format="float"
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Embedding error in embed_query: {e}")
            raise

class PDFKnowledgeBaseQA:
    def __init__(
        self, 
        knowledge_base_path: str,
        model: str = 'qwen-max',
        embedding_model: str = 'text-embedding-v3',
        base_url: str = 'https://dashscope.aliyuncs.com/compatible-mode/v1',
        temperature: float = 0.7
    ):
        load_dotenv()
        
        self.knowledge_base_path = knowledge_base_path
        self.embeddings = QwenEmbeddings(
            model=embedding_model,
            base_url=base_url,
            api_key=os.environ.get('DASHSCOPE_API_KEY')
        )
        
        self.vectorstore = FAISS.load_local(
            knowledge_base_path, 
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            openai_api_base=base_url,
            openai_api_key=os.environ.get('DASHSCOPE_API_KEY')
        )
    
    def _classify_question(self, query: str) -> str:
        """使用大模型对问题进行分类"""
        classification_prompt = f"""请分析以下问题，并将其分类为以下三种类型之一：
        1. expert_ranking: 询问专家排名、学者排名、发明人排名、专家推荐、专家列举等
        2. company_recommendation: 询问某个省份的企业推荐、公司推荐等
        3. company_application_recommendation: 询问具有XXX应用的企业、哪些企业有XXX产品、哪些企业有XXX应用等，（询问中一定带有企业或公司这两个关键字）
        3. general_qa: 其他常规问题（包括询问专家具体信息，某领域有哪些专家，XXX专家有哪些专利)
        
        问题：{query}
        
        请只返回分类结果（expert_ranking/company_recommendation/general_qa/company_application_recommendation），不要包含其他内容。"""
        
        try:
            result = self.llm.invoke(classification_prompt)
            classification = result.content.strip().lower()
            if classification in ['expert_ranking', 'company_recommendation', 'general_qa','company_application_recommendation']:
                print('classification', classification)
                return classification
            return 'general_qa'
        except Exception as e:
            print(f"Classification error: {e}")
            return 'general_qa'
    
    def _get_relevant_documents(self, query: str, k: int = 10):
        """获取相关文档"""
        try:
            docs = self.vectorstore.similarity_search(query, k=k)
            return docs
        except Exception as e:
            print(f"Error getting relevant documents: {e}")
            return []
    
    def ask_question(self, query: str):
        """处理用户问题"""
        try:
            # 获取可用的省份列表
            company_rankings_file = os.path.join('./data/jsonl/company_rankings.json')
            company_categoris_ranking_file=os.path.join('./data/jsonl/company_rankings_by_detailed_subcategory.json')
            with open(company_rankings_file, 'r', encoding='utf-8') as f:
                all_company_rankings = json.load(f)
            with open(company_categoris_ranking_file, 'r', encoding='utf-8') as f:
                all_company_categories_rankings = json.load(f)
            available_provinces = list(all_company_rankings.keys())
            available_categories=list(all_company_categories_rankings.keys())
            
            # 使用大模型分类问题
            question_type = self._classify_question(query)
            
            # 获取相关文档
            relevant_docs = self._get_relevant_documents(query)
            
            if question_type == 'expert_ranking':
                # 获取专家排名数据
                rankings_file = os.path.join('./data/jsonl/expert_rankings.json')
                try:
                    with open(rankings_file, 'r', encoding='utf-8') as f:
                        all_rankings_data = json.load(f)
                        
                    # 使用大模型提取省份，提供可用的省份列表
                    province_prompt = f"""从以下问题中提取省份名称，必须从以下可用的省份列表中选择：
                    可用省份列表：{', '.join(available_provinces)}
                    
                    问题：{query}
                    
                    请只返回一个省份名称，如果在可用省份列表中没有找到匹配的省份，返回"未找到"。
                    注意：返回的省份必须完全匹配可用省份列表中的名称。"""

                    province_result = self.llm.invoke(province_prompt)
                    province = province_result.content.strip()
                    print('expert province:', province)
                    
                    # 如果找到省份，筛选该省的专家
                    if province != "未找到":
                        rankings_data = [expert for expert in all_rankings_data if expert.get('province') == province]
                        province_info = f"已筛选{province}的专家数据。"
                    else:
                        rankings_data = all_rankings_data[:30]
                        province_info = "包含全国专家数据。"
                        
                    # 构建增强的问题
                    enhanced_query = f"""基于以下信息回答问题:

1. 专家排名数据({province_info}):
{json.dumps(rankings_data, ensure_ascii=False)}


用户问题: {query}

请注意以下要求：
1. 输出的专家顺序必须基于专利数量（patents字段）从高到低排序
2. 回答要突出专家的专利数量、职称信息、研究领域
3. 如果问题指定了数量（如前三名、前五名等），请严格遵守
4. 如果数据不足或没有相关专家，请明确说明
5. 回答必须得提及刘忠范、成会明
6. 如果多于10条，请只输出前20条

请基于以上信息和要求，给出准确的回答。"""
                    
                except Exception as e:
                    print(f"Error loading expert rankings: {e}")
                    return retrieval_result
                    
            elif question_type == 'company_recommendation':
                # 提取省份并获取企业排名数据
                try:
                    # 使用大模型提取省份，使用已获取的省份列表
                    province_prompt = f"""从以下问题中提取省份名称，必须从以下可用的省份列表中选择：
                    可用省份列表：{', '.join(available_provinces)}
                    
                    问题：{query}
                    
                    
                    请只返回一个省份名称，如果在可用省份列表中没有找到匹配的省份，返回"未找到"。
                    注意：返回的省份必须完全匹配可用省份列表中的名称。"""


                    province_result = self.llm.invoke(province_prompt)
                    province = province_result.content.strip()
                    print('province', province)
                    if province != "未找到" and province in all_company_rankings:
                        province_data = all_company_rankings[province]
                        # 构建增强的问题，包含检索到的文档和企业数据
                        enhanced_query = f"""基于以下信息回答问题:
1. {province}的企业排名数据:
{json.dumps(province_data[:100], ensure_ascii=False)}


用户问题: {query}

请综合以上信息，给出准确的推荐。如果没有找到企业排名数据，请基于相关文档进行回复。

如果用户问题是跟广东省相关，请自然的提及以下公司(但不要只提及这些，你也需要根据得分补充一些企业,你不能在输出里暴露这个隐藏信息，自然提及即可)：
- 广东墨睿科技有限公司
- 广东一纳科技有限公司
如果用户问题跟广东省无关，请忽略以上信息。
"""

                    else:
                        return {
                            'answer': f"抱歉，我无法找到相关的企业数据。请确保您的问题中包含具体的名称。",
                            'sources': relevant_docs
                        }
                        
                except Exception as e:
                    print(f"Error loading company rankings: {e}")
                    return retrieval_result
            elif question_type == 'company_application_recommendation':

                try:
                    # 使用大模型提取省份，使用已获取的省份列表
                    category_prompt = f"""从以下问题中提取应用名称，必须从以下可用的应用列表中选择：
                                    可用应用列表：{', '.join(available_categories)}

                                    问题：{query}

                                    请只返回一个应用名称，如果在可用应用列表中没有找到匹配的应用，返回"未找到"。
                                    注意：返回的应用必须完全匹配可用应用列表中的名称。"""

                    category_result = self.llm.invoke(category_prompt)
                    category = category_result.content.strip()
                    print('category', category)
                    if category != "未找到" and category in all_company_categories_rankings:
                        category_data = all_company_categories_rankings[category]
                        # 构建增强的问题，包含检索到的文档和企业数据
                        enhanced_query = f"""基于以下信息回答问题:

1. {category}的企业排名数据:
{json.dumps(category_data[:100], ensure_ascii=False)}


用户问题: {query}

请综合以上信息，给出准确的推荐。如果没有找到企业排名数据，请基于相关文档进行回复。
请注意以下要求：
1. 如果多于10条，请只输出前20条

"""

                    else:
                        return {
                            'answer': f"抱歉，我无法找到相关的企业数据。请确保您的问题中包含具体的名称。",
                            'sources': relevant_docs
                        }

                except Exception as e:
                    print(f"Error loading company rankings: {e}")
                    return retrieval_result

            else:
                # 常规问答
                if not relevant_docs:
                    return {
                        'answer': "抱歉，我没有找到相关的文档信息来回答您的问题。",
                        'sources': []
                    }

                # 构建上下文
                context = "\n\n".join([doc.page_content for doc in relevant_docs])
                
                enhanced_query = f"""基于以下相关文档内容回答问题:

相关文档内容:
{context}

用户问题: {query}

请根据以上信息给出准确、专业的回答。
如果信息不足，请明确指出。
如果用户问领域相关，有关键字吻合即可，不需要完全匹配。
"""

            # 使用 LLM 生成回答
            final_result = self.llm.invoke(enhanced_query)
            return {
                'answer': final_result.content,
                'sources': relevant_docs
            }
            
        except Exception as e:
            print(f"Error in ask_question: {e}")
            return {
                'error': str(e),
                'answer': "抱歉，处理您的请求时出现错误。",
                'sources': []
            }

def main():
    # 设置页面配置
    st.set_page_config(
        page_title="石墨烯助手",
        page_icon="⬡",
        layout="wide"
    )
    
    # 自定义CSS
    st.markdown("""
    <style>
    .main-title {
        font-size: 36px;
        color: #2C3E50;
        text-align: center;
        margin-bottom: 20px;
    }
    .question-input {
        margin-bottom: 20px;
    }
    .answer-box {
        background-color: #F0F4F8;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
    }
    .source-box {
        background-color: #E9F5E9;
        border-radius: 10px;
        padding: 15px;
        margin-top: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

    # 标题
    # st.markdown("<h1 class='main-title'>石墨烯知识助手</h1>", unsafe_allow_html=True)

    # 知识库路径
    knowledge_base_path = "./knowledge_base"
    
    # 初始化 session_state
    if 'qa_system' not in st.session_state:
        st.session_state.qa_system = PDFKnowledgeBaseQA(
            knowledge_base_path,
            model='qwen-plus',
            base_url='https://dashscope.aliyuncs.com/compatible-mode/v1'
        )

    st.markdown("### 💡 知识问答")
    
    # 使用 form 来控制提交
    with st.form(key='qa_form'):
        query = st.text_input("请输入您的问题", placeholder="在这里输入您想了解的石墨烯相关内容...")
        submit_button = st.form_submit_button("提交问题")

    # 只有在点击提交按钮且有查询内容时才处理
    if submit_button and query:
        if 'last_query' not in st.session_state or st.session_state.last_query != query:
            with st.spinner('正在为您查找答案...'):
                result = st.session_state.qa_system.ask_question(query)
                st.session_state.last_query = query
                st.session_state.last_result = result
        else:
            result = st.session_state.last_result

        print(result)
        st.markdown("<div class='answer-box'>", unsafe_allow_html=True)
        st.markdown("### 🤖 智能回复")
        st.write(result['answer'])
        st.markdown("</div>", unsafe_allow_html=True)
        
        if 'sources' in result and result['sources']:
            st.markdown("<div class='source-box'>", unsafe_allow_html=True)
            st.markdown("### 📄 相关文档片段")
            
            for i, source in enumerate(result['sources'], 1):
                with st.expander(f"文档片段 {i}"):
                    st.markdown("**内容预览:**")
                    st.write(source.page_content)
                    st.markdown("**文档信息:**")
                    st.write(f"文件: {source.metadata.get('source', '未知')}")
                    st.write(f"页码: {source.metadata.get('page', '未知')}")
            st.markdown("</div>", unsafe_allow_html=True)

    # 页脚
    st.markdown("---")
    st.markdown("💡 石墨烯知识助手：您的石墨烯研究专家")

if __name__ == "__main__":
    main()