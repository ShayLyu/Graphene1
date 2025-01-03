import os
from langchain_community.document_loaders import PyPDFLoader, JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from typing import Optional, List, Set
from dotenv import load_dotenv
from pathlib import Path
import json
from langchain_core.documents import Document
import time
import pickle
from openai import OpenAI
from langchain.embeddings.base import Embeddings

# 获取当前脚本的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 构建.env文件的完整路径
env_path = os.path.join(current_dir, '.env')
# 显式指定.env文件路径
load_dotenv(dotenv_path=env_path)

class CustomOpenAIEmbeddings(Embeddings):
    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-v3",
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        dimensions: int = 1024,
        encoding_format: str = "float"
    ):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = model
        self.dimensions = dimensions
        self.encoding_format = encoding_format

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts,
                dimensions=self.dimensions,
                encoding_format=self.encoding_format
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            print(f"Embedding error: {e}")
            raise

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=[text],
                dimensions=self.dimensions,
                encoding_format=self.encoding_format
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Embedding error: {e}")
            raise

class KnowledgeBase:
    def __init__(
        self, 
        data_dir: str, 
        embedding_model: str = 'text-embedding-v3',
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        base_url: str = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
    ):
        self.data_dir = data_dir
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.base_url = base_url
        
    def _get_pdf_files(self) -> List[str]:
        """获取目录下所有PDF文件"""
        pdf_files = []
        for filename in os.listdir(self.data_dir):
            if filename.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(self.data_dir, filename))
        return pdf_files
    
    def _get_jsonl_files(self) -> List[str]:
        """获取目录下所有JSONL文件"""
        jsonl_dir = os.path.join(self.data_dir, 'jsonl')
        if not os.path.exists(jsonl_dir):
            return []
        
        jsonl_files = []
        for filename in os.listdir(jsonl_dir):
            if filename.lower().endswith('.jsonl'):
                jsonl_files.append(os.path.join(jsonl_dir, filename))
        return jsonl_files

    def _process_jsonl_files(self) -> List[dict]:
        """处理所有JSONL文件"""
        documents = []
        jsonl_files = self._get_jsonl_files()
        
        for jsonl_file in jsonl_files:
            print(f"Processing JSONL file: {jsonl_file}")
            try:
                with open(jsonl_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        data = json.loads(line)
                        # 将JSON对象转换为字符串
                        content = json.dumps(data, ensure_ascii=False)
                        # 创建 Document 对象
                        doc = Document(
                            page_content=content,
                            metadata={
                                'source': os.path.basename(jsonl_file),
                                'type': 'jsonl'
                            }
                        )
                        documents.append(doc)
            except Exception as e:
                print(f"Error processing {jsonl_file}: {e}")
        
        return documents

    def _save_progress(self, processed_indices: Set[int], save_path: str):
        """保存处理进度"""
        progress_file = os.path.join(save_path, 'processing_progress.pkl')
        with open(progress_file, 'wb') as f:
            pickle.dump(processed_indices, f)

    def _load_progress(self, save_path: str) -> Set[int]:
        """加载处理进度"""
        progress_file = os.path.join(save_path, 'processing_progress.pkl')
        if os.path.exists(progress_file):
            with open(progress_file, 'rb') as f:
                return pickle.load(f)
        return set()

    def _process_batch_with_retry(self, 
        batch: List[Document], 
        embeddings, 
        vectorstore: Optional[FAISS], 
        max_retries: int = 3,
        retry_delay: int = 5
    ) -> Optional[FAISS]:
        """处理单个批次，带重试机制"""
        for attempt in range(max_retries):
            try:
                if vectorstore is None:
                    vectorstore = FAISS.from_documents(batch, embeddings)
                else:
                    vectorstore.add_documents(batch)
                return vectorstore
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Error processing batch (attempt {attempt + 1}/{max_retries}): {e}")
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print(f"Failed to process batch after {max_retries} attempts: {e}")
                    raise
        return None

    def build_knowledge_base(
        self, 
        save_path: Optional[str] = None, 
        batch_size: int = 100,
        max_retries: int = 3,
        retry_delay: int = 5
    ) -> FAISS:
        """构建知识库，支持断点续跑和重试机制"""
        all_documents = []
        vectorstore = None
        
        # 处理PDF文件
        pdf_files = self._get_pdf_files()
        for pdf_path in pdf_files:
            print(f"Processing PDF: {pdf_path}")
            try:
                loader = PyPDFLoader(pdf_path)
                documents = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size, 
                    chunk_overlap=self.chunk_overlap
                )
                texts = text_splitter.split_documents(documents)
                all_documents.extend(texts)
            except Exception as e:
                print(f"Error processing PDF {pdf_path}: {e}")
        
        # 处理JSONL文件
        jsonl_documents = self._process_jsonl_files()
        all_documents.extend(jsonl_documents)
        
        if not all_documents:
            raise ValueError(f"No documents found in {self.data_dir}")
        
        # 创建自定义的 Embedding
        embeddings = CustomOpenAIEmbeddings(
            api_key=os.environ.get('DASHSCOPE_API_KEY'),
            model=self.embedding_model,
            base_url=self.base_url
        )
        
        # 加载之前的处理进度
        processed_indices = set()
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            processed_indices = self._load_progress(save_path)
            if processed_indices:
                print(f"Resuming from previous progress: {len(processed_indices)} documents already processed")
                # 如果有之前的向量库，加载它
                if os.path.exists(os.path.join(save_path, 'index.faiss')):
                    vectorstore = FAISS.load_local(save_path, embeddings)
        
        # 分批处理文档
        total_docs = len(all_documents)
        for i in range(0, total_docs, batch_size):
            # 如果这个批次已经处理过，跳过
            if all(idx in processed_indices for idx in range(i, min(i + batch_size, total_docs))):
                continue
                
            batch = all_documents[i:i + batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(total_docs + batch_size - 1)//batch_size}")
            
            try:
                # 处理批次，带重试机制
                vectorstore = self._process_batch_with_retry(
                    batch, 
                    embeddings, 
                    vectorstore,
                    max_retries,
                    retry_delay
                )
                
                # 更新处理进度
                processed_indices.update(range(i, min(i + batch_size, total_docs)))
                
                # 保存进度和向量库
                if save_path:
                    self._save_progress(processed_indices, save_path)
                    vectorstore.save_local(save_path)
                    print(f"Saved progress after batch {i//batch_size + 1}")
                    
            except Exception as e:
                print(f"Fatal error processing batch: {e}")
                print(f"Progress saved. You can resume from the last successful batch.")
                if vectorstore and save_path:
                    vectorstore.save_local(save_path)
                raise
        
        # 清理进度文件
        if save_path:
            progress_file = os.path.join(save_path, 'processing_progress.pkl')
            if os.path.exists(progress_file):
                os.remove(progress_file)
        
        print(f"Processed {len(pdf_files)} PDF files and {len(self._get_jsonl_files())} JSONL files")
        print(f"Total documents: {total_docs}")
        
        return vectorstore

def main():
    # 使用示例
    data_dir = "./data"  # 指定数据目录
    
    # 创建知识库构建器
    kb_builder = KnowledgeBase(
        data_dir, 
        chunk_size=1000,  # 可调整
        chunk_overlap=50,  # 可调整
        base_url='https://dashscope.aliyuncs.com/compatible-mode/v1'
    )
    
    # 构建并保存知识库，设置批处理大小和重试参数
    vectorstore = kb_builder.build_knowledge_base(
        save_path="./knowledge_base",
        batch_size=20,  # 每批处理100个文档
        max_retries=1,   # 最大重试次数
        retry_delay=5    # 重试延迟（秒）
    )
    
    print("知识库构建完成")

if __name__ == "__main__":
    main()