## 技术细节和知识背景

### 知识背景

行业的文档内容复杂且多样，通常包含大量的专业术语和复杂的信息架构。为了实现对文档的高效问答，本项目采用了**RAG（Retrieval-Augmented Generation）**技术。

- **RAG技术原理**：通过首先检索相关文档（Retrieval），然后在此基础上生成答案（Generation）。这种方法结合了信息检索和自然语言生成的优点，能够提供更加准确和上下文相关的答案。

- **向量数据库（FAISS）**：使用`FAISS`（Faceboook AI Similarity Search），一个高效的相似性搜索库，用于存储和快速查询文档片段的嵌入向量。

- **文本嵌入（Embeddings）**：使用OpenAI的文本嵌入模型（如`text-embedding-ada-002`），将文本转换为数值向量，以便通过向量相似性进行高效检索。

### 技术细节

- **PDF加载和文本处理**：项目中使用`langchain_community`库的`PyPDFLoader`来加载PDF文档，并使用`RecursiveCharacterTextSplitter`进行文本分块。这样可以处理大文档并保留上下文信息。

- **模型调用**：通过OpenAI的API，使用诸如`gpt-4o-mini`的模型生成回答。在API调用时，可以设置不同的参数，比如`temperature`用于控制生成的随机性。

- **用户界面**：使用`Streamlit`搭建了一个简易的Web界面，方便用户输入问题并查看系统回答。

## 环境配置和安装指南

在开始之前，请确保已安装Python 3.6以上的版本。

### 安装必要的软件包

在项目目录下，可以使用以下命令安装项目所需的库：

```bash
pip install langchain-openai langchain-community streamlit python-dotenv faiss-cpu
```

- `langchain-openai` 和 `langchain-community`：用于加载PDF、文本处理、嵌入生成以及向量库操作。
- `streamlit`: 用于构建交互式Web应用。
- `python-dotenv`: 用于从`.env`文件中加载环境变量。
- `faiss-cpu`: 用于高效的向量相似性检索。

### .env 文件配置

在项目目录下，创建一个`.env`文件，并添加您的OpenAI API密钥：

```
OPENAI_API_KEY=your_openai_api_key_here
```

确保用你的实际OpenAI API密钥替换`your_openai_api_key_here`。

### 运行项目

按照以下步骤来启动项目：

1. 构建知识库：

   运行 `data_process.py` 来构建知识库（请把所有pdf文件都放在`./data`下）：

   ```bash
   python data_process.py
   ```

2. 启动问答应用：

   使用Streamlit命令运行问答服务：

   ```bash
   streamlit run rag.py
   ```

打开浏览器并访问本地运行的Streamlit应用，输入问题与系统进行互动。

