import streamlit as st
from rag import PDFKnowledgeBaseQA
import os
import json
from datetime import datetime

def save_conversation_to_json(conversation, filename):
    """Save conversation history to a JSON file."""
    os.makedirs('conversations', exist_ok=True)
    filepath = os.path.join('conversations', filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(conversation, f, ensure_ascii=False, indent=4)

def load_conversation_from_json(filepath):
    """Load conversation history from a JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def delete_conversation_file(filename):
    """Delete a conversation file."""
    filepath = os.path.join('conversations', filename)
    if os.path.exists(filepath):
        os.remove(filepath)

def reset_conversation():
    """清空会话历史记录，开始新对话。"""
    st.session_state.conversation_history = []
    st.session_state.current_conversation_file = None
    st.session_state.loaded_conversation = False

def extract_first_user_question(filepath):
    """从历史记录文件中提取用户提出的第一个问题，并限制为11个字，超出部分省略"""
    try:
        conversation = load_conversation_from_json(filepath)
        # 提取第一个用户问题
        first_user_question = next((message for role, message in conversation if role == '用户'), "No question")
        # 替换换行符为空格
        sanitized_question = first_user_question.replace("\n", " ").replace("\r", " ")
        # 限制为11个字，超出部分用省略号替代
        return sanitized_question[:6] + "..." if len(sanitized_question) > 6 else sanitized_question
    except Exception:
        return "加载失败"

def generate_filename_from_conversation(conversation):
    """根据对话内容生成文件名"""
    if conversation:
        first_user_message = next((message for role, message in conversation if role == '用户'), "new_conversation")
        sanitized_message = ''.join(c for c in first_user_message[:10] if c.isalnum() or c in (' ', '_', '-')).strip()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{sanitized_message}_{timestamp}.json"
    else:
        return f"new_conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

def main():
    # 设置页面配置
    st.set_page_config(
        page_title="石墨烯助手",
        page_icon="⬡",
        layout="wide"
    )

    # 初始化会话状态
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
        st.session_state.current_conversation_file = None
        st.session_state.loaded_conversation = False

    # 初始化知识库系统
    if 'qa_system' not in st.session_state:
        knowledge_base_path = "./knowledge_base"
        st.session_state.qa_system = PDFKnowledgeBaseQA(
            knowledge_base_path,
            model='qwen-plus',
            base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
        )

    # 创建 conversations 文件夹
    os.makedirs('conversations', exist_ok=True)

    # 侧边栏内容
    with st.sidebar:
        # 添加标题和新对话按钮
        col1, col2 = st.sidebar.columns([4, 1])  # 分为两列
        with col1:
            st.markdown("### 📜 对话记录")
        with col2:
            if st.button("➕"):
                reset_conversation()
                st.rerun()  # 强制页面刷新

        # 获取历史会话文件
        conversation_files = [f for f in os.listdir('conversations') if f.endswith('.json')]
        conversation_files_sorted = sorted(conversation_files, key=lambda x: os.path.getmtime(os.path.join('conversations', x)), reverse=True)

        # 显示历史会话
        for conv_file in conversation_files_sorted:
            col1, col2 = st.sidebar.columns([8, 2])  # 文件名占 80%，删除按钮占 20%
            with col1:
                # 提取第一个用户问题作为按钮文字，限制为11个字
                button_text = extract_first_user_question(os.path.join('conversations', conv_file))
                if st.button(button_text, key=f"load_{conv_file}"):
                    selected_filepath = os.path.join('conversations', conv_file)
                    try:
                        loaded_conversation = load_conversation_from_json(selected_filepath)
                        st.session_state.conversation_history = loaded_conversation
                        st.session_state.current_conversation_file = conv_file
                        st.session_state.loaded_conversation = True
                    except Exception as e:
                        st.sidebar.error(f"加载会话失败: {e}")
            with col2:
                if st.button("🗑", key=f"delete_{conv_file}"):
                    # 删除文件并刷新页面
                    delete_conversation_file(conv_file)
                    st.rerun()  # 强制页面刷新

    # 主内容区域
    st.markdown("### 💡 知识问答")

    # 提交问题
    with st.form(key='qa_form'):
        query = st.text_input("请输入您的问题", placeholder="在这里输入您想了解的石墨烯相关内容...", key="query_input")
        submit_button = st.form_submit_button("提交问题")

    if submit_button and query:
        if 'last_query' not in st.session_state or st.session_state.last_query != query:
            with st.spinner('正在为您查找答案...'):
                try:
                    result = st.session_state.qa_system.ask_question(query)
                    answer = result.get('answer', '未找到相关答案')
                except Exception as e:
                    answer = f"查询时出现错误: {str(e)}"

                st.session_state.last_query = query
                st.session_state.last_result = answer
                st.session_state.conversation_history.append(('用户', query))
                st.session_state.conversation_history.append(('助手', answer))

                if st.session_state.current_conversation_file:
                    save_conversation_to_json(st.session_state.conversation_history, st.session_state.current_conversation_file)
                else:
                    filename = generate_filename_from_conversation(st.session_state.conversation_history)
                    st.session_state.current_conversation_file = filename
                    save_conversation_to_json(st.session_state.conversation_history, filename)
        else:
            answer = st.session_state.last_result

        st.markdown("### 🤖 智能回复")
        st.write(answer)

    # 显示历史对话
    if st.session_state.conversation_history:
        st.markdown("### 📝 历史对话")
        for role, message in st.session_state.conversation_history:
            if role == '用户':
                st.markdown(f"**用户**: {message}")
            else:
                st.markdown(f"**助手**: {message}")
                st.divider()

if __name__ == "__main__":
    main()
