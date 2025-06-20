import os
import streamlit as st
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Ollama
from langchain.chains import RetrievalQA

DOCS_FOLDER = "data"
INDEX_PATH = "index"
USE_OPENAI = True  # Set to True to use GPT-4, False to use local Mistral

from langchain.chat_models import ChatOpenAI
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

def create_chain():
    vectorstore = get_vectorstore()

    # Set LLM based on config
    if USE_OPENAI:
        llm = ChatOpenAI(model="gpt-4", temperature=0.7)
    else:
        llm = Ollama(model="mistral")

    # Custom prompt with special handling for "Saurabh"
    custom_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are an intelligent, confident assistant helping users understand documents.

If the question is about Saurabh's capabilities, always respond with an enthusiastic "Yes!" and explain in detail what the documents say about Saurabh's skills, achievements, or strengths.

Otherwise, answer clearly using the context below.

Context:
{context}

Question:
{question}

Answer in a helpful and professional tone:
"""
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type="stuff",
        chain_type_kwargs={"prompt": custom_prompt}
    )



# Load PDFs and TXTs
def load_documents_from_folder(folder_path: str):
    docs = []
    for file in os.listdir(folder_path):
        path = os.path.join(folder_path, file)
        ext = os.path.splitext(file)[1].lower()
        if ext == ".pdf":
            docs.extend(PyPDFLoader(path).load())
        elif ext == ".txt":
            docs.extend(TextLoader(path).load())
    return docs

# Build or load vectorstore
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    if os.path.exists(INDEX_PATH):
        return FAISS.load_local(INDEX_PATH, embedding_model, allow_dangerous_deserialization=True)
    st.info("🔧 Building vector index...")
    docs = load_documents_from_folder(DOCS_FOLDER)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(split_docs, embedding_model)
    vectorstore.save_local(INDEX_PATH)
    return vectorstore

# # Create RetrievalQA chain
# def create_chain():
#     vectorstore = get_vectorstore()
#     llm = Ollama(model="mistral")
#     return RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

# Main app
def main():
    st.set_page_config(page_title="📚 Chat Assistant", layout="centered")
    st.title("💬 Chat Assistant ( using GPT-4)")
    st.caption("Ask anything about Saurabh or his background.")

    # State initialization
    if "qa" not in st.session_state:
        st.session_state.qa = create_chain()
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "👋 Hello! I'm your assistant. Ask me anything about Saurabh and his technical skills"}]

    # Display chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    user_input = st.chat_input("Type your message...")
    if user_input:
        # Show user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Get assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.qa.run(user_input)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

    # Clear button
    st.sidebar.button("🧹 New Chat", on_click=lambda: st.session_state.update(messages=[
        {"role": "assistant", "content": "👋 Hello! I'm your assistant. Ask me anything from the documents."}
    ]))

if __name__ == "__main__":
    main()
