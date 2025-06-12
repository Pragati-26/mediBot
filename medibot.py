import sys
import asyncio

# ✅ Fix for Windows asyncio event loop error
if sys.platform.startswith('win') and sys.version_info >= (3, 8):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import os

import os
streamlit_config_path = os.path.join(os.getcwd(), ".streamlit")
os.makedirs(streamlit_config_path, exist_ok=True)
os.environ["STREAMLIT_CONFIG_DIR"] = streamlit_config_path

import streamlit as st
from dotenv import load_dotenv, find_dotenv

from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# ✅ Load .env variables
load_dotenv(find_dotenv())

# Path to saved FAISS vector store
DB_FAISS_PATH = "vectorstore/db_faiss"  # ✅ Fix here


# ✅ Cache the vector store loader to avoid reloading every time
@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

# ✅ Prompt template function
def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

# ✅ Load the LLM from Hugging Face
def load_llm(huggingface_repo_id, HF_TOKEN):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        huggingfacehub_api_token=HF_TOKEN,  # ✅ correct argument name
        temperature=0.5,
        max_new_tokens=512
    )
    return llm

# ✅ Main Streamlit app
def main():
    st.title("🤖 MediBot - Your Medical Query Assistant")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display previous messages
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    # Input from user
    prompt = st.chat_input("Ask your medical question here...")

    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        CUSTOM_PROMPT_TEMPLATE = """
        Use the pieces of information provided in the context to answer the user's question.
        If you don't know the answer, say you don't know. Do NOT make up answers.
        Stick strictly to the context.

        Context: {context}
        Question: {question}

        Start your answer directly:
        """

        HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
        HF_TOKEN = os.environ.get("HF_TOKEN")

        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("❌ Could not load vector store.")
                return

            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(HUGGINGFACE_REPO_ID, HF_TOKEN),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True,
                chain_type_kwargs={"prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            response = qa_chain.invoke({"query": prompt})
            result = response["result"]
            sources = "\n\n".join([f"- {doc.metadata.get('source', 'Unknown')}" for doc in response["source_documents"]])

            final_output = f"{result}\n\n📚 **Source Documents**:\n{sources}"
            st.chat_message("assistant").markdown(final_output)
            st.session_state.messages.append({"role": "assistant", "content": final_output})

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
