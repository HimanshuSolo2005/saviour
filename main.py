import os
import gradio as gr
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

loader = PyPDFLoader("Syllabus.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, 
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " ", ""]
)
docs = text_splitter.split_documents(documents)

import shutil
if os.path.exists("chroma_db"):
    shutil.rmtree("chroma_db")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = Chroma.from_documents(docs, embeddings, persist_directory="chroma_db")

retriever = vectordb.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 10}
)

from transformers import pipeline
llm_pipeline = pipeline("text2text-generation", model="google/flan-t5-small")
llm = HuggingFacePipeline(pipeline=llm_pipeline)

prompt_template = """
Use the following context to answer the question. If you cannot find the answer in the context, say "I don't have enough information to answer this question."

Context: {context}

Question: {question}

Answer:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm, 
    retriever=retriever,
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True
)

def chatbot(query: str) -> str:
    try:
        result = qa_chain.invoke(query)
        if isinstance(result, dict):
            answer = result.get("result", "No answer found")
            sources = result.get("source_documents", [])
            
            if sources:
                source_info = f"\n\nSources: {len(sources)} relevant document(s) found"
                return answer + source_info
            return answer
        return str(result)
    except Exception as e:
        return f"Error processing query: {str(e)}"

iface = gr.Interface(
    fn=chatbot,
    inputs=gr.Textbox(lines=2, placeholder="Ask me"),
    outputs="text",
    title="Knowledge Chatbot",
    description="Chat with your OS subject."
)

if __name__ == "__main__":
    iface.launch()
