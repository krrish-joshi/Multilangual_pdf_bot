# import streamlit as st
# import os, io, requests
# from dotenv import load_dotenv
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from langchain_community.vectorstores import FAISS
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate
# import google.generativeai as genai
# from sarvamai import SarvamAI

# # -------------------- Load API Keys --------------------
# load_dotenv()
# SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# genai.configure(api_key=GOOGLE_API_KEY)
# client = SarvamAI(api_subscription_key=SARVAM_API_KEY)

# # -------------------- Language Options --------------------
# LANGUAGES = {
#     "English": "en-IN",
#     "Hindi": "hi-IN",
#     "Gujarati": "gu-IN",
#     "Bengali": "bn-IN",
#     "Kannada": "kn-IN",
#     "Punjabi": "pa-IN"
# }

# if "active_lang" not in st.session_state:
#     st.session_state.active_lang = "English"

# # -------------------- PDF Functions --------------------
# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         try:
#             pdf_reader = PdfReader(io.BytesIO(pdf.read()))
#             for page in pdf_reader.pages:
#                 page_text = page.extract_text()
#                 if page_text:
#                     text += page_text
#         except Exception as e:
#             st.error(f"Error reading PDF: {getattr(pdf, 'name', 'Unknown file')} - {e}")
#     return text

# def get_text_chunks(text):
#     splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
#     return splitter.split_text(text)

# def get_vector_store(text_chunks):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     try:
#         vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#         vector_store.save_local("faiss_index")
#     except Exception as e:
#         st.error(f"Error creating vector store: {e}")

# def get_conversational_chain():
#     prompt_template = """
#     Answer the question as detailed as possible from the provided context. 
#     If the answer is not in context, just say "answer is not available in the context".
    
#     Context:\n{context}\n
#     Question:\n{question}\n
#     Answer:
#     """
#     model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#     chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
#     return chain

# def get_answer(user_question):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     index_path = "faiss_index"

#     # Check if FAISS index exists
#     if not os.path.exists(index_path) or not os.path.exists(os.path.join(index_path, "index.faiss")):
#         return "⚠️ No knowledge base found. Please upload and process PDFs first."

#     try:
#         db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
#         docs = db.similarity_search(user_question)
#         if not docs:
#             return "⚠️ No relevant information found in the knowledge base."
#         chain = get_conversational_chain()
#         response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
#         return response.get("output_text", "⚠️ No answer generated.")
#     except Exception as e:
#         error_msg = str(e)
#         if "429" in error_msg and "quota" in error_msg:
#             return ("⚠️ Gemini API quota exceeded. Please wait for your quota to reset or add a new API key. "
#                     "See https://ai.google.dev/gemini-api/docs/rate-limits for details.")
#         return f"⚠️ Error retrieving answer: {e}"

# # -------------------- Streamlit UI --------------------
# st.set_page_config(page_title="Multilingual PDF Chatbot", layout="centered")

# # Sidebar language selection
# st.sidebar.markdown("## Select Language")
# for lang in LANGUAGES:
#     if st.sidebar.button(lang):
#         st.session_state.active_lang = lang

# current_lang = st.session_state.active_lang
# target_lang_code = LANGUAGES[current_lang]

# st.markdown(f"<h2 style='text-align:center;'>Chat in {current_lang}</h2>", unsafe_allow_html=True)
# st.divider()

# # PDF Upload
# with st.sidebar:
#     st.title("Upload PDFs")
#     pdf_docs = st.file_uploader("Upload your PDF files", accept_multiple_files=True, type=["pdf"])
#     if st.button("Submit PDFs"):
#         if pdf_docs:
#             with st.spinner("Processing..."):
#                 raw_text = get_pdf_text(pdf_docs)
#                 if not raw_text.strip():
#                     st.error("No extractable text found in uploaded PDFs.")
#                 else:
#                     chunks = get_text_chunks(raw_text)
#                     get_vector_store(chunks)
#                     st.success("PDFs processed successfully!")
#         else:
#             st.error("Please upload at least one PDF file.")

# # Chat UI
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# for chat in st.session_state.chat_history:
#     st.markdown(chat, unsafe_allow_html=True)

# with st.form("chat_form", clear_on_submit=True):
#     user_input = st.text_input("You", placeholder="Type your question here...", label_visibility="collapsed")
#     submitted = st.form_submit_button("Send")

# if submitted and user_input.strip():
#     st.session_state.chat_history.append(
#         f"<div style='text-align:right;color:#1e90ff;'><strong>You:</strong> {user_input}</div>"
#     )

#     # Get answer from PDF
#     answer = get_answer(user_input)

#     # Translate if needed
#     if answer and target_lang_code != "en-IN" and not answer.startswith("⚠️"):
#         try:
#             translation = client.text.translate(
#                 input=answer,
#                 source_language_code="en-IN",
#                 target_language_code=target_lang_code,
#                 speaker_gender="Male"
#             )
#             final_answer = translation.translated_text if hasattr(translation, 'translated_text') else answer
#         except Exception as e:
#             st.error(f"Translation error: {e}")
#             final_answer = answer
#     else:
#         final_answer = answer

#     st.session_state.chat_history.append(
#         f"<div style='text-align:left;color:#228B22;'><strong>Bot:</strong> {final_answer}</div>"
#     )
#     st.rerun()
# app.py



# app.py
# app.py
import streamlit as st
import os
import io
import pickle
import re
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

import google.generativeai as genai
from sarvamai import SarvamAI

from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# -------------------- MUST BE FIRST --------------------
st.set_page_config(page_title="Multilingual PDF Chatbot", layout="centered")

# -------------------- Load API Keys --------------------
load_dotenv()
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

client = SarvamAI(api_subscription_key=SARVAM_API_KEY) if SARVAM_API_KEY else None

# -------------------- Languages --------------------
LANGUAGES = {
    "English": "en-IN",
    "Hindi": "hi-IN",
    "Punjabi": "pa-IN",
    "Gujarati": "gu-IN",
    "Bengali": "bn-IN",
    "Kannada": "kn-IN"
}

if "active_lang" not in st.session_state:
    st.session_state.active_lang = "English"

# -------------------- Embeddings + FAISS --------------------
MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_DIR = "faiss_index"

@st.cache_resource
def load_embedder():
    return SentenceTransformer(MODEL_NAME)

embedder = load_embedder()

# -------------------- PDF Processing --------------------
def get_pdf_text(pdfs):
    text = ""
    for pdf in pdfs:
        reader = PdfReader(io.BytesIO(pdf.read()))
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    return text

def split_text(text):
    spl = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    return spl.split_text(text)

# -------------------- Build & Load FAISS --------------------
def build_faiss(chunks):
    os.makedirs(INDEX_DIR, exist_ok=True)

    embeddings = embedder.encode(chunks, convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, f"{INDEX_DIR}/index.faiss")

    with open(f"{INDEX_DIR}/meta.pkl", "wb") as f:
        pickle.dump({"texts": chunks}, f)

def load_faiss():
    if not os.path.exists(f"{INDEX_DIR}/index.faiss"):
        return None, None

    index = faiss.read_index(f"{INDEX_DIR}/index.faiss")
    meta = pickle.load(open(f"{INDEX_DIR}/meta.pkl", "rb"))
    return index, meta

def search(q, k=5):
    index, meta = load_faiss()
    if index is None:
        return []

    vec = embedder.encode([q], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(vec)

    _, ids = index.search(vec, k)
    texts = meta["texts"]

    return [texts[i] for i in ids[0] if i < len(texts)]

# -------------------- SAFE SARAM TRANSLATION (FINAL VERSION) --------------------
def sarvam_translate(text, source_lang, target_lang, max_chars=950):
    """
    Sarvam translation with safe chunking so it NEVER exceeds 1000 char limit.
    """
    if client is None:
        return text  # fallback

    # Short text = direct translate
    if len(text) <= max_chars:
        try:
            res = client.text.translate(
                input=text,
                source_language_code=source_lang,
                target_language_code=target_lang,
                speaker_gender="Male",
            )
            return getattr(res, "translated_text", text)
        except:
            return text

    # Long text = split safely
    # Split at sentence boundaries
    sentences = re.split(r'(?<=[.!?।])\s+', text)
    chunks = []
    current = ""

    for s in sentences:
        if len(current) + len(s) <= max_chars:
            current += " " + s
        else:
            chunks.append(current.strip())
            current = s

    if current:
        chunks.append(current.strip())

    final_output = []

    # Translate each chunk
    for chunk in chunks:
        try:
            res = client.text.translate(
                input=chunk,
                source_language_code=source_lang,
                target_language_code=target_lang,
                speaker_gender="Male",
            )
            translated = getattr(res, "translated_text", chunk)
        except:
            translated = chunk  # fallback on error
        final_output.append(translated)

    return " ".join(final_output)

# -------------------- English Gemini QA --------------------
def english_chain():
    template = """
Use ONLY the provided context to answer the question.
If not found, say: "Answer is not available in the context."

Context:
{context}

Question:
{question}

Answer in English:
"""
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# -------------------- MAIN LOGIC: ALWAYS RETURN SELECTED LANGUAGE --------------------
def answer_multilingual(user_question, target_code):
    # 1️⃣ First detect question language → convert question → English
    question_in_english = sarvam_translate(user_question, target_code, "en-IN")

    # 2️⃣ Retrieve context
    docs = search(question_in_english, k=5)
    if not docs:
        return sarvam_translate("Answer is not available in the context.", "en-IN", target_code)

    context = "\n\n".join(docs)

    # 3️⃣ Ask Gemini in ENGLISH (stable)
    try:
        chain = english_chain()
        res = chain(
            {"input_documents": docs, "question": question_in_english},
            return_only_outputs=True
        )
        ans_en = res.get("output_text") or res.get("answer") or str(res)
    except:
        ans_en = context[:1200]

    # 4️⃣ ALWAYS translate answer → Target Language
    final_ans = sarvam_translate(ans_en, "en-IN", target_code)
    return final_ans

# -------------------- STREAMLIT UI --------------------
st.sidebar.title("Language")
st.session_state.active_lang = st.sidebar.radio(
    "Choose language:", list(LANGUAGES.keys()),
    index=list(LANGUAGES.keys()).index(st.session_state.active_lang)
)

current_lang = st.session_state.active_lang
target_code = LANGUAGES[current_lang]

st.title(f"Chat in {current_lang}")

# Upload PDFs
with st.sidebar:
    st.header("Upload PDFs")
    pdfs = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
    if st.button("Process PDFs"):
        if pdfs:
            with st.spinner("Reading & indexing PDFs..."):
                text = get_pdf_text(pdfs)
                chunks = split_text(text)
                build_faiss(chunks)
                st.success("PDFs processed & indexed!")
        else:
            st.error("Please upload at least one PDF.")

# Chat history
if "chat" not in st.session_state:
    st.session_state.chat = []

for m in st.session_state.chat:
    st.markdown(m, unsafe_allow_html=True)

with st.form("chat_form", clear_on_submit=True):
    question = st.text_input("You:")
    submit = st.form_submit_button("Send")

if submit and question.strip():
    st.session_state.chat.append(f"<p style='text-align:right;color:cyan;'><b>You:</b> {question}</p>")

    answer = answer_multilingual(question, target_code)

    st.session_state.chat.append(f"<p style='text-align:left;color:lightgreen;'><b>Bot:</b> {answer}</p>")
    st.rerun()
