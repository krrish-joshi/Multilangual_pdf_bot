import streamlit as st
import os, io, requests
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai
from sarvamai import SarvamAI

# -------------------- Load API Keys --------------------
load_dotenv()
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=GOOGLE_API_KEY)
client = SarvamAI(api_subscription_key=SARVAM_API_KEY)

# -------------------- Language Options --------------------
LANGUAGES = {
    "English": "en-IN",
    "Hindi": "hi-IN",
    "Gujarati": "gu-IN",
    "Bengali": "bn-IN",
    "Kannada": "kn-IN",
    "Punjabi": "pa-IN"
}

if "active_lang" not in st.session_state:
    st.session_state.active_lang = "English"

# -------------------- PDF Functions --------------------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(io.BytesIO(pdf.read()))
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        except Exception as e:
            st.error(f"Error reading PDF: {getattr(pdf, 'name', 'Unknown file')} - {e}")
    return text

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    try:
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
    except Exception as e:
        st.error(f"Error creating vector store: {e}")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. 
    If the answer is not in context, just say "answer is not available in the context".
    
    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def get_answer(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    index_path = "faiss_index"

    # Check if FAISS index exists
    if not os.path.exists(index_path) or not os.path.exists(os.path.join(index_path, "index.faiss")):
        return "⚠️ No knowledge base found. Please upload and process PDFs first."

    try:
        db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        docs = db.similarity_search(user_question)
        if not docs:
            return "⚠️ No relevant information found in the knowledge base."
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        return response.get("output_text", "⚠️ No answer generated.")
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg and "quota" in error_msg:
            return ("⚠️ Gemini API quota exceeded. Please wait for your quota to reset or add a new API key. "
                    "See https://ai.google.dev/gemini-api/docs/rate-limits for details.")
        return f"⚠️ Error retrieving answer: {e}"

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="Multilingual PDF Chatbot", layout="centered")

# Sidebar language selection
st.sidebar.markdown("## Select Language")
for lang in LANGUAGES:
    if st.sidebar.button(lang):
        st.session_state.active_lang = lang

current_lang = st.session_state.active_lang
target_lang_code = LANGUAGES[current_lang]

st.markdown(f"<h2 style='text-align:center;'>Chat in {current_lang}</h2>", unsafe_allow_html=True)
st.divider()

# PDF Upload
with st.sidebar:
    st.title("Upload PDFs")
    pdf_docs = st.file_uploader("Upload your PDF files", accept_multiple_files=True, type=["pdf"])
    if st.button("Submit PDFs"):
        if pdf_docs:
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                if not raw_text.strip():
                    st.error("No extractable text found in uploaded PDFs.")
                else:
                    chunks = get_text_chunks(raw_text)
                    get_vector_store(chunks)
                    st.success("PDFs processed successfully!")
        else:
            st.error("Please upload at least one PDF file.")

# Chat UI
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for chat in st.session_state.chat_history:
    st.markdown(chat, unsafe_allow_html=True)

with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("You", placeholder="Type your question here...", label_visibility="collapsed")
    submitted = st.form_submit_button("Send")

if submitted and user_input.strip():
    st.session_state.chat_history.append(
        f"<div style='text-align:right;color:#1e90ff;'><strong>You:</strong> {user_input}</div>"
    )

    # Get answer from PDF
    answer = get_answer(user_input)

    # Translate if needed
    if answer and target_lang_code != "en-IN" and not answer.startswith("⚠️"):
        try:
            translation = client.text.translate(
                input=answer,
                source_language_code="en-IN",
                target_language_code=target_lang_code,
                speaker_gender="Male"
            )
            final_answer = translation.translated_text if hasattr(translation, 'translated_text') else answer
        except Exception as e:
            st.error(f"Translation error: {e}")
            final_answer = answer
    else:
        final_answer = answer

    st.session_state.chat_history.append(
        f"<div style='text-align:left;color:#228B22;'><strong>Bot:</strong> {final_answer}</div>"
    )

    st.rerun()
