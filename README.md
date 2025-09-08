<p align="center">
  alt="Multilingual PDF Chatbot Banner"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue.svg" />
  <img src="https://img.shields.io/badge/Streamlit-1.39-red.svg" />
  <img src="https://img.shields.io/badge/License-MIT-green.svg" />
  <img src="https://img.shields.io/badge/Made%20with-%E2%9D%A4-pink.svg" />
</p>

# 📚 **Multilingual PDF Chatbot**  
✨ *AI-powered PDF Question Answering with Multilingual Support* ✨  

---

## 📑 **Table of Contents**
- [🚀 Overview](#-overview)  
- [🎯 Problem Statement](#-problem-statement)  
- [✅ Solution](#-solution)  
- [🛠️ Tech Stack](#️-tech-stack)  
- [⚡ Features](#-features)  
- [📦 Installation Guide](#-installation-guide)  
- [🖥️ How to Use](#️-how-to-use)  
- [📊 Example](#-example)  
- [📅 Roadmap](#-roadmap)  
- [👨‍💻 Maintainer](#-maintainer)  
- [💡 Quote](#-quote)  

---

## 🚀 **Overview**
> 📘 An **AI-powered chatbot** that allows students to **chat with their PDFs** in **multiple Indian languages**.  

This project brings together:  
- 🔹 **Google Gemini embeddings**  
- 🔹 **FAISS vector search**  
- 🔹 **SarvamAI translation**  
- 🔹 **Streamlit user interface**  

---

## 🎯 **Problem Statement**
- ❌ Extracting information from **large PDFs** is time-consuming.  
- ❌ Normal chatbots → give **random / irrelevant answers**.  
- ❌ Most solutions support **English only**, ignoring regional languages.  

---

## ✅ **Solution**
- ✔️ Upload **PDFs** → chatbot builds a knowledge base.  
- ✔️ Ask questions → chatbot replies **from your PDFs only**.  
- ✔️ Responds in **English + Indian languages** (Hindi, Gujarati, Bengali, Kannada, Punjabi).  
- ✔️ Provides a **student-friendly chat UI** with history.  

---

## 🛠️ **Tech Stack**
- 🖥️ **Frontend/UI** → Streamlit  
- 📄 **PDF Processing** → PyPDF2  
- 🧠 **Embeddings** → Google Gemini API  
- 📦 **Vector Database** → FAISS  
- 🔗 **Q&A Engine** → LangChain (Gemini Flash)  
- 🌐 **Translation** → SarvamAI API  
- 🔑 **Secrets Handling** → python-dotenv  

---

## ⚡ **Features**
- 📂 Upload multiple **PDF files**  
- 🗨️ Chat in **English + regional languages**  
- 🌐 Switch language from **sidebar**  
- 🕑 Saves **chat history**  
- 🛡️ Handles errors gracefully (quota exceeded, empty PDFs, etc.)  

---

## 📦 **Installation Guide**

### 🔹 1. Clone Repository
```bash
git clone https://github.com/your-username/multilingual-pdf-chatbot.git
cd multilingual-pdf-chatbot
🔹 2. Install Dependencies
bash
Copy code
pip install -r requirements.txt
🔹 3. Setup Environment Variables
Create a .env file in the project root:

SARVAM_API_KEY=your_sarvam_api_key
GOOGLE_API_KEY=your_gemini_api_key
🔹 4. Run the Application
streamlit run sih.py
🖥️ How to Use
🌐 Open browser → http://localhost:8501

📂 Upload your PDF files

🔄 Select your preferred language

❓ Type your question

🤖 Get answers in your chosen language

📊 Example
📂 Upload → machine_learning.pdf

🌐 Select → Hindi

❓ Ask → "इस किताब में supervised learning क्या है?"

🤖 Bot → Replies in Hindi (extracted from PDF)

📅 Roadmap
✅ PDF extraction + FAISS search

✅ Multilingual chat (SarvamAI)

✅ Streamlit UI with chat history

🔜 Add Speech-to-Text & Text-to-Speech

🔜 Add support for more Indian languages

🔜 Deploy on Cloud

👨‍💻 Maintainer
👤 Krrish Joshi

🌐 Role: DevOps & Integrations

💡 Quote
“Don’t just read your PDFs — converse with them.”

---





