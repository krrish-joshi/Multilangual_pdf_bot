<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue.svg" />
  <img src="https://img.shields.io/badge/Streamlit-1.39-red.svg" />
  <img src="https://img.shields.io/badge/License-MIT-green.svg" />
  <img src="https://img.shields.io/badge/Made%20with-%E2%9D%A4-pink.svg" />
</p>

# 📚 **Multilingual PDF Chatbot**  
✨ *AI-powered PDF Question Answering with Multilingual Support* ✨  

---

## 1️⃣ **Table of Contents**
1. [🚀 Overview](#2️⃣-overview)  
2. [🎯 Problem Statement](#3️⃣-problem-statement)  
3. [✅ Solution](#4️⃣-solution)  
4. [🛠️ Tech Stack](#5️⃣-tech-stack)  
5. [⚡ Features](#6️⃣-features)  
6. [📦 Installation Guide](#7️⃣-installation-guide)  
7. [🖥️ How to Use](#8️⃣-how-to-use)  
8. [📊 Example](#9️⃣-example)  
9. [📅 Roadmap](#🔟-roadmap)  
10. [👨‍💻 Maintainer](#1️⃣1️⃣-maintainer)  
11. [💡 Quote](#1️⃣2️⃣-quote)  

---

## 2️⃣ **Overview**
> 📘 An **AI-powered chatbot** that allows students to **chat with their PDFs** in **multiple Indian languages**.  

This project combines:  
- 🔹 **Google Gemini embeddings**  
- 🔹 **FAISS vector search**  
- 🔹 **SarvamAI translation**  
- 🔹 **Streamlit user interface**  

---

## 3️⃣ **Problem Statement**
- ❌ Extracting information from **large PDFs** is time-consuming.  
- ❌ Normal chatbots → give **random / irrelevant answers**.  
- ❌ Most solutions support **English only**, ignoring regional languages.  

---

## 4️⃣ **Solution**
- ✔️ Upload **PDFs** → chatbot builds a knowledge base.  
- ✔️ Ask questions → chatbot replies **from your PDFs only**.  
- ✔️ Responds in **English + Indian languages** (Hindi, Gujarati, Bengali, Kannada, Punjabi).  
- ✔️ Provides a **student-friendly chat UI** with history.  

---

## 5️⃣ **Tech Stack**
- 🖥️ **Frontend/UI** → Streamlit  
- 📄 **PDF Processing** → PyPDF2  
- 🧠 **Embeddings** → Google Gemini API  
- 📦 **Vector Database** → FAISS  
- 🔗 **Q&A Engine** → LangChain (Gemini Flash)  
- 🌐 **Translation** → SarvamAI API  
- 🔑 **Secrets Handling** → python-dotenv  

---

## 6️⃣ **Features**
- 📂 Upload multiple **PDF files**  
- 🗨️ Chat in **English + regional languages**  
- 🌐 Switch language from **sidebar**  
- 🕑 Saves **chat history**  
- 🛡️ Handles errors gracefully (quota exceeded, empty PDFs, etc.)  

---

## 7️⃣ **Installation Guide**
Run the following commands step by step:  

# Step 1: Clone Repository
```bash
git clone https://github.com/your-username/multilingual-pdf-chatbot.git
```
# Step 2: Move into the project directory
```bash
cd multilingual-pdf-chatbot
```
# Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

# Step 4: Setup Environment Variables (create .env file in root)
# Add your keys inside .env
```bash
SARVAM_API_KEY=your_sarvam_api_key
GOOGLE_API_KEY=your_gemini_api_key
```

# Step 5: Run the Application
```bash
streamlit run sih.py
```
##8️⃣ **How to Use**
  🌐 Open browser → http://localhost:8501

  📂 Upload your PDF files

  🔄 Select your preferred language

  ❓ Type your question

  🤖 Get answers in your chosen language

##9️⃣ **Example**
  📂 Upload → machine_learning.pdf

  🌐 Select → Hindi

  ❓ Ask → "इस किताब में supervised learning क्या है?"

  🤖 Bot → Replies in Hindi (extracted from PDF)

🔟 Roadmap
 ✅ PDF extraction + FAISS search

 ✅ Multilingual chat (SarvamAI)

 ✅ Streamlit UI with chat history

 🔜 Add Speech-to-Text & Text-to-Speech

 🔜 Add support for more Indian languages

 🔜 Deploy on Cloud

1️⃣1️⃣ Maintainer
👤 [**Krrish Joshi**](https://github.com/krrish-joshi)

🌐 Role: AI & RAG

1️⃣2️⃣ Quote
“Don’t just read your PDFs — converse with them.”
