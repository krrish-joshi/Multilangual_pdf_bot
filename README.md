📚 Multilingual PDF Chatbot (Streamlit + Gemini + SarvamAI)

🚀 An AI-powered chatbot that allows users to upload PDFs and ask questions in multiple Indian languages.
The bot uses Google Gemini + FAISS + SarvamAI translation to provide contextual answers in the user’s chosen language.

📌 Problem Statement

Students often struggle with large PDF documents like notes, books, and research papers.

Traditional chatbots don’t read documents, giving random/irrelevant answers.

Most tools work only in English, limiting accessibility for regional language students.

✅ Our Solution

A Streamlit-based chatbot that:

📄 Extracts knowledge from uploaded PDFs.

🔍 Answers contextual questions (not random).

🌐 Responds in English + Indian regional languages (Hindi, Gujarati, Bengali, Kannada, Punjabi).

🖥️ Provides a simple UI for students and educators.

🛠️ Tech Stack

UI → Streamlit

PDF Processing → PyPDF2

Embeddings → Google Gemini API

Vector DB → FAISS

Q&A Engine → LangChain + Gemini Flash

Translation → SarvamAI API

Env Handling → python-dotenv

⚡ Features

Upload multiple PDFs.

Ask questions in chat → bot answers only from PDFs.

Choose response language from sidebar.

Maintains chat history.

Handles errors gracefully (empty PDFs, API quota issues, translation errors).

📦 Installation
1️⃣ Clone Repo
git clone https://github.com/your-username/multilingual-pdf-chatbot.git
cd multilingual-pdf-chatbot

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Setup API Keys

Create a .env file in the project root:

SARVAM_API_KEY=your_sarvam_api_key
GOOGLE_API_KEY=your_gemini_api_key

4️⃣ Run the App
streamlit run sih.py

🖥️ Usage

Open the app in your browser → http://localhost:8501.

Select a language from the sidebar.

Upload one or more PDF files.

Ask questions in the chat input box.

Get instant answers in your chosen language!

📊 Example

Upload → machine_learning.pdf

Select → Hindi

Ask → "इस किताब में supervised learning क्या है?"

Bot → Replies in Hindi, based on English PDF content.

📅 Roadmap

✅ PDF extraction + FAISS knowledge base

✅ Multilingual chat (SarvamAI)

✅ Streamlit UI with history

🔜 Speech-to-Text + Text-to-Speech

🔜 More Indian languages

🔜 Cloud deployment (future)

👤 Maintainer: Krrish Joshi

🌐 Role: DevOps & Integrations
