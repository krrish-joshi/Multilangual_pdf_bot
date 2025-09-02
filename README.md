# 🎙️ Voice-First Multilingual Chatbot (DevOps by Krrish Joshi)

🚀 This project is a **Voice-First Multilingual Chatbot** designed for students,  
with live deployment on cloud + WhatsApp integration.  

This repo is maintained by **Krrish Joshi** ([@krrish-joshi](https://github.com/krrish-joshi)) as part of our team project,  
where my role is **DevOps & Integrations**.

---

## 📌 Problem Statement
- A chatbot that only runs locally = ❌ useless.  
- Students need it live on the **college website** and on **WhatsApp/Telegram**.  
- If it crashes under load → failure.  
- If it’s insecure → unusable.  

---

## ✅ Our Solution
A **Voice-First Multilingual Chatbot**, deployed in Docker + Cloud, available to students online:
- Containers for **ASR, RAG, TTS, Backend**.
- Public APIs for frontend (Simran).
- Accessible via **WhatsApp (Twilio Sandbox)**.
- Logs stored in **Postgres** for monitoring & analytics.
- HTTPS enabled for secure access.

---

## 👨‍💻 Team Roles
- **AIML** → Manan + Pragy  
- **WebDev** → Simran  
- **DevOps (this repo)** → Krrish Joshi  
- **Data** → Runali  

---

## 🛠️ Tech Stack
- **Backend**: FastAPI (Python)  
- **Containers**: Docker + Docker Compose  
- **Database**: PostgreSQL  
- **Monitoring**: Prometheus + Grafana  
- **Messaging**: Twilio WhatsApp API  
- **Deployment**: Cloud VM (AWS/GCP/Azure)  

---

## ⚡ Quick Start (Local Setup)

### 1️⃣ Clone Repo
```bash
git clone https://github.com/krrish-joshi/multilingual-chatbot-devops.git
cd multilingual-chatbot-devops
2️⃣ Create .env
ini
Copy code
TWILIO_ACCOUNT_SID=your_twilio_sid
TWILIO_AUTH_TOKEN=your_twilio_auth_token
POSTGRES_USER=chat
POSTGRES_PASSWORD=changeme
POSTGRES_DB=chatdb
3️⃣ Run with Docker
bash
Copy code
docker compose up --build
Services:

Backend → http://localhost:8000

ASR → http://localhost:8001

RAG → http://localhost:8002

TTS → http://localhost:8003

Grafana → http://localhost:3000

Prometheus → http://localhost:9090

4️⃣ Health Check
bash
Copy code
curl http://localhost:8000/health
📱 WhatsApp Integration
Join Twilio WhatsApp Sandbox.

Set webhook URL → https://your-domain/webhook/twilio.

Send a WhatsApp message → chatbot replies instantly.

📊 Monitoring
Metrics exposed at /metrics (Prometheus format).

Grafana dashboards available on port 3000.

🚀 Deployment
Deploy on cloud VM:

bash
Copy code
docker compose up -d --build
Point domain → VM IP, Caddy auto-generates HTTPS.

🛡️ Security Checklist
HTTPS with Let’s Encrypt (via Caddy).

Secrets in .env (not committed).

Logs stored in Postgres.

Docker healthchecks + monitoring enabled.

📅 Roadmap
 Backend setup (FastAPI)

 ASR + RAG + TTS stubs

 Postgres logging

 Prometheus + Grafana monitoring

 Twilio webhook signature validation

 Cloud deployment (AWS/GCP/Azure)

 Kubernetes scaling (future)

👤 Maintainer: @krrish-joshi
🌐 Role: DevOps & Integrations
