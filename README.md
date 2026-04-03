# FinComply AI 🏛️

> **India's Real-Time Financial Regulatory Intelligence System**  
> Powered by a 3-Agent LangGraph Pipeline · Fine-Tuned Mistral 7B · MCP Tools on AWS Lambda

[![Live Demo](https://img.shields.io/badge/Live%20Demo-fincomply--ai.duckdns.org-teal?style=for-the-badge)](https://fincomply-ai.duckdns.org)
[![HuggingFace](https://img.shields.io/badge/Model-prem332%2Ffincomply--mistral--7b-yellow?style=for-the-badge&logo=huggingface)](https://huggingface.co/prem332/fincomply-mistral-7b)
[![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![LangSmith](https://img.shields.io/badge/Monitored-LangSmith-orange?style=for-the-badge)](https://smith.langchain.com)

---

## Problem Statement

Indian businesses — startups, CFOs, legal teams, and compliance officers — must navigate an ever-changing regulatory landscape across **GST, RBI, SEBI, MCA, and Income Tax**. Government circulars are published frequently, buried in `.gov.in` portals, and require expert interpretation. Missing a deadline or misapplying a regulation results in penalties, legal exposure, and operational disruption.

**Existing solutions** (ClearTax, Taxmann, IndiaFilings) are either expensive SaaS subscriptions, manually curated by human experts with update lag, or generic AI assistants that hallucinate regulatory details without source verification.

---

## Solution

FinComply AI is an **agentic RAG system** that provides accurate, source-verified regulatory compliance guidance across all five major Indian regulatory domains. Unlike generic LLMs, every answer is:

- Grounded in **official government circulars** from `.gov.in` sources
- Verified by an **LLM-as-Judge Critic Agent** that checks source URL, circular number, and recency
- Scored with a **92% HIGH confidence** when all verification checks pass
- Evaluated against **RAGAS metrics** (Faithfulness 1.0, Relevancy 1.0, Precision 0.99, Recall 0.81)
- Monitored end-to-end via **LangSmith** with 0% error rate across 133 production traces

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────┐
│           Safety Layer (FastAPI)            │
│  Pattern Injection Check → LLM Injection    │
│  Check → PII Guardrail                      │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│              3-Agent LangGraph Pipeline                     │
│                                                             │
│  ┌──────────────────┐   ┌──────────────────┐   ┌─────────┐ │
│  │  Research Agent  │──▶│  Critic Agent    │──▶│Supervisor│ │
│  │  (Agent 1)       │   │  (Agent 2)       │   │(Agent 3) │ │
│  │                  │   │  LLM as Judge    │   │         │ │
│  │  • pgvector RAG  │   │  • URL verify    │   │• Final  │ │
│  │  • MCP Lambda    │   │  • Circular No.  │   │  answer │ │
│  │  • Fine-tuned    │   │  • Recency check │   │• 92%    │ │
│  │    Mistral 7B    │   │  • Mistral API   │   │  conf.  │ │
│  └──────────────────┘   └──────────────────┘   └─────────┘ │
└─────────────────────────────────────────────────────────────┘
    │                           │
    ▼                           ▼
pgvector (RDS)          MCP Tools (Lambda)
213 regulatory facts    GST · RBI · SEBI
                        MCA · Income Tax
    │
    ▼
NLI Hallucination Check
(cross-encoder/nli-deberta-v3-small)
    │
    ▼
Structured Response with Confidence Score
    │
    ▼
LangSmith Monitoring (Traces · Latency · Errors)
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **LLM** | Mistral API (`mistral-small-latest`) + Fine-tuned `prem332/fincomply-mistral-7b` |
| **Fine-tuning** | QLoRA via Unsloth on Google Colab T4 GPU (213 domain examples, loss 0.58) |
| **Agent Framework** | LangGraph 3-agent Critique-Revision pipeline |
| **MCP Tools** | AWS Lambda (Python 3.11) + API Gateway — 5 domain tools |
| **Vector DB** | pgvector on AWS RDS t3.micro (213 regulatory embeddings, TOP_K=10) |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions) |
| **Hallucination Check** | `cross-encoder/nli-deberta-v3-small` (NLI entailment, pre-loaded at startup) |
| **Backend** | FastAPI + Uvicorn |
| **Frontend** | React + Vite + Tailwind CSS |
| **Deployment** | AWS EC2 t2.micro + Docker + Nginx + Let's Encrypt SSL |
| **Domain** | DuckDNS (fincomply-ai.duckdns.org) |
| **Monitoring** | LangSmith — full trace visibility per agent |
| **Evaluation** | RAGAS (custom Mistral judge — no OpenAI dependency) |

---

## Regulatory Domains

| Domain | Source | Coverage |
|---|---|---|
| **GST** | cbic-gst.gov.in | Rates, ITC, E-invoicing, Returns, Refunds |
| **RBI** | rbi.org.in | KYC, Digital Lending, NBFC, Payments, Forex |
| **SEBI** | sebi.gov.in | IPO, Insider Trading, Mutual Funds, FPI |
| **MCA** | mca.gov.in | Company Law, Directors, Annual Filings, IBC |
| **Income Tax** | incometaxindia.gov.in | Tax Slabs, TDS, Deductions, Capital Gains |

---

## Key Features

- **3-Layer Safety Pipeline** — Pattern injection detection → LLM injection classifier → PII guardrail
- **Parallel MCP Tool Calls** — All 5 domain tools called simultaneously via `ThreadPoolExecutor`
- **Rule-Based Source Verification** — Only `.gov.in` and `rbi.org.in` URLs accepted
- **Confidence Scoring** — HIGH (92%) when URL + Circular + Recency all verified
- **NLI Hallucination Check** — Answer entailment verified against source documents at startup
- **Graceful Fallback** — HuggingFace model → Mistral API → structured error response
- **Compliance Deadlines** — Urgency-colored deadline alerts in UI
- **LangSmith Tracing** — Every agent call traced with latency and error tracking

---

## RAGAS Evaluation Results

Evaluated on 20 golden test questions across all 5 domains using Mistral as judge LLM (no OpenAI dependency):

| Metric | Score | Threshold | Status |
|---|---|---|---|
| **Faithfulness** | 1.000 | ≥ 0.90 | ✅ PASS |
| **Answer Relevancy** | 1.000 | ≥ 0.85 | ✅ PASS |
| **Context Precision** | 0.990 | ≥ 0.80 | ✅ PASS |
| **Context Recall** | 0.812 | ≥ 0.75 | ✅ PASS |

---

## Production Metrics (LangSmith)

Real-time pipeline monitoring via LangSmith across 133 production traces:

| Metric | Value |
|---|---|
| **Total Traces** | 133 |
| **Error Rate** | 0% |
| **P50 Latency** | 10.80 seconds |
| **P99 Latency** | 21.24 seconds |
| **Agent Pipeline** | Research → Critic → Supervisor |
| **Monitoring** | [smith.langchain.com](https://smith.langchain.com) → Project: `fincomply-ai` |

> P99 latency of 21.24s covers cold-start scenarios (first query after idle period). Warm P50 is consistently 8-12 seconds across 3 Mistral API calls.

---

## Fine-Tuning

The Mistral 7B model was fine-tuned using **QLoRA** (Quantized LoRA) via Unsloth:

- **Base Model**: `unsloth/mistral-7b-instruct-v0.3-bnb-4bit`
- **Dataset**: 213 domain-specific Indian regulatory Q&A pairs
- **Method**: QLoRA (4-bit quantization + LoRA rank 16)
- **Training**: 60 steps, 3 epochs, T4 GPU on Google Colab
- **Final Loss**: 0.58 (down from 1.93 — 70% reduction)
- **Trainable Parameters**: 41M of 7.3B (0.58%)
- **Adapter**: [huggingface.co/prem332/fincomply-mistral-7b](https://huggingface.co/prem332/fincomply-mistral-7b)

**Agent Usage:**
- Research Agent → Fine-tuned HF model (falls back to Mistral API)
- Critic Agent → Mistral API base model (neutral evaluation)
- Supervisor Agent → Fine-tuned HF model (falls back to Mistral API)

---

## Project Structure

```
fincomply-ai-/
├── backend/
│   ├── config.py                    # Single source of truth — all config here
│   ├── api/
│   │   └── main.py                  # FastAPI app + safety pipeline + NLI pre-load
│   ├── agents/
│   │   ├── graph.py                 # LangGraph pipeline definition
│   │   ├── research_agent.py        # Agent 1: RAG + parallel MCP + HF inference
│   │   ├── critic_agent.py          # Agent 2: LLM-as-Judge + rule-based verification
│   │   ├── supervisor_agent.py      # Agent 3: Final answer + confidence scoring
│   │   └── prompts.py               # All LLM prompts
│   ├── mcp_server/
│   │   ├── server.py                # AWS Lambda handler
│   │   ├── gst_tool.py              # GST data fetcher (cbic-gst.gov.in)
│   │   ├── rbi_tool.py              # RBI data fetcher (rbi.org.in)
│   │   ├── sebi_tool.py             # SEBI data fetcher (sebi.gov.in)
│   │   ├── mca_tool.py              # MCA data fetcher (mca.gov.in)
│   │   └── income_tax_tool.py       # Income Tax data fetcher (incometaxindia.gov.in)
│   ├── database/
│   │   ├── init_db.py               # pgvector table creation
│   │   ├── seed_data.py             # 213 regulatory facts (50 per domain)
│   │   └── ingest_data.py           # Embed and load into pgvector
│   ├── fine_tuning/
│   │   ├── generate_dataset.py      # Generate QA pairs from seed data
│   │   └── data/                    # Training JSONL dataset (213 examples)
│   └── evaluation/
│       ├── ragas_eval.py            # RAGAS evaluation (Mistral judge, no OpenAI)
│       └── ragas_results.json       # Latest evaluation scores
├── frontend/
│   └── src/                         # React + Vite + Tailwind CSS
├── docker/
│   └── Dockerfile                   # Production Docker image (python:3.11-slim)
├── infra/
│   ├── ec2_setup.sh                 # EC2 setup script
│   └── lambda_deploy.sh             # Lambda deployment script
└── requirements.txt
```

---

## Local Development Setup

### Prerequisites
- Python 3.12
- Node.js 20+
- AWS account (RDS + Lambda)
- Mistral API key

### Backend Setup

```bash
# Clone repository
git clone https://github.com/prem332/fincomply-ai-.git
cd fincomply-ai-

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Initialize database
cd backend
python database/init_db.py
python database/ingest_data.py

# Start backend
python api/main.py
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev -- --host
```

Backend runs at `http://localhost:8000`  
Frontend runs at `http://localhost:5173`

---

## AWS Infrastructure Setup

For the complete step-by-step AWS deployment guide (IAM, RDS, Lambda, API Gateway, EC2, DuckDNS, HTTPS):

📄 **[AWS Deployment Guide](https://docs.google.com/document/d/1IUpwjVjX7Ixyw7cITi5FQGIu09Ep0ykA/edit?usp=drive_link&ouid=106005894817234320205&rtpof=true&sd=true)**

### Quick Overview

| Service | Purpose | Type |
|---|---|---|
| RDS PostgreSQL | pgvector database | t3.micro |
| Lambda | MCP tool endpoints | Python 3.11, 256MB |
| API Gateway | Lambda HTTP routes | HTTP API |
| EC2 | Application server | t2.micro |
| Nginx | Reverse proxy + SSL | Let's Encrypt |
| DuckDNS | Free domain | fincomply-ai.duckdns.org |

---

## Environment Variables

```env
# Mistral API
MISTRAL_API_KEY=your_key
MISTRAL_MODEL=mistral-small-latest

# HuggingFace (fine-tuned model)
HF_TOKEN=your_token
HF_USERNAME=prem332
HF_INFERENCE_URL=https://api-inference.huggingface.co/models/prem332/fincomply-mistral-7b

# AWS RDS
DB_HOST=your-rds-endpoint.ap-south-1.rds.amazonaws.com
DB_PORT=5432
DB_NAME=postgres
DB_USER=postgres
DB_PASSWORD=your_password

# MCP Lambda
MCP_BASE_URL=https://your-api-gateway-id.execute-api.ap-south-1.amazonaws.com/prod

# LangSmith Monitoring
LANGCHAIN_API_KEY=your_key
LANGCHAIN_PROJECT=fincomply-ai
LANGCHAIN_TRACING_V2=true
```

---

## API Reference

### POST /query

```json
{
  "query": "What is the GST rate on restaurant services?",
  "domain": "gst"
}
```

**Response:**

```json
{
  "success": true,
  "answer": {
    "final_answer": "The GST rate on restaurant services is 5%...",
    "circular_number": "Notification No. 11/2017-Central Tax (Rate)",
    "source_url": "https://cbic-gst.gov.in/central-tax-rate-notifications.html",
    "confidence_level": "HIGH",
    "confidence_score": 0.92,
    "is_gov_verified": true,
    "action_required": "..."
  },
  "response_time_ms": 8432
}
```

### GET /health

```json
{
  "status": "healthy",
  "service": "FinComply AI",
  "nli_model_loaded": true
}
```

---

## Built By

**Prem Kumar** — AI/ML Engineer

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://linkedin.com/in/premkumar)
[![GitHub](https://img.shields.io/badge/GitHub-prem332-black?style=flat&logo=github)](https://github.com/prem332)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-prem332-yellow?style=flat&logo=huggingface)](https://huggingface.co/prem332)

---