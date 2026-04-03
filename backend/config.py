import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))

# ─── RAG Configuration ───────────────────────────────────────────────────────
RAG_TOP_K = 10

# ─── LLM Configuration ───────────────────────────────────────────────────────
MISTRAL_API_KEY           = os.getenv("MISTRAL_API_KEY", "")
MISTRAL_MODEL             = "mistral-small-latest"          # Base model for Critic Agent
MISTRAL_FINE_TUNED_MODEL  = os.getenv(
    "MISTRAL_FINE_TUNED_MODEL", MISTRAL_MODEL               # Fallback to base model
)

# ─── HuggingFace Configuration ───────────────────────────────────────────────
HF_TOKEN        = os.getenv("HF_TOKEN", "")
HF_USERNAME     = os.getenv("HF_USERNAME", "prem332")
HF_REPO_ID      = f"{HF_USERNAME}/fincomply-mistral-7b"
BASE_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"

# Fine-tuned model inference via HuggingFace Inference API
# Used by Research Agent and Supervisor Agent
HF_INFERENCE_URL = os.getenv(
    "HF_INFERENCE_URL",
    f"https://api-inference.huggingface.co/models/{HF_USERNAME}/fincomply-mistral-7b"
)

# ─── AWS Configuration ───────────────────────────────────────────────────────
AWS_REGION            = os.getenv("AWS_REGION", "ap-south-1")
AWS_ACCESS_KEY_ID     = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
AWS_S3_BUCKET         = os.getenv("AWS_S3_BUCKET", "fincomply-ai-data")
AWS_LAMBDA_ROLE_ARN   = os.getenv("AWS_LAMBDA_ROLE_ARN", "")

# ─── Database Configuration (AWS RDS pgvector t3.micro) ──────────────────────
DB_HOST     = os.getenv("DB_HOST", "localhost")
DB_PORT     = int(os.getenv("DB_PORT", "5432"))
DB_NAME     = os.getenv("DB_NAME", "fincomply")
DB_USER     = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

DATABASE_URL = (
    f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# ─── Vector Store Configuration ──────────────────────────────────────────────
EMBEDDING_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_DIMENSION  = 384
VECTOR_TABLE_NAME = "regulatory_embeddings"

# ─── MCP Tool URLs ────────────────────────────────
MCP_BASE_URL  = os.getenv(
    "MCP_BASE_URL",
    "https://YOUR_API_GATEWAY_ID.execute-api.ap-south-1.amazonaws.com/prod",
)
GST_TOOL_URL  = f"{MCP_BASE_URL}/gst"
RBI_TOOL_URL  = f"{MCP_BASE_URL}/rbi"
SEBI_TOOL_URL = f"{MCP_BASE_URL}/sebi"
MCA_TOOL_URL  = f"{MCP_BASE_URL}/mca"
ITAX_TOOL_URL = f"{MCP_BASE_URL}/incometax"

# ─── API Server ──────────────────────────────────────────────────────────────
API_HOST     = os.getenv("API_HOST", "0.0.0.0")
API_PORT     = int(os.getenv("API_PORT", "8000"))
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")

# ─── Safety / Guardrails ─────────────────────────────────────────────────────
MAX_QUERY_LENGTH        = 1000
ALLOWED_DOMAINS         = ["gst", "rbi", "sebi", "mca", "income_tax", "all"]
CIRCULAR_FRESHNESS_DAYS = 3650

INJECTION_PATTERNS = [
    "ignore previous", "ignore above", "disregard", "forget instructions",
    "you are now", "act as", "jailbreak", "bypass", "override",
    "pretend you", "new persona", "sudo", "admin mode",
]

# ─── Confidence Thresholds ───────────────────────────────────────────────────
CONFIDENCE_HIGH_THRESHOLD   = 0.92
CONFIDENCE_MEDIUM_THRESHOLD = 0.60

# ─── RAGAS Evaluation Thresholds ─────────────────────────────────────────────
RAGAS_FAITHFULNESS_THRESHOLD      = 0.90
RAGAS_ANSWER_RELEVANCY_THRESHOLD  = 0.85
RAGAS_CONTEXT_PRECISION_THRESHOLD = 0.80
RAGAS_CONTEXT_RECALL_THRESHOLD    = 0.75

# ─── LangSmith Monitoring ────────────────────────────────────────────────────
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "true")
LANGCHAIN_API_KEY    = os.getenv("LANGCHAIN_API_KEY", "")
LANGCHAIN_PROJECT    = os.getenv("LANGCHAIN_PROJECT", "fincomply-ai")

# Auto-set env vars so LangGraph picks them up
os.environ["LANGCHAIN_TRACING_V2"] = LANGCHAIN_TRACING_V2
if LANGCHAIN_API_KEY:
    os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT

# ─── Fine-Tuning Hyperparameters ─────────────────────────────────────────────
FINE_TUNE_MAX_SEQ_LENGTH    = 2048
FINE_TUNE_BATCH_SIZE        = 2
FINE_TUNE_GRAD_ACCUMULATION = 4
FINE_TUNE_EPOCHS            = 3
FINE_TUNE_LEARNING_RATE     = 2e-4
FINE_TUNE_LORA_R            = 16
FINE_TUNE_LORA_ALPHA        = 32
FINE_TUNE_LORA_DROPOUT      = 0.05
FINE_TUNE_OUTPUT_DIR        = "./fine_tuning/checkpoints"

# ─── Government RSS Feed URLs (official .gov.in sources only) ────────────────
GST_RSS_URL  = "https://cbic-gst.gov.in/central-tax-notifications.html"
RBI_RSS_URL  = "https://www.rbi.org.in/scripts/rss.aspx"
SEBI_RSS_URL = "https://www.sebi.gov.in/sebirss.xml"
MCA_RSS_URL  = "https://www.mca.gov.in/content/mca/global/en/data-and-reports/rss-feed.html"
ITAX_RSS_URL = "https://www.incometaxindia.gov.in/pages/whats-new.aspx"