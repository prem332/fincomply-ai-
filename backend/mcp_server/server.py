import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from gst_tool import fetch_gst_data
from rbi_tool import fetch_rbi_data
from sebi_tool import fetch_sebi_data
from mca_tool import fetch_mca_data
from income_tax_tool import fetch_income_tax_data

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization",
    "Access-Control-Allow-Methods": "POST,OPTIONS",
    "Content-Type": "application/json",
}


def _success(data: dict) -> dict:
    return {"statusCode": 200, "headers": CORS_HEADERS, "body": json.dumps(data)}


def _error(message: str, status: int = 400) -> dict:
    return {"statusCode": status, "headers": CORS_HEADERS, "body": json.dumps({"error": message})}


def lambda_handler(event: dict, context) -> dict:
    if event.get("httpMethod") == "OPTIONS":
        return {"statusCode": 200, "headers": CORS_HEADERS, "body": ""}

    path = event.get("rawPath", event.get("path", "")).rstrip("/").lower()
    domain = path.split("/")[-1]

    try:
        body = json.loads(event.get("body") or "{}")
        query = body.get("query", "")
        max_results = int(body.get("max_results", 10))
    except (json.JSONDecodeError, ValueError) as e:
        return _error(f"Invalid request body: {e}")

    if not query:
        return _error("'query' field is required")

    if max_results < 1 or max_results > 20:
        max_results = 10

    logger.info(f"MCP request: domain={domain}, query={query[:60]}")

    try:
        if domain == "gst":
            results = fetch_gst_data(query, max_results)
        elif domain == "rbi":
            results = fetch_rbi_data(query, max_results)
        elif domain == "sebi":
            results = fetch_sebi_data(query, max_results)
        elif domain == "mca":
            results = fetch_mca_data(query, max_results)
        elif domain == "incometax":
            results = fetch_income_tax_data(query, max_results)
        else:
            return _error(f"Unknown domain: '{domain}'. Use: gst, rbi, sebi, mca, incometax")

        return _success({
            "domain": domain,
            "query": query,
            "results": results,
            "count": len(results),
        })

    except Exception as e:
        logger.error(f"Tool error for domain={domain}: {e}")
        return _error(f"Tool execution failed: {str(e)}", status=500)