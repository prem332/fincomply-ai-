source .env 2>/dev/null || true

FUNCTION_NAME="fincomply-mcp-tools"
REGION="${AWS_REGION:-ap-south-1}"
ROLE_ARN="${AWS_LAMBDA_ROLE_ARN}"

if [ -z "$ROLE_ARN" ]; then
    echo "ERROR: AWS_LAMBDA_ROLE_ARN not set in .env"
    exit 1
fi

echo "=== Deploying MCP Tools to AWS Lambda ==="
echo "  Region:   $REGION"
echo "  Function: $FUNCTION_NAME"
echo ""

# ── Clean previous build artifacts ───────────────────────────────────────────
echo "→ Cleaning previous build..."
rm -rf /tmp/lambda-pkg /tmp/fincomply-lambda.zip
mkdir -p /tmp/lambda-pkg

# ── Copy MCP server files ─────────────────────────────────────────────────────
echo "→ Creating deployment package..."
cp backend/mcp_server/*.py /tmp/lambda-pkg/
cp backend/config.py /tmp/lambda-pkg/

# ── Install dependencies into package ────────────────────────────────────────
pip install httpx feedparser beautifulsoup4 lxml python-dateutil python-dotenv \
    -t /tmp/lambda-pkg/ --quiet

# ── Create ZIP ────────────────────────────────────────────────────────────────
cd /tmp/lambda-pkg
zip -r /tmp/fincomply-lambda.zip . -q
cd -

echo "  Package size: $(du -sh /tmp/fincomply-lambda.zip | cut -f1)"

# ── Create or update Lambda function ─────────────────────────────────────────
echo "→ Deploying to Lambda..."

if aws lambda get-function --function-name "$FUNCTION_NAME" --region "$REGION" --no-cli-pager 2>/dev/null; then
    echo "  Updating existing function..."
    aws lambda update-function-code \
        --function-name "$FUNCTION_NAME" \
        --zip-file fileb:///tmp/fincomply-lambda.zip \
        --region "$REGION" \
        --no-cli-pager
else
    echo "  Creating new function..."
    aws lambda create-function \
        --function-name "$FUNCTION_NAME" \
        --runtime python3.11 \
        --role "$ROLE_ARN" \
        --handler server.lambda_handler \
        --zip-file fileb:///tmp/fincomply-lambda.zip \
        --timeout 30 \
        --memory-size 256 \
        --region "$REGION" \
        --no-cli-pager
fi

echo ""
echo "  Lambda deployed successfully!"
echo "”