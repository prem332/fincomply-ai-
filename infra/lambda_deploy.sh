set -e
source .env 2>/dev/null || true

FUNCTION_NAME="fincomply-mcp-tools"
REGION="${AWS_REGION:-ap-south-1}"
ROLE_ARN="${AWS_LAMBDA_ROLE_ARN}"

if [ -z "$ROLE_ARN" ]; then
    echo "ERROR: AWS_LAMBDA_ROLE_ARN not set in .env"
    echo "Create an IAM role with AWSLambdaBasicExecutionRole and paste the ARN in .env"
    exit 1
fi

echo "=== Deploying MCP Tools to AWS Lambda ==="
echo "  Region:   $REGION"
echo "  Function: $FUNCTION_NAME"
echo ""

# ── Package Lambda code ───────────────────────────────────────────────────────
echo "→ Creating deployment package..."
mkdir -p /tmp/lambda-pkg
cp mcp_server/*.py /tmp/lambda-pkg/
cp config.py /tmp/lambda-pkg/

# Install dependencies into package (Lambda needs them bundled)
pip install httpx feedparser beautifulsoup4 lxml python-dateutil \
    -t /tmp/lambda-pkg/ --quiet

# Create ZIP
cd /tmp/lambda-pkg
zip -r /tmp/fincomply-lambda.zip . -q
cd -

echo "  Package size: $(du -sh /tmp/fincomply-lambda.zip | cut -f1)"

# ── Create or update Lambda function ─────────────────────────────────────────
echo "→ Deploying to Lambda..."

# Check if function exists
FUNC_EXISTS=$(aws lambda get-function --function-name "$FUNCTION_NAME" \
    --region "$REGION" 2>/dev/null && echo "yes" || echo "no")

if [ "$FUNC_EXISTS" = "yes" ]; then
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

echo "  ✓ Lambda deployed!"
echo ""
echo "=== NEXT STEP: Create API Gateway ==="
echo "  1. Go to AWS Console → API Gateway"
echo "  2. Create HTTP API"
echo "  3. Add routes: POST /gst, POST /rbi, POST /sebi, POST /mca"
echo "  4. Set Lambda integration to: $FUNCTION_NAME"
echo "  5. Deploy API and copy the URL"
echo "  6. Paste URL into .env as MCP_BASE_URL"
echo "”