set -e

echo "=== FinComply AI — EC2 t2.micro Setup ==="
echo ""

# ── Step 1: Update system ─────────────────────────────────────────────────────
echo "→ Updating system packages..."
sudo yum update -y 2>/dev/null || sudo apt-get update -y

# ── Step 2: Install Docker ────────────────────────────────────────────────────
echo "→ Installing Docker..."
if ! command -v docker &>/dev/null; then
    # Amazon Linux 2
    sudo yum install -y docker 2>/dev/null || \
    # Ubuntu
    sudo apt-get install -y docker.io
    sudo systemctl start docker
    sudo systemctl enable docker
    sudo usermod -aG docker $USER
    echo "  Docker installed. Log out and back in if 'docker ps' gives permission error."
fi
echo "  Docker version: $(docker --version)"

# ── Step 3: Install AWS CLI ───────────────────────────────────────────────────
echo "→ Installing AWS CLI..."
if ! command -v aws &>/dev/null; then
    curl -s "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
    sudo apt-get install -y unzip 2>/dev/null || sudo yum install -y unzip
    unzip -q awscliv2.zip
    sudo ./aws/install
    rm -rf awscliv2.zip aws/
fi
echo "  AWS CLI version: $(aws --version)"

# ── Step 4: Install Git ───────────────────────────────────────────────────────
echo "→ Installing Git..."
sudo yum install -y git 2>/dev/null || sudo apt-get install -y git

# ── Step 5: Add swap space (CRITICAL for t3.micro with 1GB RAM) ───────────────
echo "→ Setting up swap space (2GB)..."
if [ ! -f /swapfile ]; then
    sudo dd if=/dev/zero of=/swapfile bs=1M count=2048
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    # Make swap permanent
    echo "/swapfile none swap sw 0 0" | sudo tee -a /etc/fstab
    echo "  ✓ 2GB swap space created (prevents OOM kills on 1GB RAM)"
else
    echo "  Swap already configured"
fi
free -h

# ── Step 6: Clone project ─────────────────────────────────────────────────────
echo "→ Cloning project from GitHub..."
PROJECT_DIR="$HOME/fincomply-ai"
if [ ! -d "$PROJECT_DIR" ]; then
    # Replace with your actual GitHub repo URL
    git clone https://github.com/YOUR_GITHUB_USERNAME/fincomply-ai.git "$PROJECT_DIR"
    echo "  Project cloned to $PROJECT_DIR"
else
    cd "$PROJECT_DIR" && git pull
    echo "  Project updated"
fi

# ── Step 7: Create .env file ──────────────────────────────────────────────────
echo "→ Setting up environment file..."
cd "$PROJECT_DIR"
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo ""
    echo "  ⚠️  IMPORTANT: Edit .env with your actual API keys:"
    echo "  nano $PROJECT_DIR/.env"
    echo ""
fi

echo ""
echo "=== EC2 Setup Complete! ==="
echo ""
echo "NEXT STEPS:"
echo "1. Edit your .env file: nano $PROJECT_DIR/.env"
echo "2. Build Docker image: cd $PROJECT_DIR && docker build -f docker/Dockerfile -t fincomply-ai ."
echo "3. Run container: docker run -d -p 8000:8000 --env-file .env --name fincomply fincomply-ai"
echo "4. Test: curl http://localhost:8000/health"
echo ""