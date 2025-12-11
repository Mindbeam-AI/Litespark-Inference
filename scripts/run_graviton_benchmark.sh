#!/bin/bash
#
# Graviton Benchmark Automation Script
#
# This script:
# 1. Launches a Graviton 3 EC2 instance (or uses existing)
# 2. Sets up the environment (clone repo, install deps)
# 3. Runs both short and long benchmarks
# 4. Downloads results locally
# 5. Optionally terminates the instance
#
# Usage:
#   ./run_graviton_benchmark.sh [--terminate] [--instance-id i-xxx]
#

set -e

# Configuration
INSTANCE_TYPE="c7g.4xlarge"  # Graviton 3, 16 vCPUs, 32GB RAM
AMI_ID="ami-0c7217cdde317cfec"  # Amazon Linux 2023 ARM64 (us-east-1)
KEY_NAME="graviton-benchmark-key"
SECURITY_GROUP="graviton-benchmark-ssh"
REGION="us-east-1"
KEY_FILE="$HOME/.ssh/${KEY_NAME}.pem"
RESULTS_DIR="./graviton_results_$(date +%Y%m%d_%H%M%S)"

# Get the local repo directory (parent of scripts/)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_REPO_DIR="$(dirname "$SCRIPT_DIR")"

# Parse arguments
TERMINATE_AFTER=false
INSTANCE_ID=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --terminate)
            TERMINATE_AFTER=true
            shift
            ;;
        --instance-id)
            INSTANCE_ID="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "========================================"
echo "Graviton Benchmark Automation"
echo "========================================"
echo "Instance type: $INSTANCE_TYPE"
echo "Region: $REGION"
echo "Results dir: $RESULTS_DIR"
echo ""

# Create results directory
mkdir -p "$RESULTS_DIR"

# Function to get instance public IP
get_instance_ip() {
    aws ec2 describe-instances \
        --instance-ids "$1" \
        --query 'Reservations[0].Instances[0].PublicIpAddress' \
        --output text \
        --region "$REGION"
}

# Function to wait for instance to be ready
wait_for_instance() {
    local instance_id="$1"
    echo "Waiting for instance $instance_id to be running..."
    aws ec2 wait instance-running --instance-ids "$instance_id" --region "$REGION"

    echo "Waiting for instance to pass status checks..."
    aws ec2 wait instance-status-ok --instance-ids "$instance_id" --region "$REGION"

    # Get IP
    local ip=$(get_instance_ip "$instance_id")
    echo "Instance IP: $ip"

    # Wait for SSH to be available
    echo "Waiting for SSH to be available..."
    local max_attempts=30
    local attempt=0
    while ! ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no -i "$KEY_FILE" ec2-user@"$ip" "echo 'SSH ready'" 2>/dev/null; do
        attempt=$((attempt + 1))
        if [ $attempt -ge $max_attempts ]; then
            echo "ERROR: SSH not available after $max_attempts attempts"
            exit 1
        fi
        echo "  Attempt $attempt/$max_attempts..."
        sleep 10
    done
    echo "SSH is ready!"
}

# Step 1: Check/create key pair
echo ""
echo "Step 1: Checking SSH key pair..."
if [ ! -f "$KEY_FILE" ]; then
    echo "Key file not found at $KEY_FILE"

    # Check if key exists in AWS - if so, delete it (we can't recover the private key)
    if aws ec2 describe-key-pairs --key-names "$KEY_NAME" --region "$REGION" >/dev/null 2>&1; then
        echo "Key pair '$KEY_NAME' exists in AWS but no local .pem file."
        echo "Deleting orphaned AWS key pair (private key is unrecoverable)..."
        aws ec2 delete-key-pair --key-name "$KEY_NAME" --region "$REGION"
    fi

    echo "Creating new key pair..."
    aws ec2 create-key-pair \
        --key-name "$KEY_NAME" \
        --query 'KeyMaterial' \
        --output text \
        --region "$REGION" > "$KEY_FILE"
    chmod 600 "$KEY_FILE"
    echo "Key pair created and saved to $KEY_FILE"
else
    echo "Using existing key file: $KEY_FILE"
fi

# Step 2: Check/create security group
echo ""
echo "Step 2: Checking security group..."
SG_ID=$(aws ec2 describe-security-groups \
    --filters "Name=group-name,Values=$SECURITY_GROUP" \
    --query 'SecurityGroups[0].GroupId' \
    --output text \
    --region "$REGION" 2>/dev/null || echo "None")

if [ "$SG_ID" == "None" ] || [ -z "$SG_ID" ]; then
    echo "Creating security group..."
    SG_ID=$(aws ec2 create-security-group \
        --group-name "$SECURITY_GROUP" \
        --description "SSH access for Graviton benchmarks" \
        --query 'GroupId' \
        --output text \
        --region "$REGION")

    # Allow SSH from anywhere (you may want to restrict this)
    aws ec2 authorize-security-group-ingress \
        --group-id "$SG_ID" \
        --protocol tcp \
        --port 22 \
        --cidr 0.0.0.0/0 \
        --region "$REGION"
    echo "Security group created: $SG_ID"
else
    echo "Using existing security group: $SG_ID"
fi

# Step 3: Launch or use existing instance
echo ""
echo "Step 3: Setting up EC2 instance..."
if [ -n "$INSTANCE_ID" ]; then
    echo "Using provided instance: $INSTANCE_ID"
    # Make sure it's running
    STATE=$(aws ec2 describe-instances \
        --instance-ids "$INSTANCE_ID" \
        --query 'Reservations[0].Instances[0].State.Name' \
        --output text \
        --region "$REGION")

    if [ "$STATE" == "stopped" ]; then
        echo "Starting stopped instance..."
        aws ec2 start-instances --instance-ids "$INSTANCE_ID" --region "$REGION"
        wait_for_instance "$INSTANCE_ID"
    elif [ "$STATE" != "running" ]; then
        echo "ERROR: Instance is in state: $STATE"
        exit 1
    fi
else
    # Check for existing graviton-benchmark instance
    echo "Checking for existing graviton-benchmark instance..."
    EXISTING_INSTANCE=$(aws ec2 describe-instances \
        --filters "Name=tag:Name,Values=graviton-benchmark" "Name=instance-state-name,Values=running,stopped" \
        --query 'Reservations[0].Instances[0].InstanceId' \
        --output text \
        --region "$REGION" 2>/dev/null)

    if [ -n "$EXISTING_INSTANCE" ] && [ "$EXISTING_INSTANCE" != "None" ]; then
        INSTANCE_ID="$EXISTING_INSTANCE"
        echo "Found existing instance: $INSTANCE_ID"

        STATE=$(aws ec2 describe-instances \
            --instance-ids "$INSTANCE_ID" \
            --query 'Reservations[0].Instances[0].State.Name' \
            --output text \
            --region "$REGION")

        if [ "$STATE" == "stopped" ]; then
            echo "Starting stopped instance..."
            aws ec2 start-instances --instance-ids "$INSTANCE_ID" --region "$REGION"
            wait_for_instance "$INSTANCE_ID"
        elif [ "$STATE" == "running" ]; then
            echo "Instance is already running"
            # Wait for SSH just in case
            INSTANCE_IP=$(get_instance_ip "$INSTANCE_ID")
            echo "Instance IP: $INSTANCE_IP"
        fi
    else
        # No existing instance, launch a new one
        # Find the right AMI for Graviton in the region
        echo "No existing instance found. Launching new one..."
        echo "Finding Amazon Linux 2023 ARM64 AMI..."
        AMI_ID=$(aws ec2 describe-images \
            --owners amazon \
            --filters "Name=name,Values=al2023-ami-2023*-arm64" "Name=state,Values=available" \
            --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' \
            --output text \
            --region "$REGION")
        echo "Using AMI: $AMI_ID"

        echo "Launching new Graviton instance..."
        INSTANCE_ID=$(aws ec2 run-instances \
            --image-id "$AMI_ID" \
            --instance-type "$INSTANCE_TYPE" \
            --key-name "$KEY_NAME" \
            --security-group-ids "$SG_ID" \
            --associate-public-ip-address \
            --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":50,"VolumeType":"gp3"}}]' \
            --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=graviton-benchmark}]" \
            --query 'Instances[0].InstanceId' \
            --output text \
            --region "$REGION")
        echo "Launched instance: $INSTANCE_ID"
        wait_for_instance "$INSTANCE_ID"
    fi
fi

INSTANCE_IP=$(get_instance_ip "$INSTANCE_ID")
echo "Instance IP: $INSTANCE_IP"

# SSH command helper
SSH_CMD="ssh -o StrictHostKeyChecking=no -i $KEY_FILE ec2-user@$INSTANCE_IP"
SCP_CMD="scp -o StrictHostKeyChecking=no -i $KEY_FILE"

# Step 4: Setup environment on instance
echo ""
echo "Step 4: Setting up environment on instance..."

# Install system dependencies first
$SSH_CMD << 'SETUP_DEPS'
set -e
echo "=== Installing system dependencies ==="
sudo dnf install -y python3-pip gcc-c++ cmake python3-devel ninja-build || \
sudo yum install -y python3-pip gcc-c++ cmake python3-devel ninja-build
echo "=== System dependencies installed ==="
SETUP_DEPS

# Copy local repo to instance via SCP (avoids git auth issues)
echo "Copying local repository to instance..."
REMOTE_REPO="ec2-user@$INSTANCE_IP:~/matmulMM"

# Create a tarball excluding large/unnecessary files
TARBALL="/tmp/matmulMM_upload.tar.gz"
echo "Creating tarball of local repo..."
tar -czf "$TARBALL" -C "$LOCAL_REPO_DIR" \
    --exclude='.git' \
    --exclude='*.pyc' \
    --exclude='__pycache__' \
    --exclude='venv' \
    --exclude='*.egg-info' \
    --exclude='.pytest_cache' \
    --exclude='build' \
    --exclude='dist' \
    --exclude='*.so' \
    .

echo "Uploading to instance..."
$SCP_CMD "$TARBALL" "ec2-user@$INSTANCE_IP:/tmp/"

# Extract and setup on remote
$SSH_CMD << 'SETUP_SCRIPT'
set -e
echo "=== Setting up Graviton benchmark environment ==="

REPO_DIR="$HOME/matmulMM"

# Remove old repo if exists, extract fresh copy
rm -rf "$REPO_DIR"
mkdir -p "$REPO_DIR"
tar -xzf /tmp/matmulMM_upload.tar.gz -C "$REPO_DIR"
rm /tmp/matmulMM_upload.tar.gz

# Create virtual environment if needed
if [ ! -d "$REPO_DIR/venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$REPO_DIR/venv"
fi

# Activate and install dependencies
source "$REPO_DIR/venv/bin/activate"
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install torch transformers safetensors datasets huggingface_hub numpy matplotlib

echo "=== Setup complete ==="
SETUP_SCRIPT

# Clean up local tarball
rm -f "$TARBALL"

# Step 5: Run benchmarks
echo ""
echo "Step 5: Running benchmarks..."

$SSH_CMD << 'BENCHMARK_SCRIPT'
set -e
cd ~/matmulMM
source venv/bin/activate

echo "=== Running Short Benchmark ==="
cd inference
python3 benchmark_comparison_short.py 2>&1 | tee ~/benchmark_short_results.txt

echo ""
echo "=== Running Long Benchmark ==="
python3 benchmark_comparison_long.py 2>&1 | tee ~/benchmark_long_results.txt

echo ""
echo "=== Benchmarks Complete ==="

# Create summary
echo "Creating summary..."
cat > ~/benchmark_summary.txt << EOF
Graviton Benchmark Results
==========================
Date: $(date)
Instance Type: $(curl -s http://169.254.169.254/latest/meta-data/instance-type)
Architecture: $(uname -m)
CPU Info: $(grep -m1 'model name' /proc/cpuinfo 2>/dev/null || echo "ARM Neoverse")
Threads: $(nproc)

Console output logs:
- benchmark_short_results.txt
- benchmark_long_results.txt

Structured results and graphs (in benchmark_results/):
- benchmark_results.json (short benchmark)
- benchmark_summary.txt (short benchmark)
- benchmark_long_results.json (long benchmark)
- benchmark_long_summary.txt (long benchmark)
- model_comparison.png
- kernel_speedup.png
- speedup_summary.png
- prefill_scaling.png
- batch_scaling.png
- generation_scaling.png
- speedup_summary_long.png
EOF

cat ~/benchmark_summary.txt

# List generated files
echo ""
echo "Generated files in benchmark_results/:"
ls -la ~/matmulMM/inference/benchmark_results/ 2>/dev/null || echo "(directory not found)"
BENCHMARK_SCRIPT

# Step 6: Download results
echo ""
echo "Step 6: Downloading results..."

# Download console output logs
$SCP_CMD "ec2-user@$INSTANCE_IP:~/benchmark_short_results.txt" "$RESULTS_DIR/" || true
$SCP_CMD "ec2-user@$INSTANCE_IP:~/benchmark_long_results.txt" "$RESULTS_DIR/" || true
$SCP_CMD "ec2-user@$INSTANCE_IP:~/benchmark_summary.txt" "$RESULTS_DIR/" || true

# Download structured results (JSON, text summaries, graphs)
# These are saved to ~/matmulMM/inference/benchmark_results/
echo "Downloading structured results and graphs..."
$SCP_CMD -r "ec2-user@$INSTANCE_IP:~/matmulMM/inference/benchmark_results/" "$RESULTS_DIR/benchmark_results/" || true

# List all downloaded files
echo ""
echo "Results downloaded to: $RESULTS_DIR"
echo ""
echo "Console output logs:"
ls -la "$RESULTS_DIR"/*.txt 2>/dev/null || echo "  (none)"
echo ""
echo "Structured results and graphs:"
ls -la "$RESULTS_DIR/benchmark_results/" 2>/dev/null || echo "  (none)"

# Step 7: Optionally terminate
if [ "$TERMINATE_AFTER" = true ]; then
    echo ""
    echo "Step 7: Terminating instance..."
    aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" --region "$REGION"
    echo "Instance $INSTANCE_ID termination initiated"
else
    echo ""
    echo "Instance is still running: $INSTANCE_ID ($INSTANCE_IP)"
    echo "To terminate later: aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region $REGION"
    echo "To SSH: ssh -i $KEY_FILE ec2-user@$INSTANCE_IP"
fi

# Save instance info
cat > "$RESULTS_DIR/instance_info.txt" << EOF
Instance ID: $INSTANCE_ID
Instance IP: $INSTANCE_IP
Instance Type: $INSTANCE_TYPE
Region: $REGION
Key File: $KEY_FILE

SSH Command:
ssh -i $KEY_FILE ec2-user@$INSTANCE_IP

Terminate Command:
aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region $REGION
EOF

echo ""
echo "========================================"
echo "Benchmark Complete!"
echo "========================================"
echo "Results saved to: $RESULTS_DIR"
echo ""
cat "$RESULTS_DIR/instance_info.txt"
