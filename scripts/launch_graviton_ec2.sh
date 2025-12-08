#!/bin/bash
# Launch a Graviton EC2 instance for testing
#
# Prerequisites:
#   - AWS CLI configured with credentials
#   - An existing key pair (or create one)
#   - An existing security group with SSH access (port 22)
#
# Usage:
#   ./launch_graviton_ec2.sh [key-name] [security-group-id]
#
# Example:
#   ./launch_graviton_ec2.sh my-key-pair sg-0123456789abcdef0

set -e

KEY_NAME="${1:-}"
SECURITY_GROUP="${2:-}"
INSTANCE_TYPE="${3:-c7g.xlarge}"  # Graviton 3, 4 vCPUs, ~$0.15/hr
REGION="${AWS_REGION:-us-east-1}"

# Amazon Linux 2023 ARM64 AMI (update for your region)
# You can find the latest AMI with:
# aws ec2 describe-images --owners amazon --filters "Name=name,Values=al2023-ami-*-arm64" --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' --output text
AMI_ID=""

if [ -z "$KEY_NAME" ]; then
    echo "Usage: $0 <key-name> <security-group-id> [instance-type]"
    echo ""
    echo "Example:"
    echo "  $0 my-key-pair sg-0123456789abcdef0 c7g.xlarge"
    echo ""
    echo "Available Graviton instance types:"
    echo "  c7g.medium   - 1 vCPU,  2 GB  (~\$0.04/hr)"
    echo "  c7g.large    - 2 vCPU,  4 GB  (~\$0.07/hr)"
    echo "  c7g.xlarge   - 4 vCPU,  8 GB  (~\$0.15/hr) [default]"
    echo "  c7g.2xlarge  - 8 vCPU,  16 GB (~\$0.29/hr)"
    echo "  c7g.4xlarge  - 16 vCPU, 32 GB (~\$0.58/hr)"
    exit 1
fi

if [ -z "$SECURITY_GROUP" ]; then
    echo "Error: Security group ID required"
    exit 1
fi

echo "Finding latest Amazon Linux 2023 ARM64 AMI in $REGION..."
AMI_ID=$(aws ec2 describe-images \
    --region "$REGION" \
    --owners amazon \
    --filters "Name=name,Values=al2023-ami-*-arm64" "Name=state,Values=available" \
    --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' \
    --output text)

if [ -z "$AMI_ID" ] || [ "$AMI_ID" == "None" ]; then
    echo "Error: Could not find AMI"
    exit 1
fi

echo "Using AMI: $AMI_ID"
echo "Instance type: $INSTANCE_TYPE"
echo "Key pair: $KEY_NAME"
echo "Security group: $SECURITY_GROUP"
echo ""

# User data script to set up the instance
USER_DATA=$(cat <<'USERDATA'
#!/bin/bash
set -e

# Update system
dnf update -y

# Install development tools
dnf groupinstall -y "Development Tools"
dnf install -y git python3.11 python3.11-pip python3.11-devel openmp

# Create symlinks
ln -sf /usr/bin/python3.11 /usr/bin/python3
ln -sf /usr/bin/pip3.11 /usr/bin/pip3

# Install Python packages
pip3 install torch numpy transformers datasets

# Clone the repo
cd /home/ec2-user
git clone https://github.com/tonymindbeam/matmulMM.git
chown -R ec2-user:ec2-user matmulMM

# Create a ready flag
touch /home/ec2-user/.setup_complete

echo "Setup complete!"
USERDATA
)

echo "Launching instance..."
INSTANCE_ID=$(aws ec2 run-instances \
    --region "$REGION" \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" \
    --security-group-ids "$SECURITY_GROUP" \
    --user-data "$USER_DATA" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=graviton-benchmark}]" \
    --query 'Instances[0].InstanceId' \
    --output text)

echo "Instance ID: $INSTANCE_ID"
echo ""
echo "Waiting for instance to be running..."
aws ec2 wait instance-running --region "$REGION" --instance-ids "$INSTANCE_ID"

# Get public IP
PUBLIC_IP=$(aws ec2 describe-instances \
    --region "$REGION" \
    --instance-ids "$INSTANCE_ID" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo ""
echo "=========================================="
echo "Instance is running!"
echo "=========================================="
echo "Instance ID: $INSTANCE_ID"
echo "Public IP: $PUBLIC_IP"
echo ""
echo "Wait ~2-3 minutes for setup to complete, then connect:"
echo ""
echo "  ssh -i ~/.ssh/${KEY_NAME}.pem ec2-user@${PUBLIC_IP}"
echo ""
echo "Once connected, run:"
echo ""
echo "  cd matmulMM"
echo "  git checkout cpu-dev"
echo "  python3 scripts/benchmark_graviton.py --verify"
echo ""
echo "To terminate when done:"
echo ""
echo "  aws ec2 terminate-instances --instance-ids $INSTANCE_ID"
echo ""
