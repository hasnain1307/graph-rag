#!/bin/bash
set -e

# Update system packages
yum update -y

# Install Docker
amazon-linux-extras install docker -y
yum install -y docker
systemctl start docker
systemctl enable docker
usermod -aG docker ec2-user

# Install Docker Compose
curl -L "https://github.com/docker/compose/releases/download/v2.21.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Install AWS CLI
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
yum install -y unzip
unzip awscliv2.zip
./aws/install

# Create app directory structure
mkdir -p /home/ec2-user/app
chown -R ec2-user:ec2-user /home/ec2-user/app

echo "Installation completed"