variable "aws_region" {
  description = "AWS region to deploy resources"
  type        = string
  default     = "us-east-1"
}

variable "aws_role_arn" {
  description = "ARN of the IAM role to assume for AWS operations"
  type        = string
}

variable "aws_access_key" {
  description = "Access key for aws"
  type = string
}

variable "aws_secret_key" {
  description = "Access secret key for aws"
  type = string
}

variable "project_name" {
  description = "Name of the project"
  type        = string
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
}

variable "public_subnet_cidrs" {
  description = "CIDR blocks for public subnets"
  type        = list(string)
}

variable "instance_type" {
  description = "EC2 instance type"
  type        = string
}

variable "ssh_public_key" {
  description = "Public SSH key for EC2 access"
  type        = string
}

variable "admin_ip_cidr" {
  description = "CIDR block for admin access (your IP address)"
  type        = string
}