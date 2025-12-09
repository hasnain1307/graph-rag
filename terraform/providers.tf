terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  cloud {
    organization = "ISSM-AI"
    workspaces {
      name = "digital-eye"
    }
  }
}

provider "aws" {
  region = var.aws_region
  access_key = var.aws_access_key
  secret_key = var.aws_secret_key
  # # IAM role authentication
  # assume_role {
  #   role_arn = var.aws_role_arn
  # }
}