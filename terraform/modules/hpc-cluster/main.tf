terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
  }
}

variable "cluster_name" {
  description = "Name of the HPC cluster"
  type        = string
  default     = "hpc-training-cluster"
}

variable "cluster_size" {
  description = "Number of nodes in the cluster"
  type        = number
  default     = 4
}

variable "gpu_per_node" {
  description = "Number of GPUs per node"
  type        = number
  default     = 8
}

variable "instance_type" {
  description = "EC2 instance type for compute nodes"
  type        = string
  default     = "p4d.24xlarge"
}

variable "availability_zones" {
  description = "Availability zones for the cluster"
  type        = list(string)
  default     = ["us-west-2a", "us-west-2b"]
}

# VPC Configuration
resource "aws_vpc" "hpc_vpc" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name = "${var.cluster_name}-vpc"
    Type = "hpc-infrastructure"
  }
}

resource "aws_subnet" "hpc_subnet" {
  count             = length(var.availability_zones)
  vpc_id            = aws_vpc.hpc_vpc.id
  cidr_block        = "10.0.${count.index + 1}.0/24"
  availability_zone = var.availability_zones[count.index]

  map_public_ip_on_launch = true

  tags = {
    Name = "${var.cluster_name}-subnet-${count.index + 1}"
    Type = "hpc-infrastructure"
  }
}

resource "aws_internet_gateway" "hpc_igw" {
  vpc_id = aws_vpc.hpc_vpc.id

  tags = {
    Name = "${var.cluster_name}-igw"
  }
}

resource "aws_route_table" "hpc_rt" {
  vpc_id = aws_vpc.hpc_vpc.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.hpc_igw.id
  }

  tags = {
    Name = "${var.cluster_name}-route-table"
  }
}

resource "aws_route_table_association" "hpc_rta" {
  count          = length(aws_subnet.hpc_subnet)
  subnet_id      = aws_subnet.hpc_subnet[count.index].id
  route_table_id = aws_route_table.hpc_rt.id
}

# EKS Cluster
resource "aws_eks_cluster" "hpc_cluster" {
  name     = var.cluster_name
  role_arn = aws_iam_role.hpc_cluster_role.arn
  version  = "1.28"

  vpc_config {
    subnet_ids              = aws_subnet.hpc_subnet[*].id
    endpoint_private_access = true
    endpoint_public_access  = true
  }

  depends_on = [
    aws_iam_role_policy_attachment.hpc_cluster_AmazonEKSClusterPolicy,
  ]

  tags = {
    Name = var.cluster_name
    Type = "hpc-infrastructure"
  }
}

# IAM Roles
resource "aws_iam_role" "hpc_cluster_role" {
  name = "${var.cluster_name}-cluster-role"

  assume_role_policy = jsonencode({
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "eks.amazonaws.com"
      }
    }]
    Version = "2012-10-17"
  })
}

resource "aws_iam_role_policy_attachment" "hpc_cluster_AmazonEKSClusterPolicy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
  role       = aws_iam_role.hpc_cluster_role.name
}

resource "aws_iam_role" "hpc_node_role" {
  name = "${var.cluster_name}-node-role"

  assume_role_policy = jsonencode({
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "ec2.amazonaws.com"
      }
    }]
    Version = "2012-10-17"
  })
}

resource "aws_iam_role_policy_attachment" "hpc_node_AmazonEKSWorkerNodePolicy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy"
  role       = aws_iam_role.hpc_node_role.name
}

resource "aws_iam_role_policy_attachment" "hpc_node_AmazonEKS_CNI_Policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy"
  role       = aws_iam_role.hpc_node_role.name
}

resource "aws_iam_role_policy_attachment" "hpc_node_AmazonEC2ContainerRegistryReadOnly" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
  role       = aws_iam_role.hpc_node_role.name
}

# Node Group for GPU instances
resource "aws_eks_node_group" "hpc_gpu_nodes" {
  cluster_name    = aws_eks_cluster.hpc_cluster.name
  node_group_name = "${var.cluster_name}-gpu-nodes"
  node_role_arn   = aws_iam_role.hpc_node_role.arn
  subnet_ids      = aws_subnet.hpc_subnet[*].id
  instance_types  = [var.instance_type]

  scaling_config {
    desired_size = var.cluster_size
    max_size     = var.cluster_size * 2
    min_size     = 1
  }

  update_config {
    max_unavailable = 1
  }

  # GPU-specific configuration
  launch_template {
    id      = aws_launch_template.hpc_gpu_template.id
    version = aws_launch_template.hpc_gpu_template.latest_version
  }

  depends_on = [
    aws_iam_role_policy_attachment.hpc_node_AmazonEKSWorkerNodePolicy,
    aws_iam_role_policy_attachment.hpc_node_AmazonEKS_CNI_Policy,
    aws_iam_role_policy_attachment.hpc_node_AmazonEC2ContainerRegistryReadOnly,
  ]

  tags = {
    Name = "${var.cluster_name}-gpu-nodes"
    Type = "hpc-compute"
  }
}

resource "aws_launch_template" "hpc_gpu_template" {
  name_prefix   = "${var.cluster_name}-gpu-"
  image_id      = data.aws_ami.eks_gpu_ami.id
  instance_type = var.instance_type

  vpc_security_group_ids = [aws_security_group.hpc_node_sg.id]

  user_data = base64encode(templatefile("${path.module}/user_data.sh", {
    cluster_name = var.cluster_name
  }))

  tag_specifications {
    resource_type = "instance"
    tags = {
      Name = "${var.cluster_name}-gpu-instance"
      Type = "hpc-compute"
    }
  }
}

data "aws_ami" "eks_gpu_ami" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["amazon-eks-gpu-node-*"]
  }
}

resource "aws_security_group" "hpc_node_sg" {
  name_prefix = "${var.cluster_name}-node-"
  vpc_id      = aws_vpc.hpc_vpc.id

  ingress {
    from_port = 0
    to_port   = 65535
    protocol  = "tcp"
    self      = true
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.cluster_name}-node-sg"
  }
}

# Outputs
output "cluster_endpoint" {
  description = "Endpoint for EKS control plane"
  value       = aws_eks_cluster.hpc_cluster.endpoint
}

output "cluster_security_group_id" {
  description = "Security group ids attached to the cluster control plane"
  value       = aws_eks_cluster.hpc_cluster.vpc_config[0].cluster_security_group_id
}

output "cluster_name" {
  description = "Kubernetes Cluster Name"
  value       = aws_eks_cluster.hpc_cluster.name
}
