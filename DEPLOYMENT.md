# VeriNews — AWS Deployment Guide

Production deployment on AWS using ECS Fargate, ALB, RDS, S3, and CloudFront.

---

## Architecture Overview

```
Internet
   │
   ▼
Route 53 (DNS)
   │
   ▼
CloudFront (CDN — caches /static/*)
   │
   ▼
Application Load Balancer (HTTPS:443)
   │
   ▼
ECS Fargate (Docker container, 2 tasks min)
   │
   ├── RDS PostgreSQL (db.t3.micro)
   ├── S3 bucket (model pkl files)
   └── Secrets Manager (API keys)
```

---

## Prerequisites

- AWS CLI v2 installed and configured (`aws configure`)
- Docker installed locally
- An AWS account with appropriate IAM permissions
- A domain name (optional but recommended)

---

## Step 1 — Set Environment Variables

```bash
export AWS_REGION=us-east-1
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export APP_NAME=verinews
export ECR_REPO=$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$APP_NAME
```

---

## Step 2 — Build & Push Docker Image to ECR

```bash
# Create ECR repository
aws ecr create-repository --repository-name $APP_NAME --region $AWS_REGION

# Authenticate Docker to ECR
aws ecr get-login-password --region $AWS_REGION | \
  docker login --username AWS --password-stdin $ECR_REPO

# Build and tag
docker build -t $APP_NAME:latest .
docker tag $APP_NAME:latest $ECR_REPO:latest

# Push
docker push $ECR_REPO:latest
```

---

## Step 3 — Upload Models to S3

```bash
# Create S3 bucket for model artifacts
aws s3 mb s3://$APP_NAME-models --region $AWS_REGION

# Upload trained models (run after python scripts/train_models.py)
aws s3 sync models/ s3://$APP_NAME-models/models/ --exclude "*.pyc"

# Verify
aws s3 ls s3://$APP_NAME-models/models/
```

Your Docker container's startup script should download models from S3 on boot:

```bash
# Add to container entrypoint (or as an ECS sidecar task):
aws s3 sync s3://$APP_NAME-models/models/ /app/models/
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 2
```

---

## Step 4 — Store Secrets in AWS Secrets Manager

```bash
# Store each API key as a separate secret
aws secretsmanager create-secret \
  --name verinews/OPENAI_API_KEY \
  --secret-string "sk-..."

aws secretsmanager create-secret \
  --name verinews/NEWS_API_KEY \
  --secret-string "your-newsapi-key"

aws secretsmanager create-secret \
  --name verinews/PERPLEXITY_API_KEY \
  --secret-string "pplx-..."
```

The ECS task IAM role must have `secretsmanager:GetSecretValue` permission.

---

## Step 5 — Create RDS PostgreSQL (optional — upgrades from SQLite)

```bash
# Create a DB subnet group first (use your VPC's private subnets)
aws rds create-db-subnet-group \
  --db-subnet-group-name verinews-db-subnet \
  --db-subnet-group-description "VeriNews DB subnet" \
  --subnet-ids subnet-xxx subnet-yyy

# Create PostgreSQL instance
aws rds create-db-instance \
  --db-instance-identifier verinews-db \
  --db-instance-class db.t3.micro \
  --engine postgres \
  --engine-version 15.4 \
  --master-username verinews \
  --master-user-password "REPLACE_ME" \
  --allocated-storage 20 \
  --db-subnet-group-name verinews-db-subnet \
  --no-publicly-accessible \
  --region $AWS_REGION
```

Update `src/storage/database.py` to use `psycopg2` when `DATABASE_URL` env var is set (the current SQLite code stays as a fallback for local dev).

---

## Step 6 — Create ECS Fargate Cluster & Service

```bash
# Create cluster
aws ecs create-cluster --cluster-name $APP_NAME --region $AWS_REGION

# Register task definition (see task-definition.json below)
aws ecs register-task-definition \
  --cli-input-json file://infra/task-definition.json

# Create service (after ALB is set up)
aws ecs create-service \
  --cluster $APP_NAME \
  --service-name $APP_NAME-service \
  --task-definition $APP_NAME \
  --desired-count 2 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx,subnet-yyy],securityGroups=[sg-xxx],assignPublicIp=DISABLED}" \
  --load-balancers "targetGroupArn=arn:aws:elasticloadbalancing:...,containerName=$APP_NAME,containerPort=8000"
```

**Minimal `infra/task-definition.json`:**
```json
{
  "family": "verinews",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::ACCOUNT:role/ecsTaskExecutionRole",
  "taskRoleArn":      "arn:aws:iam::ACCOUNT:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "verinews",
      "image": "ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/verinews:latest",
      "portMappings": [{"containerPort": 8000, "protocol": "tcp"}],
      "environment": [
        {"name": "MODELS_DIR", "value": "/app/models"}
      ],
      "secrets": [
        {"name": "OPENAI_API_KEY",     "valueFrom": "arn:aws:secretsmanager:...:verinews/OPENAI_API_KEY"},
        {"name": "NEWS_API_KEY",       "valueFrom": "arn:aws:secretsmanager:...:verinews/NEWS_API_KEY"},
        {"name": "PERPLEXITY_API_KEY", "valueFrom": "arn:aws:secretsmanager:...:verinews/PERPLEXITY_API_KEY"}
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/verinews",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8000/api/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3
      }
    }
  ]
}
```

---

## Step 7 — Application Load Balancer

```bash
# Create ALB (replace subnet/VPC IDs with yours)
aws elbv2 create-load-balancer \
  --name $APP_NAME-alb \
  --subnets subnet-pub1 subnet-pub2 \
  --security-groups sg-alb \
  --scheme internet-facing \
  --type application

# Create target group
aws elbv2 create-target-group \
  --name $APP_NAME-tg \
  --protocol HTTP \
  --port 8000 \
  --vpc-id vpc-xxx \
  --target-type ip \
  --health-check-path /api/health \
  --health-check-interval-seconds 30

# Create HTTPS listener (requires ACM certificate)
aws elbv2 create-listener \
  --load-balancer-arn <ALB_ARN> \
  --protocol HTTPS --port 443 \
  --certificates CertificateArn=<ACM_CERT_ARN> \
  --default-actions Type=forward,TargetGroupArn=<TG_ARN>
```

---

## Step 8 — CloudFront Distribution

```bash
aws cloudfront create-distribution \
  --distribution-config '{
    "Origins": {
      "Quantity": 1,
      "Items": [{
        "Id": "verinews-alb",
        "DomainName": "YOUR-ALB.us-east-1.elb.amazonaws.com",
        "CustomOriginConfig": {"HTTPSPort": 443, "OriginProtocolPolicy": "https-only"}
      }]
    },
    "DefaultCacheBehavior": {
      "TargetOriginId": "verinews-alb",
      "CachePolicyId": "658327ea-f89d-4fab-a63d-7e88639e58f6",
      "ViewerProtocolPolicy": "redirect-to-https",
      "AllowedMethods": {"Quantity": 7, "Items": ["GET","HEAD","OPTIONS","PUT","PATCH","POST","DELETE"]}
    },
    "CacheBehaviors": {
      "Quantity": 1,
      "Items": [{
        "PathPattern": "/static/*",
        "TargetOriginId": "verinews-alb",
        "CachePolicyId": "4135ea2d-6df8-44a3-9df3-4b5a84be39ad",
        "ViewerProtocolPolicy": "redirect-to-https"
      }]
    },
    "Enabled": true,
    "Comment": "VeriNews CDN"
  }'
```

---

## Step 9 — Auto-Scaling

```bash
# Register scalable target
aws application-autoscaling register-scalable-target \
  --service-namespace ecs \
  --resource-id service/$APP_NAME/$APP_NAME-service \
  --scalable-dimension ecs:service:DesiredCount \
  --min-capacity 1 \
  --max-capacity 10

# CPU-based scaling policy (scale out at 70% CPU)
aws application-autoscaling put-scaling-policy \
  --policy-name $APP_NAME-cpu-scaling \
  --service-namespace ecs \
  --resource-id service/$APP_NAME/$APP_NAME-service \
  --scalable-dimension ecs:service:DesiredCount \
  --policy-type TargetTrackingScaling \
  --target-tracking-scaling-policy-configuration '{
    "TargetValue": 70.0,
    "PredefinedMetricSpecification": {
      "PredefinedMetricType": "ECSServiceAverageCPUUtilization"
    },
    "ScaleInCooldown": 300,
    "ScaleOutCooldown": 60
  }'
```

---

## Step 10 — CI/CD via GitHub Actions

The `.github/workflows/ci.yml` already runs tests and builds Docker. Add a deploy job:

```yaml
  deploy:
    name: Deploy to ECS
    needs: [test, docker]
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id:     ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-east-1

      - name: Push new image
        run: |
          aws ecr get-login-password | docker login --username AWS --password-stdin $ECR_REPO
          docker build -t $ECR_REPO:${{ github.sha }} .
          docker push $ECR_REPO:${{ github.sha }}

      - name: Update ECS service
        run: |
          aws ecs update-service \
            --cluster verinews \
            --service verinews-service \
            --force-new-deployment
```

---

## Estimated Monthly AWS Cost

| Service              | Config            | Est. cost / month |
|----------------------|-------------------|-------------------|
| ECS Fargate (2 tasks)| 1 vCPU, 2 GB each | ~$60              |
| RDS PostgreSQL       | db.t3.micro        | ~$15              |
| ALB                  | Standard          | ~$18              |
| CloudFront           | 10 GB transfer    | ~$1               |
| S3                   | 1 GB models       | ~$0.03            |
| Secrets Manager      | 4 secrets         | ~$1.60            |
| **Total**            |                   | **~$96/month**    |

> For a demo/portfolio environment: scale tasks to 0 when idle → ~$30/month.

---

## Local Development (Quick Start)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

# 3. Set environment variables
cp .env.example .env
# Edit .env to add NEWS_API_KEY, OPENAI_API_KEY, PERPLEXITY_API_KEY

# 4. (Optional) Download datasets and train models
#    — ISOT: download True.csv and Fake.csv from Kaggle and place in data/raw/isot/
python scripts/prepare_data.py
python scripts/train_models.py

# 5. Start the server
uvicorn api.main:app --reload --port 8000

# 6. Open http://localhost:8000
```

---

## Terminal Commands — Complete Ordered List

```bash
# ── 0. Setup ──────────────────────────────────────────────────────────────
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
cp .env.example .env

# ── 1. Download ISOT dataset ──────────────────────────────────────────────
# Go to https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
# Download True.csv and Fake.csv, place them here:
mkdir -p data/raw/isot
# (Manually copy True.csv and Fake.csv to data/raw/isot/)

# ── 2. Prepare data ───────────────────────────────────────────────────────
python scripts/prepare_data.py
# Creates: data/processed/{combined_dataset,train,validation,test}.csv

# ── 3. Train models ───────────────────────────────────────────────────────
python scripts/train_models.py
# Creates: models/{tfidf,lr,nb,svm,ensemble,fake_news_model}.pkl
# Creates: models/model_metrics.json

# ── 4. Run the application ────────────────────────────────────────────────
uvicorn api.main:app --reload --port 8000
# Open http://localhost:8000

# ── 5. Run tests ──────────────────────────────────────────────────────────
pytest tests/ -v

# ── 6. Build Docker image ─────────────────────────────────────────────────
docker build -t verinews:latest .
docker run -p 8000:8000 --env-file .env verinews:latest

# ── 7. Deploy to AWS (see steps above) ───────────────────────────────────
# Step 2 (ECR push) → Step 3 (S3 models) → Step 6 (ECS) → Step 7 (ALB)
```
