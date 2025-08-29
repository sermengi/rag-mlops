# rag-mlops

Production-grade Retrieval-Augmented Generation (RAG) service with observability, automated evaluation, and CI/CD for real-world deployment.

## Overview

This project is designed to showcase how to build a **full-stack RAG system** and productionize it using modern **MLOps practices**.  
The goal is to cover the full lifecycle of a machine learning application, from ingestion and retrieval to monitoring, evaluation, and deployment.

Key features planned for this project:
- **RAG API Service**: FastAPI endpoint for chat with LLMs (using vLLM/TGI).
- **Vector Database**: Document ingestion, embedding, and retrieval with Qdrant.
- **Observability**: OpenTelemetry + Prometheus + Grafana for metrics and monitoring.
- **Evaluation Loop**: Ragas for nightly benchmarks, tracked in MLflow.
- **Kubernetes Deployment**: Containerization, Helm charts, and autoscaling.
- **CI/CD**: GitHub Actions for automated testing, evaluation, and deployment.

## Project Structure
```text
.
├── LICENSE
├── README.md
├── configs        # Configuration files (YAML, JSON, etc.)
├── infra          # Infrastructure as code
│   ├── helm       # Helm charts for Kubernetes deployment
│   └── k8s        # Raw Kubernetes manifests
├── notebooks      # Jupyter notebooks for experimentation and prototyping
├── services       # Core services for the RAG pipeline
│   ├── api        # FastAPI service
│   │   ├── __init__.py
│   │   └── app
│   │       └── __init__.py
│   ├── model
│   │   ├── __init__.py
│   │   └── src
│   │       └── __init__.py
│   └── retriever
│       ├── __init__.py
│       └── src
│           └── __init__.py
└── tests
```

## Tech Stack

- **Backend**: FastAPI, vLLM / TGI  
- **Vector Store**: Qdrant  
- **Evaluation**: Ragas, MLflow  
- **Monitoring**: OpenTelemetry, Prometheus, Grafana  
- **Deployment**: Docker, Kubernetes, Helm, GitHub Actions  

## Roadmap

- [ ] Phase 0: Minimal RAG service (local)  
- [ ] Phase 1: Observability & SLOs  
- [ ] Phase 2: Quality evaluation loop  
- [ ] Phase 3: Kubernetes & Helm deployment  
- [ ] Phase 4: CI/CD with GitHub Actions  

## Getting Started

Clone the repository:
```bash
git clone https://github.com/sermengi/rag-mlops.git
cd rag-mlops