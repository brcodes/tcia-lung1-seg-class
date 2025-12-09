# Environment Setup Guide

This document explains how to set up and reproduce the environment for the tcia-lung1-seg-class project. The environment supports local development (preprocessing, testing, exploration) and cloud deployment (Azure ML Studio, Epic Sandbox integration).

## Prerequisites

- Miniconda (recommended) or Anaconda
- Git for version control
- Azure ML Studio account (for cloud deployment)
- Epic Sandbox access (for FHIR API integration)

## Local Development Setup

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/tcia-lung1-seg-class.git
cd tcia-lung1-seg-class
```

### 2. Create the Conda environment

On macOS:
conda env create -f environment-macos.yml
conda activate tcia-lung1-seg-class-cpu

On Azure/Linux GPU:
conda env create -f environment-azure.yml
conda activate tcia-lung1-seg-class-gpu

remove if partial/stopped solve
```bash
conda env remove -n tcia-lung1-seg-class
```

Check dep presence
```bash
conda activate tcia-lung1-seg-class
conda list | grep torch
# Should see: libtorch, pytorch, torchvision
```

Check GPU support in GPU environment
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```



### 3. Activate the environment

```bash
conda activate tcia-lung1-seg-class
```

### 4. Verify installation

```bash
python -c "import torch, monai, nibabel, simpleitk; print('Environment OK')"
```

## Cloud Deployment (Azure ML Studio)

Azure ML Studio supports Conda environments directly:

1. Upload `environment.yml` to your Azure ML workspace
2. Reference it in your training/deployment job:

```yaml
environment:
    conda_file: environment.yml
    docker:
        base_image: mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu20.04
```

3. Ensure no PHI is included in environment variables or logs. Use `.env` files with `python-dotenv` for safe secret management.

## Environment Files

- **environment.yml** → Heavy ML stack (PyTorch, MONAI, CUDA, ITK, nibabel, scikit-image)
- **requirements.txt** → Lightweight extras (dev tools, FHIR client, transformers)

This hybrid approach ensures reproducibility while keeping cloud deployments lean.

## Compliance Notes

- **No PHI**: Only use TCIA NSCLC-Radiomics (public dataset)
- **Audit trail**: Keep `environment.yml` under version control
- **Secrets**: Store API keys (Epic Sandbox, Azure) in `.env` files, never in code
- **Reproducibility**: Always run experiments with a fixed environment hash

## Quick Commands

Update environment after changes:
```bash
conda env update -f environment.yml --prune
```

Export exact environment for audit:
```bash
conda env export > env-lock.yml
```