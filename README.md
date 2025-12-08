# tcia-lung1-seg-class

## 📌 Project Overview

End‑to‑end healthcare ML imaging pipeline using TCIA NSCLC‑Radiomics lung cancer CT data.

### Workflow

- **Segmentation** → nnU‑Net trained from scratch
- **Classification** → Transformer fine‑tuned with pretrained weights
- **Integration** → Epic Sandbox APIs (FHIR) for workflow alignment
- **Deployment** → Azure ML Studio with compliance‑ready environment

## 🧩 Key Features

- 🔬 **Medical Imaging**: DICOM preprocessing, resampling, normalization
- 🧠 **Deep Learning**: nnU‑Net segmentation + transformer classification
- 📊 **Evaluation**: ROC, confusion matrices, reproducible metrics
- ☁️ **Cloud Ready**: Azure ML Studio deployment artifacts
- 🏥 **Workflow Integration**: Epic Sandbox API adapters (FHIR standard)
- 🔒 **Compliance**: HIPAA/FDA notes, PHI handling policy documented

## 📂 Repository Structure

```txt
tcia-lung1-seg-class/
├── src/               # Core Python modules
├── configs/           # YAML configs for reproducibility
├── notebooks/         # Exploration & visualization
├── tests/             # Unit/integration tests
├── docs/              # Compliance + workflow documentation
├── scripts/           # CLI entry points
├── cloud/             # Azure ML + Epic Sandbox integration
├── environment.yml    # Conda environment (heavy ML stack)
├── requirements.txt   # Pip extras (dev tools, FHIR client)
└── README.md          # Project overview
```

## ⚙️ Environment Setup

Local + cloud environments are unified for reproducibility.

### Create environment

```bash
conda env create -f environment.yml
conda activate tcia-lung1-seg-class
```

### Verify installation

```bash
python -c "import torch, monai, nibabel, simpleitk; print('Environment OK')"
```

For details, see `docs/env_setup.md`.

## 🚀 Usage

### Preprocessing

```bash
python scripts/run_preprocessing.py --config configs/preprocessing.yaml
```

### Segmentation

```bash
python scripts/train_segmentation.py --config configs/nnunet_train.yaml
```

### Classification

```bash
python scripts/train_classification.py --config configs/transformer_class.yaml
```

### Deployment

```bash
python scripts/deploy_cloud.py --config configs/deploy.yaml
```

## 🔒 Compliance

- **No PHI**: Only TCIA NSCLC‑Radiomics (public dataset) used
- **Audit Trail**: Configs + logs maintained under version control
- **Secrets**: Epic Sandbox + Azure credentials stored in `.env` files
- **Documentation**: See `docs/compliance.md`

## 📊 Results

- Segmentation Dice scores (nnU‑Net)
- Classification ROC curves (transformer)
- Cloud deployment screenshots (Azure ML Studio)

## 🤝 Contributions

Pull requests welcome. Please review compliance guidelines before submitting.

## 📜 License

Explicit license included for medical/clinical reproducibility.
