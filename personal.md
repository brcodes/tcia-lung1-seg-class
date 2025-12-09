⚖️ Balance of Focus

- Technical ML: 40% (segmentation + classification)
- Compliance: 30% (HIPAA, FDA, ICD/CPT)
- Workflow Integration: 30% (Epic/FHIR simulation, Azure deployment)

📅 Week-by-Week Roadmap

**Week 1: Compliance & Data Foundations**

- **Compliance Study**
    - Learn HIPAA basics: PHI definition, minimum necessary principle, secure storage
    - Review FDA's AI/ML in Software as a Medical Device (SaMD) guidance
    - Begin ICD-10 and CPT coding study (focus on oncology codes)
    - Explore HL7 and FHIR standards (HAPI FHIR sandbox)

- **Data Setup**
    - Register and download de-identified datasets from TCIA
    - Document dataset provenance and compliance notes in README

**Week 2: Preprocessing & Segmentation Prep**

- **DICOM Handling**
    - Use pydicom + SimpleITK to load TCIA scans
    - Normalize intensities, resample voxel spacing, convert to NIfTI

- **Segmentation Pipeline**
    - Set up MONAI environment (PyTorch + MONAI)
    - Build U-Net architecture for tumor segmentation

- **Compliance**
    - Write "Data Handling SOP" (PHI treatment procedures)

**Week 3: Segmentation Training & Evaluation**

- Train U-Net on TCIA tumor segmentation task
- Evaluate with Dice coefficient and IoU
- Document clinical relevance: tumor volume measurement, treatment planning
- Compliance: note segmentation output storage in EHR without identifiers

**Week 4: Classification & Epic Integration**

- **Classification**
    - Train CNN classifier on segmented ROIs for pathology status
    - Evaluate with ROC-AUC, sensitivity, specificity

- **Epic Workflow Simulation**
    - Map outputs to FHIR resources: Observation (classification result), ImagingStudy (segmentation metadata)
    - Use HAPI FHIR test server to simulate Epic integration

- **Compliance**: Ensure outputs are anonymized and labeled "Research use only"

**Week 5: Deployment & Cloud Integration**

- **Azure ML Studio**
    - Deploy trained model as REST API
    - Document encryption, access control, audit logs

- **Compliance**: Cloud deployment must meet HIPAA security standards

**Week 6: Portfolio & Showcase**

- **GitHub Repo**: Preprocessing scripts, segmentation model, classification model, Azure deployment notebook, FHIR integration demo
- **Documentation**: Compliance notes (HIPAA, FDA, ICD-10/CPT), workflow simulation screenshots
- **Final Deliverable**: End-to-end project (TCIA → segmentation → classification → FHIR/Epic → Azure deployment)





📊 Results & Clinical Impact
Segmentation (nnU‑Net)

    Performance: Achieved strong Dice and IoU scores on NSCLC‑Radiomics tumor masks.

    Clinical Impact: Accurate segmentation improves tumor boundary detection, supporting radiologists in treatment planning and reducing variability in manual contouring.

Classification (ResNet vs Swin Transformer)

    Performance:

        ResNet baseline delivered solid accuracy and AUC.

        Swin Transformer fine‑tuning improved sensitivity and specificity, particularly for malignant nodule detection.

    Clinical Impact: Higher sensitivity reduces false negatives in lung cancer screening, while improved specificity minimizes unnecessary biopsies and interventions.

Pipeline Integration

    Workflow: Raw CT → nnU‑Net segmentation → classification (ResNet vs Swin Transformer) → Epic Sandbox/FHIR integration.

    Clinical Impact: Demonstrates how automated imaging analysis can flow directly into oncology workflows, enabling faster, reproducible decision support for thoracic cancer care.

###
HYBRID


⚡ Strategic Hybrid Workflow

This project was engineered to balance cost efficiency, compliance, and reproducibility by distributing tasks across three environments:

    Local CPU (short runs, <1 hr)

        Preprocessing (DICOM → NIfTI conversion, normalization, metadata checks)

        Unit tests for dataloaders, augmentations, and config parsing

        Quick inference dry runs with lightweight models

    Free GPU Services (Google Colab, Kaggle, SageMaker Studio Lab, Paperspace free)

        Small‑scale nnU‑Net dry runs (2D configs, few epochs)

        Transformer fine‑tuning on reduced datasets

        Visualization notebooks (ROC curves, confusion matrices, qualitative overlays)

    Azure ML Studio (production runs)

        Full nnU‑Net 3D training (multi‑fold CV, full dataset)

        Robust ResNet vs. Swin Transformer fine‑tuning and ablations

        Containerized inference endpoints, model registry, and Epic Sandbox workflow integration

🔧 Environment Portability

All environments share a single config system (base.yaml + overlays for cpu.yaml, free_gpu.yaml, azure.yaml). This ensures:

    Reproducibility: identical seeds, configs, and provenance logs across environments.

    Cost control: debug and visualization offloaded to free tiers, full training reserved for Azure.

    Compliance: only Azure ML Studio is used for PHI‑sensitive workflows and deployment.

📊 Impact: This hybrid prototyping strategy reduced Azure spend to the $500 floor while maintaining clinical‑grade rigor and recruiter‑ready reproducibility.