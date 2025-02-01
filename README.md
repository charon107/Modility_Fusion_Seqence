# Project Overview

This project, based on the **MOSI/MOSEI** database, aims to explore the impact of different modality fusion sequences (text, acoustic, and visual) on the performance of various large language models and downstream tasks.

## Key Features:

- **Audio Features**: 
  - Extracted using **OpenSmile**.
  - A custom **Transformer Encoder** is employed as the **Acoustic modality encoder** for representation learning.

- **Visual Features**:
  - Extracted using **OpenFace 2.0**.
  - A custom **Video Transformer Encoder** is used as the **Visual modality encoder** for representation learning.

- **Text Features**: 
  - **BERT** is used as the **Text modality encoder** for feature learning.

- **Modality Fusion**:
  - **Cross-attention** is utilized for modality fusion.
  - Different fusion sequences (e.g., **T-A-V**, **A-V-T**, etc.) are experimented with.

- **Final Stage**:
  - The fused features are fed into various pre-trained **large language models**(e.g., **Roberta**, **GPT 3**, etc.).
  - The goal is to investigate whether there is an optimal modality fusion sequence, and to examine whether this sequence is universally applicable across all large pre-trained language models.

![Model Architecture](images/Model Architecture.png)
## Objective:

- Investigate the impact of different modality fusion sequences on model performance.
- Evaluate the generalization of the fusion sequence across various large pre-trained language models.
