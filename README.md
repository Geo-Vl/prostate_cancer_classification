# Histological Image Analysis for Prostate Cancer Diagnosis using Patch-Based Deep Learning

## Project Overview
- This repository contains the full pipeline for analyzing large-scale histological images, specifically focusing on the **Prostate Cancer Grade Assessment (PANDA)** dataset (https://www.kaggle.com/competitions/prostate-cancer-grade-assessment).
- The project revolves around extracting meaningful patches from high-resolution whole-slide images (WSIs) and building a **patch-based deep learning model** that classifies the images (according to the presence of cancer) using the extracted patches.
- This approach is based on the "Concat tile pooling" (https://www.kaggle.com/code/iafoss/panda-concat-tile-pooling-starter-0-79-lb) (https://github.com/iafoss/PANDA).
- The novel aspect of this approach lies in the **patch handling through the model architecture**, where the entire image is represented as a **weighted sum of patches**. The weights are assigned dynamically (learnable parameters) to the most informative patches. This enables efficient computation and improved classification performance, especially in the context of histological images, where only certain regions of an image might be of diagnostic importance.

<div align="center">
  <img src="https://github.com/user-attachments/assets/8dda8cf0-c53f-46b6-9c8d-4e6a8629a742" alt="Schema" />
</div>
