# Lunar Pareidolia Detection & Classification 

A robust, high-performance machine learning pipeline designed to detect and classify lunar pareidolia. This approach uniquely combines **physics-informed Shape-from-Shading (SfS)** priors, advanced hand-crafted topographical descriptors, **self-supervised Vision Transformer embeddings (DINOv2)**, and an optimized **Stacking Ensemble**.

---

## Key Features

* **Azimuth-Aware Preprocessing:** Applies high-quality bicubic rotation to input images based on recorded sun azimuth angles to ensure rotational invariance.
* **Physics-Informed Topography:** 
  * **Frankot-Chellappa Integration:** Reconstructs dense 2.5D height maps from shading cues via Fourier domain processing.
  * **Hessian Curvature Analysis:** Computes eigenvalues ($\lambda_1, \lambda_2$), trace, and determinant to classify local surface geometries into bowls, domes, and saddles.
* **Hybrid Feature Extraction:** Combines Histogram of Oriented Gradients (HOG), Local Binary Patterns (LBP), global statistical moments, and directional gradients.
* **Deep Semantic Embeddings:** Leverages Meta's pre-trained **DINOv2-S/14** backbone, compressed efficiently via Principal Component Analysis (PCA).
* **Ensemble Stacking Architecture:** Blends XGBoost, Histogram GradientBoosting, and Random Forest using a Logistic Regression meta-classifier with automated class-weight balancing and decision threshold tuning.

---

## Project Structure

```text
Pareidolia 2.0/
├── cache_dinov2/                  # Cached features and PCA projection components
├── train/                         # Training images directory
├── val/                           # Validation images directory
├── eval_images/                   # Test evaluation images directory
├── train_metadata_split.csv       # Training split metadata
├── val_metadata_split.csv         # Validation split metadata
├── test_metadata.csv              # Test set metadata
│
├── final_model_weights.pkl        # Serialized stacking ensemble pipeline
├── optimal_threshold.pkl          # Tuned classification decision threshold
├── dino.py                        # Main script for model training and weight saving
└── sub.py                         # Script for generating test submissions
