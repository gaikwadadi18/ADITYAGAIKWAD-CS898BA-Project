# Skin Lesion Classification: A Segmentation-Assisted Deep Learning Pipeline

**Course:** CS 898BA | **Status:** Pitch Stage

## Problem Statement
Early detection of melanoma significantly improves patient survival, but dermoscopic image interpretation is subjective and requires specialist expertise. This project builds an automated pipeline that classifies dermoscopic skin lesion images into diagnostic categories (e.g., melanoma, nevus, basal cell carcinoma) using a combination of classical image processing and deep learning.

## Dataset
[ISIC Archive](https://www.isic-archive.com/) / HAM10000 — a public dermoscopic image dataset (~10,000+ labeled lesion images across multiple diagnostic classes). Free to download for research/educational use.

## Approach
A **two-stage pipeline**, chosen over a naive end-to-end CNN because it satisfies the project's image-processing requirement and is well-supported in prior literature:

1. **Preprocessing** (beyond augmentation):
   - Hair artifact removal (morphological black-hat filtering, DullRazor-style)
   - Color constancy / illumination normalization (shades-of-gray)
   - Contrast enhancement (CLAHE)
2. **Segmentation**: isolate the lesion region from surrounding skin (thresholding baseline → U-Net if time allows)
3. **Classification**: CNN (transfer learning, e.g., EfficientNet) trained on the segmented, normalized lesion crops

## Alternative Approach Considered
End-to-end transfer learning directly on lightly-augmented raw images — simpler and faster to implement, but does not isolate the lesion from background artifacts (hair, rulers, skin tone variation), which the literature shows hurts both accuracy and interpretability.

## Repository Structure
```
src/            - source code (preprocessing, segmentation, model, training scripts)
data/           - dataset (not committed; see data/README for download instructions)
notebooks/      - exploratory analysis and experiment notebooks
AI_Log.md       - required AI accountability log
```

## Setup
```bash
pip install -r requirements.txt
python src/hello_world.py
```

## Status / Next Steps
- [x] Project pitch and dataset selection
- [ ] Acquire and inspect dataset
- [ ] Implement preprocessing pipeline
- [ ] Baseline classifier (midterm milestone)
- [ ] Hyperparameter optimization and final evaluation (final milestone)

## References
- Codella et al., ISIC Challenge datasets
- Tschandl et al. (2018), HAM10000 dataset
- See pitch presentation slides for full literature review
