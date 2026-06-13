# A Comparative Security Evaluation of Side-Channel Countermeasures for Resource-Constrained IoT Devices

**Author:** Xu Xu (徐旭), Sichuan University
**Type:** Empirical Study · Springer LNCS
**Status:** Under Review

## Overview

This paper presents a systematic, reproducible evaluation of deep learning-based side-channel analysis (DL-SCA) countermeasures for IoT embedded devices. We compare three architectures (MLP, CNN, ResNet) across three escalating software protection levels—boolean masking, masking + 50-cycle desynchronization, and masking + 100-cycle desynchronization—under identical experimental conditions (27 runs total), using exclusively public ASCAD datasets.

### Key Findings

- Boolean masking alone is bypassed in 70–150 traces by DL-based SCA
- 50-cycle random desynchronization provides a 70× security gain at near-zero cost
- ResNet requires 650× more training time than MLP yet achieves zero successful attacks
- A leakage plateau phenomenon is observed where additional traces degrade attack performance

## Repository Structure

```
├── paper.tex           # LaTeX source (Springer LNCS)
├── references.bib      # BibTeX references
├── llncs.cls           # Springer LNCS document class
├── splncs04.bst        # BibTeX style
├── figures/
│   ├── ge_comparison.png
│   ├── security_gain_waterfall.png
│   └── ttr_heatmap.png
├── src/
│   ├── config.py       # Experiment configuration
│   ├── models.py       # MLP, CNN, ResNet architectures
│   ├── train.py        # Training and evaluation loop
│   ├── run_experiments.py
│   ├── visualize.py
│   ├── download_data.py
│   └── requirements.txt
└── data-info/
    └── README.md       # Dataset download instructions
```

## Reproducibility

All experiments are based on publicly available ASCAD datasets. Complete hyperparameters, random seeds, and model architectures are documented in the paper and source code.

## Build

```bash
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```

## License

This work is for academic purposes. All rights reserved.
