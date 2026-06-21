# A Comparative Security Evaluation of Side-Channel Countermeasures for Resource-Constrained IoT Devices

**Author:** Xu Xu (徐旭), Sichuan University  
**Type:** Empirical Study · Springer LNCS  
**Venue:** The 5th International Conference on Internet of Things, Communication and Intelligent Technology (IoTCIT 2026)  
**Status:** Accepted — to be published by Springer, indexed by EI Compendex

## Overview

This paper presents a systematic, reproducible evaluation of deep learning-based side-channel analysis (DL-SCA) countermeasures for IoT embedded devices. We compare three architectures (MLP, CNN, ResNet) across three escalating software protection levels under identical experimental conditions (27 runs total), using exclusively public ASCAD datasets.

### Key Findings

- Boolean masking alone is bypassed in 70–150 traces by DL-based SCA
- 50-cycle random desynchronization provides a 70× security gain at near-zero cost
- ResNet requires 650× more training time than MLP yet achieves zero successful attacks
- A leakage plateau phenomenon is observed where additional traces degrade attack performance

## Repository Structure

```
├── figures/
│   ├── ge_comparison.png
│   ├── security_gain_waterfall.png
│   ├── leakage_plateau.png
│   └── ttr_heatmap.png
├── src/
│   ├── config.py              # Experiment configuration
│   ├── models.py              # MLP, CNN, ResNet architectures
│   ├── train.py               # Training and evaluation loop
│   ├── run_experiments.py     # Experiment automation
│   ├── visualize.py           # Figure generation
│   ├── download_data.py       # ASCAD dataset downloader
│   └── requirements.txt       # Python dependencies
└── data-info/
    └── README.md              # Dataset download instructions
```

## Reproducibility

All experiments use publicly available ASCAD datasets. Complete hyperparameters, random seeds (42, 142, 242), and model architectures are documented in the source code and the published paper. Install dependencies with `pip install -r src/requirements.txt`, download ASCAD data with `python src/download_data.py`, and reproduce all experiments with `python src/run_experiments.py`.

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{xu2026comparative,
  title={A Comparative Security Evaluation of Side-Channel Countermeasures 
         for Resource-Constrained IoT Devices},
  author={Xu, Xu},
  booktitle={Proc. 5th Int. Conf. on Internet of Things, Communication 
             and Intelligent Technology (IoTCIT 2026)},
  year={2026},
  publisher={Springer}
}
```

## License

This work is for academic purposes. Code and data are provided for reproducibility. All rights reserved.
