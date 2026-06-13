# Dataset Information

## ASCADv1 (ATmega8515)
- URL: https://www.data.gouv.fr/api/1/datasets/r/e7ab6f9e-79bf-431f-a5ed-faf0ebe9b08e
- Format: ZIP (~4.4 GB), contains 3 HDF5 files
- Files: ASCAD.h5, ASCAD_desync50.h5, ASCAD_desync100.h5
- Structure: 50,000 profiling + 10,000 attack traces, 700 features each

## ASCADv2 (STM32F303)
- URL: https://object.files.data.gouv.fr/anssi/ascadv2/ascadv2-extracted.h5
- Format: HDF5 (~7.2 GB)
- Structure: 500,000 profiling + 10,000 attack traces, 15,000 features each

## Setup
pip install -r src/requirements.txt
python -m src.download_data
