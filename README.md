# FUSED-Net
## Installation
```bash
git clone https://github.com/180041123-Atiq/FUSED-Net.git
cd FUSED-Net
conda env create -f environment.yml
conda activate FUSED-Net
```
## Data Processing
```bash
mkdir datasets
cd datasets
wget --no-check-certificate "https://drive.google.com/uc?export=download&id=1FRDs9V8SXFZRhyqUmMZpTiSTyLXBiCw0" -O BDTSD.zip
unzip BDTSD.zip
cd ..
```
## MTSDD Weight Initialization
```bash
mkdir output
cd output
wget --no-check-certificate "https://drive.google.com/uc?export=download&id=1ogF_v2QdDLPgnsCimXNpQUrv-hXHsl4m" -O model_final.pth
cd ..
```
## Training and Evaluation
```bash
mkdir log
bash run.sh
```
