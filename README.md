# README
The official implementation of **Low-dose CT image super-resolution with noise suppression based on prior degradation estimator and self-guidance mechanism**.

This repository is modified from [BasicSR](https://github.com/XPixelGroup/BasicSR). Thanks for the open source code of [BasicSR](https://github.com/XPixelGroup/BasicSR).
## Installation
```bash
conda create -n new_env python=3.9.7 -y
conda activate new_env
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
pip install -e .
```
More details could be found in [the installation ducoment of BasicSR](https://github.com/XPixelGroup/BasicSR/blob/master/docs/INSTALL.md).
## Data preparation
You should modify the path in configuration files in "opations/train/\*.yml" or "opations/test/\*.yml".
## Training
### Stage 1: train the PDE
```python
python basicsr/train.py -opt options/train/train_pde.yml
```
### Stage 2: train the self-guidance SR network
```python
python basicsr/train.py -opt options/train/train_self_guidance_sr_x2_aapm.yml
```
or
```python
python basicsr/train.py -opt options/train/train_self_guidance_sr_x4_aapm.yml
```
## Testing
```python
python basicsr/train.py -opt options/test/test_self_guidance_sr_x2_aapm.yml
```
or
```python
python basicsr/train.py -opt options/test/test_self_guidance_sr_x4_aapm.yml
```
