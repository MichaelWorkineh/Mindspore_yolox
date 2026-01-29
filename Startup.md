# 0. Install miniconda
```powershell
# Option 1: Using Winget (Recommended)
winget install Anaconda.Miniconda3

# Option 2: Manual Download & Install
Invoke-WebRequest -Uri "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe" -OutFile "miniconda_installer.exe"
Start-Process -FilePath ".\miniconda_installer.exe" -ArgumentList "/S", "/AddToPath=1", "/RegisterPython=0" -Wait
Remove-Item "miniconda_installer.exe"
```

# 1. Environment Setup
```bash
conda create -n mindyolo python=3.8 -y
conda activate mindyolo

pip install mindspore
pip install pycocotools
pip install "Pillow<10.0.0"
pip install opencv-python
pip install PyYAML
```

# 2. Model Training
```bash
python yolox/train.py --data_dir ./datasets/vehicle_detection --device_target CPU --name yolox-tiny --per_batch_size 1 --max_epoch 50
```

# 3. Model Inference
```bash
python RealtimeCam.py
```
