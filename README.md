# GPU stress
Stress test all your GPUs

## Installation
```bash
python3 -m venv venv
```
```bash
source venv/bin/activate
```
Nvidia
```bash
pip install torch
```
AMD
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
```

## Usage
```bash
python stress.py
```

## Options
```
-r or --runtime - Duration of the test in seconds
-d or --delay - Delay before starting the test in seconds
-g or --gpus - Specify GPU ids to use (e.g., "0,1,3") or "all" for all GPUs. Default is "all".
```

## Support
- Nvidia GPUs
- AMD GPUs