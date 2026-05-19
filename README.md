
# Landscape-Aware Benchmarks for Multi-Modal Multi-Objective Optimization

---

## 1. Prerequisites

- **Operating System:** Windows 10/11 64-bit
- **Python:** 3.11.9
- **GPU:** NVIDIA GPU with CUDA 12.1+ 
- **VS Code**
> All commands below assume execution in the **PowerShell terminal**.

---

## 2. Install VS Code

Download and install from: https://code.visualstudio.com/download  

---

## 3. Install Python 3.11.9

Download installer: https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe

Verify installation:

```powershell
python --version
# Expected: Python 3.11.9
```

---

## 4. Install NVIDIA Driver (if using GPU)

Download: https://www.nvidia.com/en-us/drivers/  

- Choose **Game Studio Driver**
- Verify installation:

```powershell
nvidia-smi
```

---

## 5. Install CUDA Toolkit 12.1.1

Download: https://developer.nvidia.com/cuda-12-1-1-download-archive  

Verify installation:

```powershell
nvcc --version
```

---

## 6. Clone the Repository

```powershell
git clone https://github.com/shuheitnk/mmmo-algorithm-selection.git
cd mmmo-algorithm-selection
```

---

## 7. Create and Activate a Python Virtual Environment

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\activate
python --version  # should show 3.11.9
```

If you encounter a **PSSecurityException (Execution Policy Restriction)**, enable script execution for your user only:
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

Upgrade pip:

```powershell
python -m pip install --upgrade pip
```

---

## 8. Install Dependencies

Install PyTorch with CUDA support:

```powershell
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verify:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Install the remaining requirements:

```powershell
pip install -r requirements.txt
```

## 9. External Dependencies

Clone the required repositories:

```bash
git clone https://github.com/BIMK/PlatEMO.git
git clone https://github.com/Wenhua-Li/ComparativeStudyofMMOP.git
```

This project depends on the following MATLAB toolbox for using PlatEMO:

- Statistics and Machine Learning Toolbox
