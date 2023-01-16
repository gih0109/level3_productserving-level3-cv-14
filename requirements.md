정상적인 프로젝트 진행을 위해, 각자 개발환경을 통일합니다.

환경 관리는 conda를 활용합니다. (aistages 서버 기본 conda version: 4.12.0)
1. conda 환경 생성
```bash
conda create -n final python=3.8.15
```
2. conda 환경 활성화
```bash
conda activate final
```
3. conda update 및 버전 확인
```bash
conda update -n base conda
conda --version # 22.11.1
```
4. pytorch, cudatoolkit 등 설치
```bash
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 -c pytorch
```
4-1. torch 정상 설치 확인
```python
import torch
torch.cuda.is_available() # True로 나와야함
```
5. mmcv 설치
```bash
pip install -U openmim # 0.3.4
mim install mmcv-full # 1.7.1
```
5-1. 아래와 같은 에러 발생시 
```
ImportError: cannot import name 'Config' from 'mmcv' (unknown location)
ModuleNotFoundError: No module named 'mmcv'
```
```bash
mim install mmcv # 1.7.1
conda deactivate
conda activate final
```
6. mmcls 설치
```bash
pip install mmcls
```
7. wandb 설치
```bash
conda install -c conda-forge wandb
```
8. mmdetection 설치
```bash
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e .
```
9. mmdeploy 설치

9-1. ONNX (CPU)
```bash
git clone https://github.com/open-mmlab/mmdeploy.git
wget https://github.com/open-mmlab/mmdeploy/releases/download/v0.12.0/mmdeploy-0.12.0-linux-x86_64-onnxruntime1.8.1.tar.gz
tar -zxvf mmdeploy-0.12.0-linux-x86_64-onnxruntime1.8.1.tar.gz
cd mmdeploy-0.12.0-linux-x86_64-onnxruntime1.8.1
pip install dist/mmdeploy-0.12.0-py3-none-linux_x86_64.whl
pip install sdk/python/mmdeploy_python-0.12.0-cp38-none-linux_x86_64.whl
cd ..
pip install onnxruntime==1.8.1
wget https://github.com/microsoft/onnxruntime/releases/download/v1.8.1/onnxruntime-linux-x64-1.8.1.tgz
export ONNXRUNTIME_DIR=$(pwd)/onnxruntime-linux-x64-1.8.1
export LD_LIBRARY_PATH=$ONNXRUNTIME_DIR/lib:$LD_LIBRARY_PATH
```

9-2. TensorRT -> aistages server에서는 불가능합니다.
```bash
wget https://github.com/open-mmlab/mmdeploy/releases/download/v0.12.0/mmdeploy-0.12.0-linux-x86_64-cuda11.1-tensorrt8.2.3.0.tar.gz
tar -zxvf mmdeploy-0.12.0-linux-x86_64-cuda11.1-tensorrt8.2.3.0.tar.gz
cd mmdeploy-0.12.0-linux-x86_64-cuda11.1-tensorrt8.2.3.0
pip install dist/mmdeploy-0.12.0-py3-none-linux_x86_64.whl
pip install sdk/python/mmdeploy_python-0.12.0-cp38-none-linux_x86_64.whl
cd ..
# Nvidia 에서 tensorrt 8.2.3.0 다운로드 받기 (https://developer.nvidia.com/nvidia-tensorrt-8x-download)
tar -zxvf TensorRT-8.2.3.0.Linux.x86_64-gnu.cuda-11.4.cudnn8.2.tar.gz
pip install TensorRT-8.2.3.0/python/tensorrt-8.2.3.0-cp38-none-linux_x86_64.whl
pip install pycuda
export TENSORRT_DIR=$(pwd)/TensorRT-8.2.3.0
export LD_LIBRARY_PATH=${TENSORRT_DIR}/lib:$LD_LIBRARY_PATH
# !!! Download cuDNN 8.2.1 CUDA 11.x tar package from NVIDIA, and extract it to the current directory
export CUDNN_DIR=$(pwd)/cuda
export LD_LIBRARY_PATH=$CUDNN_DIR/lib64:$LD_LIBRARY_PATH
```

10. dvc
```bash
pip install 'dvc[all]'==2.8.1
```

11. opencv
```bash
conda install opencv
```