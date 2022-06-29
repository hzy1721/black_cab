## 安装步骤

```shell
conda create -n black_cab python=3.8 -y
conda activate black_cab
pip install -r docs/requirements.txt
```

根据官方指引安装 PyTorch：[https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

```shell
pip install -U openmim
mim install mmcv-full
pip install mmdet
```
