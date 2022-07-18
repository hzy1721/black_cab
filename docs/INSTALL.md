## 安装步骤

```shell
conda create -n black_cab python=3.8 -y
conda activate black_cab
pip install -r docs/requirements.txt
```

根据官方指引安装 PyTorch 1.11.0：[https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

```shell
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu115/torch1.11.0/index.html
pip install mmdet
```
