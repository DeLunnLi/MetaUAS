# MetaUAS

Unofficial PyTorch implementation of [MetaUAS: Universal Anomaly Segmentation with One-Prompt Meta-Learning](https://arxiv.org/abs/2505.09265).

## 🖼️ Framework

![MetaUAS Framework](https://raw.githubusercontent.com/gaobb/MetaUAS/main/images/MetaUAS_Framework.jpg)

## 🚀 Usage

### 1. 🔧 COCO Data Preprocessing

Install [LaMa](https://github.com/advimman/lama) and download the pretrained model:

```bash
git clone https://github.com/advimman/lama.git
pip install -r lama/requirements.txt

curl -LJO https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip
unzip big-lama.zip -d lama/big-lama/
```

Then prepare COCO train2017 images and annotations:

```bash
# Expected structure:
# /path/to/datasets/images/train2017/   # COCO train images
# /path/to/datasets/images/annotations/ # instances_train2017.json

# Generate CYWS coco-inpainted dataset
bash scripts/run_generate_cyws_dataset.sh
```

> Adjust paths in `scripts/run_generate_cyws_dataset.sh` before running.
> 
> Alternatively, you can directly download the pre-generated COCO inpainted dataset from [CYWS](https://thor.robots.ox.ac.uk/~vgg/data/cyws/coco-inpainted.tar).

### 2. 🔥 Training

```bash
pip install -r requirements.txt
bash scripts/train_on_4gpu.sh
```

## 📊 Performance & Checkpoints

> Experiments run on 4× NVIDIA RTX 3090, batch size 96/GPU, lr=1e-4, weight decay=0.005.

### Results

| Dataset | Method | I-AUROC | P-AUROC | P-PRO |
| :--- | :--- | :---: | :---: | :---: |
| MVTec AD | MetaUAS | -- | -- | -- |
| VisA | MetaUAS | -- | -- | -- |

### Pretrained Models

| Model | Link |
| :--- | :--- |
| MetaUAS | *TODO* |

## 🙏 Acknowledgements

We reference code from [MetaUAS](https://github.com/gaobb/MetaUAS), [LaMa](https://github.com/advimman/lama), and [CYWS](https://github.com/ragavsachdeva/The-Change-You-Want-to-See).

## 🤝 Contributing

- ⭐ If you find this project useful, a star would be greatly appreciated
- 🐛 Report bugs or ask questions via [Issues](https://github.com/DeLunnLi/MetaUAS/issues)
- 🔀 Fork and submit a PR with your improvements — we'll review and add you to the contributors
