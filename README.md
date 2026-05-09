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

### 3. 📈 Evaluation

Two evaluation modes are supported:
- **Oneprompt**: one random normal sample per class as prompt, multi-seed averaging
- **TopK**: per-image top-k most similar normal samples as prompts

```bash
# Download pretrained checkpoint
wget https://huggingface.co/ldl010302/MetaUAS/resolve/main/metauas-256.pth

# Run both modes on MVTec AD and VisA
bash scripts/run_eval_mvtec.sh
```

> Adjust `CKPT`, `MVTEC_ROOT`, `VISA_ROOT` in `scripts/run_eval_mvtec.sh` before running.

## 📊 Performance & Checkpoints

> Experiments run on 4× NVIDIA RTX 3090, batch size 96/GPU, lr=1e-4, weight decay=0.005.

### Results

<div align="center">

<table style="border-collapse: collapse; border-top: 1px solid #fff; border-bottom: 1px solid #fff;">
<thead>
<tr>
<th rowspan="2">Methods</th>
<th rowspan="2">Categories</th>
<th colspan="3" style="text-align: center;">Anomaly Classification</th>
<th colspan="4" style="text-align: center;">Anomaly Segmentation</th>
</tr>
<tr>
<th>I-ROC</th>
<th>I-PR</th>
<th>I-F1<sub>max</sub></th>
<th>P-ROC</th>
<th>P-PR</th>
<th>P-F1<sub>max</sub></th>
<th>P-PRO</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="15">MetaUAS</td>
<td>bottle</td>
<td>98.9±0.9</td><td>99.7±0.2</td><td>98.3±0.9</td><td>98.3±0.7</td><td>84.4±2.0</td><td>75.9±1.2</td><td>93.6±1.4</td>
</tr>
<tr><td>cable</td><td>91.1±1.9</td><td>95.2±0.9</td><td>86.9±1.4</td><td>94.3±0.3</td><td>66.7±1.0</td><td>63.4±1.2</td><td>86.7±1.3</td></tr>
<tr><td>capsule</td><td>69.9±3.7</td><td>90.6±3.5</td><td>92.1±1.4</td><td>93.8±0.6</td><td>28.4±8.9</td><td>34.5±5.5</td><td>62.4±2.7</td></tr>
<tr><td>carpet</td><td>99.6±0.2</td><td>99.9±0.0</td><td>99.0±0.4</td><td>98.1±0.2</td><td>68.8±1.0</td><td>64.3±0.5</td><td>96.0±0.6</td></tr>
<tr><td>grid</td><td>91.7±1.2</td><td>97.3±0.4</td><td>90.2±1.3</td><td>93.0±1.7</td><td>26.8±3.6</td><td>33.4±3.2</td><td>74.7±5.8</td></tr>
<tr><td>hazelnut</td><td>78.8±11.9</td><td>84.8±11.2</td><td>84.5±2.6</td><td>96.0±1.3</td><td>34.0±13.1</td><td>39.7±9.7</td><td>80.6±5.8</td></tr>
<tr><td>leather</td><td>100.0±0.0</td><td>100.0±0.0</td><td>100.0±0.0</td><td>99.4±0.1</td><td>61.3±1.9</td><td>56.4±1.4</td><td>99.0±0.1</td></tr>
<tr><td>metal nut</td><td>95.5±2.0</td><td>98.9±0.5</td><td>95.4±1.4</td><td>96.4±0.7</td><td>80.4±3.7</td><td>73.7±2.8</td><td>90.5±2.2</td></tr>
<tr><td>pill</td><td>90.2±2.8</td><td>98.1±0.6</td><td>93.2±0.9</td><td>95.6±0.9</td><td>64.3±5.2</td><td>59.8±3.5</td><td>92.5±0.7</td></tr>
<tr><td>screw</td><td>53.6±5.2</td><td>76.7±3.5</td><td>85.5±0.4</td><td>92.0±1.6</td><td>4.6±1.2</td><td>9.6±2.1</td><td>71.2±4.4</td></tr>
<tr><td>tile</td><td>96.0±1.3</td><td>98.7±0.4</td><td>94.6±1.6</td><td>94.0±1.4</td><td>75.8±1.9</td><td>69.6±1.3</td><td>87.0±2.7</td></tr>
<tr><td>toothbrush</td><td>92.8±1.6</td><td>97.4±0.6</td><td>92.0±1.7</td><td>98.4±0.3</td><td>58.8±2.4</td><td>59.6±2.9</td><td>83.8±1.2</td></tr>
<tr><td>transistor</td><td>83.9±5.4</td><td>82.4±4.6</td><td>74.6±5.9</td><td>85.7±3.4</td><td>40.6±5.3</td><td>40.7±5.4</td><td>75.2±5.2</td></tr>
<tr><td>wood</td><td>99.8±0.2</td><td>99.9±0.1</td><td>99.2±0.0</td><td>94.1±0.7</td><td>68.7±1.6</td><td>64.7±1.2</td><td>93.9±0.6</td></tr>
<tr><td>zipper</td><td>95.0±2.8</td><td>98.5±0.9</td><td>95.3±2.0</td><td>97.1±1.1</td><td>58.9±4.5</td><td>55.7±3.1</td><td>69.7±5.2</td></tr>
<tr>
<td style="border-bottom: 1px solid #fff;"></td>
<td style="text-align: left; border-bottom: 1px solid #fff;">mean</td>
<td style="border-bottom: 1px solid #fff;">89.1±1.1</td><td style="border-bottom: 1px solid #fff;">94.5±1.0</td><td style="border-bottom: 1px solid #fff;">92.1±0.6</td><td style="border-bottom: 1px solid #fff;">95.1±0.2</td><td style="border-bottom: 1px solid #fff;">54.8±0.9</td><td style="border-bottom: 1px solid #fff;">53.4±0.6</td><td style="border-bottom: 1px solid #fff;">83.8±0.6</td>
</tr>
</tbody>
</table>



</div>

<br/>

<div align="center">

<table style="border-collapse: collapse; border-top: 1px solid #fff; border-bottom: 1px solid #fff;">
<thead>
<tr>
<th rowspan="2">Methods</th>
<th rowspan="2">Categories</th>
<th colspan="3" style="text-align: center;">Anomaly Classification</th>
<th colspan="4" style="text-align: center;">Anomaly Segmentation</th>
</tr>
<tr>
<th>I-ROC</th>
<th>I-PR</th>
<th>I-F1<sub>max</sub></th>
<th>P-ROC</th>
<th>P-PR</th>
<th>P-F1<sub>max</sub></th>
<th>P-PRO</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="15">MetaUAS*</td>
<td>bottle</td>
<td>99.5</td><td>99.9</td><td>98.4</td><td>98.4</td><td>83.9</td><td>75.2</td><td>92.9</td>
</tr>
<tr><td>cable</td><td>96.3</td><td>98.0</td><td>93.4</td><td>96.4</td><td>70.5</td><td>66.1</td><td>91.2</td></tr>
<tr><td>capsule</td><td>89.3</td><td>97.5</td><td>93.7</td><td>95.4</td><td>45.7</td><td>46.4</td><td>68.5</td></tr>
<tr><td>carpet</td><td>99.7</td><td>99.9</td><td>98.9</td><td>97.9</td><td>68.2</td><td>64.0</td><td>95.4</td></tr>
<tr><td>grid</td><td>95.7</td><td>98.7</td><td>93.7</td><td>94.0</td><td>29.4</td><td>36.8</td><td>79.7</td></tr>
<tr><td>hazelnut</td><td>99.6</td><td>99.8</td><td>97.9</td><td>98.8</td><td>68.2</td><td>66.9</td><td>92.8</td></tr>
<tr><td>leather</td><td>100.0</td><td>100.0</td><td>100.0</td><td>99.4</td><td>61.1</td><td>57.0</td><td>98.8</td></tr>
<tr><td>metal nut</td><td>97.2</td><td>99.4</td><td>96.3</td><td>97.2</td><td>83.1</td><td>76.6</td><td>93.6</td></tr>
<tr><td>pill</td><td>82.5</td><td>96.7</td><td>91.6</td><td>94.0</td><td>58.6</td><td>57.4</td><td>91.0</td></tr>
<tr><td>screw</td><td>82.4</td><td>92.7</td><td>87.0</td><td>97.4</td><td>23.6</td><td>27.9</td><td>65.4</td></tr>
<tr><td>tile</td><td>96.6</td><td>98.9</td><td>94.4</td><td>94.7</td><td>76.7</td><td>71.1</td><td>88.7</td></tr>
<tr><td>toothbrush</td><td>96.4</td><td>98.6</td><td>93.8</td><td>99.0</td><td>64.3</td><td>63.2</td><td>81.2</td></tr>
<tr><td>transistor</td><td>89.0</td><td>86.3</td><td>79.1</td><td>87.9</td><td>45.2</td><td>45.0</td><td>79.3</td></tr>
<tr><td>wood</td><td>99.7</td><td>99.9</td><td>98.4</td><td>93.2</td><td>67.0</td><td>63.0</td><td>93.7</td></tr>
<tr><td>zipper</td><td>93.6</td><td>98.1</td><td>95.0</td><td>96.8</td><td>58.4</td><td>56.2</td><td>68.2</td></tr>
<tr>
<td style="border-bottom: 1px solid #fff;"></td>
<td style="text-align: left; border-bottom: 1px solid #fff;">mean</td>
<td style="border-bottom: 1px solid #fff;">94.5</td><td style="border-bottom: 1px solid #fff;">97.6</td><td style="border-bottom: 1px solid #fff;">94.1</td><td style="border-bottom: 1px solid #fff;">96.0</td><td style="border-bottom: 1px solid #fff;">60.2</td><td style="border-bottom: 1px solid #fff;">58.2</td><td style="border-bottom: 1px solid #fff;">85.4</td>
</tr>
</tbody>
</table>

</div>

<br/>

<div align="center">

<table style="border-collapse: collapse; border-top: 1px solid #fff; border-bottom: 1px solid #fff;">
<thead>
<tr>
<th rowspan="2">Methods</th>
<th rowspan="2">Categories</th>
<th colspan="3" style="text-align: center;">Anomaly Classification</th>
<th colspan="4" style="text-align: center;">Anomaly Segmentation</th>
</tr>
<tr>
<th>I-ROC</th>
<th>I-PR</th>
<th>I-F1<sub>max</sub></th>
<th>P-ROC</th>
<th>P-PR</th>
<th>P-F1<sub>max</sub></th>
<th>P-PRO</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="12">MetaUAS</td>
<td>candle</td>
<td>90.1±1.2</td><td>90.6±1.0</td><td>84.0±1.9</td><td>98.4±0.4</td><td>33.1±2.5</td><td>38.8±1.3</td><td>85.5±1.8</td>
</tr>
<tr><td>capsules</td><td>62.7±5.5</td><td>73.4±4.0</td><td>78.1±0.6</td><td>92.2±2.2</td><td>16.0±2.8</td><td>24.7±1.9</td><td>52.2±2.3</td></tr>
<tr><td>cashew</td><td>86.1±2.8</td><td>93.7±1.3</td><td>85.6±2.1</td><td>97.2±1.3</td><td>71.7±3.3</td><td>67.9±2.6</td><td>79.2±2.0</td></tr>
<tr><td>chewinggum</td><td>96.9±1.0</td><td>98.7±0.4</td><td>94.5±0.6</td><td>99.5±0.1</td><td>82.5±1.2</td><td>76.6±0.8</td><td>85.3±0.9</td></tr>
<tr><td>fryum</td><td>78.9±2.7</td><td>89.2±2.9</td><td>82.0±0.9</td><td>86.6±1.7</td><td>19.9±3.5</td><td>29.0±2.9</td><td>48.2±5.8</td></tr>
<tr><td>macaroni1</td><td>77.2±1.4</td><td>80.7±0.8</td><td>71.9±1.7</td><td>92.6±2.4</td><td>12.7±0.7</td><td>21.6±0.6</td><td>56.9±4.2</td></tr>
<tr><td>macaroni2</td><td>58.9±7.3</td><td>56.9±6.8</td><td>68.8±1.7</td><td>90.1±1.7</td><td>0.9±0.5</td><td>3.9±1.9</td><td>65.0±7.2</td></tr>
<tr><td>pcb1</td><td>64.9±21.6</td><td>70.3±13.2</td><td>75.2±9.1</td><td>98.1±0.5</td><td>63.1±4.1</td><td>60.5±4.0</td><td>68.8±13.2</td></tr>
<tr><td>pcb2</td><td>69.1±3.1</td><td>67.8±2.0</td><td>69.6±2.3</td><td>96.2±0.8</td><td>14.8±2.5</td><td>25.8±3.8</td><td>75.7±3.3</td></tr>
<tr><td>pcb3</td><td>62.4±6.7</td><td>63.7±8.0</td><td>69.0±1.4</td><td>96.8±0.3</td><td>26.1±3.6</td><td>31.2±3.3</td><td>61.9±5.2</td></tr>
<tr><td>pcb4</td><td>95.5±1.4</td><td>95.3±1.2</td><td>90.2±2.1</td><td>97.3±1.0</td><td>34.9±3.8</td><td>43.0±3.9</td><td>78.8±2.4</td></tr>
<tr><td>pipe_fryum</td><td>95.4±1.9</td><td>97.5±1.1</td><td>93.6±1.9</td><td>98.7±0.5</td><td>67.8±3.3</td><td>63.2±2.2</td><td>85.8±1.3</td></tr>
<tr>
<td style="border-bottom: 1px solid #fff;"></td>
<td style="text-align: left; border-bottom: 1px solid #fff;">mean</td>
<td style="border-bottom: 1px solid #fff;">78.2±2.0</td><td style="border-bottom: 1px solid #fff;">81.5±1.5</td><td style="border-bottom: 1px solid #fff;">80.2±1.0</td><td style="border-bottom: 1px solid #fff;">95.3±0.5</td><td style="border-bottom: 1px solid #fff;">37.0±0.9</td><td style="border-bottom: 1px solid #fff;">40.5±0.6</td><td style="border-bottom: 1px solid #fff;">70.3±1.5</td>
</tr>
</tbody>
</table>

</div>

<br/>

<div align="center">

<table style="border-collapse: collapse; border-top: 1px solid #fff; border-bottom: 1px solid #fff;">
<thead>
<tr>
<th rowspan="2">Methods</th>
<th rowspan="2">Categories</th>
<th colspan="3" style="text-align: center;">Anomaly Classification</th>
<th colspan="4" style="text-align: center;">Anomaly Segmentation</th>
</tr>
<tr>
<th>I-ROC</th>
<th>I-PR</th>
<th>I-F1<sub>max</sub></th>
<th>P-ROC</th>
<th>P-PR</th>
<th>P-F1<sub>max</sub></th>
<th>P-PRO</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="12">MetaUAS*</td>
<td>candle</td>
<td>91.2</td><td>91.2</td><td>84.7</td><td>98.6</td><td>35.2</td><td>39.1</td><td>84.9</td>
</tr>
<tr><td>capsules</td><td>62.9</td><td>77.3</td><td>76.9</td><td>94.5</td><td>30.7</td><td>35.6</td><td>58.1</td></tr>
<tr><td>cashew</td><td>85.5</td><td>93.4</td><td>84.3</td><td>98.6</td><td>77.9</td><td>71.4</td><td>76.8</td></tr>
<tr><td>chewinggum</td><td>97.3</td><td>98.8</td><td>93.7</td><td>99.5</td><td>82.2</td><td>76.4</td><td>86.6</td></tr>
<tr><td>fryum</td><td>79.5</td><td>89.9</td><td>81.7</td><td>88.9</td><td>22.3</td><td>31.7</td><td>37.7</td></tr>
<tr><td>macaroni1</td><td>75.7</td><td>77.5</td><td>72.0</td><td>93.6</td><td>11.6</td><td>20.0</td><td>49.5</td></tr>
<tr><td>macaroni2</td><td>59.2</td><td>56.3</td><td>68.1</td><td>91.7</td><td>1.1</td><td>5.6</td><td>67.8</td></tr>
<tr><td>pcb1</td><td>82.5</td><td>80.5</td><td>77.1</td><td>99.2</td><td>76.3</td><td>71.3</td><td>69.3</td></tr>
<tr><td>pcb2</td><td>70.0</td><td>69.6</td><td>67.7</td><td>97.2</td><td>14.8</td><td>26.3</td><td>79.0</td></tr>
<tr><td>pcb3</td><td>76.2</td><td>75.8</td><td>73.6</td><td>97.2</td><td>31.1</td><td>35.5</td><td>57.0</td></tr>
<tr><td>pcb4</td><td>95.3</td><td>95.6</td><td>88.5</td><td>96.8</td><td>35.6</td><td>43.8</td><td>72.2</td></tr>
<tr><td>pipe_fryum</td><td>95.7</td><td>97.7</td><td>94.2</td><td>98.4</td><td>64.1</td><td>60.2</td><td>88.6</td></tr>
<tr>
<td style="border-bottom: 1px solid #fff;"></td>
<td style="text-align: left; border-bottom: 1px solid #fff;">mean</td>
<td style="border-bottom: 1px solid #fff;">80.9</td><td style="border-bottom: 1px solid #fff;">83.6</td><td style="border-bottom: 1px solid #fff;">80.2</td><td style="border-bottom: 1px solid #fff;">96.2</td><td style="border-bottom: 1px solid #fff;">40.2</td><td style="border-bottom: 1px solid #fff;">43.1</td><td style="border-bottom: 1px solid #fff;">69.0</td>
</tr>
</tbody>
</table>

</div>


### Pretrained Models

MetaUAS: [metauas-256.pth](https://huggingface.co/ldl010302/MetaUAS/blob/main/metauas-256.pth)

## 🙏 Acknowledgements

We reference code from [MetaUAS](https://github.com/gaobb/MetaUAS), [LaMa](https://github.com/advimman/lama), and [CYWS](https://github.com/ragavsachdeva/The-Change-You-Want-to-See).

## 🤝 Contributing

- ⭐ If you find this project useful, a star would be greatly appreciated
- 🐛 Report bugs or ask questions via [Issues](https://github.com/DeLunnLi/MetaUAS/issues)
- 🔀 Fork and submit a PR with your improvements — we'll review and add you to the contributors
