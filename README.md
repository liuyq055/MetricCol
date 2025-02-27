# MetricCol: Absolute-Scale Depth and Pose Estimation in Colonoscopy via Geometric Consistency and Domain Adaptation
## Abstract
Accurate metric depth and pose estimation are critical for colonoscopic navi-gation and lesion localization. However, existing methods often struggle with scale ambiguity and domain gaps between synthetic and real datasets. To ad-dress these issues, we propose a novel framework consisting of two stages: 1) a fully supervised depth estimation model utilizing synthetic data with anatomi-cal priors to bridge the domain gap between synthetic and real datasets, and 2) a weakly supervised joint learning approach combining camera-aware depth scaling with uncertainty-driven pseudo-labeling to refine metric depth and pose estimation. We validate the framework on both synthetic and real colon-oscopy datasets, achieving superior performance in metric depth (RMSE=3.5408) and pose estimation (average ATE=0.6143). Experimental re-sults on both synthetic and real colonoscopy datasets show superior perfor-mance, robustness under challenging conditions, and demonstrate the clinical applicability of our method. 

## Results
#### Table 2. Quantitative depth comparison on C3VD dataset. 

|                                     Method                                      |  Abs Rel(↓)   |  Sq Rel(↓)   |    RMSE(↓)    |  RMSE log(↓)  |    a1(↑)    | 
|:-------------------------------------------------------------------------------:|:----------:|:------------:|:----------:|:----------:|:----------:|
|                                   DenseDepth                                    |   0.1803   |    2.0154    |   7.1911   |   0.2803   |   0.7183   | 
|                                       BTS                                       |   0.1534   |    1.2341    |   7.1054   |   0.1954   |   0.7441   | 
|                                  DepthAnythin                                   |   0.1136   |    0.5841    |   4.5747   |   0.1368   |   0.9023   | 
|                                  Ours (Stage1)                                  |   0.1105   |    0.5632    |   4.2915   |   0.1347   |   0.9040   | 
|                                 Endo-SfMLearner                                 |   0.5490   |    8.2250    |  12.7652   |   0.5186   |   0.3469   | 
|                                  AF-SfMLearner                                  |   0.2813   |    3.7943    |  12.1886   |   0.3549   |   0.4674   | 
|                                     EndoDAC                                     |   0.3432   |    6.1716    |  14.3523   |   0.4136   |   0.4159   |                                                  -                                                   |
| Ours([Baidu Netdisk](https://pan.baidu.com/s/1Osp-iavHERogxzs2XF-zng?pwd=jw3M)) | **0.0953** |  **0.4019**  | **3.5408** | **0.1158** | **0.9429** |

提取码：jw3M 

#### Figure 1. Depth results comparison on the C3VD dataset.
![depth-git](https://github.com/user-attachments/assets/86bc663a-d082-4eed-a473-694bb615ddfa)

Table 3. Quantitative pose comparison on C3VD dataset.

| Method | t3v1 APE (↓) | t3v1 ATE (↓) | t4v1 APE (↓) | t4v1 ATE (↓) | s3v2 APE (↓) | s3v2 ATE (↓) |
|--------|--------------|--------------|--------------|--------------|--------------|--------------|
| ES     | 0.7014       | 0.4902       | 10.740       | 9.6029       | 5.2610       | 4.8089       |
| AF     | 0.5972       | 0.4174       | 5.0222       | 4.2030       | 5.1370       | 4.4748       |
| ED     | 0.2586       | 0.2227       | 1.7911       | 1.6432       | 1.7673       | 1.5261       |
| Ours   | **0.1486**   | **0.1317**   | **1.2176**   | **1.1370**   | **0.6735**   | **0.5741**   |



#### Figure 2. Pose results comparison on the C3VD dataset.
![pose-git1](https://github.com/user-attachments/assets/20cb8272-7365-44cc-90e4-6e1db1def301)


## Initialization

Create an environment with conda:
```
conda env create -f conda.yaml
conda activate depth-anything
```

Install required dependencies with pip:
```
pip install -r requirements.txt
```

Download pretrained model from: [depth_anything_vitb14](https://drive.google.com/file/d/163ILZcnz_-IUoIgy1UF_r7PAQBqgDbll/view?usp=sharing). Create a folder named ```pretrained_model``` in this repo and place the downloaded model in it.

## Dataset 
### C3VD Dataset.
Download dataset from :[c3vd](https://durrlab.github.io/C3VD/).For the training set, we selected the following samples: c2v2, c3v1, c1v1, t1v2, t3v2, c1v2, s3v1, t4v2, t1v1, t2v1, c2v3, t2v2, and s2v1. The validation set in-cludes: c4v2, c2v1, t2v3, and s1v1. The test set contains: t3v1, c4v1, and d4v1.
### SimCol Dataset. 
Download dataset from:[simcol](https://github.com/anitarau/simcol). The training set consists of 25,942 images, the validation set includes 2,882 imag-es, and the test set contains 9,009 images

## Usage
### Training
The code will be released as soon as possible.

### Evaluation
```
python evaluate_depth.py --data_path <your_data_path> --eval_split <dataset> --load_weights_folder <the path of loaded weights> --visualize_depth
python evaluate_c3vd_pose.py --data_path <your_data_path> --load_weights_folder <the path of loaded weights> --eval_split <dataset>
```

## Acknowledgment
Our code is based on the implementation of [AF-SfMLearner](https://github.com/ShuweiShao/AF-SfMLearner), [Depth-Anything](https://github.com/LiheYoung/Depth-Anything), [EndoDAC](https://github.com/BeileiCui/EndoDAC). We thank their excellent works.
