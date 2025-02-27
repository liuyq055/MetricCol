![rpe_error](https://github.com/user-attachments/assets/cba36aae-9be1-4f75-8fe6-ed73123691b2)# MetricCol
MetricCol: Absolute-Scale Depth and Pose Estimation in Colonoscopy via Geometric Consistency and Domain Adaptation
## Abstract
Accurate metric depth and pose estimation are critical for colonoscopic navi-gation and lesion localization. However, existing methods often struggle with scale ambiguity and domain gaps between synthetic and real datasets. To ad-dress these issues, we propose a novel framework consisting of two stages: 1) a fully supervised depth estimation model utilizing synthetic data with anatomi-cal priors to bridge the domain gap between synthetic and real datasets, and 2) a weakly supervised joint learning approach combining camera-aware depth scaling with uncertainty-driven pseudo-labeling to refine metric depth and pose estimation. We validate the framework on both synthetic and real colon-oscopy datasets, achieving superior performance in metric depth (RMSE=3.5408) and pose estimation (average ATE=0.6143). Experimental re-sults on both synthetic and real colonoscopy datasets show superior perfor-mance, robustness under challenging conditions, and demonstrate the clinical applicability of our method. 

## Results
Table 2. Quantitative depth comparison on C3VD dataset. 
| TS   | Method                 | Error (↓)                                     | Accuracy (↑)                              |
|------|------------------------|-----------------------------------------------|-------------------------------------------|
|      |                        | AbsRel   | SqRel   | RMSE    | RMSE_log | a1      | a2      | a3      |
|------|------------------------|-----------------------------------------------|-------------------------------------------|
| S    | DD                 | 0.1803   | 2.0154  | 7.1911  | 0.2803   | 0.7183  | 0.8970  | 0.9583  |
|      | BTS                | 0.1534   | 1.2341  | 7.1054  | 0.1954   | 0.7441  | 0.9736  | 0.9967  |
|      | DA                | 0.1136   | 0.5841  | 4.5747  | 0.1368   | 0.9023  | 0.9966  | 0.9998  |
|      | Ours (Stage1)          | 0.1105   | 0.5632  | 4.2915  | 0.1347   | 0.9040  | 0.9952  | 0.9996  |
| U/W  | ES                | 0.5490   | 8.2250  | 12.7652 | 0.5186   | 0.3469  | 0.6319  | 0.8274  |
|      | AF                 | 0.2813   | 3.7943  | 12.1886 | 0.3549   | 0.4674  | 0.8043  | 0.9398  |
|      | ED                  | 0.3432   | 6.1716  | 14.3523 | 0.4136   | 0.4159  | 0.7384  | 0.8986  |
|      | Ours([Baidu Netdisk](https://pan.baidu.com/s/1Osp-iavHERogxzs2XF-zng?pwd=jw3M))                   | 0.0953   | 0.4019  | 3.5408  | 0.1158   | 0.9429  | 0.9963  | 0.9998  |
提取码：jw3M 
![depth-git](https://github.com/user-attachments/assets/86bc663a-d082-4eed-a473-694bb615ddfa)
Figure 1. Depth results comparison on the C3VD dataset.

Table 3. Quantitative pose comparison on C3VD dataset.
| Method | t3v1                    | t4v1                    | s3v2                    |
|--------|--------------------------|--------------------------|--------------------------|
|        | APE (↓) | ATE (↓)      | APE (↓) | ATE (↓)      | APE (↓) | ATE (↓)      |
|--------|----------|--------------|----------|--------------|----------|--------------|
| ES[19] | 0.7014   | 0.4902       | 10.740  | 9.6029       | 5.2610   | 4.8089       |
| AF[26] | 0.5972   | 0.4174       | 5.0222  | 4.2030       | 5.1370   | 4.4748       |
| ED[8]  | 0.2586   | 0.2227       | 1.7911  | 1.6432       | 1.7673   | 1.5261       |
| Ours   | 0.1486   | 0.1317       | 1.2176  | 1.1370       | 0.6735   | 0.5741       |


![pose-git1](https://github.com/user-attachments/assets/e4e557f2-4966-4b51-911c-71f0ace3615c)
Figure 2. Pose results comparison on the C3VD dataset.
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
If accepted, it will be made public.

### Evaluation
```
python evaluate_depth.py --data_path <your_data_path> --eval_split <dataset> --load_weights_folder <the path of loaded weights> --visualize_depth
python evaluate_c3vd_pose.py --data_path <your_data_path> --load_weights_folder <the path of loaded weights> --eval_split <dataset>
```

## Acknowledgment
Our code is based on the implementation of [AF-SfMLearner](https://github.com/ShuweiShao/AF-SfMLearner), [Depth-Anything](https://github.com/LiheYoung/Depth-Anything), [EndoDAC](https://github.com/BeileiCui/EndoDAC). We thank their excellent works.
