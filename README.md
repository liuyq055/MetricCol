# MetricCol
MetricCol: Absolute-Scale Depth and Pose Estimation in Colonoscopy via Geometric Consistency and Domain Adaptation
## Abstract
Accurate metric depth and pose estimation are critical for colonoscopic navi-gation and lesion localization. However, existing methods often struggle with scale ambiguity and domain gaps between synthetic and real datasets. To ad-dress these issues, we propose a novel framework consisting of two stages: 1) a fully supervised depth estimation model utilizing synthetic data with anatomi-cal priors to bridge the domain gap between synthetic and real datasets, and 2) a weakly supervised joint learning approach combining camera-aware depth scaling with uncertainty-driven pseudo-labeling to refine metric depth and pose estimation. We validate the framework on both synthetic and real colon-oscopy datasets, achieving superior performance in metric depth (RMSE=3.5408) and pose estimation (average ATE=0.6143). Experimental re-sults on both synthetic and real colonoscopy datasets show superior perfor-mance, robustness under challenging conditions, and demonstrate the clinical applicability of our method. 
## Results

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
