<!-- #region -->
# CT-and-Whole-Slide-Image-Features-Predict-Clinical-Outcomes-in-Lung-Neuroendocrine-Patients

<img src="figures/fig1.png" alt="drawing" width="75%"/>

## Abstract
Background: Neuroendocrine tumors (NETs) are rare and heterogeneous cancers that vary in clinical presentation, biology, and treatment response. They exhibit slow growth with varying levels of aggressiveness, highlighting the need for reliable biomarkers to guide personalized treatment. This study aims to develop predictive models for overall survival (OS) and progression-free survival (PFS) using CT scans, whole slide images (WSIs) and clinical data. 
Methods: This retrospective analysis included 83 patients. Predictive models were developed using radiomics features from CT scans and morphological or pathomics features from whole slide images (WSIs). The Cox model was trained using the most significant features from both radiomics and pathomics. By integrating these features with clinical data, we built predictive models combining clinical-radiomics and clinical-pathomics information. We also assessed how image harmonization across different acquisition parameters affects model performance.
Results: The radiomics model's concordance indices (C-indices) for predicting OS and PFS in the validation cohort were 0.623 and 0.63, respectively. Combining radiomics with clinical data slightly improved performance, with C-indices of 0.642 for OS and 0.653 for PFS. For the pathomics model, combining morphological features with clinical data also showed slight improvements, with C-indices for OS rising from 0.633 to 0.655, and for PFS from 0.65 to 0.675. Harmonizing radiomics features did not significantly enhance the model's performance for predicting survival outcomes.
Conclusion: This study developed and validated models that integrate radiomics and pathomics with clinical data, improving prognostic accuracy for OS and PFS. These multimodal approaches, supported by large datasets, offer significant potential for enhancing patient risk stratification. Further multi-institutional validation is needed, but these imaging-driven biomarkers could ultimately refine therapeutic strategies and optimize survival outcomes.


## Demonstration

<table>
  <tr>
    <td align="center">
    </td>
    <td align="center">
      <strong>Ground truth</strong><br>
    </td>
    <td align="center">
      <strong>Semantic layout</strong><br>
    </td>
    <td align="center">
        <strong><a href="https://ieeexplore.ieee.org/document/10493074">Med-DDPM</a></strong><br>
    </td>
    <td align="center">
        <strong><a href="https://link.springer.com/chapter/10.1007/978-3-031-43999-5_56">Make-A-Volume</a></strong><br>
    </td>
    <td align="center">
        <strong><a href="https://arxiv.org/abs/2403.12852">GEM-3D</a></strong><br>
    </td>
    <td align="center">
        <strong> Lung-DDPM </strong><br>
    </td>
  </tr>
  <tr>
    <td align="center">
      <strong>Case 1</strong><br>
    </td>
    <td align="center">
      <img id="gt_0" src="materials/gt_0.gif">
    </td>
    <td align="center">
      <img id="mask_0" src="materials/mask_0.gif">
    </td>
    <td align="center">
      <img id="med-ddpm_0" src="materials/med-ddpm_0.gif">
    </td>
    <td align="center">
      <img id="mav_0" src="materials/mav_0.gif">
    </td>
    <td align="center">
      <img id="gem-3d_0" src="materials/gem-3d_0.gif">
    </td>
    <td align="center">
      <img id="lung-ddpm_0" src="materials/lung-ddpm_0.gif">
    </td>
  </tr>
  <tr>
    <td align="center">
      <strong>Case 2</strong><br>
    </td>
    <td align="center">
      <img id="img_2" src="materials/gt_1.gif">
    </td>
    <td align="center">
      <img id="img_3" src="materials/mask_1.gif">
    </td>
    <td align="center">
      <img id="med-ddpm_1" src="materials/med-ddpm_1.gif">
    </td>
    <td align="center">
      <img id="mav_1" src="materials/mav_1.gif">
    </td>
    <td align="center">
      <img id="gem-3d_1" src="materials/gem-3d_1.gif">
    </td>
    <td align="center">
      <img id="lung-ddpm_1" src="materials/lung-ddpm_1.gif">
    </td>
  </tr>
</table>
<!-- #endregion -->

## Installation
```
conda env create -n lung-ddpm python=3.9
conda activate lung-ddpm
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Preparation
We prepared three sample cases at [GoogleDrive](https://drive.google.com/drive/folders/1IiWUsGyU-4aNN6-PscrcWijBNdjH9Qhz?usp=sharing), and you can download to 'data/LIDC-IDRI/' folder. Then you can easily use a command to verify the training and sampling function.

Also, you are welcome to specify '--ct_path' and '--mask_path' parameters for your own dataset, please follow the above three cases for data formats.

Download the pretrained model at [GoogleDrive](https://drive.google.com/drive/folders/1IiWUsGyU-4aNN6-PscrcWijBNdjH9Qhz?usp=sharing). And put it at the 'checkpoints' folder in the root directory.
The directory structure should look similar as follows:
```
Lung-DDPM/
│── data/
│   ├── LIDC-IDRI/    # Prepared dataset
│   │   ├── CT/
│   │   │   ├── LIDC-IDRI-0001.nii.gz
            ...
│   │   ├── SEG/
│   │   │   ├── LIDC-IDRI-0001.nii.gz
            ...
│   ...
│── checkpoints/      # Pretrained models
│   ├── Lung-DDPM-LIDC-IDRI-100000-steps.pt
│── train.py
│── sample.py
│── README.md
...
```

## Training 
```
python train.py --ct_path "data/LIDC-IDRI/CT" --mask_path "data/LIDC-IDRI/SEG" --batchsize 1 --epochs 100000 --input_size 128 --depth_size 128 --num_class_labels 3 --num_channels 64 --num_res_blocks 1 --timesteps 250 --save_and_sample_every 2
```

## Citation
If Lung-DDPM contributes to your research, please cite as follows:
```


