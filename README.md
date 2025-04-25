<!-- #region -->
# CT-and-Whole-Slide-Image-Features-Predict-Clinical-Outcomes-in-Lung-Neuroendocrine-Patients

<img src="figures/fig1.png" alt="drawing" width="75%"/>

## **Abstract**
**Background**: Neuroendocrine tumors (NETs) are rare and heterogeneous cancers that vary in clinical presentation, biology, and treatment response. They exhibit slow growth with varying levels of aggressiveness, highlighting the need for reliable biomarkers to guide personalized treatment. This study aims to develop predictive models for overall survival (OS) and progression-free survival (PFS) using CT scans, whole slide images (WSIs) and clinical data. 

**Methods**: This retrospective analysis included 83 patients. Predictive models were developed using radiomics features from CT scans and morphological or pathomics features from whole slide images (WSIs). The Cox model was trained using the most significant features from both radiomics and pathomics. By integrating these features with clinical data, we built predictive models combining clinical-radiomics and clinical-pathomics information. We also assessed how image harmonization across different acquisition parameters affects model performance.

**Results**: The radiomics model's concordance indices (C-indices) for predicting OS and PFS in the validation cohort were 0.64 ± 0.06 and 0.60 ± 0.05, respectively. Combining radiomics with clinical data slightly improved performance, with C-indices of 0.643 ± 0.04 for OS and 0.61 ± 0.04 for PFS. For the pathomics model, combining morphological features with clinical data also showed slight improvements, with C-indices for OS rising from 0.60 ± 0.15 to 0.68 ± 0.08, and for PFS from 0.65 ± 0.08 to 0.70 ± 0.03. Harmonizing radiomics features did not significantly enhance the model's performance for predicting survival outcomes.

**Conclusion**: This study developed and validated models that integrate radiomics and pathomics with clinical data, improving prognostic accuracy for OS and PFS. These multimodal approaches, supported by large datasets, offer significant potential for enhancing patient risk stratification. Further multi-institutional validation is needed, but these imaging-driven biomarkers could ultimately refine therapeutic strategies and optimize survival outcomes.


## Installation
```
pip install -r requirements.txt
```
## 📁 Data Availability
The data presented in this study are not publicly available at this time but may be obtained from the corresponding author, **Venkata Manem**, upon reasonable request.

## Training 
```
python train_wsi_model.py
python train_ct_model.py
```

## Citation
If this work contributes to your research, please cite as follows:
```


