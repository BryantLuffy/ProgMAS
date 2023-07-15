# ProgMAS：A two-step processing framework to integrate multi-omics data for cancer survival classification and biomarker identification




## Description
Predicting cancer survival rates is very important for individual patient care. And accurate prognosis is helpful in the clinical management of cancer.

ProgMAS is a model for classification of survival periods of cancer patients based on machine learning and deep learning methods.

We demonstrate that ProgMAS outperforms some traditional machine learning methods and other state-of-the-art multi-omics data integration methods on three types of cancer. Furthermore, ProgMAS can identify important biomarkers from different omics data types related to the investigated biomedical problems.

## Dependencies
ProgCAE is implemented in Python 3.8, which also requires the installation of keras, lifelines, numpy, pandas, scikit-learn, mrmr-selection,tensorflow and other packages. Their specific versions are as follows.

### packages

`keras`         `2.11.0`<br>
`lifelines`     `0.27.7`<br>
`numpy`         `1.22.3`<br>
`pandas`        `1.5.3`<br>
`scikit-learn`  `1.2.1`<br>
`tensorflow`    `2.11.0`<br> 
`mrmr-selection`    `0.2.6`<br> 


## Usage
The input to ProgMAS consists of multi-omics dictionary and survival information, where the survival information is in csv format, and various thresholds. For a particular omics matrix, its rows should represent samples (patients), its columns should represent features (genes), and the first column should be the id of each patient. The survival information consists of the patient ID, status, and time. Outputs include indicator values for categorical outcomes, p-values for rank tests and biomarkers that are important under different omics. 

Assuming that we already have a processed multi-omics dictionary, survival information and pre-configured parameters for the model. a simple example of ProgMAS is shown below:

```python
from mrmr_ae.ProgMAS import ProgMAS
import mrmr_ae
from mrmr_ae.Preprocess import MultiomicsDataProcessor
# mRMR feature selection under each omics
progmas = ProgMAS(ae_params=ae_params, mrmr_params=mrmr_params)
# Compression of features selected by mRMR using AE
merged_omics, embedded_Z = progmas.get_latent_repre(omics_dict, survival_data_processed)
# Classification in the training set
acc, auc, f1 = progmas.train_predict(model='svm', model_params=model_params)
# Log-rank test p-value
p_values = progmas.survival_analysis()
# Biomarkers under each omics
biomarker_results = progmas.biomarker_idt(omics_dict, survival_data_processed)
```

As an example of BLCA cancer multi-omics data, the specific use of ProgMAS can be found in the tutorial.ipynb file
#### Specific operational details are as shown in tutorial.ipynb.

### Data Acquisition
#### To replicate our study, you can download multi-omics datasets for various types of cancer from the links provided below.

| Cancer | Link |
| :------: | :------ |
| BRCA   | https://xenabrowser.net/datapages/?cohort=TCGA%20Breast%20Cancer%20(BRCA)&removeHub=https%3A%2F%2Fxena.treehouse.gi.ucsc.edu%3A443 |
| BLCA   | https://xenabrowser.net/datapages/?cohort=TCGA%20Bladder%20Cancer%20(BLCA)&removeHub=https%3A%2F%2Fxena.treehouse.gi.ucsc.edu%3A443 |
| LGG    | https://xenabrowser.net/datapages/?cohort=TCGA%20Lower%20Grade%20Glioma%20(LGG)&removeHub=https%3A%2F%2Fxena.treehouse.gi.ucsc.edu%3A443 |
