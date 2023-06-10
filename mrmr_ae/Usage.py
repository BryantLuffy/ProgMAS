from mrmr_ae.ProgMAS import ProgMAS
import pandas as pd
survive_data=pd.read_csv('D:\Study\学习资料\Multi-omic data\examples\data\BLCAdata\survive_BLCA.csv'
                         ,index_col=0,sep=',')
survive_data=survive_data.dropna(how='any')

import mrmr_ae
def add_survival_labels(data):
    data['label'] = 0
    data.loc[(data['OS.time'] > 1825) & (data['OS'].isin([0, 1])), 'label'] = 1
    data = data.loc[~((data['OS.time'] < 1825) & (data['OS'] == 0))]
    return data
survive_data=add_survival_labels(survive_data)
CNV_data=pd.read_table('D:\Study\学习资料\Multi-omic data\examples\data\BLCAdata\CNV_BLCA'
                         ,index_col=0)
RNA_data=pd.read_table('D:\Study\学习资料\Multi-omic data\examples\data\BLCAdata\GeneExp_BLCA'
                         ,index_col=0)
miRNA_data=pd.read_table('D:\Study\学习资料\Multi-omic data\examples\data\BLCAdata\miRNA_BLCA'
                         ,index_col=0)

omics_data = {
    'omics1': CNV_data,
    'omics2': RNA_data,
    'omics3': miRNA_data
}


from mrmr_ae.Preprocess import MultiomicsDataProcessor

processor=MultiomicsDataProcessor()
missing_thresholds = {
    'omics1': 0.2,
    'omics2': 0.2,
    'omics3': 0.2
}

variance_thresholds = {
    'omics1': 0.2,
    'omics2': 0.2,
    'omics3': 0.2
}
processor.process_multiomics_data(survive_data, omics_data, missing_thresholds, variance_thresholds)
survival_data_processed = processor.survival_data
omics1_processed = processor.get_processed_data('omics1')
omics2_processed = processor.get_processed_data('omics2')
omics3_processed = processor.get_processed_data('omics3')

model_params = {'kernel': 'linear', 'C': 0.1,'gamma':10}
ae_params = {
        'epochs': 300,
        'batch_size': 16
    }
mrmr_params = {
        'k': 100,
        'p_dict': {
            'omics1': 1,
            'omics2': 1,
            'omics3': 1
        }
    }
progmas = ProgMAS(ae_params=ae_params, mrmr_params=mrmr_params)

    # Step 2: Set random seed (optional)
progmas.set_seed(529)

    # Step 3: Get latent representation using get_latent_repre method
omics_dict = {
        'omics1': omics1_processed,
        'omics2': omics2_processed,
        'omics3': omics3_processed
    }
merged_omics, embedded_Z = progmas.get_latent_repre(omics_dict, survival_data_processed)

    # Step 4: Split data into training and testing sets
progmas.split_train_test(survival_data_processed, test_size=0.3)

    # Step 5: Train and predict using SVM model

acc, auc, f1 = progmas.train_predict(model='svm', model_params=model_params)
print("SVM Model - AUC: {:.4f}, Accuracy: {:.4f}, F1-Score: {:.4f}".format(auc, acc, f1))

    # Step 6: Perform survival analysis
p_values = progmas.survival_analysis()
print("Log-Rank Test - p-values:\n", p_values)

    # Step 7: Perform biomarker identification
biomarker_results = progmas.biomarker_idt(omics_dict, survival_data_processed)
for omics_name, top_features in biomarker_results.items():
    print("Top features for {}: {}".format(omics_name, top_features))