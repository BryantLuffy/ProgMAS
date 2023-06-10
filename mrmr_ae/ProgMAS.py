from mrmr_ae.AE_mrmr import mrmr_ae
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import random
import numpy as np
import warnings
import os
from lifelines.statistics import multivariate_logrank_test
from sklearn.model_selection import cross_val_score

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

class ProgMAS:
    def __init__(self, ae_params, mrmr_params, seed=None):
        """
        ProgMAS (Progression Modeling with Autoencoders and mRMR) class for training and evaluating a survival prediction model.

        Args:
            ae_params (dict): Parameters for the Autoencoder.
            mrmr_params (dict): Parameters for the mRMR feature selection.
            seed (int): Random seed for reproducibility. Default is None.
        """
        self.ae_params = ae_params
        self.mrmr_params = mrmr_params
        self.seed = seed

        self.mrmr_ae = None
        self.embedded_Z = None
        self.train_X = None
        self.train_y = None
        self.test_X = None
        self.test_y = None

    def set_seed(self, seed):
        """
        Set the random seed for reproducibility.

        Args:
            seed (int): Random seed.
        """
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

    def get_latent_repre(self, omics_dict, survival_data):
        """
        Get the latent representations of the input data using mrmr_ae.

        Args:
            omics_dict (dict): Dictionary of omics data.
            survival_data (ndarray): Survival data.

        Returns:
            ndarray: mRMR-selected features.
            ndarray: Embedded latent representations.
        """
        # Process data using mrmr_ae
        self.mrmr_ae = mrmr_ae(omics_dict, survival_data.label, self.ae_params, self.mrmr_params)
        self.mRMR_features, self.embedded_Z = self.mrmr_ae.fit_transform()

        return self.mRMR_features, self.embedded_Z

    def split_train_test(self, survival_data, test_size=0.3):
        """
        Split the embedded latent representations and survival data into training and testing sets.

        Args:
            survival_data (ndarray): Survival data.
            test_size (float): Size of the test set as a fraction of the whole dataset. Default is 0.3.
        """
        # Split data into training and testing sets
        self.survival_data = survival_data
        self.train_X, self.test_X, self.train_y, self.test_y = train_test_split(
            self.embedded_Z, survival_data, test_size=test_size, random_state=self.seed
        )

    def train_predict(self, model='svm', model_params=None):
        """
        Train and predict using a survival prediction model.

        Args:
            model (str): Type of model to use. Supported models: 'svm', 'logistic'. Default is 'svm'.
            model_params (dict): Parameters for the model. Default is None.

        Returns:
            float: Mean accuracy.
            float: Mean AUC.
            float: Mean F1 score.
        """
        if model == 'svm':
            clf = SVC(probability=True, random_state=self.seed, **model_params)
        elif model == 'logistic':
            clf = LogisticRegression(random_state=self.seed, **model_params)
        else:
            raise ValueError("Invalid model type. Supported models: 'svm', 'logistic'")

        # Calculate five-fold evaluation metrics
        acc = cross_val_score(clf, self.embedded_Z, self.survival_data.label, cv=5)
        auc = cross_val_score(clf, self.embedded_Z, self.survival_data.label, cv=5, scoring='roc_auc')
        f1 = cross_val_score(clf, self.embedded_Z, self.survival_data.label, cv=5, scoring='f1_weighted')

        clf.fit(self.train_X, self.train_y.label)

        # Predict on the test set
        self.y_pred = clf.predict(self.test_X)

        return acc.mean(), auc.mean(), f1.mean()

    def survival_analysis(self):
        """
        Perform survival analysis using the predicted labels.

        Returns:
            float: P-value from the multivariate log-rank test.
        """
        self.test_y.label = self.y_pred
        pvalue = multivariate_logrank_test(self.test_y['OS.time'], self.test_y['label'], self.test_y['OS'])
        return pvalue.p_value

    def biomarker_idt(self, omics_dict, survival_data):
        """
        Perform biomarker identification using mrmr_ae.

        Args:
            omics_dict (dict): Dictionary of omics data.
            survival_data (ndarray): Survival data.

        Returns:
            dict: Dictionary of biomarker results.
        """
        biomarker_results = {}

        for omics_name, omics_data in omics_dict.items():
            merged_omics, embedded_Z = self.get_latent_repre({omics_name: omics_data}, survival_data)
            selected_features = merged_omics.columns.tolist()
            biomarker_results[omics_name] = selected_features[:10]

        return biomarker_results


