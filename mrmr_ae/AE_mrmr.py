import pandas as pd
from mrmr_ae.multi_mrmr import Multi_mRMR
from mrmr_ae.AE_model import Autoencoder

class mrmr_ae:
    def __init__(self, omics_dict, labels, ae_params, mrmr_params):
        """
        Class for performing mRMR feature selection followed by Autoencoder-based dimensionality reduction.

        Args:
            omics_dict (dict): Dictionary of omics data.
            labels (ndarray): Labels for the omics data.
            ae_params (dict): Parameters for the Autoencoder.
            mrmr_params (dict): Parameters for the mRMR feature selection.
        """
        self.omics_dict = omics_dict
        self.labels = labels
        self.ae_params = ae_params
        self.mrmr_params = mrmr_params

    def fit_transform(self):
        """
        Perform mRMR feature selection followed by Autoencoder-based dimensionality reduction.

        Returns:
            DataFrame: Selected mRMR features.
            DataFrame: Embedded latent representations.
        """
        omics_dict = self.omics_dict
        labels = self.labels
        mrmr_params = self.mrmr_params
        ae_params = self.ae_params

        # Instantiate the Multi_mRMR object with k=100
        print("omics-specific mRMR begin")
        multi_mrmr = Multi_mRMR(omics_dict, labels, k=mrmr_params['k'])

        # Perform mrmr feature selection and merge the selected features
        mRMR_features = multi_mrmr.fit_transform(omics_dict, labels, p_dict=mrmr_params['p_dict'])

        # Instantiate the Autoencoder object with input_dim equal to the number of selected features
        autoencoder = Autoencoder(input_dim=mRMR_features.shape[1], **ae_params)
        print("omics-specific mRMR end")
        # Fit the Autoencoder on the selected features
        print("Auto-encoder has ready")
        autoencoder.fit(mRMR_features)

        # Extract the reduced features using the Autoencoder
        embedded_z = autoencoder.extract_features(mRMR_features)
        print("Auto-encoder end")
        embedded_z = pd.DataFrame(embedded_z)
        embedded_z = embedded_z.rename(columns=lambda x: 'hidden{}'.format(x))
        embedded_z.index = mRMR_features.index

        return mRMR_features, embedded_z


