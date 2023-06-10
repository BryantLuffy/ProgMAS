

import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold


class MultiomicsDataProcessor:
    def __init__(self):
        self.intersection = None
        self.data = {}

    def calculate_intersection(self, survival_data, *omics_data):
        intersection = set(survival_data.index)
        for data in omics_data:
            intersection = intersection.intersection(data.T.index)
        self.intersection = intersection
        self.survival_data = survival_data.loc[self.intersection]
    def get_intersection_data(self, data):
        return data.T.loc[self.intersection]

    def remove_missing_features(self, omics_data, missing_threshold):
        data = omics_data.copy()
        missing_percentages = data.isnull().mean()
        columns_to_remove = missing_percentages[missing_percentages > missing_threshold].index
        data.drop(columns=columns_to_remove, inplace=True)
        return data

    def impute_missing_values(self, omics_data):
        imputer = KNNImputer(n_neighbors=5)
        data = omics_data.copy()
        data_imputed = pd.DataFrame(imputer.fit_transform(data), index=data.index, columns=data.columns)
        return data_imputed

    def apply_variance_filter(self, omics_data, variance_threshold):
        selector = VarianceThreshold(threshold=variance_threshold)
        data = omics_data.copy()
        data_filtered = pd.DataFrame(selector.fit_transform(data), index=data.index)
        selected_features = data.columns[selector.get_support()]
        data_filtered.columns = selected_features
        return data_filtered, selected_features

    def min_max_normalize(self, omics_data):
        scaler = MinMaxScaler()
        data = omics_data.copy()
        data_normalized = pd.DataFrame(scaler.fit_transform(data), index=data.index, columns=data.columns)
        return data_normalized

    def process_multiomics_data(self, survival_data, omics_data, missing_thresholds, variance_thresholds):
        self.calculate_intersection(survival_data, *omics_data.values())

        for omics_name, omics_data in omics_data.items():
            # Step 1: Get intersection data
            intersection_data = self.get_intersection_data(omics_data)

            # Step 2: Remove missing features
            missing_threshold = missing_thresholds.get(omics_name, 0)
            omics_data_filtered = self.remove_missing_features(intersection_data, missing_threshold)

            # Step 3: Impute missing values
            omics_data_imputed = self.impute_missing_values(omics_data_filtered)

            # Step 4: Apply variance filter
            variance_threshold = variance_thresholds.get(omics_name, 0)
            omics_data_filtered, selected_features = self.apply_variance_filter(omics_data_imputed, variance_threshold)

            # Step 5: Min-max normalization
            omics_data_normalized = self.min_max_normalize(omics_data_filtered)

            self.data[omics_name] = omics_data_normalized

    def get_processed_data(self, omics_name):
        return self.data.get(omics_name, None)


