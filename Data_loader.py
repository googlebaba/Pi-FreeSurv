import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
class Data:
    def __init__(self, data_name = 'HCC', outcome="OS", duration_col="Survival.months", event_col="Survival.status"):

        if data_name == 'HCC':
            train = pd.read_csv("./datasets/HCC/HCC_train_zscore_top400.csv", index_col=0)
            train[duration_col] = train[duration_col]/30.0

            test1 = pd.read_csv("./datasets/HCC/HCC_test1_zscore_top400.csv", index_col=0)
            test2 = pd.read_csv("./datasets/HCC/HCC_test2_zscore_top400.csv", index_col=0)
            train_sample = train
            train_label = train_sample[[duration_col, event_col]]
            max_val = train_sample[duration_col].max()
            test1_sample = test1[test1[duration_col]<=max_val]

            test1_label = test1_sample[[duration_col, event_col]]
            test1_sample = test1_sample.drop([duration_col, event_col], axis=1)

            test2_sample = test2[test2[duration_col]<=max_val]
            test2_label = test2_sample[[duration_col, event_col]]
            test2_sample = test2_sample.drop([duration_col, event_col], axis=1)

            train_sample = train_sample.drop([duration_col, event_col], axis=1)

            columns = train.columns
            scaler = StandardScaler()
            scaler.fit(train_sample.iloc[:, :])
            self.train = pd.DataFrame(np.concatenate((scaler.transform(train_sample.iloc[:, :]), train_label), axis=1), columns=columns)
            self.test1 = pd.DataFrame(np.concatenate((scaler.transform(test1_sample.iloc[:, :]), test1_label), axis=1), columns=columns)
            self.test2 = pd.DataFrame(np.concatenate((scaler.transform(test2_sample.iloc[:, :]), test2_label), axis=1), columns=columns)

        elif data_name == 'BreastCancer':
            if outcome=="OS":
                train = pd.read_csv("./datasets/breast_cancer/train_Cox_zscore_top400_1030.csv", index_col=0)
                train = train.drop(["RFS.months", "RFS.status"], axis=1)
                test1 = pd.read_csv("./datasets/breast_cancer/test1_Cox_zscore_top400_1030.csv", index_col=0)
                test1 = test1.drop(["RFS.months", "RFS.status"], axis=1)
             
            if outcome=="RFS":
                train = pd.read_csv("./datasets/breast_cancer/train_Cox_zscore_top400_1030.csv", index_col=0)
                train = train.drop(["Survival.months", "Survival.status"], axis=1)
                test1 = pd.read_csv("./datasets/breast_cancer/test1_Cox_zscore_top400_1030.csv", index_col=0)
                test1 = test1.drop(["Survival.months", "Survival.status"], axis=1)
            train_sample = train
            train_label = train_sample[[event_col, duration_col]]
            max_val = train_sample[duration_col].max()
            test1_sample = test1[test1[duration_col]<=max_val]
            
            test1_label = test1_sample[[event_col, duration_col]]
            test1_sample = test1_sample.drop([duration_col, event_col], axis=1)
            train_sample = train_sample.drop([duration_col, event_col], axis=1)

            columns = list(train.columns[:-2]) + [event_col, duration_col]
            
            scaler = StandardScaler()
            scaler.fit(train_sample.iloc[:, :])

            self.train = pd.DataFrame(np.concatenate((scaler.transform(train_sample.iloc[:, :]), train_label), axis=1), columns=columns)
            self.test1 = pd.DataFrame(np.concatenate((scaler.transform(test1_sample.iloc[:, :]), test1_label), axis=1), columns=columns)

        elif data_name == 'PDAC':
            if outcome=="OS":
                train = pd.read_csv("./datasets/PDAC/Hyeon_survival_PDAC_400.csv", index_col=0)
                train = train.drop(["Recurr.months", "Recurr.status"], axis=1)
                test1 = pd.read_csv("./datasets/PDAC/Cao_survival_PDAC_400.csv", index_col=0)
                test1 = test1.drop(["Recurr.months", "Recurr.status"], axis=1)
            if outcome=="RFS":
                train = pd.read_csv("./datasets/PDAC/Hyeon_survival_PDAC_400.csv", index_col=0)
                train = train.drop(["Survival.months", "Survival.status"], axis=1)
                test1 = pd.read_csv("./datasets/PDAC/Cao_survival_PDAC_400.csv", index_col=0)
                test1 = test1.drop(["Survival.months", "Survival.status"], axis=1)
            train_sample = train
            train_label = train_sample[[duration_col, event_col]]
            max_val = train_sample[duration_col].max()
            test1_sample = test1[test1[duration_col]<=max_val]
            
            test1_label = test1_sample[[duration_col, event_col]]
            test1_sample = test1_sample.drop([duration_col, event_col], axis=1)
            train_sample = train_sample.drop([duration_col, event_col], axis=1)

            columns = list(train.columns[2:]) + [duration_col, event_col]
            
            scaler = StandardScaler()
            scaler.fit(train_sample.iloc[:, :])
            self.train = pd.DataFrame(np.concatenate((scaler.transform(train_sample.iloc[:, :]), train_label), axis=1), columns=columns)
            self.test1 = pd.DataFrame(np.concatenate((scaler.transform(test1_sample.iloc[:, :]), test1_label), axis=1), columns=columns)

        #self.data_name = data_name
    def get_data(self, data_name, set_name, duration_col, event_col, outcome="OS"):
        if data_name == 'HCC':

            if set_name == 'train':
                return self.train[duration_col],  self.train[event_col], self.train.drop([duration_col, event_col], axis=1)
            elif set_name == "test1":
                return self.test1[duration_col],  self.test1[event_col], self.test1.drop([duration_col, event_col], axis=1)
            elif set_name == "test2":
                return self.test2[duration_col],  self.test2[event_col], self.test2.drop([duration_col, event_col], axis=1)

        elif data_name == 'BreastCancer':
            if outcome=="OS":
                if set_name == 'train':
                    return self.train[duration_col],  self.train[event_col], self.train.drop([duration_col, event_col], axis=1)
                elif set_name == "test":
                    return self.test1[duration_col],  self.test1[event_col], self.test1.drop([duration_col, event_col], axis=1)
            if outcome=="RFS":
                if set_name == 'train':
                    return self.train[duration_col],  self.train[event_col], self.train.drop([duration_col, event_col], axis=1)
                elif set_name == "test":
                    return self.test1[duration_col],  self.test1[event_col], self.test1.drop([duration_col, event_col], axis=1)

        elif data_name == 'PDAC':
            if outcome=="OS":
                if set_name == 'train':
                    return self.train[duration_col],  self.train[event_col], self.train.drop([duration_col, event_col], axis=1)
                elif set_name == "test":
                    return self.test1[duration_col],  self.test1[event_col], self.test1.drop([duration_col, event_col], axis=1)
            if outcome=="RFS":
                if set_name == 'train':
                    return self.train[duration_col],  self.train[event_col], self.train.drop([duration_col, event_col], axis=1)
                elif set_name == "test":
                    return self.test1[duration_col],  self.test1[event_col], self.test1.drop([duration_col, event_col], axis=1)
        else:
            print("Wrong data! Data name is invalid!")












