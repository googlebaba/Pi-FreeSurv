import pandas as pd
class Data:
    def __init__(self):
        #self.data_name = data_name
        pass
    def get_data(self, data_name, set_name, duration_col, event_col, outcome="OS"):
        if data_name == 'HCC':
            if set_name == 'train':
                train = pd.read_csv("./datasets/HCC/HCC_train_zscore_top400.csv", index_col=0)
                train[duration_col] = train[duration_col]/30.0
                return train[duration_col],  train[event_col], train.drop([duration_col, event_col], axis=1)
            elif set_name == "test1":
                test1 = pd.read_csv("./datasets/HCC/HCC_test1_zscore_top400.csv", index_col=0)
                return test1[duration_col],  test1[event_col], test1.drop([duration_col, event_col], axis=1)
            elif set_name == "test2":
                test2 = pd.read_csv("./datasets/HCC/HCC_test2_zscore_top400.csv", index_col=0)
                return test2[duration_col],  test2[event_col], test2.drop([duration_col, event_col], axis=1)

        elif data_name == 'BreastCancer':
            if outcome=="OS":
                if set_name == 'train':
                    train = pd.read_csv("./datasets/breast_cancer/train_Cox_zscore_top400_1030.csv", index_col=0)
                    train = train.drop(["RFS.months", "RFS.status"], axis=1)
                    return train[duration_col],  train[event_col], train.drop([duration_col, event_col], axis=1)
                elif set_name == "test":
                    test1 = pd.read_csv("./datasets/breast_cancer/test1_Cox_zscore_top400_1030.csv", index_col=0)
                    test1 = test1.drop(["RFS.months", "RFS.status"], axis=1)
                    return test1[duration_col],  test1[event_col], test1.drop([duration_col, event_col], axis=1)
            if outcome=="RFS":
                if set_name == 'train':
                    train = pd.read_csv("./datasets/breast_cancer/train_Cox_zscore_top400_1030.csv", index_col=0)
                    train = train.drop(["Survival.months", "Survival.status"], axis=1)
                    return train[duration_col],  train[event_col], train.drop([duration_col, event_col], axis=1)
                elif set_name == "test":
                    test1 = pd.read_csv("./datasets/breast_cancer/test1_Cox_zscore_top400_1030.csv", index_col=0)
                    test1 = test1.drop(["Survival.months", "Survival.status"], axis=1)
                    return test1[duration_col],  test1[event_col], test1.drop([duration_col, event_col], axis=1)

        elif data_name == 'PDAC':
            if outcome=="OS":
                if set_name == 'train':
                    train = pd.read_csv("./datasets/PDAC/Hyeon_survival_PDAC_400.csv", index_col=0)
                    train = train.drop(["Recurr.months", "Recurr.status"], axis=1)
                    return train[duration_col],  train[event_col], train.drop([duration_col, event_col], axis=1)
                elif set_name == "test":
                    test1 = pd.read_csv("./datasets/PDAC/Cao_survival_PDAC_400.csv", index_col=0)
                    test1 = test1.drop(["Recurr.months", "Recurr.status"], axis=1)
                    return test1[duration_col],  test1[event_col], test1.drop([duration_col, event_col], axis=1)
            if outcome=="RFS":
                if set_name == 'train':
                    train = pd.read_csv("./datasets/PDAC/Hyeon_survival_PDAC_400.csv", index_col=0)
                    train = train.drop(["Survival.months", "Survival.status"], axis=1)
                    return train[duration_col],  train[event_col], train.drop([duration_col, event_col], axis=1)
                elif set_name == "test":
                    test1 = pd.read_csv("./datasets/PDAC/Hyeon_survival_PDAC_400.csv", index_col=0)
                    test1 = test1.drop(["Survival.months", "Survival.status"], axis=1)
                    return test1[duration_col],  test1[event_col], test1.drop([duration_col, event_col], axis=1)
        else:
            print("Wrong data! Data name is invalid!")












