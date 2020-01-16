# coding: utf-8
# 读取各个数据集并做简单预处理

import numpy as np
import pandas as pd
from scipy.io import arff
import os
from tensorflow.examples.tutorials.mnist import input_data
from sklearn import preprocessing
import sklearn.datasets as dataset


def class_map_encode(feature):
    class_mapping = {label: idx for idx, label in enumerate(set(feature))}
    return feature.map(class_mapping)


def one_hot(feature):
    encode = pd.get_dummies(feature, prefix=None, prefix_sep='_', dummy_na=False,
                            columns=None, sparse=False, drop_first=False)
    return encode


def see_null(df):
    return df.isnull().any()


def see_feature_null(df, feature):
    return df[df[feature].isnull()]


def load_Car_Evaluation(return_X_y=True,
                        root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "Car_Evaluation"):
    train_file_name = root + os.sep + "car.data"
    train_data = pd.read_csv(train_file_name, header=None)
    train_data.loc[train_data[6] == "unacc", 6] = 0
    train_data.loc[train_data[6] == "acc", 6] = 1
    train_data.loc[train_data[6] == "good", 6] = 2
    train_data.loc[train_data[6] == "vgood", 6] = 3
    labels = train_data[6]
    del train_data[6]
    train_data = one_hot(train_data)

    if return_X_y:
        return np.array(train_data.values, np.float32), np.array(labels.values, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_letter_recognition_data_set(return_X_y=True,
                                     root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "letter_recognition_data_set"):
    train_file_name = root + os.sep + "data.csv"
    train_data = pd.read_csv(train_file_name, header=None)
    train_data.loc[train_data[0] == "A", 0] = 0
    train_data.loc[train_data[0] == "B", 0] = 1
    train_data.loc[train_data[0] == "C", 0] = 2
    train_data.loc[train_data[0] == "D", 0] = 3
    train_data.loc[train_data[0] == "E", 0] = 4
    train_data.loc[train_data[0] == "F", 0] = 5
    train_data.loc[train_data[0] == "G", 0] = 6
    train_data.loc[train_data[0] == "H", 0] = 7
    train_data.loc[train_data[0] == "I", 0] = 8
    train_data.loc[train_data[0] == "J", 0] = 9
    train_data.loc[train_data[0] == "K", 0] = 10
    train_data.loc[train_data[0] == "L", 0] = 11
    train_data.loc[train_data[0] == "M", 0] = 12
    train_data.loc[train_data[0] == "N", 0] = 13
    train_data.loc[train_data[0] == "O", 0] = 14
    train_data.loc[train_data[0] == "P", 0] = 15
    train_data.loc[train_data[0] == "Q", 0] = 16
    train_data.loc[train_data[0] == "R", 0] = 17
    train_data.loc[train_data[0] == "S", 0] = 18
    train_data.loc[train_data[0] == "T", 0] = 19
    train_data.loc[train_data[0] == "U", 0] = 20
    train_data.loc[train_data[0] == "V", 0] = 21
    train_data.loc[train_data[0] == "W", 0] = 22
    train_data.loc[train_data[0] == "X", 0] = 23
    train_data.loc[train_data[0] == "Y", 0] = 24
    train_data.loc[train_data[0] == "Z", 0] = 25

    labels = train_data[0]
    del train_data[0]

    if return_X_y:
        return np.array(train_data.values, np.float32), np.array(labels.values, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_Mushroom(return_X_y=True,
                  root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "Mushroom"):
    train_file_name = root + os.sep + "agaricus-lepiota.data"
    train_data = pd.read_csv(train_file_name, header=None)

    train_data.loc[train_data[0] == "e", 0] = 0
    train_data.loc[train_data[0] == "p", 0] = 1
    labels = train_data[0]
    del train_data[0]
    train_data = one_hot(train_data)
    if return_X_y:
        return np.array(train_data.values, np.float32), np.array(labels.values, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_Crowdsourced_Mapping_Data_Set(return_X_y=True,
                                       root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "Crowdsourced_Mapping_Data_Set"):
    train_file_name = root + os.sep + "training.csv"
    train_data = pd.read_csv(train_file_name, header=None)
    test_file_name = root + os.sep + "testing.csv"
    test_data = pd.read_csv(test_file_name, header=None)
    train_data.loc[train_data[0] == "impervious", 0] = 0
    train_data.loc[train_data[0] == "farm", 0] = 1
    train_data.loc[train_data[0] == "forest", 0] = 2
    train_data.loc[train_data[0] == "grass", 0] = 3
    train_data.loc[train_data[0] == "orchard", 0] = 4
    train_data.loc[train_data[0] == "water", 0] = 5

    test_data.loc[test_data[0] == "impervious", 0] = 0
    test_data.loc[test_data[0] == "farm", 0] = 1
    test_data.loc[test_data[0] == "forest", 0] = 2
    test_data.loc[test_data[0] == "grass", 0] = 3
    test_data.loc[test_data[0] == "orchard", 0] = 4
    test_data.loc[test_data[0] == "water", 0] = 5

    labels_train = train_data[0]
    del train_data[0]
    labels_test = test_data[0]
    del test_data[0]
    return np.array(train_data[1:].values, np.float32), np.array(labels_train[1:].values, np.int), \
           np.array(test_data[1:].values, np.float32), np.array(labels_test[1:].values, np.int)


def load_optdigits(return_X_y=True,
                   root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "optdigits"):
    train_file_name = root + os.sep + "optdigits.tra"
    test_file_name = root + os.sep + "optdigits.tes"

    train_data = pd.read_csv(train_file_name, header=None)
    test_data = pd.read_csv(test_file_name, header=None)

    all_data = pd.concat([train_data, test_data])

    all_labels = all_data.loc[:, 64]
    all_data = all_data.loc[:, :63]

    if return_X_y:
        y = np.array(all_labels.values, np.int)
        del all_labels
        X = np.array(all_data, np.float32)
        del all_data
        return X, y
    else:
        return train_data, test_data


def load_Mnist(return_X_y=True, root="datasets" + os.sep + "mnist"):
    mnist_data = input_data.read_data_sets('/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets/mnist')
    train_data = mnist_data.train.images
    labels_train = mnist_data.train.labels
    test_data = mnist_data.test.images
    labels_test = mnist_data.test.labels
    print(train_data.shape)  # 55000*784
    print(test_data.shape)  # 10000*784
    return np.array(train_data, np.float32), np.array(labels_train, np.int), \
           np.array(test_data, np.float32), np.array(labels_test, np.int)


def load_frogs_mfcc_data_set(return_X_y=True,
                             root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "frogs_mfcc"):
    train_file_name = root + os.sep + "data.csv"
    train_data = pd.read_csv(train_file_name, header=None)
    train_data.loc[train_data[0] == "AdenomeraAndre", 0] = 0
    train_data.loc[train_data[0] == "Ameeregatrivittata", 0] = 1
    train_data.loc[train_data[0] == "AdenomeraHylaedactylus", 0] = 2
    train_data.loc[train_data[0] == "HylaMinuta", 0] = 3
    train_data.loc[train_data[0] == "HypsiboasCinerascens", 0] = 4
    train_data.loc[train_data[0] == "HypsiboasCordobae", 0] = 5
    train_data.loc[train_data[0] == "LeptodactylusFuscus", 0] = 6
    train_data.loc[train_data[0] == "OsteocephalusOophagus", 0] = 7
    train_data.loc[train_data[0] == "Rhinellagranulosa", 0] = 8
    train_data.loc[train_data[0] == "ScinaxRuber", 0] = 9

    labels = train_data[0]
    del train_data[0]

    if return_X_y:
        return np.array(train_data.values, np.float32), np.array(labels.values, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_CTG_data_set(return_X_y=True,
                      root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "Cardiotocography"):
    train_file_name = root + os.sep + "data.csv"
    train_data = pd.read_csv(train_file_name, header=None)
    labels = train_data[0]
    del train_data[0]

    if return_X_y:
        return np.array(train_data.values, np.float32), np.array(labels.values, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_dr_debrecen_data_set(return_X_y=True,
                              root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "DR_Debrecen"):
    train_file_name = root + os.sep + "data.csv"
    train_data = pd.read_csv(train_file_name, header=None)
    labels = train_data[0]
    del train_data[0]

    if return_X_y:
        return np.array(train_data.values, np.float32), np.array(labels.values, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_phishing_websites_data_set(return_X_y=True,
                                    root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "phishing_websites"):
    train_file_name = root + os.sep + "data.csv"
    train_data = pd.read_csv(train_file_name, header=None)
    labels = train_data[0]
    del train_data[0]

    if return_X_y:
        return np.array(train_data.values, np.float32), np.array(labels.values, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_turkiye_student_evaluation_data_set(return_X_y=True,
                                             root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "Turkiye_Student_Evaluation"):
    train_file_name = root + os.sep + "data.csv"
    train_data = pd.read_csv(train_file_name, header=None)
    labels = train_data[0]
    del train_data[0]

    if return_X_y:
        return np.array(train_data.values, np.float32), np.array(labels.values, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_winequality_white_data_set(return_X_y=True,
                                    root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "winequality_white"):
    train_file_name = root + os.sep + "data.csv"
    train_data = pd.read_csv(train_file_name, header=None)
    labels = train_data[0]
    del train_data[0]

    if return_X_y:
        return np.array(train_data.values, np.float32), np.array(labels.values, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_cv_firm_teacher_data_set(return_X_y=True,
                                  root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "ClaveVectors_Firm_Teacher_Model"):
    train_file_name = root + os.sep + "data.csv"
    train_data = pd.read_csv(train_file_name, header=None)
    labels = train_data[0]
    del train_data[0]

    if return_X_y:
        return np.array(train_data.values, np.float32), np.array(labels.values, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_htru_data_set(return_X_y=True,
                       root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "HTRU_2"):
    train_file_name = root + os.sep + "data.csv"
    train_data = pd.read_csv(train_file_name, header=None)
    labels = train_data[0]
    del train_data[0]

    if return_X_y:
        return np.array(train_data.values, np.float32), np.array(labels.values, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_wilt_data_set(return_X_y=True,
                       root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "Wilt"):
    train_file_name = root + os.sep + "training.csv"
    train_data = pd.read_csv(train_file_name, header=None)
    test_file_name = root + os.sep + "testing.csv"
    test_data = pd.read_csv(test_file_name, header=None)
    train_data.loc[train_data[0] == "w", 0] = 0
    train_data.loc[train_data[0] == "n", 0] = 1

    test_data.loc[test_data[0] == "w", 0] = 0
    test_data.loc[test_data[0] == "n", 0] = 1

    labels_train = train_data[0]
    del train_data[0]
    labels_test = test_data[0]
    del test_data[0]
    return np.array(train_data[1:].values, np.float32), np.array(labels_train[1:].values, np.int), \
           np.array(test_data[1:].values, np.float32), np.array(labels_test[1:].values, np.int)


def load_image_segmentation_data_set(return_X_y=True,
                                     root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "Image_Segmentation"):
    train_file_name = root + os.sep + "training.csv"
    train_data = pd.read_csv(train_file_name, header=None)
    test_file_name = root + os.sep + "testing.csv"
    test_data = pd.read_csv(test_file_name, header=None)
    train_data.loc[train_data[0] == "GRASS", 0] = 0
    train_data.loc[train_data[0] == "PATH", 0] = 1
    train_data.loc[train_data[0] == "WINDOW", 0] = 2
    train_data.loc[train_data[0] == "CEMENT", 0] = 3
    train_data.loc[train_data[0] == "FOLIAGE", 0] = 4
    train_data.loc[train_data[0] == "SKY", 0] = 5
    train_data.loc[train_data[0] == "BRICKFACE", 0] = 6

    test_data.loc[test_data[0] == "GRASS", 0] = 0
    test_data.loc[test_data[0] == "PATH", 0] = 1
    test_data.loc[test_data[0] == "WINDOW", 0] = 2
    test_data.loc[test_data[0] == "CEMENT", 0] = 3
    test_data.loc[test_data[0] == "FOLIAGE", 0] = 4
    test_data.loc[test_data[0] == "SKY", 0] = 5
    test_data.loc[test_data[0] == "BRICKFACE", 0] = 6

    labels_train = train_data[0]
    del train_data[0]
    labels_test = test_data[0]
    del test_data[0]
    return np.array(train_data[1:].values, np.float32), np.array(labels_train[1:].values, np.int), \
           np.array(test_data[1:].values, np.float32), np.array(labels_test[1:].values, np.int)


def load_pr_handwritten_data_set(return_X_y=True,
                                 root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "Pen_Recognition_Handwritten"):
    train_file_name = root + os.sep + "training.csv"
    train_data = pd.read_csv(train_file_name, header=None)
    test_file_name = root + os.sep + "testing.csv"
    test_data = pd.read_csv(test_file_name, header=None)
    labels_train = train_data[0]
    del train_data[0]
    labels_test = test_data[0]
    del test_data[0]
    return np.array(train_data[1:].values, np.float32), np.array(labels_train[1:].values, np.int), \
           np.array(test_data[1:].values, np.float32), np.array(labels_test[1:].values, np.int)


def load_abalone_data_set(return_X_y=True,
                          root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "abalone"):
    train_file_name = root + os.sep + "abalone.data"
    train_data = pd.read_csv(train_file_name, header=None)

    train_data.loc[train_data[0] == "M", 0] = 0
    train_data.loc[train_data[0] == "F", 0] = 1
    train_data.loc[train_data[0] == "I", 0] = 2
    labels = train_data[train_data.shape[1] - 1]
    del train_data[train_data.shape[1] - 1]
    # train_data = one_hot(train_data)
    if return_X_y:
        return np.array(train_data.values, np.float32), np.array(labels.values, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_glass_identify_data_set(return_X_y=True,
                                 root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "glass_identify"):
    train_file_name = root + os.sep + "glass.data"
    train_data = pd.read_csv(train_file_name, header=None)
    labels = train_data[train_data.shape[1] - 1]
    del train_data[0], train_data[train_data.shape[1] - 1]
    # train_data = one_hot(train_data)
    if return_X_y:
        return np.array(train_data.values, np.float32), np.array(labels.values, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_ILPD_data_set(return_X_y=True,
                       root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "ILPD"):
    train_file_name = root + os.sep + "ILPD.csv"
    train_data = pd.read_csv(train_file_name, header=None)
    labels = train_data[train_data.shape[1] - 1]
    train_data.loc[train_data[1] == "Female", 1] = 0
    train_data.loc[train_data[1] == "Male", 1] = 1
    del train_data[train_data.shape[1] - 1]
    columns = train_data.columns.tolist()
    for c in columns:
        d = train_data[c]
        MAX = d.max()
        MIN = d.min()
        train_data[c] = ((d - MIN) / (MAX - MIN)).tolist()
    if return_X_y:
        return np.array(train_data.values, np.float32), np.array(labels.values, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_magic_data_set(return_X_y=True,
                        root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "magic"):
    train_file_name = root + os.sep + "magic.data"
    train_data = pd.read_csv(train_file_name, header=None)
    train_data.loc[train_data[train_data.shape[1] - 1] == "g", train_data.shape[1] - 1] = 0
    train_data.loc[train_data[train_data.shape[1] - 1] == "h", train_data.shape[1] - 1] = 1
    labels = train_data[train_data.shape[1] - 1]
    del train_data[train_data.shape[1] - 1]
    columns = train_data.columns.tolist()
    for c in columns:
        d = train_data[c]
        MAX = d.max()
        MIN = d.min()
        train_data[c] = ((d - MIN) / (MAX - MIN)).tolist()
    if return_X_y:
        return np.array(train_data.values, np.float32), np.array(labels.values, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_tic_data_set(return_X_y=True,
                      root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "tic_tac_toc"):
    train_file_name = root + os.sep + "tic_tac_toe.data"
    train_data = pd.read_csv(train_file_name, header=None)
    for i in range(train_data.shape[1]):
        train_data.loc[train_data[i] == "x", i] = 0
        train_data.loc[train_data[i] == "o", i] = 1
        train_data.loc[train_data[i] == "b", i] = 2
    train_data.loc[train_data[train_data.shape[1] - 1] == "positive", train_data.shape[1] - 1] = 0
    train_data.loc[train_data[train_data.shape[1] - 1] == "negative", train_data.shape[1] - 1] = 1
    labels = train_data[train_data.shape[1] - 1]
    del train_data[train_data.shape[1] - 1]
    if return_X_y:
        return np.array(train_data.values, np.float32), np.array(labels.values, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_blood_data_set(return_X_y=True,
                        root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "blood"):
    train_file_name = root + os.sep + "blood.arff"
    data = arff.loadarff(train_file_name)
    train_data = pd.DataFrame(data[0])
    # train_data.loc[train_data['Class'] == "b'1'"] = 0
    # train_data.loc[train_data['Class'] == "b'2'"] = 1
    labels = train_data['Class']
    del train_data['Class']
    if return_X_y:
        return np.array(train_data.values, np.float32), np.array(labels.values, np.int)


def load_mfeat_data_set(return_X_y=True,
                        root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "mfeat"):
    train_file_name = root + os.sep + "mfeat.arff"
    data = arff.loadarff(train_file_name)
    train_data = pd.DataFrame(data[0])
    # train_data.loc[train_data['Class'] == "b'1'"] = 0
    # train_data.loc[train_data['Class'] == "b'2'"] = 1
    labels = train_data['class']
    del train_data['class']
    # train_data = train_data[1:]
    columns = train_data[0:1]
    for c in columns:
        d = train_data[c]
        MAX = d.max()
        MIN = d.min()
        train_data[c] = ((d - MIN) / (MAX - MIN)).tolist()
    if return_X_y:
        return np.array(train_data.values, np.float32), np.array(labels.values, np.int)


def load_ilpd_data_set(return_X_y=True,
                       root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "ilpd"):
    train_file_name = root + os.sep + "ilpd.csv"
    train_data = pd.read_csv(train_file_name, header=None)
    train_data = train_data[1:]
    train_data.loc[train_data[1] == "Female", 1] = 0
    train_data.loc[train_data[1] == "Male", 1] = 1
    train_data = train_data.astype("float")
    labels = train_data[train_data.shape[1] - 1]
    del train_data[train_data.shape[1] - 1]
    columns = train_data.columns.tolist()
    for c in columns:
        d = train_data[c]
        MAX = d.max()
        MIN = d.min()
        train_data[c] = ((d - MIN) / (MAX - MIN)).tolist()
    if return_X_y:
        return np.array(train_data.values, np.float32), np.array(labels.values, np.int)


def load_churn_data_set(return_X_y=True,
                        root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "churn"):
    train_file_name = root + os.sep + "churn.csv"
    train_data = pd.read_csv(train_file_name, header=None)
    train_data = train_data[1:]
    train_data = train_data.astype("float")
    labels = train_data[train_data.shape[1] - 1]
    del train_data[train_data.shape[1] - 1]
    columns = train_data.columns.tolist()
    for c in columns:
        d = train_data[c]
        MAX = d.max()
        MIN = d.min()
        train_data[c] = ((d - MIN) / (MAX - MIN)).tolist()
    if return_X_y:
        return np.array(train_data.values, np.float32), np.array(labels.values, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_miniboone_data_set(return_X_y=True,
                            root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "miniboone"):
    train_file_name = root + os.sep + "MiniBooNE.csv"
    train_data = pd.read_csv(train_file_name, header=None)
    labels = train_data[0]
    temp = train_data[1]
    train_data = train_data[train_data[1] != -999.0]
    del train_data[0]
    train_data = train_data.astype("float")
    columns = train_data.columns.tolist()
    for c in columns:
        d = train_data[c]
        MAX = d.max()
        MIN = d.min()
        train_data[c] = ((d - MIN) / (MAX - MIN)).tolist()
    if return_X_y:
        return np.array(train_data.values, np.float32), np.array(labels.values, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_airline_data_set(return_X_y=True,
                          root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "airline"):
    train_file_name = root + os.sep + "airline.csv"
    train_data = pd.read_csv(train_file_name, header=None)
    # train_data = train_data[:train_data.shape[1]]

    mapping_0 = {label: idx for idx, label in enumerate(set(train_data[0]))}
    train_data[0] = train_data[0].map(mapping_0)
    mapping_2 = {label: idx for idx, label in enumerate(set(train_data[2]))}
    train_data[2] = train_data[2].map(mapping_2)
    mapping_3 = {label: idx for idx, label in enumerate(set(train_data[3]))}
    train_data[3] = train_data[3].map(mapping_3)

    train_data = train_data.astype("float")
    labels = train_data[train_data.shape[1] - 1]
    del train_data[train_data.shape[1] - 1]
    # columns = train_data.columns.tolist()
    # for c in columns:
    #     d = train_data[c]
    #     d.unique()
    #     MAX = d.max()
    #     MIN = d.min()
    #     train_data[c] = ((d - MIN) / (MAX - MIN)).tolist()
    if return_X_y:
        return np.array(train_data.values, np.float32), np.array(labels.values, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_banknote_data_set(return_X_y=True,
                           root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "banknote"):
    train_file_name = root + os.sep + "banknote.csv"
    train_data = pd.read_csv(train_file_name, header=None)
    train_data = train_data[1:]
    train_data = train_data.astype("float")
    labels = train_data[train_data.shape[1] - 1]
    del train_data[train_data.shape[1] - 1]
    columns = train_data.columns.tolist()
    for c in columns:
        d = train_data[c]
        MAX = d.max()
        MIN = d.min()
        train_data[c] = ((d - MIN) / (MAX - MIN)).tolist()
    if return_X_y:
        return np.array(train_data.values, np.float32), np.array(labels.values, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_vowel_data_set(return_X_y=True,
                        root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "vowel"):
    train_file_name = root + os.sep + "vowel.csv"
    train_data = pd.read_csv(train_file_name, header=None)
    train_data = train_data[1:]
    mapping_0 = {label: idx for idx, label in enumerate(set(train_data[0]))}
    train_data[0] = train_data[0].map(mapping_0)
    mapping_1 = {label: idx for idx, label in enumerate(set(train_data[1]))}
    train_data[1] = train_data[1].map(mapping_1)
    mapping_12 = {label: idx for idx, label in enumerate(set(train_data[12]))}
    train_data[12] = train_data[12].map(mapping_12)
    train_data = train_data.astype("float")
    labels = train_data[train_data.shape[1] - 1]
    del train_data[train_data.shape[1] - 1]
    if return_X_y:
        return np.array(train_data.values, np.float32), np.array(labels.values, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_fri_c2_500_data_set(return_X_y=True,
                             root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "fri_c2_500"):
    train_file_name = root + os.sep + "fri_c2_500_5.csv"
    train_data = pd.read_csv(train_file_name, header=None)
    train_data = train_data[1:]
    mapping_5 = {label: idx for idx, label in enumerate(set(train_data[5]))}
    train_data[5] = train_data[5].map(mapping_5)
    train_data = train_data.astype("float")
    labels = train_data[train_data.shape[1] - 1]
    del train_data[train_data.shape[1] - 1]
    if return_X_y:
        return np.array(train_data.values, np.float32), np.array(labels.values, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_disclosure_data_set(return_X_y=True,
                             root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "disclosure"):
    train_file_name = root + os.sep + "disclosure_z.csv"
    train_data = pd.read_csv(train_file_name, header=None)
    train_data = train_data[1:]
    mapping_3 = {label: idx for idx, label in enumerate(set(train_data[3]))}
    train_data[3] = train_data[3].map(mapping_3)
    train_data = train_data.astype("float")
    labels = train_data[train_data.shape[1] - 1]
    del train_data[train_data.shape[1] - 1]
    columns = train_data.columns.tolist()
    for c in columns:
        d = train_data[c]
        MAX = d.max()
        MIN = d.min()
        train_data[c] = ((d - MIN) / (MAX - MIN)).tolist()
    if return_X_y:
        return np.array(train_data.values, np.float32), np.array(labels.values, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_socmob_data_set(return_X_y=True,
                         root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "socmob"):
    train_file_name = root + os.sep + "socmob.csv"
    train_data = pd.read_csv(train_file_name, header=None)
    train_data = train_data[1:]
    mapping_0 = {label: idx for idx, label in enumerate(set(train_data[0]))}
    train_data[0] = train_data[0].map(mapping_0)
    mapping_1 = {label: idx for idx, label in enumerate(set(train_data[1]))}
    train_data[1] = train_data[1].map(mapping_1)
    mapping_2 = {label: idx for idx, label in enumerate(set(train_data[2]))}
    train_data[2] = train_data[2].map(mapping_2)
    mapping_3 = {label: idx for idx, label in enumerate(set(train_data[3]))}
    train_data[3] = train_data[3].map(mapping_3)
    mapping_5 = {label: idx for idx, label in enumerate(set(train_data[5]))}
    train_data[5] = train_data[5].map(mapping_5)
    train_data = train_data.astype("float")
    labels = train_data[train_data.shape[1] - 1]
    del train_data[train_data.shape[1] - 1]
    c = 4
    d = train_data[c]
    MAX = d.max()
    MIN = d.min()
    train_data[c] = ((d - MIN) / (MAX - MIN)).tolist()
    if return_X_y:
        return np.array(train_data.values, np.float32), np.array(labels.values, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_autouniv_data_set(return_X_y=True,
                           root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "autoUniv"):
    train_file_name = root + os.sep + "autoUniv.csv"
    train_data = pd.read_csv(train_file_name, header=None)
    train_data = train_data[1:]

    mapping_4 = {label: idx for idx, label in enumerate(set(train_data[4]))}
    train_data[4] = train_data[4].map(mapping_4)
    mapping_5 = {label: idx for idx, label in enumerate(set(train_data[5]))}
    train_data[5] = train_data[5].map(mapping_5)
    mapping_7 = {label: idx for idx, label in enumerate(set(train_data[7]))}
    train_data[7] = train_data[7].map(mapping_7)
    mapping_9 = {label: idx for idx, label in enumerate(set(train_data[9]))}
    train_data[9] = train_data[9].map(mapping_9)
    mapping_12 = {label: idx for idx, label in enumerate(set(train_data[12]))}
    train_data[12] = train_data[12].map(mapping_12)

    train_data = train_data.astype("float")
    labels = train_data[train_data.shape[1] - 1]
    del train_data[train_data.shape[1] - 1]
    columns = train_data.columns.tolist()
    for c in columns:
        d = train_data[c]
        MAX = d.max()
        MIN = d.min()
        train_data[c] = ((d - MIN) / (MAX - MIN)).tolist()
    if return_X_y:
        return np.array(train_data.values, np.float32), np.array(labels.values, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_autouniv700_data_set(return_X_y=True,
                              root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "autoUniv-700"):
    train_file_name = root + os.sep + "autouniv.csv"
    train_data = pd.read_csv(train_file_name, header=None)
    train_data = train_data[1:]

    mapping_4 = {label: idx for idx, label in enumerate(set(train_data[4]))}
    train_data[4] = train_data[4].map(mapping_4)
    mapping_5 = {label: idx for idx, label in enumerate(set(train_data[5]))}
    train_data[5] = train_data[5].map(mapping_5)
    mapping_7 = {label: idx for idx, label in enumerate(set(train_data[7]))}
    train_data[7] = train_data[7].map(mapping_7)
    mapping_9 = {label: idx for idx, label in enumerate(set(train_data[9]))}
    train_data[9] = train_data[9].map(mapping_9)
    mapping_12 = {label: idx for idx, label in enumerate(set(train_data[12]))}
    train_data[12] = train_data[12].map(mapping_12)

    train_data = train_data.astype("float")
    labels = train_data[train_data.shape[1] - 1]
    del train_data[train_data.shape[1] - 1]
    columns = train_data.columns.tolist()
    for c in columns:
        d = train_data[c]
        MAX = d.max()
        MIN = d.min()
        train_data[c] = ((d - MIN) / (MAX - MIN)).tolist()
    if return_X_y:
        return np.array(train_data.values, np.float32), np.array(labels.values, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_acute_inflammation_data_set(return_X_y=True,
                                     root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "acute_inflammation"):
    train_file_name = root + os.sep + "acute-inflammation_R.dat"
    train_data = pd.read_table(train_file_name)
    train_data = train_data.values
    labels = train_data[:, train_data.shape[1] - 1]
    train_data = train_data[:, [1, 2, 3, 4, 5, 6]]
    # train_data = train_data[1:train_data.shape[1] - 1]
    # train_data = train_data.astype("float")
    if return_X_y:
        return np.array(train_data, np.float32), np.array(labels, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_acute_nephritis_data_set(return_X_y=True,
                                  root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "acute_nephritis"):
    train_file_name = root + os.sep + "acute-nephritis.dat"
    train_data = pd.read_table(train_file_name)
    train_data = train_data[0:]
    train_data = train_data.values
    labels = train_data[:, train_data.shape[1] - 1]
    train_data = train_data[:, [1, 2, 3, 4, 5, 6, 7]]
    # train_data = train_data[1:train_data.shape[1] - 1]
    # train_data = train_data.astype("float")
    if return_X_y:
        return np.array(train_data, np.float32), np.array(labels, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_analcatdata_asbestos_data_set(return_X_y=True,
                                       root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "analcatdata_asbestos"):
    train_file_name = root + os.sep + "analcatdata_asbestos.csv"
    train_data = pd.read_csv(train_file_name, header=None)

    train_data = train_data[1:]

    mapping_0 = {label: idx for idx, label in enumerate(set(train_data[0]))}
    train_data[0] = train_data[0].map(mapping_0)
    mapping_2 = {label: idx for idx, label in enumerate(set(train_data[2]))}
    train_data[2] = train_data[2].map(mapping_2)
    mapping_3 = {label: idx for idx, label in enumerate(set(train_data[3]))}
    train_data[3] = train_data[3].map(mapping_3)

    train_data = train_data.astype("float")
    labels = train_data[train_data.shape[1] - 1]
    del train_data[train_data.shape[1] - 1]
    columns = train_data.columns.tolist()
    for c in columns:
        d = train_data[c]
        MAX = d.max()
        MIN = d.min()
        train_data[c] = ((d - MIN) / (MAX - MIN)).tolist()
    if return_X_y:
        return np.array(train_data.values, np.float32), np.array(labels.values, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_analcatdata_boxing_data_set(return_X_y=True,
                                     root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "analcatdata_boxing"):
    train_file_name = root + os.sep + "analcatdata_boxing1.csv"
    train_data = pd.read_csv(train_file_name, header=None)

    train_data = train_data[1:]

    mapping_0 = {label: idx for idx, label in enumerate(set(train_data[0]))}
    train_data[0] = train_data[0].map(mapping_0)
    mapping_1 = {label: idx for idx, label in enumerate(set(train_data[1]))}
    train_data[1] = train_data[1].map(mapping_1)
    mapping_3 = {label: idx for idx, label in enumerate(set(train_data[3]))}
    train_data[3] = train_data[3].map(mapping_3)

    train_data = train_data.astype("float")
    labels = train_data[train_data.shape[1] - 1]
    del train_data[train_data.shape[1] - 1]

    if return_X_y:
        return np.array(train_data.values, np.float32), np.array(labels.values, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_analcatdata_broadwaymult_data_set(return_X_y=True,
                                           root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "analcatdata_broadwaymult"):
    train_file_name = root + os.sep + "analcatdata_broadwaymult.csv"
    train_data = pd.read_csv(train_file_name, header=None)

    train_data = train_data[1:]

    mapping_0 = {label: idx for idx, label in enumerate(set(train_data[0]))}
    train_data[0] = train_data[0].map(mapping_0)
    mapping_1 = {label: idx for idx, label in enumerate(set(train_data[1]))}
    train_data[1] = train_data[1].map(mapping_1)
    mapping_2 = {label: idx for idx, label in enumerate(set(train_data[2]))}
    train_data[2] = train_data[2].map(mapping_2)
    mapping_6 = {label: idx for idx, label in enumerate(set(train_data[6]))}
    train_data[6] = train_data[6].map(mapping_6)

    train_data = train_data.astype("float")
    labels = train_data[train_data.shape[1] - 1]
    del train_data[train_data.shape[1] - 1]

    if return_X_y:
        return np.array(train_data.values, np.float32), np.array(labels.values, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_analcatdata_germangss_data_set(return_X_y=True,
                                        root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "analcatdata_germangss"):
    train_file_name = root + os.sep + "analcatdata_germangss.csv"
    train_data = pd.read_csv(train_file_name, header=None)

    train_data = train_data[1:]

    mapping_0 = {label: idx for idx, label in enumerate(set(train_data[0]))}
    train_data[0] = train_data[0].map(mapping_0)
    mapping_1 = {label: idx for idx, label in enumerate(set(train_data[1]))}
    train_data[1] = train_data[1].map(mapping_1)
    mapping_2 = {label: idx for idx, label in enumerate(set(train_data[2]))}
    train_data[2] = train_data[2].map(mapping_2)
    mapping_3 = {label: idx for idx, label in enumerate(set(train_data[3]))}
    train_data[3] = train_data[3].map(mapping_3)
    mapping_4 = {label: idx for idx, label in enumerate(set(train_data[4]))}
    train_data[4] = train_data[4].map(mapping_4)

    train_data = train_data.astype("float")
    labels = train_data[train_data.shape[1] - 1]
    del train_data[train_data.shape[1] - 1]

    if return_X_y:
        return np.array(train_data.values, np.float32), np.array(labels.values, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_analcatdata_lawsuit_data_set(return_X_y=True,
                                      root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "analcatdata_lawsuit"):
    train_file_name = root + os.sep + "analcatdata_lawsuit.csv"
    train_data = pd.read_csv(train_file_name, header=None)

    train_data = train_data[1:]

    train_data = train_data.astype("float")
    labels = train_data[train_data.shape[1] - 1]
    del train_data[train_data.shape[1] - 1]

    if return_X_y:
        return np.array(train_data.values, np.float32), np.array(labels.values, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_annealing_data_set(return_X_y=True,
                            root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "annealing"):
    train_file_name = root + os.sep + "annealing.dat"
    train_data = pd.read_table(train_file_name)

    train_data = train_data[0:]
    train_data = train_data.values
    labels = train_data[:, train_data.shape[1] - 1]
    train_data = train_data[:, [i for i in range(1, train_data.shape[1] - 1)]]
    # train_data = train_data[1:train_data.shape[1] - 1]
    # train_data = train_data.astype("float")
    if return_X_y:
        return np.array(train_data, np.float32), np.array(labels, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_ar4_data_set(return_X_y=True,
                      root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "ar4"):
    train_file_name = root + os.sep + "ar4.csv"
    train_data = pd.read_csv(train_file_name, header=None)

    train_data = train_data[1:]

    mapping_n = {label: idx for idx, label in enumerate(set(train_data[train_data.shape[1] - 1]))}
    train_data[train_data.shape[1] - 1] = train_data[train_data.shape[1] - 1].map(mapping_n)

    train_data = train_data.astype("float")
    labels = train_data[train_data.shape[1] - 1]
    del train_data[train_data.shape[1] - 1]
    columns = train_data.columns.tolist()
    for c in columns:
        d = train_data[c]
        MAX = d.max()
        MIN = d.min()
        train_data[c] = ((d - MIN) / (MAX - MIN)).tolist()
    if return_X_y:
        return np.array(train_data.values, np.float32), np.array(labels.values, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_arrhythmia_data_set(return_X_y=True,
                             root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "arrhythmia"):
    train_file_name = root + os.sep + "arrhythmia_R.dat"
    train_data = pd.read_table(train_file_name)

    train_data = train_data[0:]
    train_data = train_data.values
    labels = train_data[:, train_data.shape[1] - 1]
    train_data = train_data[:, [i for i in range(1, train_data.shape[1] - 1)]]
    # train_data = train_data[1:train_data.shape[1] - 1]
    # train_data = train_data.astype("float")
    if return_X_y:
        return np.array(train_data, np.float32), np.array(labels, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_audiology_data_set(return_X_y=True,
                            root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "audiology_std"):
    train_file_name = root + os.sep + "audiology-std.dat"
    train_data = pd.read_table(train_file_name)

    train_data = train_data[0:]
    train_data = train_data.values
    labels = train_data[:, train_data.shape[1] - 1]
    train_data = train_data[:, [i for i in range(1, train_data.shape[1] - 1)]]
    # train_data = train_data[1:train_data.shape[1] - 1]
    # train_data = train_data.astype("float")
    if return_X_y:
        return np.array(train_data, np.float32), np.array(labels, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_balance_scale_data_set(return_X_y=True,
                                root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "balance_scale"):
    train_file_name = root + os.sep + "balance-scale_R.dat"
    train_data = pd.read_table(train_file_name)

    train_data = train_data[0:]
    train_data = train_data.values
    labels = train_data[:, train_data.shape[1] - 1]
    train_data = train_data[:, [i for i in range(1, train_data.shape[1] - 1)]]
    # train_data = train_data[1:train_data.shape[1] - 1]
    # train_data = train_data.astype("float")
    if return_X_y:
        return np.array(train_data, np.float32), np.array(labels, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_baseball_data_set(return_X_y=True,
                           root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "baseball"):
    train_file_name = root + os.sep + "baseball.csv"
    train_data = pd.read_csv(train_file_name, header=None)

    train_data = train_data[1:]

    mapping_15 = {label: idx for idx, label in enumerate(set(train_data[15]))}
    train_data[15] = train_data[15].map(mapping_15)

    train_data = train_data.astype("float")
    labels = train_data[train_data.shape[1] - 1]
    del train_data[train_data.shape[1] - 1]
    columns = train_data.columns.tolist()
    for c in columns:
        d = train_data[c]
        MAX = d.max()
        MIN = d.min()
        train_data[c] = ((d - MIN) / (MAX - MIN)).tolist()
    if return_X_y:
        return np.array(train_data.values, np.float32), np.array(labels.values, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_bodyfat_data_set(return_X_y=True,
                          root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "bodyfat"):
    train_file_name = root + os.sep + "bodyfat.csv"
    train_data = pd.read_csv(train_file_name, header=None)

    train_data = train_data[1:]

    mapping_n = {label: idx for idx, label in enumerate(set(train_data[train_data.shape[1] - 1]))}
    train_data[train_data.shape[1] - 1] = train_data[train_data.shape[1] - 1].map(mapping_n)

    train_data = train_data.astype("float")
    labels = train_data[train_data.shape[1] - 1]
    del train_data[train_data.shape[1] - 1]
    columns = train_data.columns.tolist()
    for c in columns:
        d = train_data[c]
        MAX = d.max()
        MIN = d.min()
        train_data[c] = ((d - MIN) / (MAX - MIN)).tolist()
    if return_X_y:
        return np.array(train_data.values, np.float32), np.array(labels.values, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_chatfield_4_data_set(return_X_y=True,
                              root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "chatfield_4"):
    train_file_name = root + os.sep + "chatfield_4.csv"
    train_data = pd.read_csv(train_file_name, header=None)

    train_data = train_data[1:]

    mapping_n = {label: idx for idx, label in enumerate(set(train_data[train_data.shape[1] - 1]))}
    train_data[train_data.shape[1] - 1] = train_data[train_data.shape[1] - 1].map(mapping_n)

    train_data = train_data.astype("float")
    labels = train_data[train_data.shape[1] - 1]
    del train_data[train_data.shape[1] - 1]
    columns = train_data.columns.tolist()
    for c in columns:
        d = train_data[c]
        MAX = d.max()
        MIN = d.min()
        train_data[c] = ((d - MIN) / (MAX - MIN)).tolist()
    if return_X_y:
        return np.array(train_data.values, np.float32), np.array(labels.values, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_chess_krvkp_data_set(return_X_y=True,
                              root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "chess_krvkp"):
    train_file_name = root + os.sep + "chess-krvkp_R.dat"
    train_data = pd.read_table(train_file_name)

    train_data = train_data[0:]
    train_data = train_data.values
    labels = train_data[:, train_data.shape[1] - 1]
    train_data = train_data[:, [i for i in range(1, train_data.shape[1] - 1)]]
    # train_data = train_data[1:train_data.shape[1] - 1]
    # train_data = train_data.astype("float")
    if return_X_y:
        return np.array(train_data, np.float32), np.array(labels, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_chscase_vine_data_set(return_X_y=True,
                               root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "chscase_vine"):
    train_file_name = root + os.sep + "chscase_vine1.csv"
    train_data = pd.read_csv(train_file_name, header=None)

    train_data = train_data[1:]

    mapping_n = {label: idx for idx, label in enumerate(set(train_data[train_data.shape[1] - 1]))}
    train_data[train_data.shape[1] - 1] = train_data[train_data.shape[1] - 1].map(mapping_n)

    train_data = train_data.astype("float")
    labels = train_data[train_data.shape[1] - 1]
    del train_data[train_data.shape[1] - 1]
    columns = train_data.columns.tolist()
    for c in columns:
        d = train_data[c]
        MAX = d.max()
        MIN = d.min()
        train_data[c] = ((d - MIN) / (MAX - MIN)).tolist()
    if return_X_y:
        return np.array(train_data.values, np.float32), np.array(labels.values, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_cloud_data_set(return_X_y=True,
                        root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "cloud"):
    train_file_name = root + os.sep + "cloud.csv"
    train_data = pd.read_csv(train_file_name, header=None)

    train_data = train_data[1:]

    mapping_0 = {label: idx for idx, label in enumerate(set(train_data[0]))}
    train_data[0] = train_data[0].map(mapping_0)
    mapping_n = {label: idx for idx, label in enumerate(set(train_data[train_data.shape[1] - 1]))}
    train_data[train_data.shape[1] - 1] = train_data[train_data.shape[1] - 1].map(mapping_n)

    train_data = train_data.astype("float")
    labels = train_data[train_data.shape[1] - 1]
    del train_data[train_data.shape[1] - 1]
    columns = train_data.columns.tolist()
    for c in columns:
        d = train_data[c]
        MAX = d.max()
        MIN = d.min()
        train_data[c] = ((d - MIN) / (MAX - MIN)).tolist()
    if return_X_y:
        return np.array(train_data.values, np.float32), np.array(labels.values, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_congressional_voting_data_set(return_X_y=True,
                                       root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "congressional_voting"):
    train_file_name = root + os.sep + "congressional-voting_R.dat"
    train_data = pd.read_table(train_file_name)

    train_data = train_data[0:]
    train_data = train_data.values
    labels = train_data[:, train_data.shape[1] - 1]
    train_data = train_data[:, [i for i in range(1, train_data.shape[1] - 1)]]
    # train_data = train_data[1:train_data.shape[1] - 1]
    # train_data = train_data.astype("float")
    if return_X_y:
        return np.array(train_data, np.float32), np.array(labels, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_diabetes_data_set(return_X_y=True,
                           root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "diabetes"):
    train_file_name = root + os.sep + "diabetes.csv"
    train_data = pd.read_csv(train_file_name, header=None)

    train_data = train_data[1:]

    mapping_n = {label: idx for idx, label in enumerate(set(train_data[train_data.shape[1] - 1]))}
    train_data[train_data.shape[1] - 1] = train_data[train_data.shape[1] - 1].map(mapping_n)

    train_data = train_data.astype("float")
    labels = train_data[train_data.shape[1] - 1]
    del train_data[train_data.shape[1] - 1]
    columns = train_data.columns.tolist()
    for c in columns:
        d = train_data[c]
        MAX = d.max()
        MIN = d.min()
        train_data[c] = ((d - MIN) / (MAX - MIN)).tolist()
    if return_X_y:
        return np.array(train_data.values, np.float32), np.array(labels.values, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_diabetes_data_set(return_X_y=True,
                           root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "diabetes"):
    train_file_name = root + os.sep + "diabetes.csv"
    train_data = pd.read_csv(train_file_name, header=None)

    train_data = train_data[1:]

    mapping_n = {label: idx for idx, label in enumerate(set(train_data[train_data.shape[1] - 1]))}
    train_data[train_data.shape[1] - 1] = train_data[train_data.shape[1] - 1].map(mapping_n)

    train_data = train_data.astype("float")
    labels = train_data[train_data.shape[1] - 1]
    del train_data[train_data.shape[1] - 1]
    columns = train_data.columns.tolist()
    for c in columns:
        d = train_data[c]
        MAX = d.max()
        MIN = d.min()
        train_data[c] = ((d - MIN) / (MAX - MIN)).tolist()
    if return_X_y:
        return np.array(train_data.values, np.float32), np.array(labels.values, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_elusage_data_set(return_X_y=True,
                          root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "elusage"):
    train_file_name = root + os.sep + "elusage.csv"
    train_data = pd.read_csv(train_file_name, header=None)

    train_data = train_data[1:]

    mapping_n = {label: idx for idx, label in enumerate(set(train_data[train_data.shape[1] - 1]))}
    train_data[train_data.shape[1] - 1] = train_data[train_data.shape[1] - 1].map(mapping_n)

    train_data = train_data.astype("float")
    labels = train_data[train_data.shape[1] - 1]
    del train_data[train_data.shape[1] - 1]
    columns = train_data.columns.tolist()
    for c in columns:
        d = train_data[c]
        MAX = d.max()
        MIN = d.min()
        train_data[c] = ((d - MIN) / (MAX - MIN)).tolist()
    if return_X_y:
        return np.array(train_data.values, np.float32), np.array(labels.values, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_hill_valley_data_set(return_X_y=True,
                              root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "hill_valley"):
    train_file_name = root + os.sep + "hill-valley.dat"
    train_data = pd.read_table(train_file_name)

    train_data = train_data[0:]
    train_data = train_data.values
    labels = train_data[:, train_data.shape[1] - 1]
    train_data = train_data[:, [i for i in range(1, train_data.shape[1] - 1)]]
    # train_data = train_data[1:train_data.shape[1] - 1]
    # train_data = train_data.astype("float")
    if return_X_y:
        return np.array(train_data, np.float32), np.array(labels, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_kc3_data_set(return_X_y=True,
                      root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "kc3"):
    train_file_name = root + os.sep + "kc3.csv"
    train_data = pd.read_csv(train_file_name, header=None)

    train_data = train_data[1:]

    mapping_n = {label: idx for idx, label in enumerate(set(train_data[train_data.shape[1] - 1]))}
    train_data[train_data.shape[1] - 1] = train_data[train_data.shape[1] - 1].map(mapping_n)

    train_data = train_data.astype("float")
    labels = train_data[train_data.shape[1] - 1]
    del train_data[train_data.shape[1] - 1]
    columns = train_data.columns.tolist()
    for c in columns:
        d = train_data[c]
        MAX = d.max()
        MIN = d.min()
        train_data[c] = ((d - MIN) / (MAX - MIN)).tolist()
    if return_X_y:
        return np.array(train_data.values, np.float32), np.array(labels.values, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_kidney_data_set(return_X_y=True,
                         root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "kidney"):
    train_file_name = root + os.sep + "kidney.csv"
    train_data = pd.read_csv(train_file_name, header=None)

    train_data = train_data[1:]

    mapping_4 = {label: idx for idx, label in enumerate(set(train_data[4]))}
    train_data[4] = train_data[4].map(mapping_4)
    mapping_5 = {label: idx for idx, label in enumerate(set(train_data[5]))}
    train_data[5] = train_data[5].map(mapping_5)
    mapping_6 = {label: idx for idx, label in enumerate(set(train_data[6]))}
    train_data[6] = train_data[6].map(mapping_6)
    mapping_n = {label: idx for idx, label in enumerate(set(train_data[train_data.shape[1] - 1]))}
    train_data[train_data.shape[1] - 1] = train_data[train_data.shape[1] - 1].map(mapping_n)

    train_data = train_data.astype("float")
    labels = train_data[train_data.shape[1] - 1]
    del train_data[train_data.shape[1] - 1]
    columns = train_data.columns.tolist()
    for c in columns:
        d = train_data[c]
        MAX = d.max()
        MIN = d.min()
        train_data[c] = ((d - MIN) / (MAX - MIN)).tolist()
    if return_X_y:
        return np.array(train_data.values, np.float32), np.array(labels.values, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_lowbwt_data_set(return_X_y=True,
                         root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "lowbwt"):
    train_file_name = root + os.sep + "lowbwt.csv"
    train_data = pd.read_csv(train_file_name, header=None)

    train_data = train_data[1:]

    mapping_n = {label: idx for idx, label in enumerate(set(train_data[train_data.shape[1] - 1]))}
    train_data[train_data.shape[1] - 1] = train_data[train_data.shape[1] - 1].map(mapping_n)

    train_data = train_data.astype("float")
    labels = train_data[train_data.shape[1] - 1]
    del train_data[train_data.shape[1] - 1]
    columns = train_data.columns.tolist()
    for c in columns:
        d = train_data[c]
        MAX = d.max()
        MIN = d.min()
        train_data[c] = ((d - MIN) / (MAX - MIN)).tolist()
    if return_X_y:
        return np.array(train_data.values, np.float32), np.array(labels.values, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_lung_cancer_data_set(return_X_y=True,
                              root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "lung_cancer"):
    train_file_name = root + os.sep + "lung-cancer_R.dat"
    train_data = pd.read_table(train_file_name)

    train_data = train_data[0:]
    train_data = train_data.values
    labels = train_data[:, train_data.shape[1] - 1]
    train_data = train_data[:, [i for i in range(1, train_data.shape[1] - 1)]]
    # train_data = train_data[1:train_data.shape[1] - 1]
    # train_data = train_data.astype("float")
    if return_X_y:
        return np.array(train_data, np.float32), np.array(labels, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_lupus_data_set(return_X_y=True,
                        root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "lupus"):
    train_file_name = root + os.sep + "lupus.csv"
    train_data = pd.read_csv(train_file_name, header=None)

    train_data = train_data[1:]

    mapping_n = {label: idx for idx, label in enumerate(set(train_data[train_data.shape[1] - 1]))}
    train_data[train_data.shape[1] - 1] = train_data[train_data.shape[1] - 1].map(mapping_n)

    train_data = train_data.astype("float")
    labels = train_data[train_data.shape[1] - 1]
    del train_data[train_data.shape[1] - 1]
    columns = train_data.columns.tolist()
    for c in columns:
        d = train_data[c]
        MAX = d.max()
        MIN = d.min()
        train_data[c] = ((d - MIN) / (MAX - MIN)).tolist()
    if return_X_y:
        return np.array(train_data.values, np.float32), np.array(labels.values, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_meta_data_set(return_X_y=True,
                       root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "meta"):
    train_file_name = root + os.sep + "meta.csv"
    train_data = pd.read_csv(train_file_name, header=None)

    train_data = train_data[1:]

    mapping_n = {label: idx for idx, label in enumerate(set(train_data[train_data.shape[1] - 1]))}
    train_data[train_data.shape[1] - 1] = train_data[train_data.shape[1] - 1].map(mapping_n)

    train_data = train_data.astype("float")
    labels = train_data[train_data.shape[1] - 1]
    del train_data[train_data.shape[1] - 1]
    columns = train_data.columns.tolist()
    for c in columns:
        d = train_data[c]
        MAX = d.max()
        MIN = d.min()
        train_data[c] = ((d - MIN) / (MAX - MIN)).tolist()
    if return_X_y:
        return np.array(train_data.values, np.float32), np.array(labels.values, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_monks_1_data_set(return_X_y=True,
                          root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "monks_1"):
    train_file_name = root + os.sep + "monks-1.dat"
    train_data = pd.read_table(train_file_name)

    train_data = train_data[0:]
    train_data = train_data.values
    labels = train_data[:, train_data.shape[1] - 1]
    train_data = train_data[:, [i for i in range(1, train_data.shape[1] - 1)]]
    # train_data = train_data[1:train_data.shape[1] - 1]
    # train_data = train_data.astype("float")
    if return_X_y:
        return np.array(train_data, np.float32), np.array(labels, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_no2_data_set(return_X_y=True,
                      root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "no2"):
    train_file_name = root + os.sep + "no2.csv"
    train_data = pd.read_csv(train_file_name, header=None)

    train_data = train_data[1:]

    mapping_n = {label: idx for idx, label in enumerate(set(train_data[train_data.shape[1] - 1]))}
    train_data[train_data.shape[1] - 1] = train_data[train_data.shape[1] - 1].map(mapping_n)

    train_data = train_data.astype("float")
    labels = train_data[train_data.shape[1] - 1]
    del train_data[train_data.shape[1] - 1]
    columns = train_data.columns.tolist()
    for c in columns:
        d = train_data[c]
        MAX = d.max()
        MIN = d.min()
        train_data[c] = ((d - MIN) / (MAX - MIN)).tolist()
    if return_X_y:
        return np.array(train_data.values, np.float32), np.array(labels.values, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_pima_data_set(return_X_y=True,
                       root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "pima"):
    train_file_name = root + os.sep + "pima_R.dat"
    train_data = pd.read_table(train_file_name)

    train_data = train_data[0:]
    train_data = train_data.values
    labels = train_data[:, train_data.shape[1] - 1]
    train_data = train_data[:, [i for i in range(1, train_data.shape[1] - 1)]]
    # train_data = train_data[1:train_data.shape[1] - 1]
    # train_data = train_data.astype("float")
    if return_X_y:
        return np.array(train_data, np.float32), np.array(labels, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_planning_data_set(return_X_y=True,
                           root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "planning"):
    train_file_name = root + os.sep + "planning_R.dat"
    train_data = pd.read_table(train_file_name)

    train_data = train_data[0:]
    train_data = train_data.values
    labels = train_data[:, train_data.shape[1] - 1]
    train_data = train_data[:, [i for i in range(1, train_data.shape[1] - 1)]]
    # train_data = train_data[1:train_data.shape[1] - 1]
    # train_data = train_data.astype("float")
    if return_X_y:
        return np.array(train_data, np.float32), np.array(labels, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_pm10_data_set(return_X_y=True,
                       root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "pm10"):
    train_file_name = root + os.sep + "pm10.csv"
    train_data = pd.read_csv(train_file_name, header=None)

    train_data = train_data[1:]

    mapping_n = {label: idx for idx, label in enumerate(set(train_data[train_data.shape[1] - 1]))}
    train_data[train_data.shape[1] - 1] = train_data[train_data.shape[1] - 1].map(mapping_n)

    train_data = train_data.astype("float")
    labels = train_data[train_data.shape[1] - 1]
    del train_data[train_data.shape[1] - 1]
    columns = train_data.columns.tolist()
    for c in columns:
        d = train_data[c]
        MAX = d.max()
        MIN = d.min()
        train_data[c] = ((d - MIN) / (MAX - MIN)).tolist()
    if return_X_y:
        return np.array(train_data.values, np.float32), np.array(labels.values, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_rabe_131_data_set(return_X_y=True,
                           root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "rabe_131"):
    train_file_name = root + os.sep + "rabe_131.csv"
    train_data = pd.read_csv(train_file_name, header=None)

    train_data = train_data[1:]

    mapping_n = {label: idx for idx, label in enumerate(set(train_data[train_data.shape[1] - 1]))}
    train_data[train_data.shape[1] - 1] = train_data[train_data.shape[1] - 1].map(mapping_n)

    train_data = train_data.astype("float")
    labels = train_data[train_data.shape[1] - 1]
    del train_data[train_data.shape[1] - 1]
    columns = train_data.columns.tolist()
    for c in columns:
        d = train_data[c]
        MAX = d.max()
        MIN = d.min()
        train_data[c] = ((d - MIN) / (MAX - MIN)).tolist()
    if return_X_y:
        return np.array(train_data.values, np.float32), np.array(labels.values, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_schlvote_data_set(return_X_y=True,
                           root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "schlvote"):
    train_file_name = root + os.sep + "schlvote.csv"
    train_data = pd.read_csv(train_file_name, header=None)

    train_data = train_data[1:]

    mapping_n = {label: idx for idx, label in enumerate(set(train_data[train_data.shape[1] - 1]))}
    train_data[train_data.shape[1] - 1] = train_data[train_data.shape[1] - 1].map(mapping_n)

    train_data = train_data.astype("float")
    labels = train_data[train_data.shape[1] - 1]
    del train_data[train_data.shape[1] - 1]
    columns = train_data.columns.tolist()
    for c in columns:
        d = train_data[c]
        MAX = d.max()
        MIN = d.min()
        train_data[c] = ((d - MIN) / (MAX - MIN)).tolist()
    if return_X_y:
        return np.array(train_data.values, np.float32), np.array(labels.values, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_soybean_data_set(return_X_y=True,
                          root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "soybean"):
    train_file_name = root + os.sep + "soybean.dat"
    train_data = pd.read_table(train_file_name)

    train_data = train_data[0:]
    train_data = train_data.values
    labels = train_data[:, train_data.shape[1] - 1]
    train_data = train_data[:, [i for i in range(1, train_data.shape[1] - 1)]]
    # train_data = train_data[1:train_data.shape[1] - 1]
    # train_data = train_data.astype("float")
    if return_X_y:
        return np.array(train_data, np.float32), np.array(labels, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_spambase_data_set(return_X_y=True,
                           root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "spambase"):
    train_file_name = root + os.sep + "spambase_R.dat"
    train_data = pd.read_table(train_file_name)

    train_data = train_data[0:]
    train_data = train_data.values
    labels = train_data[:, train_data.shape[1] - 1]
    train_data = train_data[:, [i for i in range(1, train_data.shape[1] - 1)]]
    # train_data = train_data[1:train_data.shape[1] - 1]
    # train_data = train_data.astype("float")
    if return_X_y:
        return np.array(train_data, np.float32), np.array(labels, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_spect_data_set(return_X_y=True,
                        root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "spect"):
    train_file_name = root + os.sep + "spect.dat"
    train_data = pd.read_table(train_file_name)

    train_data = train_data[0:]
    train_data = train_data.values
    labels = train_data[:, train_data.shape[1] - 1]
    train_data = train_data[:, [i for i in range(1, train_data.shape[1] - 1)]]
    # train_data = train_data[1:train_data.shape[1] - 1]
    # train_data = train_data.astype("float")
    if return_X_y:
        return np.array(train_data, np.float32), np.array(labels, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_tae_data_set(return_X_y=True,
                      root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "tae"):
    train_file_name = root + os.sep + "tae.csv"
    train_data = pd.read_csv(train_file_name, header=None)

    train_data = train_data[1:]

    mapping_n = {label: idx for idx, label in enumerate(set(train_data[train_data.shape[1] - 1]))}
    train_data[train_data.shape[1] - 1] = train_data[train_data.shape[1] - 1].map(mapping_n)

    train_data = train_data.astype("float")
    labels = train_data[train_data.shape[1] - 1]
    del train_data[train_data.shape[1] - 1]
    columns = train_data.columns.tolist()
    for c in columns:
        d = train_data[c]
        MAX = d.max()
        MIN = d.min()
        train_data[c] = ((d - MIN) / (MAX - MIN)).tolist()
    if return_X_y:
        return np.array(train_data.values, np.float32), np.array(labels.values, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_transplant_data_set(return_X_y=True,
                             root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "transplant"):
    train_file_name = root + os.sep + "transplant.csv"
    train_data = pd.read_csv(train_file_name, header=None)

    train_data = train_data[1:]

    mapping_n = {label: idx for idx, label in enumerate(set(train_data[train_data.shape[1] - 1]))}
    train_data[train_data.shape[1] - 1] = train_data[train_data.shape[1] - 1].map(mapping_n)

    train_data = train_data.astype("float")
    labels = train_data[train_data.shape[1] - 1]
    del train_data[train_data.shape[1] - 1]
    if return_X_y:
        return np.array(train_data.values, np.float32), np.array(labels.values, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_triazines_data_set(return_X_y=True,
                            root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "triazines"):
    train_file_name = root + os.sep + "triazines.csv"
    train_data = pd.read_csv(train_file_name, header=None)

    train_data = train_data[1:]

    mapping_n = {label: idx for idx, label in enumerate(set(train_data[train_data.shape[1] - 1]))}
    train_data[train_data.shape[1] - 1] = train_data[train_data.shape[1] - 1].map(mapping_n)

    train_data = train_data.astype("float")
    labels = train_data[train_data.shape[1] - 1]
    del train_data[train_data.shape[1] - 1]
    if return_X_y:
        return np.array(train_data.values, np.float32), np.array(labels.values, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_veteran_data_set(return_X_y=True,
                          root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "veteran"):
    train_file_name = root + os.sep + "veteran.csv"
    train_data = pd.read_csv(train_file_name, header=None)

    train_data = train_data[1:]

    mapping_n = {label: idx for idx, label in enumerate(set(train_data[train_data.shape[1] - 1]))}
    train_data[train_data.shape[1] - 1] = train_data[train_data.shape[1] - 1].map(mapping_n)

    train_data = train_data.astype("float")
    labels = train_data[train_data.shape[1] - 1]
    del train_data[train_data.shape[1] - 1]
    columns = train_data.columns.tolist()
    for c in columns:
        d = train_data[c]
        MAX = d.max()
        MIN = d.min()
        train_data[c] = ((d - MIN) / (MAX - MIN)).tolist()
    if return_X_y:
        return np.array(train_data.values, np.float32), np.array(labels.values, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_waveform_data_set(return_X_y=True,
                           root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "waveform"):
    train_file_name = root + os.sep + "waveform_R.dat"
    train_data = pd.read_table(train_file_name)

    train_data = train_data[0:]
    train_data = train_data.values
    labels = train_data[:, train_data.shape[1] - 1]
    train_data = train_data[:, [i for i in range(1, train_data.shape[1] - 1)]]
    # train_data = train_data[1:train_data.shape[1] - 1]
    # train_data = train_data.astype("float")
    if return_X_y:
        return np.array(train_data, np.float32), np.array(labels, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_white_clover_data_set(return_X_y=True,
                               root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "white_clover"):
    train_file_name = root + os.sep + "white-clover.csv"
    train_data = pd.read_csv(train_file_name, header=None)

    train_data = train_data[1:]

    mapping_0 = {label: idx for idx, label in enumerate(set(train_data[0]))}
    train_data[0] = train_data[0].map(mapping_0)
    mapping_1 = {label: idx for idx, label in enumerate(set(train_data[1]))}
    train_data[1] = train_data[1].map(mapping_1)
    mapping_n = {label: idx for idx, label in enumerate(set(train_data[train_data.shape[1] - 1]))}
    train_data[train_data.shape[1] - 1] = train_data[train_data.shape[1] - 1].map(mapping_n)

    train_data = train_data.astype("float")
    labels = train_data[train_data.shape[1] - 1]
    del train_data[train_data.shape[1] - 1]
    columns = train_data.columns.tolist()
    for c in columns:
        d = train_data[c]
        MAX = d.max()
        MIN = d.min()
        train_data[c] = ((d - MIN) / (MAX - MIN)).tolist()
    if return_X_y:
        return np.array(train_data.values, np.float32), np.array(labels.values, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_yeast_data_set(return_X_y=True,
                        root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "yeast"):
    train_file_name = root + os.sep + "yeast_R.dat"
    train_data = pd.read_table(train_file_name)

    train_data = train_data[0:]
    train_data = train_data.values
    labels = train_data[:, train_data.shape[1] - 1]
    train_data = train_data[:, [i for i in range(1, train_data.shape[1] - 1)]]
    # train_data = train_data[1:train_data.shape[1] - 1]
    # train_data = train_data.astype("float")
    if return_X_y:
        return np.array(train_data, np.float32), np.array(labels, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_zoo_data_set(return_X_y=True,
                      root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "zoo"):
    train_file_name = root + os.sep + "zoo_R.dat"
    train_data = pd.read_table(train_file_name)

    train_data = train_data[0:]
    train_data = train_data.values
    labels = train_data[:, train_data.shape[1] - 1]
    train_data = train_data[:, [i for i in range(1, train_data.shape[1] - 1)]]
    # train_data = train_data[1:train_data.shape[1] - 1]
    # train_data = train_data.astype("float")
    if return_X_y:
        return np.array(train_data, np.float32), np.array(labels, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_seed_data_set(return_X_y=True,
                       root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "seed"):
    train_file_name = root + os.sep + "seeds_R.dat"
    train_data = pd.read_table(train_file_name)

    train_data = train_data[0:]
    train_data = train_data.values
    labels = train_data[:, train_data.shape[1] - 1]
    train_data = train_data[:, [i for i in range(1, train_data.shape[1] - 1)]]
    # train_data = train_data[1:train_data.shape[1] - 1]
    # train_data = train_data.astype("float")
    if return_X_y:
        return np.array(train_data, np.float32), np.array(labels, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_titanic_data_set(return_X_y=True,
                          root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "titanic"):
    train_file_name = root + os.sep + "titanic_R.dat"
    train_data = pd.read_table(train_file_name)

    train_data = train_data[0:]
    train_data = train_data.values
    labels = train_data[:, train_data.shape[1] - 1]
    train_data = train_data[:, [i for i in range(1, train_data.shape[1] - 1)]]
    # train_data = train_data[1:train_data.shape[1] - 1]
    # train_data = train_data.astype("float")
    if return_X_y:
        return np.array(train_data, np.float32), np.array(labels, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_statlog_heart_data_set(return_X_y=True,
                                root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "statlog_heart"):
    train_file_name = root + os.sep + "statlog-heart_R.dat"
    train_data = pd.read_table(train_file_name)

    train_data = train_data[0:]
    train_data = train_data.values
    labels = train_data[:, train_data.shape[1] - 1]
    train_data = train_data[:, [i for i in range(1, train_data.shape[1] - 1)]]
    # train_data = train_data[1:train_data.shape[1] - 1]
    # train_data = train_data.astype("float")
    if return_X_y:
        return np.array(train_data, np.float32), np.array(labels, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_steel_plates_data_set(return_X_y=True,
                               root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "steel_plates"):
    train_file_name = root + os.sep + "steel-plates_R.dat"
    train_data = pd.read_table(train_file_name)

    train_data = train_data[0:]
    train_data = train_data.values
    labels = train_data[:, train_data.shape[1] - 1]
    train_data = train_data[:, [i for i in range(1, train_data.shape[1] - 1)]]
    # train_data = train_data[1:train_data.shape[1] - 1]
    # train_data = train_data.astype("float")
    if return_X_y:
        return np.array(train_data, np.float32), np.array(labels, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_libras_data_set(return_X_y=True,
                         root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "libras"):
    train_file_name = root + os.sep + "libras_R.dat"
    train_data = pd.read_table(train_file_name)

    train_data = train_data[0:]
    train_data = train_data.values
    labels = train_data[:, train_data.shape[1] - 1]
    train_data = train_data[:, [i for i in range(1, train_data.shape[1] - 1)]]
    # train_data = train_data[1:train_data.shape[1] - 1]
    # train_data = train_data.astype("float")
    if return_X_y:
        return np.array(train_data, np.float32), np.array(labels, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_mammographic_data_set(return_X_y=True,
                               root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "mammographic"):
    train_file_name = root + os.sep + "mammographic_R.dat"
    train_data = pd.read_table(train_file_name)

    train_data = train_data[0:]
    train_data = train_data.values
    labels = train_data[:, train_data.shape[1] - 1]
    train_data = train_data[:, [i for i in range(1, train_data.shape[1] - 1)]]
    # train_data = train_data[1:train_data.shape[1] - 1]
    # train_data = train_data.astype("float")
    if return_X_y:
        return np.array(train_data, np.float32), np.array(labels, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_monks_2_data_set(return_X_y=True,
                          root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "monks_2"):
    train_file_name = root + os.sep + "monks-2.dat"
    train_data = pd.read_table(train_file_name)

    train_data = train_data[0:]
    train_data = train_data.values
    labels = train_data[:, train_data.shape[1] - 1]
    train_data = train_data[:, [i for i in range(1, train_data.shape[1] - 1)]]
    # train_data = train_data[1:train_data.shape[1] - 1]
    # train_data = train_data.astype("float")
    if return_X_y:
        return np.array(train_data, np.float32), np.array(labels, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_monks_3_data_set(return_X_y=True,
                          root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "monks_3"):
    train_file_name = root + os.sep + "monks-3.dat"
    train_data = pd.read_table(train_file_name)

    train_data = train_data[0:]
    train_data = train_data.values
    labels = train_data[:, train_data.shape[1] - 1]
    train_data = train_data[:, [i for i in range(1, train_data.shape[1] - 1)]]
    # train_data = train_data[1:train_data.shape[1] - 1]
    # train_data = train_data.astype("float")
    if return_X_y:
        return np.array(train_data, np.float32), np.array(labels, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_musk_1_data_set(return_X_y=True,
                         root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "musk_1"):
    train_file_name = root + os.sep + "musk-1_R.dat"
    train_data = pd.read_table(train_file_name)

    train_data = train_data[0:]
    train_data = train_data.values
    labels = train_data[:, train_data.shape[1] - 1]
    train_data = train_data[:, [i for i in range(1, train_data.shape[1] - 1)]]
    # train_data = train_data[1:train_data.shape[1] - 1]
    # train_data = train_data.astype("float")
    if return_X_y:
        return np.array(train_data, np.float32), np.array(labels, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_ozone_data_set(return_X_y=True,
                        root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "ozone"):
    train_file_name = root + os.sep + "ozone_R.dat"
    train_data = pd.read_table(train_file_name)

    train_data = train_data[0:]
    train_data = train_data.values
    labels = train_data[:, train_data.shape[1] - 1]
    train_data = train_data[:, [i for i in range(1, train_data.shape[1] - 1)]]
    # train_data = train_data[1:train_data.shape[1] - 1]
    # train_data = train_data.astype("float")
    if return_X_y:
        return np.array(train_data, np.float32), np.array(labels, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_parkinsons_data_set(return_X_y=True,
                             root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "parkinsons"):
    train_file_name = root + os.sep + "parkinsons_R.dat"
    train_data = pd.read_table(train_file_name)

    train_data = train_data[0:]
    train_data = train_data.values
    labels = train_data[:, train_data.shape[1] - 1]
    train_data = train_data[:, [i for i in range(1, train_data.shape[1] - 1)]]
    # train_data = train_data[1:train_data.shape[1] - 1]
    # train_data = train_data.astype("float")
    if return_X_y:
        return np.array(train_data, np.float32), np.array(labels, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_plant_margin_data_set(return_X_y=True,
                               root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "plant_margin"):
    train_file_name = root + os.sep + "plant-margin_R.dat"
    train_data = pd.read_table(train_file_name)

    train_data = train_data[0:]
    train_data = train_data.values
    labels = train_data[:, train_data.shape[1] - 1]
    train_data = train_data[:, [i for i in range(1, train_data.shape[1] - 1)]]
    # train_data = train_data[1:train_data.shape[1] - 1]
    # train_data = train_data.astype("float")
    if return_X_y:
        return np.array(train_data, np.float32), np.array(labels, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_plant_shape_data_set(return_X_y=True,
                              root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "plant_shape"):
    train_file_name = root + os.sep + "plant-shape_R.dat"
    train_data = pd.read_table(train_file_name)

    train_data = train_data[0:]
    train_data = train_data.values
    labels = train_data[:, train_data.shape[1] - 1]
    train_data = train_data[:, [i for i in range(1, train_data.shape[1] - 1)]]
    # train_data = train_data[1:train_data.shape[1] - 1]
    # train_data = train_data.astype("float")
    if return_X_y:
        return np.array(train_data, np.float32), np.array(labels, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_post_operative_data_set(return_X_y=True,
                                 root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "post_operative"):
    train_file_name = root + os.sep + "post-operative_R.dat"
    train_data = pd.read_table(train_file_name)

    train_data = train_data[0:]
    train_data = train_data.values
    labels = train_data[:, train_data.shape[1] - 1]
    train_data = train_data[:, [i for i in range(1, train_data.shape[1] - 1)]]
    # train_data = train_data[1:train_data.shape[1] - 1]
    # train_data = train_data.astype("float")
    if return_X_y:
        return np.array(train_data, np.float32), np.array(labels, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_semeion_data_set(return_X_y=True,
                          root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "semeion"):
    train_file_name = root + os.sep + "semeion_R.dat"
    train_data = pd.read_table(train_file_name)

    train_data = train_data[0:]
    train_data = train_data.values
    labels = train_data[:, train_data.shape[1] - 1]
    train_data = train_data[:, [i for i in range(1, train_data.shape[1] - 1)]]
    # train_data = train_data[1:train_data.shape[1] - 1]
    # train_data = train_data.astype("float")
    if return_X_y:
        return np.array(train_data, np.float32), np.array(labels, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_horse_colic_data_set(return_X_y=True,
                              root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "horse_colic"):
    train_file_name = root + os.sep + "horse-colic.dat"
    train_data = pd.read_table(train_file_name)

    train_data = train_data[0:]
    train_data = train_data.values
    labels = train_data[:, train_data.shape[1] - 1]
    train_data = train_data[:, [i for i in range(1, train_data.shape[1] - 1)]]
    # train_data = train_data[1:train_data.shape[1] - 1]
    # train_data = train_data.astype("float")
    if return_X_y:
        return np.array(train_data, np.float32), np.array(labels, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_hepatitis_data_set(return_X_y=True,
                            root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "hepatitis"):
    train_file_name = root + os.sep + "hepatitis_R.dat"
    train_data = pd.read_table(train_file_name)

    train_data = train_data[0:]
    train_data = train_data.values
    labels = train_data[:, train_data.shape[1] - 1]
    train_data = train_data[:, [i for i in range(1, train_data.shape[1] - 1)]]
    # train_data = train_data[1:train_data.shape[1] - 1]
    # train_data = train_data.astype("float")
    if return_X_y:
        return np.array(train_data, np.float32), np.array(labels, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_hayes_roth_data_set(return_X_y=True,
                             root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "hayes_roth"):
    train_file_name = root + os.sep + "hayes-roth.dat"
    train_data = pd.read_table(train_file_name)

    train_data = train_data[0:]
    train_data = train_data.values
    labels = train_data[:, train_data.shape[1] - 1]
    train_data = train_data[:, [i for i in range(1, train_data.shape[1] - 1)]]
    # train_data = train_data[1:train_data.shape[1] - 1]
    # train_data = train_data.astype("float")
    if return_X_y:
        return np.array(train_data, np.float32), np.array(labels, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_ionosphere_data_set(return_X_y=True,
                             root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "ionosphere"):
    train_file_name = root + os.sep + "ionosphere_R.dat"
    train_data = pd.read_table(train_file_name)

    train_data = train_data[0:]
    train_data = train_data.values
    labels = train_data[:, train_data.shape[1] - 1]
    train_data = train_data[:, [i for i in range(1, train_data.shape[1] - 1)]]
    # train_data = train_data[1:train_data.shape[1] - 1]
    # train_data = train_data.astype("float")
    if return_X_y:
        return np.array(train_data, np.float32), np.array(labels, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_led_display_data_set(return_X_y=True,
                              root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "led_display"):
    train_file_name = root + os.sep + "led-display_R.dat"
    train_data = pd.read_table(train_file_name)

    train_data = train_data[0:]
    train_data = train_data.values
    labels = train_data[:, train_data.shape[1] - 1]
    train_data = train_data[:, [i for i in range(1, train_data.shape[1] - 1)]]
    # train_data = train_data[1:train_data.shape[1] - 1]
    # train_data = train_data.astype("float")
    if return_X_y:
        return np.array(train_data, np.float32), np.array(labels, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_low_res_spect_data_set(return_X_y=True,
                                root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "low_res_spect"):
    train_file_name = root + os.sep + "low-res-spect_R.dat"
    train_data = pd.read_table(train_file_name)

    train_data = train_data[0:]
    train_data = train_data.values
    labels = train_data[:, train_data.shape[1] - 1]
    train_data = train_data[:, [i for i in range(1, train_data.shape[1] - 1)]]
    # train_data = train_data[1:train_data.shape[1] - 1]
    # train_data = train_data.astype("float")
    if return_X_y:
        return np.array(train_data, np.float32), np.array(labels, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_wall_following_data_set(return_X_y=True,
                                 root="/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/code/datasets" + os.sep + "wall_following"):
    train_file_name = root + os.sep + "wall-following_R.dat"
    train_data = pd.read_table(train_file_name)

    train_data = train_data[0:]
    train_data = train_data.values
    labels = train_data[:, train_data.shape[1] - 1]
    train_data = train_data[:, [i for i in range(1, train_data.shape[1] - 1)]]
    # train_data = train_data[1:train_data.shape[1] - 1]
    # train_data = train_data.astype("float")
    if return_X_y:
        return np.array(train_data, np.float32), np.array(labels, np.int)
    else:
        train_data["labels"] = labels
        return train_data


if __name__ == "__main__":
    # root = "E:\\PySpace\\datasets\\Mushroom"
    X, y = load_analcatdata_germangss_data_set(return_X_y=True)
    print(X.shape)
    print(y.shape)
    print(X[0:2])
    print(y[0:6])
    # data = [load_frogs_mfcc_data_set,  # 0   7195
    #         dataset.load_breast_cancer,  # 1     569
    #         load_Car_Evaluation,  # 2        1728
    #         load_CTG_data_set,  # 3     2126
    #         load_cv_firm_teacher_data_set,  # 4  10800
    #         load_Crowdsourced_Mapping_Data_Set,  # 5   10545  *
    #         load_dr_debrecen_data_set,  # 6    1151
    #         dataset.load_digits,  # 7   1797
    #         load_htru_data_set,  # 8  17898
    #         load_image_segmentation_data_set,  # 9   2310   *
    #         dataset.load_iris,  # 10   150
    #         load_letter_recognition_data_set,  # 11  20000
    #         load_optdigits,  # 12   5620
    #         load_pr_handwritten_data_set,  # 13   10992   *
    #         load_phishing_websites_data_set,  # 14  11055
    #         load_turkiye_student_evaluation_data_set,  # 15  5820
    #         load_wilt_data_set,  # 16   4889   *
    #         load_winequality_white_data_set,  # 17   4898
    #
    #         dataset.load_boston,  # 0  506
    #         dataset.load_diabetes,  # 1  442
    #         load_tic_data_set,  # 2  958
    #         load_magic_data_set,  # 3  19020
    #         load_ILPD_data_set,  # 4  583
    #         load_glass_identify_data_set,  # 5  214
    #         load_abalone_data_set,  # 6   4177
    #         load_autouniv700_data_set,  # 7   700
    #         load_autouniv_data_set,  # 8   500
    #         load_mfeat_data_set,  # 9  2000
    #         load_churn_data_set,  # 10  5000
    #         load_blood_data_set,  # 11  748
    #         load_ilpd_data_set,  # 12  583
    #         load_banknote_data_set,  # 13  1372
    #         load_vowel_data_set,  # 14  990
    #         load_Mushroom,  # 15  8124
    #         load_Mnist]
    # for k, load_dataset in enumerate(data):
    #     if k != 5 and k != 9 and k != 13 and k != 16:
    #         X, y = load_dataset(return_X_y=True)
    #         print(X.shape)
    #         print(y.shape)
    #         print(X[0:2])
    #         print(y[0:6])
    # pi = np.random.permutation(len(X))
    # X, y = X[pi], y[pi]
    # pi = np.random.permutation(len(X))
    # X, x = X[pi], x[pi]
    # pi = np.random.permutation(len(y))
    # Y, y = Y[pi], y[pi]
    # print(X[0:2])
    # print(y[0:6])
