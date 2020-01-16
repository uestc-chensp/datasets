# coding: utf-8
import numpy as np
import struct
import matplotlib.pyplot as plt
import sklearn.datasets as skdata
import RL.AutoDataAnalyst_PPO_xgb_v3.code.datamanager.read_data as RD


# 数据管理类
class DataManager(object):

    # 该类用于管理数据，包括训练，测试。其中训练集分为训练数据和得分数据（用于Reward）。
    # 为此该类实现：
    # 1.储存数据（包括pickle）
    # 2.分割数据（分割train_data，使其变为训练数据和得分数据）

    def __init__(self, data_set_index=6):
        self.data_cv = None
        path = "/home/shawn/PycharmProjects/AutoDataAnalyst/MNIST_data/"
        # 训练集文件
        self.train_images_idx3_ubyte_file = path + 'train-images.idx3-ubyte'
        # 训练集标签文件
        self.train_labels_idx1_ubyte_file = path + 'train-labels.idx1-ubyte'

        # 测试集文件
        self.test_images_idx3_ubyte_file = path + 't10k-images.idx3-ubyte'
        # 测试集标签文件
        self.test_labels_idx1_ubyte_file = path + 't10k-labels.idx1-ubyte'

        # 获取完整版的mnist数据集
        # self.read_data()

        # 获取成年人收入数据集
        # self.read_data(3)  # RD.load_Adult
        self.data_set_index = data_set_index
        self.read_data(data_set_index)

    def decode_idx3_ubyte(self, idx3_ubyte_file):
        """
        解析idx3文件的通用函数
        :param idx3_ubyte_file: idx3文件路径
        :return: 数据集
        """
        # 读取二进制数据
        file_x = open(idx3_ubyte_file, 'rb')
        bin_data = file_x.read()
        file_x.close()

        # 解析文件头信息，依次为魔数、图片数量、每张图片高、每张图片宽
        offset = 0
        fmt_header = '>iiii'
        magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
        # print('魔数:%d, 图片数量: %d张, 图片大小: %d*%d' % (magic_number, num_images, num_rows, num_cols))

        # 解析数据集
        image_size = num_rows * num_cols
        offset += struct.calcsize(fmt_header)
        fmt_image = '>' + str(image_size) + 'B'
        images = np.empty((num_images, num_rows * num_cols))
        for i in range(num_images):
            images[i] = (np.array(struct.unpack_from(fmt_image, bin_data, offset)) > 0).astype(int)
            offset += struct.calcsize(fmt_image)
        return images

    def decode_idx1_ubyte(self, idx1_ubyte_file):
        """
        解析idx1文件的通用函数
        :param idx1_ubyte_file: idx1文件路径
        :return: 数据集
        """
        # 读取二进制数据
        file_x = open(idx1_ubyte_file, 'rb')
        bin_data = file_x.read()
        file_x.close()

        # 解析文件头信息，依次为魔数和标签数
        offset = 0
        fmt_header = '>ii'
        magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
        # print('魔数:%d, 图片数量: %d张' % (magic_number, num_images))

        # 解析数据集
        offset += struct.calcsize(fmt_header)
        fmt_image = '>B'
        # labels = np.empty((num_images, 10))
        labels = np.empty((num_images,))
        for i in range(num_images):
            index = struct.unpack_from(fmt_image, bin_data, offset)[0]
            # labels[i] = np.zeros([10])
            # labels[i][int(index)] = 1
            labels[i] = int(index)
            offset += struct.calcsize(fmt_image)
        return labels

    def load_train_data(self):
        """
        :param idx_ubyte_file: idx文件路径
        :return: n*1维np.array对象，n为图片数量
        """
        train_data, train_labels = self.decode_idx3_ubyte(self.train_images_idx3_ubyte_file), self.decode_idx1_ubyte(
            self.train_labels_idx1_ubyte_file)

        return train_data, train_labels

    def load_test_data(self):
        """
        :param idx_ubyte_file: idx文件路径
        :return: n*row*col维np.array对象，n为图片数量
        """
        test_data, test_labels = self.decode_idx3_ubyte(self.test_images_idx3_ubyte_file), self.decode_idx1_ubyte(
            self.test_labels_idx1_ubyte_file)

        return test_data, test_labels

    # 从指定文件读取数据
    def read_data(self, data_set_id):
        dataset = [
            RD.load_image_segmentation_data_set,
            RD.load_wilt_data_set,
            RD.load_Crowdsourced_Mapping_Data_Set,
            RD.load_pr_handwritten_data_set,

            RD.load_abalone_data_set,  # 1 4177
            RD.load_acute_inflammation_data_set,
            RD.load_acute_nephritis_data_set,
            RD.load_airline_data_set,  # 27
            RD.load_analcatdata_asbestos_data_set,
            RD.load_analcatdata_boxing_data_set,
            RD.load_analcatdata_broadwaymult_data_set,
            RD.load_analcatdata_germangss_data_set,
            RD.load_analcatdata_lawsuit_data_set,
            RD.load_annealing_data_set,
            RD.load_ar4_data_set,
            RD.load_arrhythmia_data_set,
            RD.load_audiology_data_set,
            RD.load_autouniv_data_set,
            RD.load_autouniv700_data_set,
            RD.load_balance_scale_data_set,
            RD.load_banknote_data_set,  # 23 1372 元训练
            RD.load_baseball_data_set,
            RD.load_blood_data_set,  # 21 748
            RD.load_bodyfat_data_set,
            RD.load_Car_Evaluation,  # 11
            RD.load_chatfield_4_data_set,
            RD.load_chess_krvkp_data_set,
            RD.load_chscase_vine_data_set,
            RD.load_churn_data_set,  # 20 5000  元训练
            RD.load_cloud_data_set,
            RD.load_congressional_voting_data_set,

            RD.load_CTG_data_set,
            RD.load_cv_firm_teacher_data_set,
            RD.load_diabetes_data_set,
            RD.load_disclosure_data_set,
            RD.load_dr_debrecen_data_set,
            RD.load_elusage_data_set,
            RD.load_fri_c2_500_data_set,
            RD.load_frogs_mfcc_data_set,
            RD.load_glass_identify_data_set,
            RD.load_hayes_roth_data_set,
            RD.load_hepatitis_data_set,
            RD.load_hill_valley_data_set,
            RD.load_horse_colic_data_set,
            RD.load_htru_data_set,
            RD.load_ilpd_data_set,

            RD.load_ionosphere_data_set,
            RD.load_kc3_data_set,
            RD.load_kidney_data_set,
            RD.load_led_display_data_set,
            RD.load_letter_recognition_data_set,
            RD.load_libras_data_set,
            RD.load_low_res_spect_data_set,
            RD.load_lowbwt_data_set,
            RD.load_lung_cancer_data_set,
            RD.load_lupus_data_set,
            RD.load_magic_data_set,
            RD.load_mammographic_data_set,
            RD.load_mfeat_data_set,
            RD.load_miniboone_data_set,
            RD.load_Mnist,
            RD.load_monks_1_data_set,
            RD.load_monks_2_data_set,
            RD.load_monks_3_data_set,
            RD.load_Mushroom,
            RD.load_musk_1_data_set,
            RD.load_no2_data_set,
            RD.load_optdigits,
            RD.load_ozone_data_set,
            RD.load_parkinsons_data_set,
            RD.load_phishing_websites_data_set,
            RD.load_pima_data_set,
            RD.load_planning_data_set,
            RD.load_plant_margin_data_set,
            RD.load_plant_shape_data_set,
            RD.load_pm10_data_set,
            RD.load_post_operative_data_set,

            RD.load_rabe_131_data_set,
            RD.load_schlvote_data_set,
            RD.load_seed_data_set,
            RD.load_semeion_data_set,
            RD.load_socmob_data_set,
            RD.load_soybean_data_set,
            RD.load_spambase_data_set,
            RD.load_spect_data_set,
            RD.load_statlog_heart_data_set,
            RD.load_steel_plates_data_set,
            RD.load_tae_data_set,
            RD.load_tic_data_set,
            RD.load_titanic_data_set,
            RD.load_transplant_data_set,
            RD.load_triazines_data_set,
            RD.load_turkiye_student_evaluation_data_set,
            RD.load_veteran_data_set,
            RD.load_vowel_data_set,
            RD.load_wall_following_data_set,
            RD.load_waveform_data_set,
            RD.load_white_clover_data_set,

            RD.load_winequality_white_data_set,
            RD.load_yeast_data_set,
            RD.load_zoo_data_set
        ]

        if data_set_id != 0 and data_set_id != 3 and data_set_id != 2 and data_set_id != 3 and data_set_id != 59:
            data, labels = dataset[data_set_id](return_X_y=True)

            # 对数据集进行归一化
            # data = self.data_mean_norm(data)
            # labels = (labels - np.mean(labels)) / np.std(labels)

            for _ in range(20):
                pi = np.random.permutation(len(data))
                data, labels = data[pi], labels[pi]

            self.data_cv = {'data_cv': data[:int(len(data) * 0.8)],
                            'labels_cv': labels[:int(len(data) * 0.8)],
                            'data_test': data[int(len(data) * 0.8):],
                            'labels_test': labels[int(len(data) * 0.8):]
                            }
            print("data_cv:", np.shape(self.data_cv['data_cv']), 'labels_cv:', np.shape(self.data_cv['labels_cv']))
            print("data_test:", np.shape(self.data_cv['data_test']), 'labels_test:',
                  np.shape(self.data_cv['labels_test']))
        else:
            train_x, train_y, test_x, test_y = dataset[data_set_id](return_X_y=True)

            # 对数据集进行归一化
            # train_x = self.data_mean_norm(train_x)
            # test_x = self.data_mean_norm(test_x)
            # train_y = (train_y - np.mean(train_y)) / np.std(train_y)
            # test_y = (test_y - np.mean(test_y)) / np.std(test_y)

            for _ in range(20):
                pi = np.random.permutation(len(train_x))
                train_x, train_y = train_x[pi], train_y[pi]
                pi = np.random.permutation(len(test_x))
                test_x, test_y = test_x[pi], test_y[pi]

            self.data_cv = {'data_cv': train_x,
                            'labels_cv': train_y,
                            'data_test': test_x,
                            'labels_test': test_y
                            }
            print("data_cv:", np.shape(self.data_cv['data_cv']), 'labels_cv:',
                  np.shape(self.data_cv['labels_cv']))
            print("data_test:", np.shape(self.data_cv['data_test']), 'labels_test:',
                  np.shape(self.data_cv['labels_test']))


def test():
    datamanager = DataManager()
    data = datamanager.data
    print("Train:", np.shape(data['train_data']), np.shape(data['train_labels']))
    print("Test:", np.shape(data['test_data']), np.shape(data['test_labels']))

    im = np.array(data['train_data'][20])
    im = im.reshape(28, 28)
    print(im)
    print(data['train_labels'][20])
    plt.imshow(im, cmap='gray')
    plt.show()

    im = np.array(data['test_data'][30])
    im = im.reshape(28, 28)
    print(im)
    print(data['test_labels'][30])
    plt.imshow(im, cmap='gray')
    plt.show()

    print('done')


if __name__ == "__main__":
    test()
