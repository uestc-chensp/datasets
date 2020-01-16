# coding:utf-8
import tensorflow as tf
import numpy as np
import time

from RL.AutoDataAnalyst_PPO_xgb_v3.code.configFile.AgentConfigFile import AgentConfig as Config
from RL.AutoDataAnalyst_PPO_xgb_v3.code.utils.XBGoostCliper import XGBoostCliper

# 类的功能：Agent.py文件的核心类，用于Agent的初始化和参数更新等；
# 拥有函数： __init__(): 构建LSTM类,并初始化；
#          getArgParams(): 获取机器学习模型的参数配置数据，用于构建机器学习模型和LSTM参数更新；
#          learn(): 学习更新LSTM的参数；
METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),  # KL penalty
    dict(name='clip', epsilon=0.2),  # Clipped surrogate objective, find this is better
][1]
MIN_E = 1e-6
A_UPDATE_STEPS = 10
agr_params_file_name = "../validate_time/params_data_agent(chen)/params.txt"
prob_file_name = "../validate_time/params_data_agent(chen)/prob.txt"


class LSTM:
    # 构建LSTM类,并初始化；
    # 输入参数：params: 机器学习算法所要搜索的超参数值的范围；
    def __init__(self, params):
        # 不添加下面这行代码，构建另一个LSTM模型结构的时候会报错！！！
        self.top_x = list(np.zeros([Config.batch_size * 2]))
        self.top_rewards = list(np.zeros([Config.batch_size * 2]))
        self.top_agr_params = list(np.zeros([Config.batch_size * 2]))
        tf.reset_default_graph()
        self.n_step = len(params)  # 该算法需要优化的超参数的个数
        print("self.n_step = ", self.n_step)
        self.x = []  # 输入数据
        self.labels = []  # 输出数据对应的标签
        self.reward = tf.placeholder(tf.float32, [Config.batch_size, 1], name='reward')  # 奖励值
        self.baseline = tf.placeholder(tf.float32, [], name='baseline')
        self.jj = tf.placeholder(tf.int16, name='jj')

        self.mu, self.sigma, self.y_, self.y, self.pi, self.pi_params = self.build_net('pi_net', trainable=True)
        self.old_mu, self.old_sigma, y_o_, y_o, oldpi, self.oldpi_params = self.build_net('opi_net', trainable=False)
        self.sample_op = [1] * self.n_step
        self.a = []
        self.b = []
        with tf.variable_scope('sample_action'):
            for i in range(self.n_step):
                self.sample_op[i] = oldpi[i].sample(Config.batch_size)  # choosing action

        with tf.variable_scope('update_oldpi'):
            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(self.pi_params, self.oldpi_params)]

        with tf.variable_scope('loss'):
            with tf.variable_scope('surrogate'):
                ratio = []
                for i in range(self.n_step):
                    ratio.append(tf.reduce_mean(
                        tf.exp(self.pi[i].prob(self.labels[i]) - oldpi[i].prob(self.labels[i]))))
                self.avg_ratio = tf.reduce_mean(ratio)
                self.surr = self.avg_ratio * (self.reward - self.baseline)
                self.aloss = -tf.reduce_mean(tf.minimum(
                    self.surr,
                    tf.clip_by_value(self.avg_ratio, 1. - METHOD['epsilon'], 1. + METHOD['epsilon']) * (
                                self.reward - self.baseline)))

        # 模型更新操作 train
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(Config.lr).minimize(self.aloss)

    def build_net(self, name, trainable):
        with tf.variable_scope(name):
            norm_dist = []
            # 1.输入层
            with tf.name_scope('input_layer'):
                for i in range(self.n_step):
                    x_temp = tf.placeholder(tf.float32, [1, 2], name='input_' + str(i + 1))
                    self.x.append(x_temp)
                    labels_temp = tf.placeholder(tf.float32, [Config.batch_size, 1], name='labels_' + str(i + 1))
                    self.labels.append(labels_temp)

            # 2.输入全连接层；因为LSTM每个时刻输入数据的维度不一致，添加此层的主要目的是使输入到LSTM_cell层的数据维度统一
            with tf.name_scope('layer_1'):
                # X_in = X*W + b 输入输出断的参数，要不要设置为一样的值？？？
                weights_in = []  # 输入层与LSTM_cell层之间的连接权重
                biases_in = []  # LSTM_cell层神经元的偏置参数
                for i in range(self.n_step):
                    weights_in_temp = tf.Variable(tf.random_uniform([2, Config.n_hidden_units], -0.1, 0.1),
                                                  trainable=trainable, name="weights_in_" + str(i + 1))
                    weights_in.append(weights_in_temp)
                    biases_in_temp = tf.Variable(tf.constant(0.1, shape=[Config.n_hidden_units]), trainable=trainable,
                                                 name="biases_in_" + str(i + 1))
                    biases_in.append(biases_in_temp)
                input_lstm = []  # LSTM_cell层的输入
                for i in range(self.n_step):
                    input_lstm_temp = tf.nn.bias_add(tf.matmul(self.x[i], weights_in[i]), biases_in[i])
                    input_lstm.append(input_lstm_temp)

            # 3.循环神经网络层（核心层）
            with tf.name_scope('LSTM_cell'):
                # 3层 每层35个单元
                stacked_lstm = tf.contrib.rnn.MultiRNNCell(
                    [tf.contrib.rnn.BasicLSTMCell(Config.n_hidden_units, forget_bias=1.0, state_is_tuple=True) for _ in
                     range(Config.n_layers)])
                state = stacked_lstm.zero_state(1, tf.float32)
                output = []  # LSTM_cell层的输出
                for i in range(self.n_step):
                    (output_temp, state) = stacked_lstm(input_lstm[i], state)  # 按照顺序向stacked_lstm输入数据
                    output.append(output_temp)
            if not trainable:
                a = tf.trainable_variables(name)
                l = len(a)
                for i in range(len(a)):
                    a.remove(a[l - i - 1])

            # 4.输出全连接层；因为LSTM每个时刻需要预测的数据维度是不一样的，为了达到要求，特在LSTM_cell层的后面加上这层进行数据转换，以满足需要
            with tf.name_scope('layer_4'):
                # X_in = X*W + b
                weights_out = []  # LSTM_cell层与输入层之间的连接权重
                biases_out = []  # 输入层神经元的偏置参数
                mu1 = []
                sigma1 = []
                y = []  # 输出数据
                y_ = []
                for i in range(self.n_step):
                    weights_out_temp = tf.Variable(tf.random_uniform([Config.n_hidden_units, 100], -0.1, 0.1),
                                                   trainable=trainable, name="weights_out_" + str(i + 1))
                    weights_out.append(weights_out_temp)
                    biases_out_temp = tf.Variable(tf.constant(0.1, shape=[100]), trainable=trainable,
                                                  name="biases_out_" + str(i + 1))
                    biases_out.append(biases_out_temp)
                    y_temp = tf.nn.bias_add(tf.matmul(output[i], weights_out[i]), biases_out[i])
                    y.append(y_temp)

            for i in range(self.n_step):
                mu = tf.layers.dense(y[i], 1, tf.nn.tanh, trainable=trainable)
                mu1.append(mu)
                sigma = tf.layers.dense(y[i], 1, tf.nn.softplus, trainable=trainable)
                sigma1.append(sigma)
                norm_dist1 = tf.distributions.Normal(loc=mu, scale=sigma)
                norm_dist.append(norm_dist1)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return mu1, sigma1, y_, y, norm_dist, params

    # 函数功能：获取机器学习模型参数配置数据；
    # 输入参数：sess: Session()会话对象；
    # 输出参数：x： 输入数据，用于LSTM参数更新；
    #         agr_params： 所选中的参数对应的索引位置，用于构建机器学习模型和LSTM参数更新；
    def getArgParams(self, sess, init_input_c):
        x = []  # 输入数据 x
        sample_actions = []  # 各参数概率
        action = []  # 各参数对应的值
        pi_s = []
        init_input = init_input_c
        xgboostCliper = XGBoostCliper()
        for i in range(self.n_step):
            x.append(init_input)
            feed_dict = {}
            for j in range(i + 1):
                feed_dict[self.x[j]] = x[j]
            # 上一超参数的输出 等于 下一超参数的输入  mu sigma
            dist_param = sess.run([self.mu[i], self.sigma[i]], feed_dict=feed_dict)
            init_input = np.array(dist_param).reshape(1, 2)
            pi_s.append(init_input)
            # 通过均值、方差进行采样，得到output  sample_op[i] 8*1
            sample_action = sess.run(self.sample_op[i], feed_dict=feed_dict)
            sample_actions.append(sample_action)
            # clip动作  先softmax
            # a = np.exp(agr_params_temp)
            # agr_params_temp = a / np.sum(a)
            action.append(xgboostCliper.convert_to_action(sample_action, i, init_input))
            # 把最后一个超参数的输出赋值给第一个x
            # if(i==self.n_step-1):
            #     x[0]=init_input
        # 转置后 agr_params为8*5,每一行都是一组超参数    这样便于获取reward
        sample_actions = (np.array(sample_actions).reshape(self.n_step, Config.batch_size)).T
        action = (np.array(action).reshape(self.n_step, Config.batch_size)).T
        return x, sample_actions, action, init_input, pi_s

    # 函数功能：学习更新LSTM的参数；
    # 输入参数：sess: Session()会话对象
    #         x： 输入数据，用于LSTM参数更新； [batch_size, params]
    #         labels： 每一时刻选取参数的索引位置
    #         rewards: 每个算法结构训练完成后，在验证数据集上得到的准确率
    #         baseline_reward： 奖励基准值
    #         agr_params： 所选中的参数对应的索引位置，用于构建机器学习模型和LSTM参数更新；
    # 输入参数：loss: LSTM更新后得到的代价函数值；
    def learn(self, isguid, sess, x1, labels1, rewards1, baseline_reward, jj):
        sess.run(self.update_oldpi_op)
        if isguid:
            feed_dict = {}
            for i in range(Config.batch_size * 2):
                x = x1[i]
                labels = np.reshape([labels1[i] for _ in range(Config.batch_size)], [Config.batch_size, self.n_step])
                for j in range(self.n_step):
                    feed_dict[self.x[j]] = np.array(x[j]).reshape(1, 2)
                    feed_dict[self.labels[j]] = (np.array(labels[:, j])).reshape(Config.batch_size, 1)
                feed_dict[self.reward] = np.reshape([rewards1[i] for _ in range(Config.batch_size)],
                                                    [Config.batch_size, 1])
                feed_dict[self.baseline] = baseline_reward
                feed_dict[self.jj] = jj
                _, loss, ratio, surr = sess.run([self.train_op, self.aloss, self.avg_ratio, self.surr],
                                                feed_dict=feed_dict)
                print("loss:" + str(loss) + " ratio:" + str(ratio) + " surr:" + str(surr))
            return loss, ratio
        else:
            feed_dict = {}  # self.train_op更新过程中传递的参数
            for j in range(self.n_step):
                feed_dict[self.x[j]] = np.array(x1[j]).reshape(1, 2)
                feed_dict[self.labels[j]] = (np.array(labels1[:, j])).reshape(Config.batch_size, 1)
            feed_dict[self.reward] = (np.array(rewards1)).reshape(Config.batch_size, 1)
            feed_dict[self.baseline] = baseline_reward
            feed_dict[self.jj] = jj
            for i in range(A_UPDATE_STEPS):
                _, loss, ratio= sess.run(
                    [self.train_op, self.aloss, self.avg_ratio], feed_dict=feed_dict)
                # with open(prob_file_name, 'a') as f:
                #     f.write("[" + str(jj) + "," + str(i) + "]: \n" +
                #             str(aa) + "," + str(bb) + "\n" + str(ratio) + "\n ")
            print("loss:" + str(loss) + " ratio:" + str(ratio))
            return loss, ratio

    # top_reward  top_agr_params 中永远是从大到小排序的 永远是最优的
    def check_topData(self, x, agr_params, rewards):
        # x=np.array(x).reshape(Config.batch_size,self.n_step)
        for i in range(Config.batch_size):
            for j in range(Config.batch_size * 2):
                if rewards[i] in self.top_rewards:
                    break
                if rewards[i] > self.top_rewards[j]:
                    self.top_rewards[j + 1:] = self.top_rewards[j:-1]
                    self.top_rewards[j] = rewards[i]

                    self.top_x[j + 1:] = self.top_x[j:-1]
                    self.top_x[j] = x

                    self.top_agr_params[j + 1:] = self.top_agr_params[j:-1]
                    self.top_agr_params[j] = agr_params[i]
                    break

    # kk 代表第几个算法
    # 双引导池
    def getInput(self):
        x = []  # 输入数据 x
        top_rewards = self.top_rewards
        top_agr_params = self.top_agr_params
        # top_reward 为最好的 reward 8*1
        top_rewards = top_rewards
        # top_agr_params 为最好的reward对应的超参数  8*5
        top_agr_params = np.array(top_agr_params).reshape(Config.batch_size * 2, self.n_step)
        x = np.array(self.top_x)[0:16, :]
        # 返回最优的并且是完整的  x
        return x, top_agr_params, top_rewards
