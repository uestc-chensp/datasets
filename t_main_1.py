# coding:utf-8
import tensorflow as tf
import pandas as pd
import numpy as np
import time
import warnings

from RL.AutoDataAnalyst_PPO_xgb_v3.code.DataManager import DataManager
from RL.AutoDataAnalyst_PPO_xgb_v3.code.t_Agent_1 import LSTM
from RL.AutoDataAnalyst_PPO_xgb_v3.code.EnvironmentManager import EnvironmentManager
from RL.AutoDataAnalyst_PPO_xgb_v3.code.configFile.MainConfigureFile import MainConfig
from RL.AutoDataAnalyst_PPO_xgb_v3.code.configFile.AgentConfigFile import AgentConfig
from RL.AutoDataAnalyst_PPO_xgb_v3.code.NNet import NNet


def t_main_1(data_manager, plot_time_reward, log_path):
    data_manager = data_manager
    envManager = EnvironmentManager(data_manager)
    envManager.auto_create_multi_singleprocess_envs()
    plot_data = {"time": [], "rewards_max": [], "rewards_mean": [], "reward_min": [],"action":[]}
    for i in range(1):
        Env, params = envManager.next_environment()
        agent = LSTM(params)
        nnet = NNet(len(params))
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            baseline_reward = 0
            a = [0, 1]
            init_input = np.array(a).reshape(1, 2)
            start_time = time.time()
            summary_writter = tf.summary.FileWriter(log_path, sess.graph)
            final_max_acc=0
            final_max_acc_time=0
            for j in range(MainConfig.num_train1):
                x, agr_params, action, _, _ = agent.getArgParams(sess, init_input)
                # #不使用神经网络
                time_start=time.time()
                rewards = Env.run(action)
                rewards=rewards[0]
                time_end=time.time()
                # 5:定义log写入流
                summarize(summary_writter, np.max(rewards), (i * MainConfig.num_train) + j, 'reward')
                summarize(summary_writter, np.mean(rewards), (i * MainConfig.num_train) + j, 'mean_reward')
                one_time=time_end-time_start
                summarize(summary_writter, one_time / 60, (i * MainConfig.num_train) + j, 'run_time')
                plot_data["time"].append(one_time)
                plot_data["rewards_max"].append(np.max(rewards))
                np.set_printoptions(suppress=True)
                plot_data["action"].append(action)
                plot_data["rewards_mean"].append(np.mean(rewards))
                plot_data["reward_min"].append(np.min(rewards))
                if j % 5 == 0:
                    plot = pd.DataFrame(data=plot_data)
                    plot.to_csv(plot_time_reward, index=False)
                if j == 0:
                    baseline_reward = np.mean(rewards)

                if np.mean(rewards)>final_max_acc:
                    final_max_acc=np.mean(rewards)
                    final_max_acc_time=one_time
                summarize(summary_writter, final_max_acc, (i * MainConfig.num_train) + j, 'final_max_acc')
                summarize(summary_writter, final_max_acc_time, (i * MainConfig.num_train) + j, 'final_max_acc_time')
                print("-------Max_Rewards:" + str(final_max_acc) + "--------")
                print("-------Max_Rewards_Time:" + str(final_max_acc_time) + "----------")
                print("else: normal training, rewards:", rewards)
                loss, ratio = agent.learn(False, sess, x, agr_params, rewards, baseline_reward, j)
                print("i=", i, " j=", j, "average_reward=", np.mean(rewards), " baseline_reward=", baseline_reward,
                      " loss=", loss, "\n")
                summarize(summary_writter, loss, j, 'loss')
                summarize(summary_writter, ratio, j, 'ratio')
                reward_c = np.mean(rewards)
                # init_input=x[-1]
                baseline_reward = baseline_reward * AgentConfig.dr + (1 - AgentConfig.dr) * reward_c
            plot = pd.DataFrame(data=plot_data)
            plot.to_csv(plot_time_reward, index=False)

    print("---------训练结束!----------")


# 未通过scalar的，使用该函数
def summarize(summary_writter, value, step, tag):
    summary = tf.Summary()
    summary.value.add(tag=tag, simple_value=value)
    summary_writter.add_summary(summary, step)
    summary_writter.flush()


# data_manager = DataManager()
# if __name__ == '__main__':
#     warnings.filterwarnings(action='ignore', category=DeprecationWarning)
#     plot_time_reward = "../validate_time/params_data_agent(chen)/plot_time_data.csv"
#     t_main_1(data_manager, plot_time_reward)
