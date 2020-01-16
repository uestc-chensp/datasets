import numpy as np
import os
import pandas as pd
import time
import tensorflow as tf
import warnings
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import cross_val_score
from RL.AutoDataAnalyst_PPO_xgb_v3.code.datamanager.DataManager import DataManager
from RL.AutoDataAnalyst_PPO_xgb_v3.code.utils.test_set_result import test_set_result
import chocolate as choco
import matplotlib.pyplot as plt
import seaborn as sns;

sns.set()
from chocolate import SQLiteConnection

top_5_config = {"reward": [0, 0, 0, 0, 0], "runtime": [0, 0, 0, 0, 0], "config": [0, 0, 0, 0, 0]}
test_result_path = "/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/validate_time/top_5_config_test_result.csv"


def t_main_cmaes(data_manager, n, file_name, log_path, conn):
    data_cv, labels_cv = data_manager.data_cv['data_cv'], data_manager.data_cv['labels_cv']
    data_test, labels_test = data_manager.data_cv['data_test'], data_manager.data_cv['labels_test']
    sess = tf.Session()
    summary_writter = tf.summary.FileWriter(log_path, sess.graph)

    def score_gbt(params):
        xgb = XGBClassifier(**params)
        start_t = time.time()
        results = cross_val_score(xgb, data_cv, labels_cv, cv=5, n_jobs=1)
        one_step_time = time.time() - start_t
        val = np.mean(results)
        return -val, one_step_time

    space = {
        'max_depth': choco.uniform(1, 25),
        'learning_rate': choco.uniform(0.001, 0.1),
        'n_estimators': choco.uniform(50, 1200),
        'gamma': choco.uniform(0.05, 0.9),
        'min_child_weight': choco.uniform(1, 9),
        'subsample': choco.uniform(0.5, 1.0),
        'colsample_bytree': choco.uniform(0.5, 1.0),
        'colsample_bylevel': choco.uniform(0.5, 1.0),
        'reg_alpha': choco.uniform(0.1, 0.9),
        'reg_lambda': choco.uniform(0.01, 0.1)  # 2
    }
    sampler = choco.CMAES(conn, space)
    plot_data = {"clock_time": [], "reward": [], "param": [], "run_time": []}
    start_time = time.time()
    max_reward = 0
    min_time = 0
    for i in range(n):
        token, params = sampler.next()
        print(params)
        params['max_depth'] = int(params['max_depth'])
        params['n_estimators'] = int(params['n_estimators'])
        params['min_child_weight'] = int(params['min_child_weight'])
        loss, run_time = score_gbt(params)
        print(loss)
        sampler.update(token, loss)
        end_time = time.time()
        one_time = end_time - start_time
        summarize(summary_writter, -loss, i, 'reward')
        summarize(summary_writter, one_time, i, 'time')
        update_top_5_config(-loss, params, one_time)
        # if -loss>max_reward:
        #     max_reward=-loss
        #     min_time=run_time
        # summarize(summary_writter, max_reward, i, 'final_max_acc')
        # summarize(summary_writter, min_time, i, 'final_max_acc_time')
        plot_data["clock_time"].append(one_time)
        plot_data["run_time"].append(run_time)
        plot_data["reward"].append(-loss)
        plot_data["param"].append(params)
        plot = pd.DataFrame(data=plot_data)
        plot.to_csv(file_name, index=False)
    test_set_top_5_mean = test_set_result(data_manager, top_5_config["config"])
    test_result = {"method": ["cmaes"],
                   "dataset": [data_manager.data_set_index],
                   "top_5_mean_test_reward": [test_set_top_5_mean],
                   "top_5_mean_val_reward": [np.mean(top_5_config["reward"])],
                   "top_5_mean_time": [str(np.mean(top_5_config["runtime"])) + "s"]}
    test_result_df = pd.DataFrame(test_result)
    test_result_df.to_csv(test_result_path, mode="a", header=False)
    print("---------训练结束!----------")


def summarize(summary_writter, value, step, tag):
    summary = tf.Summary()
    summary.value.add(tag=tag, simple_value=value)
    summary_writter.add_summary(summary, step)
    summary_writter.flush()


def update_top_5_config(reward, config, runtime):
    config = list(config.values())
    for index in range(5):
        if reward > top_5_config["reward"][index]:
            top_5_config["reward"][index + 1:] = top_5_config["reward"][index:-1]
            top_5_config["reward"][index] = reward
            top_5_config["runtime"][index + 1:] = top_5_config["runtime"][index:-1]
            top_5_config["runtime"][index] = runtime
            top_5_config["config"][index + 1:] = top_5_config["config"][index:-1]
            top_5_config["config"][index] = config
            break


data_manager = DataManager(1)
if __name__ == '__main__':
    #     n=1
    # conn = choco.SQLiteConnection(url="sqlite:///db.db")
    # results = conn.results_as_dataframe()
    # results = results['_loss']
    # results=np.array(results).reshape(len(results),1)
    # plot_time_reward = "../data.csv"
    # plot = pd.DataFrame(data=results)
    # plot.to_csv(plot_time_reward, index=False)
    n = 6
    warnings.filterwarnings("ignore")
    file_name = "../../../validate_time/params_data_agent(chen)/plot_time_data.csv"
    log_path = "../../../logs/log_agent_model"
    url_path = "sqlite:///mnistdb.db"
    conn = choco.SQLiteConnection(url=url_path)
    t_main_cmaes(data_manager, n, file_name, log_path, conn)
    # os.mkdir("./val/mnist_cmaes_"+str(i))
    # path="mnist_cmaes_"+str(i)
    # file_name = "./val/"+path+"/data.csv"
    # url_path="sqlite:///mnistdb.db"+str(i)
    # conn = choco.SQLiteConnection(url=url_path)
    # CMAES(data_manager,n,file_name,conn)
