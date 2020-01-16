# coding:utf-8
import numpy as np
import pandas as pd
import warnings
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import cross_val_score
from hyperopt import fmin, tpe, hp, space_eval
import tensorflow as tf
from RL.AutoDataAnalyst_PPO_xgb_v3.code.datamanager.DataManager import DataManager
import time
import os
from RL.AutoDataAnalyst_PPO_xgb_v3.code.utils.test_set_result import test_set_result
import warnings

top_5_config = {"reward": [0, 0, 0, 0, 0], "runtime": [0, 0, 0, 0, 0], "config": [0, 0, 0, 0, 0]}
test_result_path = "/home/chensp/code/RL/RL/AutoDataAnalyst_PPO_xgb_v3/validate_time/top_5_config_test_result.csv"


def t_main_tpe(data_manager, plot_data_path, log_path):  # file_name, data_file_name, data_dict_file
    plot_data = {"time": [], "reward": [], "param": []}

    global hot_method
    global data_cv, labels_cv
    global params, rewards
    global a
    global max_reward, min_time
    max_reward = 0
    min_time = 0
    a = 0
    hot_method = {"paras": [], "rewards": [], "time": []}
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    data_manager = data_manager
    data_cv, labels_cv = data_manager.data_cv['data_cv'], data_manager.data_cv['labels_cv']
    data_test, labels_test = data_manager.data_cv['data_test'], data_manager.data_cv['labels_test']
    params = []
    rewards = []
    global times
    start_time = time.time()
    times = []
    summary_writter = tf.summary.FileWriter(log_path, sess.graph)

    # def save_data_dict(hot_method_p):
    #     data = pd.DataFrame(data=hot_method_p)
    #     data.to_csv(data_dict_file, index=False)
    #     data_length = len(hot_method_p["paras"])
    #     print("successfull !！！ save total ", data_length, " data!")
    #     return data_length
    #
    # def restore_data_dict():
    #     global hot_method
    #     data_dict = pd.read_csv(data_dict_file, index_col=False)
    #     hot_method["paras"] = list(data_dict["paras"].values)
    #     hot_method["rewards"] = list(data_dict["rewards"].values)
    #     data_length = len(hot_method["paras"])
    #     print("successfull !！！ restore total ", data_length, " data!")
    #     return data_length

    # 未通过scalar的，使用该函数
    def summarize(summary_writter, value, step, tag):
        summary = tf.Summary()
        summary.value.add(tag=tag, simple_value=value)
        summary_writter.add_summary(summary, step)
        summary_writter.flush()

    def update_top_5_config(reward, config, runtime):
        for index in range(5):
            if reward > top_5_config["reward"][index]:
                top_5_config["reward"][index + 1:] = top_5_config["reward"][index:-1]
                top_5_config["reward"][index] = reward
                top_5_config["runtime"][index + 1:] = top_5_config["runtime"][index:-1]
                top_5_config["runtime"][index] = runtime
                top_5_config["config"][index + 1:] = top_5_config["config"][index:-1]
                top_5_config["config"][index] = config
                break

    def func(args, flag=None):
        global hot_method
        global params, times, rewards
        global data_cv, labels_cv
        global a
        global max_reward, min_time
        max_depth = args["max_depth"]
        learning_rate = args["learning_rate"]
        n_estimators = args["n_estimators"]
        gamma = args["gamma"]
        min_child_weight = args["min_child_weight"]
        subsample = args["subsample"]
        colsample_bytree = args["colsample_bytree"]
        colsample_bylevel = args["colsample_bylevel"]
        reg_alpha = args["reg_alpha"]
        reg_lambda = args["reg_lambda"]

        agr_params = [int(max_depth), float(learning_rate), int(n_estimators)
            , float(gamma), int(min_child_weight), float(subsample)
            , float(colsample_bytree), float(colsample_bylevel), float(reg_alpha)
            , float(reg_lambda)]
        agr_params = np.array(agr_params).reshape(1, len(agr_params))
        val = 0
        if np.any(str(agr_params[0]) in hot_method["paras"]):
            index = hot_method["paras"].index(str(agr_params[0]))
            val = hot_method["rewards"][index]
            one_step_time = hot_method["time"][index]
            print("if!!!")
        else:
            print("else!!!")
            xgb = XGBClassifier(max_depth=int(max_depth),
                                learning_rate=float(learning_rate),
                                n_estimators=int(n_estimators),
                                gamma=float(gamma),
                                min_child_weight=int(min_child_weight),
                                subsample=float(subsample),
                                colsample_bytree=float(colsample_bytree),
                                colsample_bylevel=float(colsample_bylevel),
                                reg_alpha=float(reg_alpha),
                                reg_lambda=float(reg_lambda),
                                nthread=-1)
            one_step_start_time = time.time()
            results = cross_val_score(xgb, data_cv, labels_cv, cv=2, n_jobs=1)
            one_step_time = time.time() - one_step_start_time
            val = np.mean(results)
            hot_method["paras"].append(str(agr_params[0]))
            hot_method["rewards"].append(val)
            hot_method["time"].append(one_step_time)
        print("val:" + str(val))
        print("time:" + str(one_step_time))
        if flag == None:
            params.append(args)
            time_p = time.time()
            times.append((time_p - start_time) / 60)
            rewards.append(val)
            # times.append(time_p - start_time)
            update_top_5_config(val, agr_params[0], (time_p - start_time) / 60)
            plot_data["time"].append((time_p - start_time) / 60)
            plot_data["reward"].append(val)
            plot_data["param"].append(str(agr_params[0]))
            a = a + 1
            if a % 5 == 0:
                data = pd.DataFrame(plot_data)
                data.to_csv(plot_data_path, index=False)
            # if val>max_reward:
            #     max_reward=val
            #     min_time=one_step_time
            summarize(summary_writter, val, a, 'reward')
            summarize(summary_writter, (time_p - start_time) / 60, a, 'time')
        return -val

    space = {
        'max_depth': hp.uniform('max_depth', 1, 25),
        'learning_rate': hp.uniform('learning_rate', 0.001, 0.1),
        'n_estimators': hp.uniform('n_estimators', 50, 1200),
        'gamma': hp.uniform('gamma', 0.05, 0.9),
        'min_child_weight': hp.uniform('min_child_weight', 1, 9),
        'subsample': hp.uniform('subsample', 0.5, 1.0),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),
        'colsample_bylevel': hp.uniform('colsample_bylevel', 0.5, 1.0),
        'reg_alpha': hp.uniform('reg_alpha', 0.1, 0.9),
        'reg_lambda': hp.uniform('reg_lambda', 0.01, 0.1)  # 2
    }
    # ------------huifu lishi shuju---------------;
    # restore_data_dict()
    start_time = time.time()
    best = fmin(func, space, algo=tpe.suggest, max_evals=200)
    test_set_top_5_mean = test_set_result(data_manager, top_5_config["config"])
    test_result = {"method": ["tpe"],
                   "dataset": [data_manager.data_set_index],
                   "top_5_mean_test_reward": [test_set_top_5_mean],
                   "top_5_mean_val_reward": [np.mean(top_5_config["reward"])],
                   "top_5_mean_time": [str(np.mean(top_5_config["runtime"])) + "s"]}
    test_result_df = pd.DataFrame(test_result)
    test_result_df.to_csv(test_result_path, mode="a", header=False)
    print("---------训练结束!----------")
    # params = pd.DataFrame(params)
    # params["time"] = times
    # params["accuracy"] = rewards
    # params.to_csv(data_file_name, index=False)

    # def print_test_accuracy(args, data_cv, labels_cv, data_test, labels_test):
    #     xbgoost = XGBClassifier(max_depth=int(args['n_estimators']),
    #                             learning_rate=float(args['learning_rate']),
    #                                             n_estimators=int(args['n_estimators']),
    #                                             gamma=float(args['gamma']),
    #                                             min_child_weight=int(args['min_child_weight']),
    #                                             subsample=float(args['subsample']),
    #                                             colsample_bytree=float(args['colsample_bytree']),
    #                                             colsample_bylevel=float(args['colsample_bylevel']),
    #                                             reg_alpha=float(args['reg_alpha']),
    #                                             reg_lambda=float(args['reg_lambda']),
    #                                             nthread=-1)
    #     xbgoost.fit(data_cv, labels_cv)
    #     val = xbgoost.score(data_test, labels_test)
    #     return val
    #
    # test_accuracy = print_test_accuracy(space_eval(space, best), data_cv, labels_cv, data_test, labels_test)
    #
    # with open(file_name, 'a') as f:
    #     f.write("\n params=\n " + str(params))
    # with open(file_name, 'a') as f:
    #     f.write("\n best_action_index= " + str(best) + "\n best_action_param= " + str(
    #         space_eval(space, best)) + "\n best_action_accuracy= " + str(
    #         func(space_eval(space, best), 1)) + "\n test_accuracy= " + str(test_accuracy))
    # over_time = time.time()
    # sum_time = over_time - start_time
    # with open(file_name, 'a') as f:
    #     f.write("\n finish ---- search hyperParams of the algorithm ," + "start_time= " + str(start_time) +
    #             ", over_time= " + str(over_time) + ", sum_time = " + str(sum_time) + "\n")
    # # -------------------baocun lishi shuju-----------------------;
    # save_data_dict(hot_method)
    #
    # print("best_action_index", best)
    # print("-----best_action_param:", space_eval(space, best))
    # print("-----best_action_accuracy,", func(space_eval(space, best), 1))
    # print('RFC, test_accuracy=', test_accuracy)
    # print("----------TPE 算法运行结束！----------")
    # # print("params:", params)
    # del data_cv, labels_cv, data_test, labels_test
    # del params, rewards, times, start_time


data_manager = DataManager(1)
if __name__ == '__main__':
    warnings.filterwarnings(action='ignore', category=DeprecationWarning)
    plot_time_reward = "../../../validate_time/params_data_agent(chen)/plot_time_data.csv"
    t_main_tpe(data_manager, plot_time_reward, "../../../logs/log_agent_model")
