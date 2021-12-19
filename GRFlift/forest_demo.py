
import sys
sys.path.append("./CausalModel")
import os
import arrow
import pickle as pkl
from TreeModel.randomForest import UpliftRandomForestClassifier
from TreeModel.model.roiTree import UpliftTreeRegressor
from TreeModel.evaluate.auuc import auuc,perfect_uplift_curve,uplift_curve,plot_uplift_curve
import datetime
import time
import pandas as pd
import numpy as np
# from demo import data_preprocess
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import json
import random



def forest_train(df_train, n_estimators, n_jobs, datas_mode='random', max_datas=1.0, max_features=0.5, save_path='treelift.pkl'):  # 森林设置不同树个数、不同线程数
    x_names = df_train.columns.tolist()[:-3]

    uplift_model = UpliftRandomForestClassifier(
        n_estimators=n_estimators,
        datas_mode = datas_mode,
        max_features=max_features,
        max_datas = max_datas,
        max_depth=6,
        min_samples_leaf=1000,
        min_samples_treatment=100,
        n_reg=100,
        evaluationFunction='Gmv',
        control_name='control',
        x_names=x_names,
        n_jobs=n_jobs
    )

    uplift_model.fit(X=df_train[x_names].values,
                        treatment=df_train['group_type'].values,
                        y=df_train['y'].values,
                        c=df_train['cost'].values)

    return uplift_model


def test_model(df_test, uplift_model):
    x_names = df_test.columns.tolist()[:-3]
    treatment_optimal, p_hat_optimal = uplift_model.predict(df_test[x_names].values)

    y_true = df_test['y'].values
    treatment_true = df_test['group_type'].values
    treatment_optimal = np.array(treatment_optimal)
    uplift_optimal = np.array(p_hat_optimal)

    for i in range(len(treatment_true)):
        if treatment_true[i] == 'control':
            treatment_true[i] = 0
        else:
            treatment_true[i] = 1

    Figure = plot_uplift_curve(y_true, uplift_optimal, treatment_true, random=True, perfect=False)
    AUUC = auuc(y_true, uplift_optimal, treatment_true)

    return AUUC, Figure



def save_model_local(uplift_model=None, save_path='model.pkl'):
    if uplift_model is None:
        raise ValueError("待保存的模型不能为空")

    if save_path == 'model.pkl':
        timestamp = arrow.now().format('YYYYMMDDHHmm')
        save_path = './saved_model/' + f'{timestamp}.pkl'
    else:
        save_path = './saved_model/' + save_path
    with open(save_path, 'wb') as inf:
        pkl.dump(uplift_model, inf)

    logger.warning(f"Model Saved in {save_path}")







    # # 1颗树，100%数据，100%特征
    # start_time = datetime.datetime.now()
    # model = forest_train(train_data, n_estimators=1, n_jobs=1, 
    #                         datas_mode='random', max_datas=1.0, 
    #                         max_features=1.0, save_path='treelift.pkl')
    # delta = datetime.datetime.now() - start_time
    # delta_gmtime = time.gmtime(delta.total_seconds())
    # duration_str_3 = time.strftime("%H:%M:%S", delta_gmtime)
    # # 每次测试前要重新读测试数据
    # test_data = data_preprocess('test_data1_20211123.csv')
    # AUUC, Figure = test_model_udf(test_data, model)
    # # 保存结果
    # record_id = os.environ.get('EXECUTE_RECORD_ID', str(random.randint(1, 10000000)))
    # experiment_name = '1颗树100%数据100%特征'
    # image_path = f"{experiment_name}_{record_id}.png"
    # auuc_path = f'{experiment_name}_{record_id}.json'
    # Figure.figure.savefig(image_path)
    # with open(auuc_path, 'w') as file:
    #     json.dump({'auuc': AUUC, 'experiment_name': experiment_name, 'time': duration_str_3}, file)
    # # 保存oss
    # store_res_to_oss(image_path, auuc_path)


    # # 5颗树，100%数据，20%特征
    # start_time = datetime.datetime.now()
    # model = forest_train(train_data, n_estimators=5, n_jobs=5, 
    #                         datas_mode='random', max_datas=1.0, 
    #                         max_features=0.2, save_path='treelift.pkl')
    # delta = datetime.datetime.now() - start_time
    # delta_gmtime = time.gmtime(delta.total_seconds())
    # duration_str_3 = time.strftime("%H:%M:%S", delta_gmtime)
    # # 每次测试前要重新读测试数据
    # test_data = data_preprocess('test_data1_20211123.csv')
    # AUUC, Figure = test_model_udf(test_data, model)
    # # 保存结果
    # record_id = os.environ.get('EXECUTE_RECORD_ID', str(random.randint(1, 10000000)))
    # experiment_name = '5颗树100%数据20%特征3'
    # image_path = f"{experiment_name}_{record_id}.png"
    # auuc_path = f'{experiment_name}_{record_id}.json'
    # Figure.figure.savefig(image_path)
    # with open(auuc_path, 'w') as file:
    #     json.dump({'auuc': AUUC, 'experiment_name': experiment_name, 'time': duration_str_3}, file)
    # # 保存oss
    # store_res_to_oss(image_path, auuc_path)

    
    # 5颗树，100%数据，50%特征
    start_time = datetime.datetime.now()
    model = forest_train(train_data, n_estimators=5, n_jobs=5, 
                            datas_mode='random', max_datas=1.0, 
                            max_features=0.5, save_path='treelift.pkl')
    delta = datetime.datetime.now() - start_time
    delta_gmtime = time.gmtime(delta.total_seconds())
    duration_str_3 = time.strftime("%H:%M:%S", delta_gmtime)
    # 每次测试前要重新读测试数据
    test_data = data_preprocess('test_data1_20211123.csv')
    AUUC, Figure = test_model_udf(test_data, model)
    # 保存结果
    record_id = os.environ.get('EXECUTE_RECORD_ID', str(random.randint(1, 10000000)))
    experiment_name = '5颗树100%数据50%特征1'
    image_path = f"{experiment_name}_{record_id}.png"
    auuc_path = f'{experiment_name}_{record_id}.json'
    Figure.figure.savefig(image_path)
    with open(auuc_path, 'w') as file:
        json.dump({'auuc': AUUC, 'experiment_name': experiment_name, 'time': duration_str_3}, file)
    # 保存oss
    store_res_to_oss(image_path, auuc_path)


    # # 5颗树，100%数据，80%特征
    # start_time = datetime.datetime.now()
    # model = forest_train(train_data, n_estimators=5, n_jobs=5, 
    #                         datas_mode='random', max_datas=1.0, 
    #                         max_features=0.8, save_path='treelift.pkl')
    # delta = datetime.datetime.now() - start_time
    # delta_gmtime = time.gmtime(delta.total_seconds())
    # duration_str_3 = time.strftime("%H:%M:%S", delta_gmtime)
    # # 每次测试前要重新读测试数据
    # test_data = data_preprocess('test_data1_20211123.csv')
    # AUUC, Figure = test_model_udf(test_data, model)
    # # 保存结果
    # record_id = os.environ.get('EXECUTE_RECORD_ID', str(random.randint(1, 10000000)))
    # experiment_name = '5颗树100%数据80%特征3'
    # image_path = f"{experiment_name}_{record_id}.png"
    # auuc_path = f'{experiment_name}_{record_id}.json'
    # Figure.figure.savefig(image_path)
    # with open(auuc_path, 'w') as file:
    #     json.dump({'auuc': AUUC, 'experiment_name': experiment_name}, file)
    # # 保存oss
    # store_res_to_oss(image_path, auuc_path)


    # # 10颗树，100%数据，20%特征
    # start_time = datetime.datetime.now()
    # model = forest_train(train_data, n_estimators=10, n_jobs=10, 
    #                         datas_mode='random', max_datas=1.0, 
    #                         max_features=0.2, save_path='treelift.pkl')
    # delta = datetime.datetime.now() - start_time
    # delta_gmtime = time.gmtime(delta.total_seconds())
    # duration_str_3 = time.strftime("%H:%M:%S", delta_gmtime)
    # # 每次测试前要重新读测试数据
    # test_data = data_preprocess('test_data1_20211123.csv')
    # AUUC, Figure = test_model_udf(test_data, model)
    # # 保存结果
    # record_id = os.environ.get('EXECUTE_RECORD_ID', str(random.randint(1, 10000000)))
    # experiment_name = '10颗树100%数据20%特征3'
    # image_path = f"{experiment_name}_{record_id}.png"
    # auuc_path = f'{experiment_name}_{record_id}.json'
    # Figure.figure.savefig(image_path)
    # with open(auuc_path, 'w') as file:
    #     json.dump({'auuc': AUUC, 'experiment_name': experiment_name, 'time': duration_str_3}, file)
    # # 保存oss
    # store_res_to_oss(image_path, auuc_path)


    # # 10颗树，100%数据，50%特征
    # start_time = datetime.datetime.now()
    # model = forest_train(train_data, n_estimators=10, n_jobs=10, 
    #                         datas_mode='random', max_datas=1.0, 
    #                         max_features=0.5, save_path='treelift.pkl')
    # delta = datetime.datetime.now() - start_time
    # delta_gmtime = time.gmtime(delta.total_seconds())
    # duration_str_3 = time.strftime("%H:%M:%S", delta_gmtime)
    # # 每次测试前要重新读测试数据
    # test_data = data_preprocess('test_data1_20211123.csv')
    # AUUC, Figure = test_model_udf(test_data, model)
    # # 保存结果
    # record_id = os.environ.get('EXECUTE_RECORD_ID', str(random.randint(1, 10000000)))
    # experiment_name = '10颗树100%数据50%特征3'
    # image_path = f"{experiment_name}_{record_id}.png"
    # auuc_path = f'{experiment_name}_{record_id}.json'
    # Figure.figure.savefig(image_path)
    # with open(auuc_path, 'w') as file:
    #     json.dump({'auuc': AUUC, 'experiment_name': experiment_name, 'time': duration_str_3}, file)
    # # 保存oss
    # store_res_to_oss(image_path, auuc_path)


    # # 10颗树，100%数据，80%特征
    # start_time = datetime.datetime.now()
    # model = forest_train(train_data, n_estimators=10, n_jobs=10, 
    #                         datas_mode='random', max_datas=1.0, 
    #                         max_features=0.8, save_path='treelift.pkl')
    # delta = datetime.datetime.now() - start_time
    # delta_gmtime = time.gmtime(delta.total_seconds())
    # duration_str_3 = time.strftime("%H:%M:%S", delta_gmtime)
    # # 每次测试前要重新读测试数据
    # test_data = data_preprocess('test_data1_20211123.csv')
    # AUUC, Figure = test_model_udf(test_data, model)
    # # 保存结果
    # record_id = os.environ.get('EXECUTE_RECORD_ID', str(random.randint(1, 10000000)))
    # experiment_name = '10颗树100%数据80%特征3'
    # image_path = f"{experiment_name}_{record_id}.png"
    # auuc_path = f'{experiment_name}_{record_id}.json'
    # Figure.figure.savefig(image_path)
    # with open(auuc_path, 'w') as file:
    #     json.dump({'auuc': AUUC, 'experiment_name': experiment_name, 'time': duration_str_3}, file)
    # # 保存oss
    # store_res_to_oss(image_path, auuc_path)