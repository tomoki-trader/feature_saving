# 特徴量の指定
features = [
    "age_mis_val_median",
    "family__size",
    "cabin",
    "fare_mis_val_median"] 
run_name = 'lgb_1102'
# 使用する特徴量リストの保存
with open(LOG_DIR_NAME + run_name + "_features.txt",'wt') as f:
    for ele in features:
        f.write(ele+'\n')
params_lgb = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'early_stopping_rounds': 20,
    'verbose': 10,
    'random_state': 99,
    'num_round': 100}
# 使用するパラメータの保存
with open(LOG_DIR_NAME + run_name + "_param.txt",'wt') as f:
    for key,value in sorted(params_lgb.items()):
        f.write(f'{key}:{value}\n')
runner = Runner(run_name, ModelLGB, features, params_lgb, n_fold, name_prefix)
runner.run_train_cv() # 学習
runner.run_predict_cv() # 推論
Submission.create_submission(run_name) # submit作成

