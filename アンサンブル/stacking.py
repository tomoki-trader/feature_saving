X_train, X_test, y_train, y_test = train_test_split(df,target,stratify=target,random_state=0)

from sklearn.metrics import log_loss
from sklearn.model_selection import KFold
 
# 学習データに対する「目的変数を知らない」予測値と、テストデータに対する予測値を返す関数
def predict_cv(model, train_x, train_y, test_x):
    preds = []
    preds_test = []
    va_idxes = []
    
　　#kfoldパート　分割数は多い方が良いそうなので10を選択
    kf = KFold(n_splits=5, shuffle=True, random_state=71)
    
    # クロスバリデーションで学習・予測を行い、予測値とインデックスを保存する
    for i, (tr_idx, va_idx) in enumerate(kf.split(train_x)):
        tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
        tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]
        model.fit(tr_x, tr_y, va_x, va_y)
        pred = model.predict(va_x)
        preds.append(pred)
        pred_test = model.predict(test_x)
        preds_test.append(pred_test)
        va_idxes.append(va_idx)
        
    # バリデーションデータに対する予測値を連結し、その後元の順序に並べ直す
    va_idxes = np.concatenate(va_idxes)
    preds = np.concatenate(preds, axis=0)
    order = np.argsort(va_idxes)
    pred_train = preds[order]
    
    # テストデータに対する予測値の平均をとる
    preds_test = np.mean(preds_test, axis=0)
    
    return pred_train, preds_test

 
# 1層目のモデル（Github参照）
#XGBoost
model_1a = Model1Xgb()
pred_train_1a, pred_test_1a = predict_cv(model_1a, X_train, y_train, X_test)
 
#KerasNN
model_1b = Model1NN()
pred_train_1b, pred_test_1b = predict_cv(model_1b, X_train, y_train, X_test)
 
#valid評価
pred_xbg1a = pred_train_1a > 0.5
pred_nn1a  = pred_train_1b > 0.5
 
#valid正答率
print("xbg valid accuracy：",accuracy_score(y_train, pred_xbg1a))
print("nn valid accuracy：",accuracy_score(y_train, pred_nn1a))
 
#test評価
pred_xbg1_test = pred_test_1a > 0.5
pred_nn1a_test  = pred_test_1b > 0.5
 
#test正答率
print("xbg test accuracy：",accuracy_score(y_test, pred_xbg1_test))
print("nn test accuracy：",accuracy_score(y_test, pred_nn1a_test))

train_x_2 = pd.DataFrame({'pred_1a': pred_train_1a, 'pred_1b': pred_train_1b})
test_x_2 = pd.DataFrame({'pred_1a': pred_test_1a, 'pred_1b': pred_test_1b})
 
# 2層目のモデル　Linerモデルで提出データを予測
model_2 = Model2Linear()
pred_train_2, pred_test_2 = predict_cv(model_2,  train_x_2, y_train, test_x_2)
 
#valid評価
pred_liner2_valid = pred_train_2 > 0.5
#test評価
pred_liner2_test = pred_test_2 > 0.5
 
#valid正答率
print("liner valid accuracy：",accuracy_score(y_train, pred_liner2_valid))
#test正答率
print("liner test accuracy：",accuracy_score(y_test, pred_liner2_test))