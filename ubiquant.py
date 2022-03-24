import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

#importing ubiquant environment for sending predictions
import ubiquant

#Getting most valuable informations based on corellation matrix
def get_features(dataframe):
    dataframe= dataframe[['f_231','f_250','f_265','f_280','f_197','f_65','f_25','f_155','f_71','f_15','f_212','f_179','f_237','f_297','f_190','f_286','f_137','f_165','f_255','f_174','f_109','f_5','f_153','f_194','f_169','f_17','f_150','f_145','f_225','f_264','f_93','f_76','f_270','f_119']]
    return dataframe


df2 = pd.read_parquet("../input/ubiquant-parquet/train_low_mem.parquet")
df2.head()
X = np.array(get_features(df2))
y = np.array(df2['target'])
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33)

model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.01,
    max_depth=30,
    subsample=0.9,
    colsample_bytree=0.7,
    #colsample_bylevel=0.75,
    missing=-999,
    random_state=1111,
    tree_method='hist'  
    )
model.fit(X_train, y_train)
predict = model.predict(X_test)
print(mean_absolute_error(predict,y_test))


#sending answers to ubiqant API
env = ubiquant.make_env()
iter_test = env.iter_test()
for i, (df_test, df_pred) in enumerate(iter_test):
    for j , row in df_test.iterrows():
        try:
            x_test = get_features(row)
            y_pred = model.predict(pd.DataFrame([x_test]))[0]
            df_pred.loc[df_pred['row_id'] == row['row_id'], 'target'] = y_pred
        except:
            df_pred.loc[df_pred['row_id'] == row['row_id'], 'target'] = 0
    env.predict(df_pred)