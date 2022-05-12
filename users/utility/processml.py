import pandas as pd
from django.conf import settings
import os
from sklearn.model_selection import train_test_split
path =os.path.join(settings.MEDIA_ROOT, 'ada.csv')
df = pd.read_csv(path)
df = df[['Open', 'High', 'Low', 'Close']]
X = df.iloc[:,:-1].values
y = df.iloc[:,-1].values
print(f'X Len{len(X)} y len {len(y)}')
X_train,X_test,y_train,y_test = train_test_split(X,y, train_size=0.80,random_state=0)


def calc_linear_regression():
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    from sklearn.metrics import mean_absolute_error
    mae = mean_absolute_error(y_pred,y_test)
    from sklearn.metrics import mean_squared_error
    mse= mean_squared_error(y_pred,y_test)
    from sklearn.metrics import r2_score
    r2 = r2_score(y_pred,y_test)
    return mae,mse,r2


def calc_random_forest_regressor():
    # from sklearn.ensemble import RandomForestRegressor
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor()
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    from sklearn.metrics import mean_absolute_error
    mae = mean_absolute_error(y_pred,y_test)
    from sklearn.metrics import mean_squared_error
    mse= mean_squared_error(y_pred,y_test)
    from sklearn.metrics import r2_score
    r2 = r2_score(y_pred,y_test)
    return mae,mse,r2


def calc_svm():
    # from sklearn.ensemble import RandomForestRegressor
    from sklearn.svm import SVR
    model = SVR()
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    from sklearn.metrics import mean_absolute_error
    mae = mean_absolute_error(y_pred,y_test)
    from sklearn.metrics import mean_squared_error
    mse= mean_squared_error(y_pred,y_test)
    from sklearn.metrics import r2_score
    r2 = r2_score(y_pred,y_test)
    return mae,mse,r2



