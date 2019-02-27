import numpy
import pandas
import time
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn import ensemble
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR

# define base model
# def baseline_model():
# 	# create model
# 	model = Sequential()
# 	model.add(Dense(300, input_dim=10, kernel_initializer='normal', activation='relu'))
# 	model.add(Dense(300, kernel_initializer='normal', activation='relu'))
# 	model.add(Dense(300, kernel_initializer='normal', activation='relu'))
# 	# model.add(Dense(50, kernel_initializer='normal', activation='relu'))
# 	model.add(Dense(1, kernel_initializer='normal'))
# 	# Compile model
# 	model.compile(loss='mean_squared_error', optimizer='adam')
# 	return model

# Data.
start_time=time.time()
data=pandas.read_csv('angin_jawa_50M.csv')
data=data[['WS50M_RANGE','WD50M','T2M_RANGE','TS','T2MDEW','T2M','PRECTOT','QV2M','RH2M','PS','WS50M']]

y=data['WS50M']
X=data.drop(['WS50M'],axis=1)

# zscore.
cols=list(X.columns)
for col in cols:
	X[col]=(X[col]-X[col].mean())/X[col].std(ddof=0)

# # fix random seed for reproducibility
# seed = 7
# numpy.random.seed(seed)
# # evaluate model with standardized dataset
# estimator = KerasRegressor(build_fn=baseline_model, epochs=500, batch_size=1000000, verbose=1,validation_split=0.2)
# kfold = KFold(n_splits=2, random_state=seed)
# results = cross_val_score(estimator, X, y, cv=kfold)
# print(results)
# print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
# exit()

kfold = KFold(n_splits=10, random_state=42)
clf=ensemble.RandomForestRegressor()
results = cross_val_score(clf, X, y, cv=kfold)
print(results)
print("Random Forest Regressor Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

kfold = KFold(n_splits=10, random_state=42)
clf=ensemble.GradientBoostingRegressor()
results = cross_val_score(clf, X, y, cv=kfold)
print(results)
print("Gradient Boosting Regressor Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

kfold = KFold(n_splits=10, random_state=42)
clf=DecisionTreeRegressor()
results = cross_val_score(clf, X, y, cv=kfold)
print(results)
print("Decision Tree Regressor Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

kfold = KFold(n_splits=10, random_state=42)
clf=ensemble.AdaBoostRegressor()
results = cross_val_score(clf, X, y, cv=kfold)
print(results)
print("Ada Boost Regressor Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

kfold = KFold(n_splits=10, random_state=42)
clf=LinearRegression()
results = cross_val_score(clf, X, y, cv=kfold)
print(results)
print("Linear Regression Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

# ----------------------------------------
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4,random_state=42,shuffle=True)

# clf = ensemble.GradientBoostingRegressor()
# clf.fit(X_train,y_train)
# mse = mean_squared_error(y_test, clf.predict(X_test))
# print('Gradient Boosting Regressor')
# print("RMSE: %.4f" % sqrt(mse))
# print('------------------------------')

# clf=DecisionTreeRegressor()
# clf.fit(X_train,y_train)
# mse = mean_squared_error(y_test, clf.predict(X_test))
# print('Decision Tree Regressor')
# print("RMSE: %.4f" % sqrt(mse))
# print('------------------------------')

# clf=ensemble.AdaBoostRegressor()
# clf.fit(X_train,y_train)
# mse = mean_squared_error(y_test, clf.predict(X_test))
# print('Ada Boost Regressor')
# print("RMSE: %.4f" % sqrt(mse))
# print('------------------------------')

# clf=LinearRegression()
# clf.fit(X_train,y_train)
# mse = mean_squared_error(y_test, clf.predict(X_test))
# print('Linear Regressor')
# print("RMSE: %.4f" % sqrt(mse))
# print('------------------------------')

# clf=ensemble.RandomForestRegressor()
# clf.fit(X_train,y_train)
# mse = mean_squared_error(y_test, clf.predict(X_test))
# print('Random Forest Regressor')
# print("RMSE: %.4f" % sqrt(mse))
# print('------------------------------')

# clf=SVR(kernel='rbf')
# clf.fit(X_train,y_train)
# mse = mean_squared_error(y_test, clf.predict(X_test))
# print('SVR rbf')
# print("RMSE: %.4f" % sqrt(mse))
# print('------------------------------')

# clf=SVR(kernel='linear')
# clf.fit(X_train,y_train)
# mse = mean_squared_error(y_test, clf.predict(X_test))
# print('SVR linear')
# print("RMSE: %.4f" % sqrt(mse))
# print('------------------------------')

# clf=SVR(kernel='poly')
# clf.fit(X_train,y_train)
# mse = mean_squared_error(y_test, clf.predict(X_test))
# print('SVR poly')
# print("RMSE: %.4f" % sqrt(mse))
# print('------------------------------')

elapsed_time = time.time() - start_time
print(elapsed_time)