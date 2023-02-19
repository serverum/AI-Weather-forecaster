import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = pd.read_excel("weather.xls", skiprows=6)

data['dates'] = pd.to_datetime(data['Местное время в Москве (ВДНХ)'], dayfirst=True)

# print(data['dates'][0])

# plt.plot(data['dates'], data['T'])
# plt.show()

# data_train = data[data['dates'] < "2020-01-01"]
# data_test = data[data['dates'] > '2020-01-01']

# print(data_train['dates'], data_train['T'])
#
# plt.figure(figsize=[20,5])
# plt.grid()
# plt.plot(data_train['dates'], data_train['T'], color="r", label="Train data")
# plt.plot(data_test['dates'], data_test['T'], color="green", label="Test data")
# plt.legend()
# plt.show()

data = data[data['T'].notna()]

data['dayofyear'] = data['dates'].dt.dayofyear

# Данные поля ниже после сортировки уже попадают data_train = data[data['dates'] < "2020-01-01"] в массив data_train
data['scaled_dayofyear'] = (data['dayofyear'] - 1) / 365 * 2 * np.pi
data['cos_dayofyear'] = (np.cos(data['scaled_dayofyear']))

data_train = data[data['dates'] < "2020-01-01"]
data_test = data[data['dates'] > '2020-01-01']

X_train = pd.DataFrame(data_train['cos_dayofyear'])
y_train = data_train['T']

X_test = pd.DataFrame(data_test['cos_dayofyear'])
y_test = data_test['T']

model = LinearRegression()
model.fit(X_train, y_train)

pred_train = model.predict(X_train)
pred_test = model.predict(X_test)



# print(X_train['cos_dayofyear'])
#
# # print(data['cos_dayofyear'].min(), data['cos_dayofyear'].max())
#
# data_train = data[data['dates'] < "2020-01-01"]
# data_test = data[data['dates'] >= '2020-01-01']
#
#
# X_train = pd.DataFrame(data['cos_dayofyear'])
# y_train = data['T']




# X_train = pd.DataFrame(data_train['dayofyear'])
# y_train = data_train['T']
#
# X_test = pd.DataFrame(data_test['dayofyear'])
# y_test = data_test['T']

# print(X_train)
# print(data_train['dayofyear'])
#
# model = LinearRegression()
# model.fit(X_train, y_train)

# прогнозируем результат модели
# pred_train = model.predict(X_train)
# pred_test = model.predict(X_test)

#наши данные

# plt.figure(figsize=[20,5])
# plt.grid()
# plt.scatter(data_train['dates'], data_train['T'], color="r", label='Train Data')
# plt.scatter(data_test['dates'], data_test['T'], color="blue", label='Test Data')
# plt.legend()
# plt.show()


# plt.figure(figsize=[20,5])
# plt.grid()
# plt.scatter(data_train['dates'], data_train['T'], color="r", label='Train Data')
# plt.scatter(data_test['dates'], data_test['T'], color="blue", label='Test Data')
# plt.scatter(data_train['dates'], pred_train, color="green", label='Train Data')
# plt.scatter(data_test['dates'], pred_test, color="orange", label='Test Data')
# plt.legend()
# plt.show()


plt.figure(figsize=[20,5])
plt.grid()
plt.scatter(data_train['dates'], data_train['T'], color="r", label='Train Data')
plt.scatter(data_test['dates'], data_test['T'], color="blue", label='Test Data')
plt.scatter(data_train['dates'], pred_train, color="green", label='Train Data', alpha=0.5)
plt.scatter(data_test['dates'], pred_test, color="orange", label='Test Data', alpha=0.5)
plt.legend()
plt.show()

print(mean_squared_error(y_train, pred_train), mean_squared_error(y_test, pred_test))
