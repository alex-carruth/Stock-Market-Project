import numpy as np
import pandas as pd
from illumina_combiner import ilmn_final
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


Y = np.array(ilmn_final['Open'])
X = np.array(ilmn_final.drop(['Open'], 1))


scalerX = MinMaxScaler()
scalerY = MinMaxScaler()
X = scalerX.fit_transform(X)
Y = Y.reshape(-1, 1)
Y = scalerY.fit_transform(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)
X_pred = X_test[0:20]
Y_check = Y_test[0:20]
Y_train = Y_train.flatten()


#Performing the Regression on the training data
svr_model = SVR(kernel = 'rbf', gamma='auto')
svr_model.fit(X_train, Y_train)
prediction = (svr_model.predict(X_pred))
Y_pred = svr_model.predict(X_test)
#print(prediction)
prediction = prediction.reshape(-1,1)
print(scalerY.inverse_transform(prediction))
print('Prediction Score: ', svr_model.score(X_test, Y_test))
print('Mean Squared Error : ', mean_squared_error(Y_test, Y_pred))
diff = 0
for i in range(0, len(Y_check)):
    diff = diff+abs(prediction[i]-Y_check[i])
percent_error = diff/20
print(percent_error)

for i in range(0, len(Y_test)):
    diff = diff+abs(Y_pred[i]-Y_test[i])
percent_error = diff/20
print(percent_error)