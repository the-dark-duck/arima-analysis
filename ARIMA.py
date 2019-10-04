#!/usr/bin/env python
# coding: utf-8

# In[38]:


import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from pandas import datetime
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from pandas import read_csv
from pandas import datetime
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from matplotlib import pyplot


# In[39]:



# from sklearn.utils import check_arrays
def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
  


# In[40]:


a = np.random.rand(5)
b = np.random.rand(5)

mean_absolute_percentage_error(a,b)


# In[23]:



 # evaluate an ARIMA model for a given order (p,d,q)
def evaluate_arima_model(X, arima_order):
   # prepare training dataset
   train_size = int(len(X) - 7)
   train, test = X[0:train_size], X[train_size:]
   history = [x for x in train]
   # make predictions
   predictions = list()
   for t in range(len(test)):
      model = ARIMA(history, order=arima_order)
      model_fit = model.fit(disp=0)
      yhat = model_fit.forecast()[0]
      predictions.append(yhat)
      obs = test[t]
      history.append(test[t])
      print('predicted=%f, expected=%f' % (yhat, obs))
   # calculate out of sample error
   #test = np.array(test)
   #predictions = np.array(predictions)
   error = mean_absolute_percentage_error(test, predictions)
   #error = error.item()
   return error


# In[24]:


c = np.random.rand(1000)
evaluate_arima_model(c, (1,0,0))


# In[34]:



# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models_test(X):
	p_values = [0, 5]
	d_values = range(0, 2)
	q_values = range(0, 5)
	dataset = X.astype('float32')
	best_score, best_cfg = float("inf"), 0
	for p in p_values:
		for d in d_values:
			for q in q_values:
				order = (p,d,q)
				try:
					mape = evaluate_arima_model(dataset, order)
					if mape < best_score:
						best_score, best_cfg = mape, order
					print('ARIMA%s MAPE=%.3f' % (order,mape))
				except:
					continue
	return best_cfg


# In[ ]:


sm.graphics.tsa.plot_acf(dta.values.squeeze(), lags=40)
plt.show()


# In[35]:


d = np.random.rand(100)
evaluate_models_test(d)


# In[43]:


plot_acf(d, ax=pyplot.gca())
pyplot.show()
plot_pacf(d, ax=pyplot.gca())
pyplot.show()


# In[ ]:




