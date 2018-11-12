import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.stattools import pacf
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import stats
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose


def diff(X,t):
    diff_x=[]
    for i in range(t,len(X)):
        diff_x.append(X[i]-X[i-t])
    return diff_x

def inverse_difference(last_ob, value):
	return value + last_ob

def check_stationarity(data):
    X = data["X"]
    #rollMean = pd.rolling_mean(X,window=12, center=False)
   # rollStd = pd.rolling_std(X,window=12, center=False)
    plt.figure()
    plt.plot(data["X"])
    #plt.plot(rollMean,linestyle='--',color='green')
    #plt.plot(rollStd,linestyle = '--', color = 'gray')
    plt.title("Original Data")
    plt.savefig("Original_data")

    '''seasonality = seasonal_decompose(np.asarray(X),freq=10)
    plt.figure()
    plt.plot(seasonality)
    plt.title("Seasonality Check for original data")
    plt.savefig("seasonality check orig")'''



    diff_x=diff(X,1)
    diff_x2=diff(diff_x,1)
    #print("diff",diff_x.head(3))

    plt.figure()
    plt.plot(diff_x)
    plt.title("Series obtained by taking  difference of data (interval=1)")
    plt.savefig("diff_x")

    plt.figure()
    plt.title("Series obtained after 2nd order differencing")
    plt.plot(diff_x2)
    plt.savefig("diff2")
    #inv_X=[X[i]+diff_x[i] for i in range(len(diff_x))]

    plt.figure()
    plt.plot(np.log(data["X"]))
    plt.title("Series obtained by performing logarithmic transformation of data")
    plt.savefig("log")

    plt.figure()
    plt.title("Series obtained by performing square root transformation of given data")
    plt.plot(np.sqrt(data["X"]))
    plt.savefig("sqrt")
    #plt.show()




def SMA(data_in,k):
    st=[]
    act=[]
    err=[]


    for t in range(k,len(data_in)):
        sum=0
        for j in range(t-k,t):
            sum+=data_in[j]
        st.append(sum/k)
        act.append(data_in[t])
        err.append(np.square((sum/k)-data_in[t]))
    sma_rmse_k=np.sqrt(np.mean(err))

    return st,act,sma_rmse_k


def SMA_check(X,data_in,k,i):

    st, act, sma_rmse_k=SMA(data_in,k)
    print("RMSE for k="+str(k)+" is: "+str(sma_rmse_k))
    inverted_in = [inverse_difference(X[i], data_in[i]) for i in range(len(st))]
    inverted_st=[inverse_difference(X[i], st[i]) for i in range(len(st))]
    plt.figure()
    plt.plot(act,color='grey',label="actual value")
    plt.plot(st,color='blue',label="predicted value")
    plt.title("Comparison of predicted and actual values for SMA model, k="+str(k))
    plt.savefig("SMA check"+str(i))

    plt.figure()
    plt.plot(inverted_in, color='red', label="actual value")
    plt.plot(inverted_st, color='blue', label="predicted value")
    plt.legend(loc='upper left')
    plt.title("Comparison of predicted and actual values for SMA model, k=" + str(k))
    plt.savefig("SMA check_actual_value"+str(i))


def SMA_plot(train):

    sma_rmse=[]
    for k in range(1,500):
        st,act, sma_rmse_k = SMA(train, k)
        sma_rmse.append(sma_rmse_k)
    sma_k=sma_rmse.index(min(sma_rmse))
    print(" Minimum RMSE for Simple Moving Average Model:",min(sma_rmse))
    print("k value corresponding to min rmse simple moving average: ",sma_k)
    plt.figure()
    plt.plot(sma_rmse)
    plt.title("RMSE vs k for Simple Moving Average Model")
    plt.xlabel("k")
    plt.ylabel("RMSE")
    plt.savefig("SMA RMSE vs k")
    #plt.show()
    return sma_k,min(sma_rmse)





#Simple Exponential Smoothing

'''model = SimpleExpSmoothing(train)
model_fit = model.fit()
# make prediction
yhat = model_fit.predict(model_fit,1,len(train))
print(yhat)
#plt.plot(train)
plt.plot(yhat)
plt.savefig("SES")
plt.show()'''

def SES(data_in,a):
    exp_st=[]
    exp_act=[]
    exp_err=[]
    for t in range(1,len(data_in)):
        if t==1:
            val=a*data_in[t-1]
            exp_st.append(val)
            exp_act.append(data_in[t])
            exp_err.append(np.square(val-data_in[t]))
        else:
            val=(a*data_in[t-1])+((1-a)*exp_st[t-2])
            exp_st.append(val)
            exp_act.append(data_in[t])
            exp_err.append(np.square(val - data_in[t]))
    ses_rmse_k = np.sqrt(np.mean(exp_err))
    return exp_st,exp_act,ses_rmse_k

def SES_check(X,data_in,a,i):

    exp_st,exp_act,ses_rmse_k =SES(data_in,a)
    plt.figure()
    plt.plot(exp_act,color='grey',label="actual value")
    plt.plot(exp_st,color='blue',label="predicted value")

    plt.title("Comparison of predicted and actual values for exponential model (not rescaled), a="+str(a))
    plt.savefig("SES check"+str(i))
   # plt.show()
    print("RMSE for a="+str(a)+" is: "+str(ses_rmse_k))
    inverted_in = [inverse_difference(X[i], data_in[i]) for i in range(len(exp_st))]
    inverted_st=[inverse_difference(X[i], exp_st[i]) for i in range(len(exp_st))]


    plt.figure()
    plt.plot(inverted_in, color='red', label="actual value")
    plt.plot(inverted_st, color='blue', label="predicted value")
    plt.legend(loc='upper left')
    plt.title("Comparison of predicted and actual values for SES model, a=" + str(a))
    plt.savefig("SES check_scaled_value"+str(i))


def SES_plot(train):

    ses_rmse=[]
    for i in range(1,10):
        a=i*0.1
        exp_st, exp_act, ses_rmse_k = SES(train, a)
        ses_rmse.append(ses_rmse_k)
    ses_a=(ses_rmse.index(min(ses_rmse))+1)*0.1
    print("Minimum RMSE value for exponential smoothing model:",min(ses_rmse))
    print("a value for simple exponential smoothing model: ",ses_a)

    plt.figure()
    plt.plot(ses_rmse,)
    #plt.fill_betweenx(0,1,0.1)
    plt.title("RMSE vs a for Simple Exponential Smoothing Model")
    plt.xlabel("a")
    plt.ylabel("RMSE")
    plt.savefig("SES RMSE vs a")
    #plt.show()
    return ses_a,min(ses_rmse)






# AR Model:

def PACF_fn(data_in):
    delta = 0.1
    lag_pacf = pacf(data_in, nlags=30, method='yw')
    #print (lag_pacf)
    upperInt = 1.96 / np.sqrt(len(data_in))
    intPoint = -1
    plt.figure()
    plt.plot(lag_pacf)
    plt.axhline(y=0, linestyle='--', color='gray')
    plt.axhline(y=-1.96 / np.sqrt(len(data_in)), linestyle='--', color='gray')
    plt.axhline(y=1.96 / np.sqrt(len(data_in)), linestyle='--', color='gray')
    plt.title('Partial Autocorrelation Function')
    plt.xlabel("lags")
    plt.ylabel("PACF")
    plt.tight_layout()
    plt.savefig("PACF")
    #plt.show()

    for i in range(0, len(lag_pacf)):
        if abs(lag_pacf[i] - upperInt) <= delta:
            print("p value using PACF is " + str(i))
            intPoint = i
            break
    return i


def AR_model(X,data_in,lag,i):
    model=AR(data_in)
    results_AR = model.fit(maxlag=lag,disp=0)
    AR_data=results_AR.fittedvalues
    act=data_in[3:]
    print("Parameters of Autoregressive Model AR(%d) are:" % lag)
    print(results_AR.params)
    plt.figure()
    plt.plot(act, color='blue', label='Actual Value')
    plt.plot(results_AR.fittedvalues, color='red', label="Predicted Value")
    plt.legend(loc='best')
    plt.xlabel("Time")
    plt.ylabel("Time series values")
    plt.title('AR('+str(lag)+")"+"Model with RMSE:" +str(np.sqrt((np.sum(np.square(AR_data - act))) / len(act))))
    plt.title("AR Fit (not scaled)")
    plt.savefig("AR fit not scaled"+str(i))
    #plt.show()
    inverted_in = [inverse_difference(X[i], data_in[i]) for i in range(len(AR_data))]
    inverted_AR=[inverse_difference(X[i], AR_data[i]) for i in range(len(AR_data))]

    plt.figure()
    plt.plot(inverted_in, color='red', label="actual value")
    plt.plot(inverted_AR, color='blue', label="predicted value")
    plt.legend(loc='upper left')
    plt.title("Comparison of predicted and actual values for Autoregression model, lag" + str(lag))
    plt.savefig(" AR Fit Final"+str(i))

    print("RMSE on the Data is:"+str(np.sqrt((np.sum(np.square(AR_data - act))) / len(act))))



    residuals=results_AR.resid
    plt.figure()
    plt.title("Residual Scatter Plot")
    plt.scatter(AR_data, residuals)
    plt.savefig("residuals"+str(i))
    #plt.show()

    plt.figure()
    qqplot(residuals)
    plt.title("Residual Q-Q Plot")
    plt.savefig("QQ"+str(i))

    plt.figure()
    plt.hist(residuals)
    plt.title("Residual Histogram")
    plt.savefig("Hist"+str(i))

    k2, p = stats.normaltest(residuals)
    alpha = 0.001
    print("Chi-Square Test : k2 = %.4f  p = %.4f" % (k2, p))
    print("two sided chi squared probability :"+str(p))



if __name__=="__main__":
    data = pd.read_csv("srajend2.csv", header=-1)
    data.columns = ["X"]
    #print(data.head(3))
    X = data["X"]
    i=0
# Task 1: check_stationarity(data)
    check_stationarity(data)

    #Create new series with differencing

    diff_x = diff(X, 1)
    train, test = train_test_split(diff_x, test_size=0.25, shuffle=False)
    #print(len(train))
    #print(len(test))

#Task2: SMA
    print("SMA: \n")
    SMA_check(X,train,2,i+1)
    sma_k,min_sma_rmse= SMA_plot(train)
    SMA_check(X,train,sma_k,i+1)

#Task 3 : SES
    print("SES: \n")
    SES_check(X,train,0.1,i+1)
    ses_a, min_ses_rmse = SES_plot(train)
    SES_check(X, train, ses_a, i + 1)

#Task 4: AR
    print("AR: \n")
    max_lag=PACF_fn(train)
    AR_model(X,train,max_lag,i+1)

#Task 5: Compare Models:

    print("Testing: \n")
    SMA_check(X,test,sma_k,i+1)
    SES_check(X, test, ses_a, i + 1)
    AR_model(X, test, max_lag, i + 1)


