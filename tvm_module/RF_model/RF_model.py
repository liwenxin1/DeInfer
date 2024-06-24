import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
df = pd.read_csv("tvm_module/model_test/profiles/model_interference_feature.csv")
x1=df["mean_l2_utilization1"]*df["model_GFLOPS1"]
x2=df["mean_l2_utilization2"]*df["model_GFLOPS2"]

y=df["single_coefficient"]

# Create an instance of LinearRegression
regression_model = LinearRegression()

# Reshape the x1 and x2 arrays to 2D
x1 = x1.values.reshape(-1, 1)
x2 = x2.values.reshape(-1, 1)

# Fit the model using x1, x2, and y
regression_model.fit(np.concatenate((x1, x2), axis=1), y)

# Get the coefficients and intercept
coefficients = regression_model.coef_
intercept = regression_model.intercept_
print(coefficients, intercept)
df2=pd.read_csv("tvm_module/model_test/profiles/model_interference_feature2.csv")
data=df2[["batchsize1","mean_l2_hit_rate1","mean_l2_utilization1","mean_sm_efficiency1","model_GFLOPS1","model_mps1",\
    "batchsize2","mean_l2_hit_rate2","mean_l2_utilization2","mean_sm_efficiency2","model_GFLOPS2","model_mps2","model_num","interference"]]



feature1=(coefficients[0]*data["mean_l2_utilization1"]*data["model_GFLOPS1"]+coefficients[1]*data["mean_l2_utilization2"]*data["model_GFLOPS2"])
feature2=data["model_mps1"]
feature3=data["model_mps2"]
feature4=data["model_num"]
xdata=np.concatenate((feature1.values.reshape(-1, 1), feature2.values.reshape(-1, 1), feature3.values.reshape(-1, 1), feature4.values.reshape(-1, 1)), axis=1)
ydata=data["interference"]-100


x_train, x_test, y_train, y_test = train_test_split(xdata, ydata, test_size=0.3)

model = RandomForestRegressor(n_estimators=100,oob_score=True)
model.fit(x_train, y_train)

# model = LinearRegression()
# model.fit(xdata, ydata)
# y_pred = model.predict(xdata)
# x_test=xdata
# y_test=ydata

y_pred = model.predict(x_test)
print(np.mean((np.abs(y_pred-y_test)/(y_test+100)))
)

# #对比参数的影响
# model2=RandomForestRegressor(n_estimators=100,oob_score=True)
# feature1=data["mean_l2_utilization1"]
# feature2=data["mean_l2_utilization2"]
# feature3=data["model_num"]
# xdata=np.concatenate((feature1.values.reshape(-1, 1), feature2.values.reshape(-1, 1), feature3.values.reshape(-1, 1)), axis=1)
# x_train, x_test, y_train, y_test = train_test_split(xdata, ydata, test_size=0.3)
# model2.fit(x_train, y_train)
# y_pred = model2.predict(x_test)
# print(np.mean((np.abs(y_pred-y_test)/(y_test+100)))
# )


data=df2[df2["modelname1"]=="vgg-19"][df2["modelname2"]=="resnet-18"][df2["model_mps1"]==20][df2["model_mps2"]==80][["modelname1","batchsize1","mean_l2_hit_rate1","mean_l2_utilization1","mean_sm_efficiency1","model_GFLOPS1","model_mps1",\
    "modelname2","batchsize2","mean_l2_hit_rate2","mean_l2_utilization2","mean_sm_efficiency2","model_GFLOPS2","model_mps2","model_num","interference"]]
feature1=(coefficients[0]*data["mean_l2_utilization1"]*data["model_GFLOPS1"]+coefficients[1]*data["mean_l2_utilization2"]*data["model_GFLOPS2"])
feature2=data["model_mps1"]
feature3=data["model_mps2"]
feature4=data["model_num"]
xdata=np.concatenate((feature1.values.reshape(-1, 1), feature2.values.reshape(-1, 1), feature3.values.reshape(-1, 1), feature4.values.reshape(-1, 1)), axis=1)
ydata=data["interference"]-100
y_pred = model.predict(xdata)
for i,j in zip(ydata,y_pred):
    print(i,j)

#Save the trained model
joblib.dump(model, "tvm_module/RF_model/random_forest_model.pkl")





