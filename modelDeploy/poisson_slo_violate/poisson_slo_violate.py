#当到达率是poisson的一个随机过程的时候，当处理能力等于平均到达率的时候，slo违反的概率是多少
#实验测试在不同到达率，不同slo要求下的slo违反概率
from modelDeploy.dispatch.model_info import Model_Info,Model_Runtime_Info
import numpy as np
import random
model_name="resnet-18"
lam=15
slo=2
model_info=Model_Info("modelDeploy/modelRepository")
model_runtime_info=Model_Runtime_Info("modelDeploy/modelRepository")

#构建poisson序列
poisson_list=np.random.poisson(lam,1000)
deal_num=0
for data in poisson_list:
    if data<=lam:
        deal_num+=data
    else:
        deal_num+=lam

print(deal_num/np.sum(poisson_list))






