import tvm
from tvm.contrib import graph_executor
import numpy as np
import matplotlib.pyplot as plt
#获取不同batch，不同优化因子下的模型运行时间，绘制图像到tvm_module/pic/model_runtime_test.png
model_name="resnet-18"
batch_size_list=[1,2,4,8,16]
factor_list=["untuned",50,100,200,400]

widths=[-0.2,-0.1,0,0.1,0.2]
fig,axs=plt.subplots(2,1,figsize=(20,10))
for index,batch_size in enumerate(batch_size_list):
    optimization_list=[]
    deal_one_data_time=[]
    for factor in factor_list:
        file_name="../tune_file/{}-NCHW-B{}-{}-cuda.so".format(model_name,batch_size,factor)
        lib=tvm.runtime.load_module(file_name)
        dev=tvm.cuda(0)
        m=graph_executor.GraphModule(lib["default"](dev))

        m.set_input("data",tvm.nd.array((np.random.uniform(size=(batch_size,3,224,224))).astype("float32")))
        
        value=m.benchmark(dev, repeat=5, min_repeat_ms=500).mean*1000
        optimization_list.append(value)
        deal_one_data_time.append(value/batch_size)
    
    x_label=np.arange(len(factor_list))
    
    axs[0].plot(["0","50","100","200","400"],optimization_list,label="batch_size={}".format(batch_size))
    
    axs[1].set_xticks(x_label)
    axs[1].bar(x_label+widths[index],deal_one_data_time,width=0.1,label="batch_size={}".format(batch_size))
    


axs[0].grid()
axs[0].legend(loc="upper right",fontsize="small")
axs[0].set_ylabel("inference of a batchsize(ms)")

axs[1].set_xticklabels(["0","50","100","200","400"])
axs[1].legend(loc="upper right",fontsize="small")
axs[1].set_ylabel("per data time used(ms)")
plt.xlabel("the optimazation factor of tvm")
plt.savefig("../pic/model_runtime_test.png")
    
    
        



        