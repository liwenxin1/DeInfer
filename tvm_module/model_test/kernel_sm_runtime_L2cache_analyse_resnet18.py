import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
import tvm
from tvm.contrib import graph_executor
import os
#只能对resnet18进行分析，只有resnet18有多factor的分析数据
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--model_name",type=str,default="resnet-18")
    parser.add_argument("--batch_size",type=int,default=1)

    args=parser.parse_args()
    model_name=args.model_name
    batch_size=args.batch_size
    
    factor_list=[50,100,200,400]
    fig,axs=plt.subplots(4,1,figsize=(10,10))
    mean_gpu_time_duration=[]
    mean_l2_hit_rate=[]
    mean_l2_utilization=[]
    mean_sm_efficiency=[]
    for index,factor in enumerate(factor_list):
        #csv_name="tvm_module/model_test/profiles/{}_{}_{}_kernel_sm_runtime_L2cache.csv".format(model_name,batch_size,factor)
        csv_name="./profiles/{}_{}_{}_kernel_sm_runtime_L2cache.csv".format(model_name,batch_size,factor)
        
        file_name="../tune_file/{}-NCHW-B{}-{}-cuda.so".format(model_name,batch_size,factor)
        lib=tvm.runtime.load_module(file_name)
        dev=tvm.cuda(0)
        m=graph_executor.GraphModule(lib["default"](dev))
        m.set_input("data",tvm.nd.array((np.random.uniform(size=(batch_size,3,224,224))).astype("float32")))
        value=m.benchmark(dev, repeat=1, number=1).mean
        mean_gpu_time_duration.append(value)
        
        pd_data=pd.read_csv(csv_name)
        pd_gpu_time_duration=pd_data[pd_data["Metric Name"]=="gpu__time_duration.sum"].reset_index(drop=True)
        pd_l2_hit_rate=pd_data[pd_data["Metric Name"]=="lts__t_sector_hit_rate.pct"].reset_index(drop=True)
        pd_l2_utilization=pd_data[pd_data["Metric Name"]=="lts__t_sectors.avg.pct_of_peak_sustained_elapsed"].reset_index(drop=True)
        pd_sm_efficiency=pd_data[pd_data["Metric Name"]=="sm__throughput.avg.pct_of_peak_sustained_elapsed"].reset_index(drop=True)
        
        sum_pd_gpu_time_duration=pd_gpu_time_duration["Metric Value"].sum()
        temp=pd_gpu_time_duration["Metric Value"]/sum_pd_gpu_time_duration
        mean_l2_hit_rate.append((pd_l2_hit_rate["Metric Value"]*temp).sum())
        mean_l2_utilization.append((pd_l2_utilization["Metric Value"]*temp).sum())
        mean_sm_efficiency.append((pd_sm_efficiency["Metric Value"]*temp).sum())
        
        kernel_x=np.arange(len(pd_gpu_time_duration["Metric Value"]))
        
        axs[0].scatter(kernel_x,pd_gpu_time_duration["Metric Value"],label="factor={}".format(factor))
        axs[1].scatter(kernel_x,pd_l2_hit_rate["Metric Value"],label="factor={}".format(factor))
        axs[2].scatter(kernel_x,pd_l2_utilization["Metric Value"],label="factor={}".format(factor))
        axs[3].scatter(kernel_x,pd_sm_efficiency["Metric Value"],label="factor={}".format(factor))
    axs[0].set_ylabel("gpu time duration(us)")
    axs[1].set_ylabel("l2 hit rate(%)")
    axs[2].set_ylabel("l2 utilization(%)")
    axs[3].set_ylabel("sm efficiency(%)")
    axs[0].legend(loc="upper right")
    axs[1].legend(loc="upper right")
    axs[2].legend(loc="upper right")
    axs[3].legend(loc="upper right")
    plt.xlabel("{} batchsize={} factor={} kernel profile".format(model_name,batch_size,factor))
    
    plt.savefig("../pic/resnet-18_kernel_analyse/{}_{}_kernel_sm_runtime_L2cache_1.png".format(model_name,batch_size))
    #plt.show()

    mean_gpu_time_duration=np.array(mean_gpu_time_duration)
    mean_l2_hit_rate=np.array(mean_l2_hit_rate)
    mean_l2_utilization=np.array(mean_l2_utilization)
    mean_sm_efficiency=np.array(mean_sm_efficiency)
    mean_gpu_time_duration=(mean_gpu_time_duration/mean_gpu_time_duration[0])*100

    model_data_list=[]
    for i in range(4):
        model_data_list.append([mean_gpu_time_duration[i],mean_l2_hit_rate[i],mean_l2_utilization[i],mean_sm_efficiency[i]])

    plt.figure(figsize=(10,5))
    xlabels=["inference time","l2 hit rate","l2 utilization","sm efficiency"]
    xlims=np.arange(len(xlabels))
    plt.bar(xlims-0.3,model_data_list[0],width=0.15,label="factor=50")
    plt.bar(xlims-0.15,model_data_list[1],width=0.15,label="factor=100")
    plt.bar(xlims,model_data_list[2],width=0.15,label="factor=200")
    plt.bar(xlims+0.15,model_data_list[3],width=0.15,label="factor=400")
    plt.xticks(xlims,xlabels)
    plt.legend()
    plt.savefig("../pic/resnet-18_kernel_analyse/{}_{}_kernel_sm_runtime_L2cache_2.png".format(model_name,batch_size))
    #plt.show()
    
    
    
    

