import sys
import numpy as np
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import re
import torchvision.models as models
# 获取当前脚本所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取父目录
parent_dir = os.path.dirname(current_dir)
# 将父目录添加到sys.path中
sys.path.append(parent_dir)

from kernel_sm_runtime_L2cache_analyse import kernel_sm_runtime_L2cache_analyse

from utils.utils import model_params_count,model_gflops_count,model_para_get
from draw import model_mps_no_interference

folder_path = './profiles/exectime_txt/'  # 替换为你的文件夹路径



#获取single_coefficient和multi_coefficient以及一些其他参数的函数
def data2csv():
    # 使用glob模块获取所有以"mps"开头的文件
    #mps_files = glob.glob(os.path.join(folder_path, 'mps*'))
    interference_files = glob.glob(os.path.join(folder_path, 'interference*'))
    #multi_interference_files = glob.glob(os.path.join(folder_path, 'multi-interference*'))

    column_names=["modelname1","batchsize1","kernel_num1","mean_gpu_time_duration1","mean_l2_hit_rate1","mean_l2_utilization1","mean_sm_efficiency1","model_gflops1","model_para1","model_GFLOPS1",\
            "modelname2","batchsize2","kernel_num2","mean_gpu_time_duration2","mean_l2_hit_rate2","mean_l2_utilization2","mean_sm_efficiency2","model_gflops2","model_para2","model_GFLOPS2","single_coefficient","multi_coefficient"]
    df=pd.DataFrame(columns=column_names)
    #模型干扰特征提取
    for filename in interference_files:
        name1=filename.split("_")[-2]
        name2=filename.split("_")[-1][:-4]
        filename1="./profiles/exectime_txt/mps_{}.txt".format(name1)
        filename2=filename
        filename3="./profiles/exectime_txt/multi-interference_{}_{}.txt".format(name1,name2)
        if os.path.exists(filename1) and os.path.exists(filename2) and os.path.exists(filename3):
            pass
        else:
            continue
        
        data_no_interference=[]
        with open(filename1,'r') as f:
            for line in f.readlines():
                if line[-3:-1]=="ms":
                    data=line.split()[-2]
                    data_no_interference.append(float(data))
        data_no_interference=np.array(data_no_interference)
        
        data_interference=[]
        with open(filename2,'r') as f:
            temp=[]
            for line in f.readlines():
                if line[-3:-1]=="ms":
                    data=line.split()[-2]
                    temp.append(float(data))
                if line[0]=='-':
                    data_interference.append(temp)
                    temp=[]
        data_interference = [(np.array(data)/np.array(data).min())*100 for data in data_interference]
        # 使用np.zeros创建一个形状适当的数组，填充缺失值
        # max_len = max(len(arr) for arr in data_interference)
        # data_interference = np.array([np.pad(arr, (0, max_len - len(arr))) for arr in data_interference])
        
        data_multi_interference=[]
        with open(filename3,'r') as f:
            temp=[]
            for line in f.readlines():
                if line[-3:-1]=="ms":
                    data=line.split()[-2]
                    temp.append(float(data))
                if line[0]=='-':
                    data_multi_interference.append(temp)
                    temp=[]
        data_multi_interference = [np.array(data) for data in data_multi_interference]
        
        #single和multi系数分别代表了特定干扰模型数量和mps数量对干扰增加的影响程度
        #single系数 干扰增加比率（随mps每增加10%）斜率系数的平均值
        temp=[]
        for data in data_interference:
            if len(data)==1:
                continue
            x=np.arange(1,len(data)+1)
            coefficient=np.polyfit(x,data,1)[0]
            temp.append(coefficient)
        coefficient1=np.mean(temp)
        
        #multi系数 干扰增加比率（随模型数量）斜率系数的平均值
        temp=[]
        x=[1,2,3]
        for data in data_multi_interference:
            data=data/data[0]*100
            coefficient=np.polyfit(x,data,1)[0]
            temp.append(coefficient)
        coefficient2=np.mean(temp)
        
        #获取模型的特征
        def model_feature_get(modelname):
            factor=modelname.split("-")[-1]
            batch_size=modelname.split("-")[-2]
            batch_size=int(batch_size)
            pattern = re.compile(r'(.*)(-{}-{})'.format(batch_size,factor))
            match = pattern.match(modelname)
            name=match.group(1)
            file="./profiles/model_feature/{}_{}_{}_kernel_sm_runtime_L2cache.csv".format(name,batch_size,factor)
            kernel_num,mean_gpu_time_duration,mean_l2_hit_rate,mean_l2_utilization,mean_sm_efficiency=kernel_sm_runtime_L2cache_analyse(file)
            return name,batch_size,kernel_num,mean_gpu_time_duration,mean_l2_hit_rate,mean_l2_utilization,mean_sm_efficiency

        feature1=model_feature_get(name1)
        feature2=model_feature_get(name2)
        
        
            
        #获取模型的gflops和参数量，并计算模型的GFLOPS
        
        def find_name(filename):
            index=len(filename)-1
            count=0
            while index>=0:
                if filename[index]=="-":
                    count+=1
                if count==2:
                    break
                index-=1
            return filename[:index]
        
        model_para1=model_para_get(feature1[0],feature1[1])
        filename="./profiles/exectime_txt/mps_{}.txt".format(name1)
        sh_batch,sh_factor=name1.split("-")[-2:]
        sh_name=find_name(name1)
        shell_commend=("bash ./model_mps_no_interference.sh {} {} {}").format(sh_name,sh_batch,sh_factor)
        os.system(shell_commend)
        runtime=model_mps_no_interference(filename)[-1]/1000
        model1_GFLOPS=model_para1[0]/runtime
        
        model_para2=model_para_get(feature2[0],feature2[1])
        filename="./profiles/exectime_txt/mps_{}.txt".format(name2)
        sh_batch,sh_factor=name2.split("-")[-2:]
        sh_name=find_name(name2)
        shell_commend=("bash ./model_mps_no_interference.sh {} {} {}").format(sh_name,sh_batch,sh_factor)
        os.system(shell_commend)
        runtime=model_mps_no_interference(filename)[-1]/1000
        model2_GFLOPS=model_para2[0]/runtime
        
        new_row=feature1+model_para1+(model1_GFLOPS,)+feature2+model_para2+(model2_GFLOPS,)+(coefficient1,coefficient2,)
        print(new_row)
        df=df._append(pd.Series(new_row,index=df.columns),ignore_index=True)
        

    df.to_csv("./profiles/model_interference_feature.csv",index=False)
    

#获取模型数量、模型GFLOPS、模型L2利用率、模型资源分配数量以及模型干扰的参数数据
def data2csv_2():
    # 使用glob模块获取所有以"mps"开头的文件
    #mps_files = glob.glob(os.path.join(folder_path, 'mps*'))
    interference_files = glob.glob(os.path.join(folder_path, 'interference*'))
    #multi_interference_files = glob.glob(os.path.join(folder_path, 'multi-interference*'))

    column_names=["modelname1","batchsize1","kernel_num1","mean_gpu_time_duration1","mean_l2_hit_rate1","mean_l2_utilization1","mean_sm_efficiency1","model_gflops1","model_para1","model_GFLOPS1","model_mps1",\
            "modelname2","batchsize2","kernel_num2","mean_gpu_time_duration2","mean_l2_hit_rate2","mean_l2_utilization2","mean_sm_efficiency2","model_gflops2","model_para2","model_GFLOPS2","model_mps2","model_num","interference"]
    df=pd.DataFrame(columns=column_names)
    #模型干扰特征提取
    for filename in interference_files:
        name1=filename.split("_")[-2]
        name2=filename.split("_")[-1][:-4]
        filename1="./profiles/exectime_txt/mps_{}.txt".format(name1)
        filename2=filename
        filename3="./profiles/exectime_txt/multi-interference_{}_{}.txt".format(name1,name2)
        if os.path.exists(filename1) and os.path.exists(filename2) and os.path.exists(filename3):
            pass
        else:
            continue
        
        #没有干扰的情况下的runtime
        data_no_interference=[]
        with open(filename1,'r') as f:
            for line in f.readlines():
                if line[-3:-1]=="ms":
                    data=line.split()[-2]
                    data_no_interference.append(float(data))
        data_no_interference=np.array(data_no_interference)
        
        data_interference=[]
        with open(filename2,'r') as f:
            temp=[]
            for line in f.readlines():
                if line[-3:-1]=="ms":
                    data=line.split()[-2]
                    temp.append(float(data))
                if line[0]=='-':
                    data_interference.append(temp)
                    temp=[]
        
        #有干扰情况下的runtime,将干扰转化为百分比
        for index,data in enumerate(data_no_interference):
            if index==len(data_interference):
                break
            data_interference[index]=np.array(data_interference[index])/data*100
        
        # 使用np.zeros创建一个形状适当的数组，填充缺失值
        # max_len = max(len(arr) for arr in data_interference)
        # data_interference = np.array([np.pad(arr, (0, max_len - len(arr))) for arr in data_interference])
        
        #多个模型干扰的情况下的runtime
        data_multi_interference=[]
        with open(filename3,'r') as f:
            temp=[]
            for line in f.readlines():
                if line[-3:-1]=="ms":
                    data=line.split()[-2]
                    temp.append(float(data))
                if line[0]=='-':
                    data_multi_interference.append(temp)
                    temp=[]
        #多模型干扰情况下的runtime，将干扰转化为百分比
        #multi_interference的数据是干扰模型占用剩余的全部资源得到的runtime，模型本身占用的资源是10,20,30,40,50,60,干扰模型的数量是1,2,3
        for i in range(6):
            data_multi_interference[i] = np.array(data_multi_interference[i])/data_no_interference[i]*100
        
        
        #获取模型的特征
        def model_feature_get(modelname):
            factor=modelname.split("-")[-1]
            batch_size=modelname.split("-")[-2]
            batch_size=int(batch_size)
            pattern = re.compile(r'(.*)(-{}-{})'.format(batch_size,factor))
            match = pattern.match(modelname)
            name=match.group(1)
            file="./profiles/model_feature/{}_{}_{}_kernel_sm_runtime_L2cache.csv".format(name,batch_size,factor)
            kernel_num,mean_gpu_time_duration,mean_l2_hit_rate,mean_l2_utilization,mean_sm_efficiency=kernel_sm_runtime_L2cache_analyse(file)
            return name,batch_size,kernel_num,mean_gpu_time_duration,mean_l2_hit_rate,mean_l2_utilization,mean_sm_efficiency

        feature1=model_feature_get(name1)
        feature2=model_feature_get(name2)
        
        #获取模型的gflops和参数量，并计算模型的GFLOPS        
        model_para1=model_para_get(feature1[0],feature1[1])
        filename="./profiles/exectime_txt/mps_{}.txt".format(name1)
        runtime=model_mps_no_interference(filename)[-1]/1000
        model1_GFLOPS=model_para1[0]/runtime
        
        model_para2=model_para_get(feature2[0],feature2[1])
        filename="./profiles/exectime_txt/mps_{}.txt".format(name2)
        runtime=model_mps_no_interference(filename)[-1]/1000
        model2_GFLOPS=model_para2[0]/runtime
        
        for i,datas in enumerate(data_interference):
            for j,data in enumerate(datas):
                new_row=feature1+model_para1+(model1_GFLOPS,(i+1)*10)+feature2+model_para2+(model2_GFLOPS,(j+1)*10,1,data)
                df=df._append(pd.Series(new_row,index=df.columns),ignore_index=True)
        
        for i,datas in enumerate(data_multi_interference):
            for j,data in enumerate(datas):
                new_row=feature1+model_para1+(model1_GFLOPS,(i+1)*10)+feature2+model_para2+(model2_GFLOPS,100-(i+1)*10,j+1,data)
                df=df._append(pd.Series(new_row,index=df.columns),ignore_index=True)
        

    df.to_csv("./profiles/model_interference_feature2.csv",index=False)

if __name__=="__main__":
    data2csv()
    data2csv_2()
    
    

    
    
    
    
    
    
    
    
