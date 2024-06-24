import matplotlib.pyplot as plt
import numpy as np
from kernel_sm_runtime_L2cache_analyse import kernel_sm_runtime_L2cache_analyse
def model_compare():
    #不同模型的参数比较
    # lstm_data=[1690,23.537,54.885,7.5846,12.20]
    # transformer=[198,26.042727272727276,48.02696969696971,33.5719191919192,24.36318181818182]
    # resnet_1_200_data=[32 ,88.87406250000001 ,68.47375 ,20.964062499999997 ,40.10625]
    # resnet_2_200_data=[32 ,149.7875 ,68.181875, 20.456249999999997 ,45.2053125]
    # resnet_4_200_data=[32 ,232.16875 ,70.7915625 ,23.167187499999997 ,43.8478125]
    # resnet_8_200_data=[32 ,314.4871875 ,68.8575 ,26.1375 ,42.620937500000004]
    lstm_data=kernel_sm_runtime_L2cache_analyse("./profiles/LSTM-2_100_untuned_kernel_sm_runtime_L2cache.csv")
    transformer=kernel_sm_runtime_L2cache_analyse("./profiles/transformer_32_untuned_kernel_sm_runtime_L2cache.csv")
    resnet_1_200_data=kernel_sm_runtime_L2cache_analyse("./profiles/resnet-18_1_200_kernel_sm_runtime_L2cache.csv")
    resnet_2_200_data=kernel_sm_runtime_L2cache_analyse("./profiles/resnet-18_2_200_kernel_sm_runtime_L2cache.csv")
    resnet_4_200_data=kernel_sm_runtime_L2cache_analyse("./profiles/resnet-18_4_200_kernel_sm_runtime_L2cache.csv")
    resnet_8_200_data=kernel_sm_runtime_L2cache_analyse("./profiles/resnet-18_8_200_kernel_sm_runtime_L2cache.csv")


    resnet_1_200_data=np.array(resnet_1_200_data)/lstm_data
    resnet_2_200_data=np.array(resnet_2_200_data)/lstm_data
    resnet_4_200_data=np.array(resnet_4_200_data)/lstm_data
    resnet_8_200_data=np.array(resnet_8_200_data)/lstm_data
    transformer=np.array(transformer)/lstm_data
    lstm_data=np.array(lstm_data)/lstm_data
    x=np.array([i*3 for i in range(5)])
    fig=plt.figure(figsize=(10,5))
    plt.bar(x-0.9,transformer,width=0.3,label='transformer')
    plt.bar(x-0.6,lstm_data,width=0.3,label='lstm-2')
    plt.bar(x-0.3,resnet_1_200_data,width=0.3,label='resnet-18_1_200')
    plt.bar(x,resnet_2_200_data,width=0.3,label='resnet-18_2_200')
    plt.bar(x+0.3,resnet_4_200_data,width=0.3,label='resnet-18_4_200')
    plt.bar(x+0.6,resnet_8_200_data,width=0.3,label='resnet-18_8_200')
    plt.xticks(x,('kernel num','gpu time duration','l2 hit rate','l2 utilization','sm efficiency'),fontsize=8)
    plt.legend(fontsize=8)
    plt.savefig('../pic/model_compare.png')
    plt.close()
    
    
def model_compare_resnet(*args, **kwargs):
    #resnet簇的参数比较
    modelname,batchsize,factor_flag=None,None,None
    for key, value in kwargs.items():
        if key == 'modelname':
            modelname = value
        if key == 'batchsize':
            batchsize = value 
        if key == 'factor':
            factor_flag = value
    
    #同一模型不同batchsize比较
    if modelname!=None and batchsize==None and factor_flag==None:   
        modelname_list=["resnet-18","resnet-34","resnet-50","resnet-101","resnet-152"]
        if modelname not in modelname_list:
            print("modelname error")
            exit(-1)
            
        batch_size=[1,2,4,8,16]
        data_list=[]
        
        for size in batch_size:
            data_list.append(kernel_sm_runtime_L2cache_analyse("./profiles/{}_{}_200_kernel_sm_runtime_L2cache.csv".format(modelname,size)))
        
        data_list=np.array(data_list)
        data_list=data_list/data_list[0]
        x=np.array([i*3 for i in range(5)])
        fig=plt.figure(figsize=(10,5))
        width=0.3
        begin=-width*3
        for index,batch in enumerate(batch_size):
            plt.bar(x+begin,data_list[index],width=width,label="{} batchsize={}".format(modelname,batch))
            begin+=width
        plt.xticks(x,('kernel num','gpu time duration','l2 hit rate','l2 utilization','sm efficiency'),fontsize=8)
        plt.legend(fontsize=8)
        plt.show()
    
    #同一batchsize不同模型比较
    elif modelname==None and batchsize!=None and factor_flag==None:
        modelname_list=["resnet-18","resnet-34","resnet-50","resnet-101","resnet-152"]
        data_list=[]
         
        for name in modelname_list:
            data_list.append(kernel_sm_runtime_L2cache_analyse("./profiles/{}_{}_200_kernel_sm_runtime_L2cache.csv".format(name,batchsize)))
        data_list=np.array(data_list)
        data_list=data_list/data_list[0]
        x=np.array([i*3 for i in range(5)])
        fig=plt.figure(figsize=(10,5))
        width=0.3
        begin=-width*2
        for index,modelname in enumerate(modelname_list):
            plt.bar(x+begin,data_list[index],width=width,label="{} batchsize={}".format(modelname,batchsize))
            begin+=width
        plt.xticks(x,('kernel num','gpu time duration','l2 hit rate','l2 utilization','sm efficiency'),fontsize=8)
        plt.legend(fontsize=8)
        plt.show()
        
    #同一模型不同优化程度比较
    elif modelname!=None and batchsize!=None and factor_flag!=None:
        #只支持resnet-18
        data_list=[]
        factor_list=["untuned",50,200,400]
        for factor in factor_list:
            data_list.append(kernel_sm_runtime_L2cache_analyse("./profiles/{}_{}_{}_kernel_sm_runtime_L2cache.csv".format(modelname,batchsize,factor)))
        data_list=np.array(data_list)
        data_list=data_list/data_list[0]
        x=np.array([i*3 for i in range(5)])
        fig=plt.figure(figsize=(10,5))
        width=0.3
        begin=-width*2
        for index,factor in enumerate(factor_list):
            plt.bar(x+begin,data_list[index],width=width,label="{} batchsize={} factor={}".format(modelname,batchsize,factor))
            begin+=width
        plt.xticks(x,('kernel num','gpu time duration','l2 hit rate','l2 utilization','sm efficiency'),fontsize=8)
        plt.legend(fontsize=8)
        plt.show()
        
        
    
    #plt.savefig('../pic/model_compare.png')
    plt.close()
    
    

def model_mps_no_interference(filename):
    data_no_interference=[]
    with open(filename,'r') as f:
        for line in f.readlines():
            if line[-3:-1]=="ms":
                data=line.split()[-2]
                data_no_interference.append(float(data))
    return data_no_interference
def resnet_18_mps_interference(filename):
    data_no_interference=[0.02220066494,0.00818814067,0.00629747512,0.0047732172,\
        0.00429408114,0.0034276351400000004,0.0030648441400000005,0.00291074748,0.0028428182000000005,0.00238310268]
    data_interference=[]
    with open(filename,'r') as f:
        temp=[]
        for line in f.readlines():
            if line[-3:-1]=="ms":
                data=line.split()[-2]
                temp.append(float(data))
            if line[0]=='-':
                data_interference.append(temp)
                temp=[]
    data_interference = [np.array(data) for data in data_interference]

    # 使用np.zeros创建一个形状适当的数组，填充缺失值
    max_len = max(len(arr) for arr in data_interference)
    
    data_interference = np.array([np.pad(arr, (0, max_len - len(arr))) for arr in data_interference])
    #data_interference的每一行代表干扰相同，资源分配不同的情况
    data_interference = data_interference.T
    
    width=0.2
    plt.bar(np.array([i*3 for i in range(len(data_no_interference))])-width*4,data_no_interference,width=width,label='no interference')
    begin=-width*3
    for index,data in enumerate(data_interference):
        x=np.array([i*3 for i in range(0,len(data))])
        plt.bar(x+begin,data,width=width,label='interference={}'.format((index+1)*10))
        begin+=width
    
    x_label=["10","20","30","40","50","60","70","80","90","100"]
    plt.xlabel("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE")
    plt.ylabel("time duration(ms)")
    plt.xticks([i*3 for i in range(0,10)], x_label)
    plt.legend()
    name=filename.split("_")[-1][:-4]
    
    plt.savefig('../pic/resnet_18_mps_interference_with_{}.png'.format(name))

def model_mps_interference(filename):
    #两个模型共置的干扰模型的比较
    name1=filename.split("_")[-2]
    name2=filename.split("_")[-1][:-4]
    
    filename1="./profiles/exectime_txt/mps_{}.txt".format(name1)
    filename2=filename
    data_no_interference=[]
    with open(filename1,'r') as f:
        for line in f.readlines():
            if line[-3:-1]=="ms":
                data=line.split()[-2]
                data_no_interference.append(float(data))
                
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
    data_interference = [np.array(data) for data in data_interference]
    
    
    # 使用np.zeros创建一个形状适当的数组，填充缺失值
    max_len = max(len(arr) for arr in data_interference)

    
    data_interference = np.array([np.pad(arr, (0, max_len - len(arr))) for arr in data_interference])
    
    #data_interference的每一行代表干扰相同，资源分配不同的情况
    data_interference = data_interference.T
    
    
    width=0.2
    plt.bar(np.array([i*3 for i in range(len(data_no_interference))])-width*4,data_no_interference,width=width,label='no interference')
    begin=-width*3
    
    for index,data in enumerate(data_interference):
        x=np.array([i*3 for i in range(0,len(data))])
        plt.bar(x+begin,data,width=width,label='interference={}'.format((index+1)*10))
        begin+=width
        
    
    x_label=["10","20","30","40","50","60","70","80","90","100"]
    plt.xlabel("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE")
    plt.ylabel("time duration(ms)")
    plt.xticks([i*3 for i in range(0,10)], x_label)
    plt.legend()
    name1=filename1.split("_")[-1][:-4]
    name2=filename2.split("_")[-1][:-4]
    
    plt.savefig('../pic/single_interference_test/single_{}_mps_interference_with_{}.png'.format(name1,name2))
    #plt.show()
    plt.close()
    # data_no_inter=data_interference[0]
    
    # for index in range(1,len(data_interference)):
    #     plt.plot([1,2,3,4],(data_interference[index]/data_no_inter-1)[:4])
    #     plt.show()
    #     print((data_interference[index]/data_no_inter-1)[:4])
        
def model_mps_interference_multi(filename):
    #一个模型和多个模型共置的干扰模型比较
    #data_interference中的每一个元素代表资源分配相同，干扰模型的数目不同，干扰模型的总资源相同的数据，分别有1,2,3,4个干扰模型
    plt.figure(figsize=(10,5))
    data_interference=[]
    with open(filename,'r') as f:
        temp=[]
        for line in f.readlines():
            if line[-3:-1]=="ms":
                data=line.split()[-2]
                temp.append(float(data))
            if line[0]=='-':
                data_interference.append(temp)
                temp=[]
    data_interference=np.array(data_interference)
    data_interference=data_interference.T
    
    width=0.2
    begin=-width
    for index,data in enumerate(data_interference):
        x=np.array([i*3 for i in range(0,len(data))])
        plt.bar(x+begin,data,width=width,label='interference_model_num={}'.format(index+1))
        begin+=width
    
    x_label=["10","20","30","40","50","60"]
    plt.xlabel("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE")
    plt.ylabel("time duration(ms)")
    plt.xticks([i*3 for i in range(0,len(data))], x_label)
    plt.legend()
    name1=filename.split("_")[-2]
    name2=filename.split("_")[-1][:-4]
    
    
    plt.savefig('../pic/multi_test/multi_{}_mps_interference_with_{}.png'.format(name1,name2))
    plt.close()

    
    
            
            

if __name__=="__main__":
    #model_compare()
    import os
    import sys
    # 获取当前脚本所在的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 获取父目录
    parent_dir = os.path.dirname(current_dir)
    # 将父目录添加到sys.path中
    sys.path.append(parent_dir)
    from model_test.test import find_name
   
    x = find_name("interference")
    root="./profiles/exectime_txt/"
    for name in x:
        name=os.path.join(root,name)
        model_mps_interference(name)
        
    #model_mps_interference("./profiles/exectime_txt/interference_vgg-19-8-untuned_resnet-50-16-200.txt")
    #model_mps_interference_multi("./profiles/exectime_txt/multi-interference_resnet-18-1-50_resnet-18-4-untuned.txt")
    #model_compare_resnet(modelname='resnet-18',batchsize=1,factor=True)
