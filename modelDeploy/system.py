# -*- coding: utf-8 -*-
# @Time : 2023/11/12 10:42
# @Author : liwenxin
# @File : system
# @Project : modelDeployment
import math
import threading
import time

from modelDeploy.dispatch.dispatch import (DataGenerator, DataTimeoutAdder,
                                           Queue_Cluster)
from modelDeploy.dispatch.model_info import Model_Info, Model_Runtime_Info
from modelDeploy.schedule.scheduler import Scheduler


class Time_Slice_System:
    def __init__(self,GPU_number:int,modelRepository) -> None:
        self.scheduler=Scheduler(GPU_number)
        self.MI=Model_Info(modelRepository)
        self.MRI=Model_Runtime_Info(modelRepository)
        self.data_generator=DataGenerator(self.MI)
        self.data_timeout_adder=DataTimeoutAdder(self.MI)
        self.model_queue=Queue_Cluster(self.MI)
        
        
        
    def put_data_into_queue(self,modelname,length,arrival_time=0.1):
        """将数据均匀放置在队列中

        Args:
            modelname (_type_): 模型队列的名称
            length (_type_): 每次放置的长度
            arrival_time (float, optional): 数据到达的时间间隔. Defaults to 0.1.
        """
        while True:
            for _ in range(length):
                data=self.data_generator.generate_data(modelname)
                time_out=self.data_timeout_adder.data_add_timeout(modelname)
                self.model_queue.put(modelname,data,time_out)
                
            time.sleep(arrival_time)
        
    
        
    #获取满足SLO的batchsize最大，mps最小的资源参数
    def search_params(self,modelname,split_factor):
        #获取模型的SLO，并且将其转化为秒
        slo=self.MI.get_model_SLO(modelname)/1000
        mps_set=split_factor
        batchsize_support_list=self.MI.get_model_batchsize_support(modelname)
        while(mps_set<=100):
            #batchsize_support_list逆序遍历
            for batchsize in batchsize_support_list[::-1]:
                runtime=self.MRI.get_model_mps_runtime(model_name=modelname,batch_size=batchsize,mps=mps_set)
                if runtime<slo/2:
                    return batchsize,mps_set
                
                mps_set+=split_factor
        return None,None
            
    #返回值是两部分，第一部分是可以推理的数据区间，第二部分是无法推理的数据区间
    def interval_acquisition(self,
                             data_length:int,
                             t:list[int,int,int,int]
                             ):
        #获取数据区间
        #如果数据长度为1，直接返回
        if data_length==1:
            if t[0]<t[2]:
                return [0,1],None
            else:
                return None,[0,1]
        
        param=(t[1]-t[3])/(t[2]-t[0])
        factor=data_length/(1+param)
        if t[2]-t[0]>t[3]-t[1]:
            #向下取整
            factor=math.floor(factor)
            if factor>=data_length:
                return [0,data_length],None
            elif factor>0:
                return [0,factor],[factor,data_length]
            else:
                return None,[0,data_length]
        
        else:
            #向上取整
            factor=math.ceil(factor)
            if factor<=0:
                return [0,data_length],None
            elif factor<data_length:
                return [factor,data_length],[0,factor]
            else:
                return None,[0,data_length]
    
    

    def SLO_policy(self,split_factor:int):
        """冷启动的时候，通过模型的slo创建一个能满足SLO的最小资源模型
        每次将队列中的所有数据取出，假设一小段时间段数据到达是匀速的，threshold>=当前时间
        Args:
            split_factor (int): 资源的划分因子，划分的资源值必须是split_factor的整数倍
        """
        def model_initialize():
            for modelname in self.MI.model_info_dict.keys():
                is_create=False
                for GID in range(self.scheduler.GPU_number):
                    batchsize,mps_set=self.search_params(modelname,split_factor)
                    mps_resource=self.scheduler.get_gpu_resource(GID)
                    #资源充足才能进行分配
                    if mps_set<=mps_resource:
                        self.scheduler.create_model([modelname],[batchsize],self.MI,mps_set,GID)
                        is_create=True
                        break
                if not is_create:
                    print("create model{} failed!,maybe the resource is not enough!".format(modelname))
                    exit(-1)
        
        def model_create(modelname):
            is_create=False
            for GID in range(self.scheduler.GPU_number):
                batchsize,mps_set=self.search_params(modelname,split_factor)
                mps_resource=self.scheduler.get_gpu_resource(GID)
                #资源充足才能进行分配
                if mps_set<=mps_resource:
                    self.scheduler.create_model([modelname],[batchsize],self.MI,mps_set,GID)
                    is_create=True
                    break
            if not is_create:
                raise Exception("create model{} failed!,maybe the resource is not enough!".format(modelname))
            return GID,len(self.scheduler.get_GPU_info()[GID])-1
                
        #1.---------------------------------------------------------
        #对data_cluster中的每一个模型建立一个满足slo的最小资源模型
        
        model_initialize()
        
        #2.---------------------------------------------------------
        #第一部分保证了模型全部创建于GPU中
        #开始进行推理
        merge_time=time.time()
        merge_interval=30
        while(True):
            time.sleep(1)#给数据一点时间，让数据到达
            
            #轮询每一个模型
            #这个for循环的一次运行时间大概是0.018s
            for modelname in self.MI.model_info_dict.keys():
                #获取队列中的所有数据
                if self.model_queue.qsize(modelname)!=0:
                    data_length=self.model_queue.qsize(modelname)
                    data,time_out=self.model_queue.multi_get(modelname,data_length)
                    
                    #设置目前的模型能否完成所有推理的标志位
                    finish_flag=False
                    #获取对应的模型的索引
                    model_list=self.scheduler.find_model_TRP(modelname)
                    for GID,TRP_ID,batch_size,mps_set in model_list:
                        model_threshold=self.scheduler.get_TRP_threshold(GID,TRP_ID)
                        
                        #一个batch的数据的推理时间
                        inference_time=self.MRI.get_model_mps_runtime(
                            model_name=modelname,
                            batch_size=batch_size,
                            mps=mps_set)
                        
                        #threshold>=当前时间
                        model_threshold=model_threshold if model_threshold>time.time() else time.time()
                        #第一个数据的完成时间
                        t1=model_threshold+inference_time
                        #最后一个数据的完成时间
                        t2=model_threshold+inference_time*math.ceil(data_length/batch_size)
                        
                        #第一个数据超时时间
                        t3=time_out[0]
                        #最后一个数据超时时间
                        t4=time_out[-1]
                        
                        t=[t1,t2,t3,t4]
                        
                        #获取数据区间,i_interval是可以推理的数据区间，n_interval是无法推理的数据区间
                        i_interval,n_interval=self.interval_acquisition(data_length,t)
                        print(i_interval)
                        if i_interval!=None:
                            infer_data=data[i_interval[0]:i_interval[1]]
                            infer_time_out=time_out[i_interval[0]:i_interval[1]]
                            print("TRP_ID{} modelname:{},data_length:{},finish data:{}".format(TRP_ID,modelname,data_length,i_interval[1]-i_interval[0]))
                            #将数据交给进程进行推理
                            #1.设置模型完成时间
                            model_threshold=model_threshold+inference_time*math.ceil(((i_interval[1]-i_interval[0])/batch_size))
                            
                            #print("{} predict time:{}".format(modelname,model_threshold))
                            self.scheduler.set_TRP_threshold(GID,TRP_ID,model_threshold)
                            #2.模型推理
                            self.scheduler.data_inference(modelname,infer_data,infer_time_out,GID,TRP_ID)
                            
                            #表示无法处理的数据为空，所有的数据都处理完成
                            if n_interval==None:
                                finish_flag=True
                                break
                            #否则继续处理无法处理的数据
                            else:
                                data=data[n_interval[0]:n_interval[1]]
                                time_out=time_out[n_interval[0]:n_interval[1]]
                                data_length=n_interval[1]-n_interval[0]
                        
                        #当前模型无法处理数据，继续下一个模型
                        else:
                            continue
                    
                    #如果数据没有都处理完成，就新建一个模型处理数据
                    if not finish_flag:
                        try:
                            GID,TRP_ID=model_create(modelname)
                            self.scheduler.data_inference(modelname,data,time_out,GID,TRP_ID)
                            
                        except Exception as e:
                            print(e)
            
            #每隔一段时间进行merge操作,进行merge的时候，所有的模型都是running状态
            if time.time()-merge_time>merge_interval:
                #对每一个模型进行merge操作
                #1.获取所有的模型，释放没有数据的模型
                #2.将碎片化的模型进行合并
                #3.将不同的模型进行合并（目前先不做这个）
                
                #1
                GPU_info=self.scheduler.get_GPU_info()
                for GID in range(self.scheduler.GPU_number):
                    #逆序遍历
                    for TRP_ID in range(len(GPU_info[GID])-1,-1,-1):
                        threshold=self.scheduler.get_TRP_threshold(GID,TRP_ID)
                        if threshold<time.time():
                            self.scheduler.kill_TRP(GID,TRP_ID)
                self.scheduler.update_GPU_info()
                
                #2.装箱问题的改进算法
                #找到所有需要合并的模型,意味着模型数量大于1
                model_merge_list=[]
                for modelname in self.MI.model_info_dict.keys():
                    model_list=self.scheduler.find_model_TRP(modelname)
                    if len(model_list)>1:
                        temp=[modelname]
                        for GID,TRP_ID,batch_size,mps_set in model_list:
                            temp.append([GID,TRP_ID,batch_size,mps_set])
                        model_merge_list.append(temp)
                    
                    
                #对模型资源合并过后的资源进行计算，得到的列表
                model_merge_resource_list=[]
                #获取每一个模型的ID所用的列表
                del_model_list=[]
                for model_info in model_merge_list:
                    modelname=model_info[0]
                    model_list=model_info[1:]
                    batch_size_choose=0
                    mps_set=0
                    for GID,TRP_ID,batch_size,mps_set in model_list:
                        if batch_size_choose<batch_size:
                            batch_size_choose=batch_size
                        mps_set+=mps_set
                        del_model_list.append([GID,TRP_ID])
                    model_merge_resource_list.append([modelname,batch_size_choose,mps_set])
                #对模型资源进行从小到大排序
                model_merge_resource_list.sort(key=lambda x:x[2])
                #对del_model_list进行排序,按照GID从小到大，TRP_ID从大到小
                del_model_list.sort(key=lambda x:(x[0],-x[1]))
                
                
                #删除需要资源合并的模型
                for GID,TRP_ID in del_model_list:
                    self.scheduler.kill_TRP(GID,TRP_ID)
                self.scheduler.update_GPU_info()
                
                    
                #获取当前剩余资源
                GPU_resource_list=[]
                for GID in range(self.scheduler.GPU_number):
                    resource=self.scheduler.get_gpu_resource(GID)
                    GPU_resource_list.append(resource)
                
                    
                #装箱问题，采用首次适宜降序法，模型资源是可分割的
                #对model_merge_resource_list逆序遍历
                for model_info in model_merge_resource_list[::-1]:
                    modelname=model_info[0]
                    batch_size=model_info[1]
                    mps_set=model_info[2]
                    for index,resource in enumerate(GPU_resource_list):
                        #资源充足，直接进行模型创建
                        if mps_set<=resource:
                            self.scheduler.create_model([modelname],[batch_size],self.MI,mps_set,GID)
                            GPU_resource_list[index]-=mps_set
                            model_merge_resource_list.remove(model_info)
                            break
                
                #如果存在模型不能被创建，说明单个GPU的资源不足，需要将模型分割
                if model_merge_resource_list!=[]:
                    for model_info in model_merge_resource_list[::-1]:
                        modelname=model_info[0]
                        batch_size=model_info[1]
                        mps_set=model_info[2]
                        for index,resource in enumerate(GPU_resource_list):
                            if mps_set>=resource:
                                self.scheduler.create_model([modelname],[batch_size],self.MI,resource,GID)
                                mps_set-=resource
                                continue
                            else:
                                self.scheduler.create_model([modelname],[batch_size],self.MI,mps_set,GID)
                                break
            

                merge_time=time.time()
                            
                    
                        
                        
                        
                        
            
            
    
    def close(self):
        self.scheduler.close()

def producer(tts:Time_Slice_System,modelname:str,length:int,arrival_time=0.1):
    while True:
        tts.put_data_into_queue(modelname,length,arrival_time)
                    
if __name__=="__main__":
    tss=Time_Slice_System(1,"modelDeploy/modelRepository")
    threading.Thread(target=producer,args=(tss,"resnet-18",20,0.1)).start()
    #threading.Thread(target=producer,args=(tss,"transformer",30,0.1)).start()
    
    tss.SLO_policy(10)
    print("end")
                        
                        
                        
                        
                        
                        
                        

                            
                        
                        
                            
                        
                                
                            
                                
                            
                                
                            
                        
                        
                    
                    
                    
                    
            
            
        
        
                    
            
            


    


