import math
import threading
import time

from modelDeploy.dispatch.dispatch import (DataGenerator, DataTimeoutAdder,
                                           Queue_Cluster,Cusum_Possion_Queue)

from modelDeploy.dispatch.model_info import Model_Info, Model_Runtime_Info
from modelDeploy.schedule.scheduler import Scheduler
from modelDeploy.forecasting_model.RF_model import ForcastModel
import numpy as np
class DeInfer_system:
    def __init__(self,GPU_number:int,modelRepository,forcast_p) -> None:
        self.scheduler=Scheduler(GPU_number)#记录GPU的调度信息，并提供调度功能
        self.MI=Model_Info(modelRepository)#记录模型的信息
        self.MRI=Model_Runtime_Info(modelRepository)#记录模型的运行时信息
        self.data_generator=DataGenerator(self.MI)#生成数据
        self.data_timeout_adder=DataTimeoutAdder(self.MI)#生成数据的超时时间
        self.model_queue=Queue_Cluster(self.MI)#存储队列
        self.possion_algorithm=Cusum_Possion_Queue(self.MI)#记录模型的到达率
        
        self.forcast_model=ForcastModel()#时延预测模型
        self.forcast_model.load_model(forcast_p)#加载时延预测模型
        
    
        
        
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
    
    def config(self,GID):
        #列表的元素包含模型的名称，资源分配，batch，l2 cache利用率，GFLOPS
        info=self.scheduler.GPU_info[GID]
        conf=[]
        for i in info:
            modelname,batch,mps,_=i
            l2cache,gflops=self.MI.repositorySearch(modelname,batch)
            conf.append([l2cache,gflops,mps])
        return conf
    
    #返回当前模型的资源分配情况
    def resourceAllocation(self,modelname,slo,arrival_rate):
        gpu_resource=[]
        for i in range(self.scheduler.GPU_number):
            gpu_resource.append(self.scheduler.get_gpu_resource(i))
        gpu_index=np.argsort(gpu_resource)
        result=[]
        for index in gpu_index:
            #获取剩余GPU资源
            r=gpu_resource[index]
            if r<10:
                continue
            #获取当前模型已分配的负载的参数信息
            info_list=self.config(index)
            #获取当前模型的资源分配
            temp_result=self.resourceCompute(modelname,r,info_list,arrival_rate,slo,index)
            
            if temp_result[1]>=arrival_rate:
                result.append(temp_result)
                
                return result
            
            else:
                result.append(temp_result)
                arrival_rate-=temp_result[1]
        
        return -1
                

    def resourceCompute(self,modelname,r,info_list,arrivalrate,slo,index_j):
        #找到满足条件的batch最大值
        batch_list=self.MI.get_model_batchsize_support(modelname)
        b_ij=0
        for batch in batch_list[::-1]:
            if self.MRI.get_model_mps_runtime(modelname,batch,r)<slo/2:
                b_ij=batch
                break
            
        l2cache,gflops=self.MI.repositorySearch(modelname,batch)
       
        MaxD_ij=(arrivalrate*slo+2-math.sqrt(arrivalrate**2*slo**2+4))/(2*arrivalrate)
        
        X=[[l2cache,gflops,r]]+info_list
        I_ij=self.forcast_model.predict(X)
        D_ij=self.MRI.get_model_mps_runtime(modelname,b_ij,r)*(1+I_ij)/batch
        if D_ij<MaxD_ij:
            #资源足够
            #这里用二分法应该更快，但是算法的瓶颈不在这里，所以暂时不做优化
            resource_list=np.arange(10,r,1)
            mps_ij=0
            for mps in resource_list:
                #不同资源下的干扰不一样，所以要重新计算
                X=[[l2cache,gflops,mps]]+info_list
                I_ij=self.forcast_model.predict(X)
                D_ij=self.MRI.get_model_mps_runtime(modelname,b_ij,mps)*(1+I_ij)/batch
                if D_ij<MaxD_ij*0.95:
                    mps_ij=mps
                    break
            #返回GPU的索引，到达率，资源分配，batch
            return (index_j,arrivalrate,mps_ij,b_ij)
        
        else:
            #资源不够，把剩下的所有资源都给它
            lambda_ij=(2*D_ij-slo)/(D_ij**2-D_ij*slo)
            return (index_j,lambda_ij,r,b_ij)
    
    
    def run(self):
        #首先根据每个模型的数据到达率，进行资源的分配
        time.sleep(1)
        for modelname in self.MI.model_info_dict.keys():
            #获取队列中的所有数据
            #if self.model_queue.qsize(modelname)!=0:
            if True:
                data_length=self.model_queue.qsize(modelname)
                data,time_out=self.model_queue.multi_get(modelname,data_length)
                lambda_stress=self.possion_algorithm.get_possion(model_name=modelname,x=data_length)
                allocation_list=self.resourceAllocation(modelname,self.MI.get_model_SLO(modelname)/1000,lambda_stress)
                for allocation in allocation_list:
                    GID,lambda_ij,mps_ij,b_ij=allocation
                    self.scheduler.create_model([modelname],[b_ij],self.MI,mps_ij,GID,lambda_ij)
                    
                        
                    
                    
        
        while(True):
            #每隔一秒进行一次数据的放置
            time.sleep(1)
            for modelname in self.MI.model_info_dict.keys():
                #获取队列中的所有数据
                if self.model_queue.qsize(modelname)!=0:
                    data_length=self.model_queue.qsize(modelname)
                    data,time_out=self.model_queue.multi_get(modelname,data_length)
                    model_list=self.scheduler.find_model_TRP(modelname)
                   
                    
                    #数据的索引位置
                    data_index=0
                    for GID,TRP_ID,batch_size,mps_set in model_list:
                        model_threshold=int(self.scheduler.get_TRP_threshold(GID,TRP_ID))
                        
                        if data_index<data_length:
                            #将model_threshold个数据放入GPU中
                            if data_length-data_index>model_threshold:
                                infer_data=data[data_index:data_index+model_threshold]
                                infer_time_out=time_out[data_index:data_index+model_threshold]
                                self.scheduler.data_inference(modelname,infer_data,infer_time_out,GID,TRP_ID)
                                data_index+=model_threshold
                            #将剩余的数据放入GPU中
                            else:
                                infer_data=data[data_index:]
                                
                                infer_time_out=time_out[data_index:]
                                self.scheduler.data_inference(modelname,infer_data,infer_time_out,GID,TRP_ID)
                        
                    
                    
def producer(tts:DeInfer_system,modelname:str,length:int,arrival_time=0.1):
    while True:
        tts.put_data_into_queue(modelname,length,arrival_time)

if __name__=="__main__":
    system=DeInfer_system(2,"modelDeploy/modelRepository",forcast_p="modelDeploy/forecasting_model/random_forest_model.pkl")
    threading.Thread(target=producer,args=(system,"mobilenet",20,0.1)).start()
    system.run()
    
        