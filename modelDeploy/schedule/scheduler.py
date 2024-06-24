# -*- coding: utf-8 -*-
# @Time : 2023/11/12 12:41
# @Author : liwenxin
# @File : schedule
# @Project : modelDeployment
from typing import Dict, Union
import numpy as np
import os
import time
from modelDeploy.modelFactory.modelInterface import ModelInterface
from modelDeploy.schedule.GPU_observer import GPU_observer
from modelDeploy.modelFactory.factoryMethed import Model_Creator
from modelDeploy.dispatch.model_info import Model_Info,Model_Runtime_Info
from modelDeploy.dispatch.model_info import Model_Info
from modelDeploy.dispatch.dispatch import DataGenerator,DataTimeoutAdder,Queue_Cluster

class Scheduler:
    def __init__(self, GPU_number:int):
        self.GPU_observer = GPU_observer(GPU_number)
        self.model_creator=Model_Creator()
        
        """
        GPU_info结构
        [
            [#GPU0
                [#TRP0
                    modelname,batchsize,mps_set,threshold
                ],
                [#TRP1
                    modelname,batchsize,mps_set,threshold
                ]
                ...
            ]
        ]
        GPU__info用于监控GPU上的进程信息
        
        """
        self.GPU_info=[]
        for _ in range(GPU_number):
            self.GPU_info.append([])
        self.GPU_number=GPU_number
    
    #获取某一个GPU上的剩余资源
    def get_gpu_resource(self, GID:int)->int:
        return self.GPU_observer.get_gpu_resoure(GID)
    
    #获取某一个GPU上，所有的GPU进程信息
    def get_process_info_by_GID(self, GID:int)->list:
        return self.GPU_observer.get_process_info_by_GID(GID)
    
    #创建一个新的GPU进程
    def _create_TRP(self,
                    model_dict: Dict[str, ModelInterface],
                    mps_set:int,
                    GID:int,
                    threshold:int):
        self.GPU_observer.create_TRP(model_dict,mps_set,GID,threshold)
        self.update_GPU_info()
    
    #清除某一个GPU上的进程
    def kill_TRP(self,GID:int,TRP_ID:int):
        self.GPU_observer.kill_TRP(GID,TRP_ID)
        
    
    #清除某GPU上的所有进程
    def close(self):
        for GID in range(self.GPU_number):
            self.GPU_observer.kill_all_TRP(GID)
    
    #获取某一个GPU上的某一个进程
    def _get_TRP(self,GID:int,TRP_ID:int):
        return self.GPU_observer.get_TRP(GID,TRP_ID)
    
    #获取某个进程的阈值
    def set_TRP_threshold(self,GID:int,TRP_ID:int,threshold):
        self.GPU_observer.set_TRP_threshold(GID,TRP_ID,threshold)
    
    #设置某个进程的阈值
    def get_TRP_threshold(self,GID:int,TRP_ID:int):
        return self.GPU_observer.get_TRP_threshold(GID,TRP_ID)
    
    #获取某个进程的状态
    def get_TRP_state(self,GID:int,TRP_ID:int):
        return self.GPU_observer.get_TRP_state(GID,TRP_ID)
    
    #设置某个进程的状态
    def set_TRP_state(self,GID:int,TRP_ID:int,state:int):
        self.GPU_observer.set_TRP_state(GID,TRP_ID,state)
    
    #查找含有某个模型的进程
    def find_model_TRP(self,modelname:str)->list:
        TRP_list=[]
        for GID in range(self.GPU_number):
            for TRP_ID,TRP_info in enumerate(self.GPU_info[GID]):
                TRP_state=self.get_TRP_state(GID,TRP_ID)
                if modelname in TRP_info and TRP_state==1:
                    batch_size=TRP_info[TRP_info.index(modelname)+1]
                    #GID TRP_ID batch_size mps_set
                    TRP_list.append([GID,TRP_ID,batch_size,TRP_info[-2]])
        return TRP_list
    
    
    #更新GPU_info的信息
    def update_GPU_info(self):
        #清空GPU_info
        self.GPU_info=[]
        #重新获取所有的信息
        for GID in range(self.GPU_number):
            GPU_model_info=self.get_process_info_by_GID(GID)
            self.GPU_info.append(GPU_model_info)
    
    #获取GPU_info的信息
    def get_GPU_info(self):
        return self.GPU_info
    
    #创建一个新的模型，并将模型信息更新
    def create_model(self,modelname_list:list[str],
                     batch_size_list:list[int],
                     model_info:Model_Info,
                     mps_set:int,
                     GID:int,
                     threshold:int)->None:
        for modelname,batch_size in zip(modelname_list,batch_size_list):
            batch_size_support=model_info.get_model_batchsize_support(modelname)
            #判断batch_size是否支持
            assert batch_size in batch_size_support,"batch_size is not supported!"
        
        dict={}
        for modelname,batch_size in zip(modelname_list,batch_size_list):
            input_shape=model_info.get_model_input(modelname)
            output_shape=model_info.get_model_output(modelname)
            lib_file=model_info.get_model_lib(modelname,batch_size)
            model=self.model_creator.create_model(modelname,lib_file,input_shape,output_shape,batch_size)
            dict[modelname]=model
            
        self._create_TRP(dict,mps_set,GID,threshold)
    
    
    #将数据交给进程进行推理
    def data_inference(self,modelname:str,data:list[np.ndarray],time_out:list[float],GID:int,TRP_ID:int):
        self.GPU_observer.inference(modelname, data,time_out,GID,TRP_ID)
        
        
    def default_policy(self,modelname:str
                       ,data:list[np.ndarray],
                       time_out:list[float],
                       model_info:Model_Info):
        """_summary_
        这一个策略不包含任何的SLO，只是简单的将模型分配到目标GPU上
        每次处理同一个模型的一组数据
        1.查看目前的GPU上是否有对应的模型
        2如果有，直接进行推理。
        3.如果没有，查看目前的GPU上是否有空余的资源，如果有，创建一个新的进程，进行推理
        4.如果没有，查看是否有空闲的GPU，如果有，创建一个新的进程，进行推理
        5.如果没有，返回错误信息
        Args:
            modelname (str): the name of model
            data (list[np.ndarray]): the data to deal with
            time_out (list[float]): timeout for each data
            model_info (Model_Info): the model information, used to get the model information,
                                    which is used to create model
        """
        
        #判断目前的GPU上是否有对应的模型
        for GID,GPU in enumerate(self.GPU_info):
            for TRP_ID,TRP_info in enumerate(GPU):
                if modelname in TRP_info:
                    #直接进行推理
                    self.data_inference(modelname, data,time_out,GID,TRP_ID)
                    return
        
        #如果没有，查看目前的GPU上是否有空余的资源，如果有，创建一个新的进程，进行推理
        for GID in range(self.GPU_number):
            resource=self._get_gpu_resource(GID)
            if resource>20:
                #创建一个新的模型
                batchsize=model_info.get_model_batchsize_support(modelname)[1]
                input_shape=model_info.get_model_input(modelname)
                output_shape=model_info.get_model_output(modelname)
                lib_file=model_info.get_model_lib(modelname,batchsize)
                model=self.model_creator.create_model(modelname,lib_file,input_shape,output_shape,batchsize)
                #创建一个新的进程,资源设置为20
                self._create_TRP({modelname:model},20,GID)
                #进行推理
                TRP_ID=len(self.GPU_info[GID])-1
                self.data_inference(modelname, data,time_out,GID,TRP_ID)
                return
                
    
    # def SLO_policy(self,
    #                modelname:str,
    #                queue_cluster:Queue_Cluster,
    #                model_info:Model_Info):
        
    #     """
    #     modelname:模型的名称
    #     queue_cluster:队列集群，用于获取模型的数据，queue中保存有每一个数据的时间片
    #     model_info:模型信息，用于获取模型的信息,用于创建对应模型
        
    #     时间片轮转策略
    #     这个策略符合公平性，以及具有清晰的计费方式
    #     资源比例x时间片大小x时间片数=花费
    #     由于每一个队列的到达率都是在变化的，所以时间片也是动态变化的
    #     问题一、如何确定时间片的大小
    #     问题二、如何调度
    #     假设问题一已经解决，时间片大小为T
    #     问题二的解决方案：
    #     1.根据时间片的大小，通过知晓每一个数据的处理时间，就知道要传输多少个数据
        
    #     2.查看GPU的状态，如果存在对应的合适模型，直接进行推理
    #     (每一个模型有一个threshold=开始推理时间+时间片，
    #     当下一批数据到达且模型还在处理数据，则threshold=threshold+时间片，如果threshold<time_out，直接将数据交给进程）
        
    #     3.如果没有合适的模型，或者之前的模型已经处于超载状态，查看GPU是否有空余的资源，
    #     如果有，创建一个新的进程（要计算进程创建的时间开销），进行推理（创建的方式为满足推理时延的最小资源分配）
        
    #     4.如果没有空余的资源，丢掉数据
        
    #     """
    #     pass
    
    
        

if __name__=="__main__":
    scheduler=Scheduler(1)
    
    modelInfo=Model_Info("modelDeploy/modelRepository")
    data_generator=DataGenerator(modelInfo)
    data_timeout_adder=DataTimeoutAdder(modelInfo)
    model_queue=Queue_Cluster(modelInfo)
    
    modelname1="resnet-18"
    
    modelname2="transformer"

    for i in range(1000):
        data=data_generator.generate_data(modelname1)
        time_out=data_timeout_adder.data_add_timeout(modelname1)
        model_queue.put(modelname1,data,time_out)
    
    for i in range(1000):
        data=data_generator.generate_data(modelname2)
        time_out=data_timeout_adder.data_add_timeout(modelname2)
        model_queue.put(modelname2,data,time_out)
    
    data1,timeout1=model_queue.multi_get(modelname1,100)
    data2,timeout2=model_queue.multi_get(modelname2,100)
    
    print(scheduler.get_GPU_info())
    scheduler.default_policy(modelname1, data1, timeout1, modelInfo)
    scheduler.default_policy(modelname2, data2, timeout2, modelInfo)
        
    print(scheduler.get_GPU_info())
    scheduler.default_policy(modelname1, data1, timeout1, modelInfo)
    print(scheduler.get_GPU_info())
    scheduler.close()
    
    
    
    
        
        
        
        

    
    


    
    
    
        
    
   