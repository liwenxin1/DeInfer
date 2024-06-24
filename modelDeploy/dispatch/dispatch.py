# -*- coding: utf-8 -*-
# @Time : 2023/11/10 21:41
# @Author : liwenxin
# @File : dispatch
# @Project : modelDeployment
from typing import Union
import numpy as np
import os
import time
import torch
import threading
from queue import Queue
from modelDeploy.dispatch.model_info import Model_Info
from modelDeploy.dispatch.autoCusum import Cusum_Possion_Algorithm
"""_summary_
dispatch负责为每一个模型构建数据等待队列，每次发送1/n的数据，数据包括modelname,data,arrived_time,timeout
"""
#给每一个模型加上超时时间
class DataTimeoutAdder(object):
    def __init__(self,model_info:Model_Info):
        self.model_SLO={}
        for model_name in model_info.model_info_dict.keys():
            self.model_SLO[model_name]=model_info.get_model_SLO(model_name)
        
    def data_add_timeout(self,model_name):
        #slo的单位是ms
        timeout=time.time()+self.model_SLO[model_name]/1000
        return timeout

#数据生成类,根据输入的名称生成随机数据
class DataGenerator(object):
    def __init__(self,model_info:Model_Info):
        self.model_input_dict={}
        for model_name in model_info.model_info_dict.keys():
            self.model_input_dict[model_name]=model_info.get_model_input(model_name)
            
    def generate_data(self,model_name:str):
        data_shape=self.model_input_dict[model_name]
        data=np.array(
            np.random.random(data_shape),
            dtype=np.float32
            )
        return data

#数据队列
class DataQueue(object):
    def __init__(self, shape):
        self.queue = Queue()
        self.time_out_list=[]
        self.shape = shape
        
        
    def put(self, data,time_out):
        assert data.shape == self.shape
        self.queue.put(data)
        self.time_out_list.append(time_out)

    def get(self):
        return self.queue.get(),self.time_out_list.pop(0)
    
    def qsize(self)->int:
        return self.queue.qsize()
    
    def empty(self)->bool:
        return self.queue.empty()
    
    def arrived_rate(self)->float:
        return self.qsize()/(self.time_out_list[-1]-self.time_out_list[0])

    

#数据队列簇
class Cusum_Possion_Queue(object):
    def __init__(self,model_info:Model_Info):
        self.model_queue_dict={}
        #为每一个模型初始化一个泊松强度检测器
        for model_name in model_info.model_info_dict.keys():
            self.model_queue_dict[model_name]=Cusum_Possion_Algorithm(50)
    
    
    def get_possion(self,model_name:str,x:int):
        return self.model_queue_dict[model_name].possion_get(x)
            
            
class Queue_Cluster(object):
    def __init__(self,model_info:Model_Info):
        self.model_queue_dict={}
        self.lock=threading.Lock()
        for model_name in model_info.model_info_dict.keys():
            self.model_queue_dict[model_name]=DataQueue(model_info.get_model_input(model_name))
        
        #print("model_queue init success!")
    
    def get_queues_keys(self):
        return self.model_queue_dict.keys()
    #将数据放入模型的数据队列
    def put(self,model_name,data,time_out):
        with self.lock:
            self.model_queue_dict[model_name].put(data,time_out)
    
    #从数据队列中取出数据
    def get(self,model_name):
        with self.lock:
            return self.model_queue_dict[model_name].get()
    
    #从数据队列中取出一批数据
    def multi_get(self,model_name,length)->list[np.ndarray,np.ndarray,int]:
        with self.lock:
            data=[]
            time_out=[]
            
            #如果数据队列中的数据不够，直接取出所有数据
            if length>self.model_queue_dict[model_name].qsize():
                print("there has no enough data in DataQueue!")
                print("get all data in DataQueue, the length is {}".format(self.model_queue_dict[model_name].qsize()))
                while not self.model_queue_dict[model_name].empty():
                    temp_data,temp_time_out=self.model_queue_dict[model_name].get()
                    data.append(temp_data)
                    time_out.append(temp_time_out)
            else:
                for _ in range(length):
                    temp_data,temp_time_out=self.model_queue_dict[model_name].get()
                    data.append(temp_data)
                    time_out.append(temp_time_out)
                    
            data=np.array(data)
            time_out=np.array(time_out)
            
            return data,time_out
    
    #获取数据队列的长度
    def qsize(self,model_name):
        with self.lock:
            return self.model_queue_dict[model_name].qsize()
    
    #获取数据队列的到达率
    def arrived_rate(self,model_name):
        return self.model_queue_dict[model_name].arrived_rate()


# class DataQueue:
#     def __init__(self):
#         self.queue=[]
#         self.queue_size=0

#     def append(self,data):
#         self.queue.append(data)
#         self.queue_size+=1

#     def pop_batch_data(self,pop_number):
#         """
#         :return: if the list has enough data to pop(the number of batchsize), then pop data
#         else return -1
#         """
#         if self.queue_size>=pop_number:
#             temp=self.queue[:pop_number]
#             self.queue=self.queue[pop_number:]
#             self.queue_size-=pop_number
#             # 获取第一列元素
#             data = np.array([row[0] for row in temp])
#             timeout=np.array([row[1] for row in temp])
#             return data,timeout
            
#         else:
#             print("there has no enough data in DataQueue!")
#             return -1

#     def getQueueSize(self):
#         return self.queue_size
    
# class NearlineModelQueue(object):
    
#     def __init__(self,modelRepository):
#         """
#         each model has a unique queue
#         dataDict的格式为{"resnet-18":DataQueue(),"transformer":DataQueue()}
#         """
#         self.model_queue_dict={}
#         self._init_nearline_model_queue(modelRepository)

#     def _init_nearline_model_queue(self,modelRepository):
#         model_dir=list_only_directories(modelRepository)
#         for filename in model_dir:
#             json_path=os.path.join("{}/{}/config.json".format(modelRepository,filename))
#             model_inf=load_json_file(json_path)
#             if model_inf["flag"]==1:
#                 self.model_queue_dict[model_inf["name"]]=DataQueue()
#         print("dataDict init success!")
        

#     #获取某个模型的队列长度
#     def queue_length(self,model_name:str):
#         try:
#             return self.model_queue_dict[model_name].getQueueSize()
#         except Exception as e:
#             print("has no model named {}".format(model_name))

#     def receive_data(self,data,time_out,model_name):
#         """
#         :param data: put data into dataDict <data,arrived time,end time>
#         :param model_name: the key to specify the queue
#         :return: None
#         """
#         self.model_queue_dict[model_name].append([data,time_out])
        

#     def dispatch_data(self,model_name:str,pop_number:int):
#         """
#         :param model_name: the model's name to pop the data out
#         :return: a batch of data, use Numpy to save
#         """

#         queue_size=self.queue_length(model_name)
#         if queue_size is None:
#             print("something error in monitor_model function")

#         try:
#             return self.dataDict[model_name].pop_batch_data(pop_number)
#         except Exception as e:
#             print("has something wrong in dispatch_data!")
#             print(e)

#     #以下两个函数要实现调度器过后才能实现
#     #获取综合调度器的数据
#     def get_model_info(self,
#                        schedule:Schedule,
#                        model_name:Union[str,list[str]])-> dict:
#         schedule.get_info(model_name)
    
    
#     #调用scheduler函数，生成数据队列
#         """
#         数据队列的格式为：
#         mps_set[10,20,30]数组长度代表进程的个数,每一个进程都有一个数据队列,以及一个lib queue
#         lib_queue[[modelname,lib,batch_size],...]
#         dataqueue[[modelname,data,time_out],...]]
#         """
#     #按照数据队列顺序执行
#     #不同GPU对应不同的GPU数据队列
#     def data_scheduler(self):
#         pass



    
    









