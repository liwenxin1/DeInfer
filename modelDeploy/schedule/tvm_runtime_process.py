from typing import Dict
from modelDeploy.modelFactory.modelInterface import ModelInterface
import os
import multiprocessing as mp
import numpy as np
import time
"""_summary_
tvm_runtime目前还不能工作，因为数据切分还没有完成
"""
class TVM_runtime:
    def __init__(self, model_dict: Dict[str, ModelInterface],mps_set:int,mp_queue:mp.Queue,GID:int):
        self.model_dict = model_dict
        self.mps_set = mps_set
        self.queue=mp_queue
        self.GID=GID
    
    def _activate_all_models(self):
        for modelname in self.model_dict.keys():
            self.model_dict[modelname].load()
            self.model_dict[modelname].activate(self.GID)
    
    def data_deal(self,modelname,data):
        #取batch_size大于data.shape[0]的最小倍数，对数据进行padding
        batch_size=self.model_dict[modelname].get_batch_size()
        padding_para=((data.shape[0]-1)//batch_size+1)*batch_size
        pad_width = [(0, padding_para - data.shape[0])] + [(0, 0)] * (data.ndim - 1)
        data=np.pad(data,pad_width)
        
        #将数据按照batch_size进行切块
        n = int(np.ceil(len(data) / batch_size))
        chunks = np.array_split(data, n)
        return chunks
    
    def run(self):
        os.environ['CUDA_MPS_ACTIVE_THREAD_PERCENTAGE'] = str(self.mps_set)
        #初始化操作
        self._activate_all_models()
        #print("activate success")
        
        #进程通信运行，基于进程间队列
        while True:
            data=self.queue.get()
            
            #check for termination
            if data=="terminate":
                break
            modelname=data[0]
            infer_data=data[1]
            time_out=data[2]
            data_length=len(data[1])
            infer_data=self.data_deal(modelname, infer_data)
            
            
            for data in infer_data:
                
                y=self.inference(modelname, data)

                #返回结果
                #TODO
                #print(y.shape)
            print("{} data_length:{}".format(modelname,data_length))
            
        
    def inference(self, modelname, data):
        return self.model_dict[modelname].inference(data)
    

#tvm_runtime_process一旦创建，所有参数默认不会更改
class TVM_runtime_process:
    def __init__(self, model_dict: Dict[str, ModelInterface],mps_set:int,GID:int,threshold):
        
        self._tvm_runtime=None
        self._mps_set=mps_set
        self._queue=mp.Queue()
        self._GID=GID
        self._threshold=threshold
        #flag表示标志位，0表示未准备好，1表示running，2.表示超载,默认为1
        self._state=1
        self._model_list=[]
        for value in model_dict.values():
            self._model_list.append(value.get_model_name())
            self._model_list.append(value.get_batch_size())
        
        self._start_process(model_dict)
    
    def _start_process(self,model_dict):
        tvm_runtime=TVM_runtime(model_dict,self._mps_set,self._queue,self._GID)
        self._tvm_runtime=mp.Process(target=tvm_runtime.run)
        self._tvm_runtime.start()
    
    def set_threshold(self,threshold):
        self._threshold=threshold
    
    def get_threshold(self):
        return self._threshold
    
    def get_state(self):
        return self._state
    
    def set_state(self,state:int):
        if state not in [0,1,2]:
            raise ValueError("state must be 0,1,2!")
        self._state=state
        
    
    def inference(self,modelname,data,time_out):
        if self._state!=1:
            raise Exception("the process is not ready!")
        self._queue.put([modelname,data,time_out])
    
    def get_mps_set(self):
        return self._mps_set
    
    def get_model_list(self):
        return self._model_list
    
    def kill(self):
        self._queue.put("terminate")
        self._tvm_runtime.join()
        print("end")

        
        
if __name__=="__main__":
    from modelDeploy.modelFactory.factoryMethed import Resnet_Factory
    from modelDeploy.dispatch.model_info import Model_Info
    from modelDeploy.dispatch.dispatch import DataGenerator,DataTimeoutAdder
    from modelDeploy.dispatch.dispatch import Queue_Cluster
    modelInfo=Model_Info("modelDeploy/modelRepository")
    modelname="resnet-18"
    batchsize=1
    model_input=modelInfo.get_model_input(modelname)
    model_output=modelInfo.get_model_output(modelname)
    model_lib=modelInfo.get_model_lib(modelname,batchsize)
    model=Resnet_Factory().create_model(modelName=modelname,
                                        lib_file=model_lib,
                                        input_shape=model_input,
                                        output_shape=model_output,
                                        batch_size=batchsize)
    data_generator=DataGenerator(modelInfo)
    data_timeout_adder=DataTimeoutAdder(modelInfo)
    data=data_generator.generate_data(modelname)
    time_out=data_timeout_adder.data_add_timeout(modelname)
    model_queue=Queue_Cluster(modelInfo)
    for i in range(100):
        model_queue.put(modelname,data,time_out)
    
    data,timeout=model_queue.multi_get(modelname,100)
    
    model_dict={modelname:model}
    mps_set=100
    
    process=TVM_runtime_process(model_dict=model_dict,mps_set=mps_set,GID=0)
    
    
    process.inference(modelname, data, time_out)
    
    process.kill()
    
    

