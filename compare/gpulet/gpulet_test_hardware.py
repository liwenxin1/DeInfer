import os
from modelDeploy.modelFactory.factoryMethed import Model_Creator
from modelDeploy.dispatch.model_info import Model_Info
from modelDeploy.dispatch.dispatch import DataGenerator
import argparse
import time
import multiprocessing as mp
import numpy as np
import tvm
from tvm.contrib import graph_executor
def data_batch(data_generator:DataGenerator,modelname:str,batch_size:int):
    data = []
    for i in range(batch_size):
        data.append(data_generator.generate_data(modelname))
    return np.array(data)
if __name__=="__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--model_name', type=str)
    argparse.add_argument('--batch_size', type=int)
    argparse.add_argument('--mps_set',type=int)
    model_name = argparse.parse_args().model_name
    batch_size=argparse.parse_args().batch_size
    mps_set=argparse.parse_args().mps_set
    os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(mps_set)
    # 模型信息仓库
    model_info = Model_Info("modelDeploy/modelRepository")
    # 数据生成器
    data_generator = DataGenerator(model_info)
    # 模型构建器
    model_creator=Model_Creator()
    # 模型构建
    
    model_input = model_info.get_model_input(model_name)
    model_output = model_info.get_model_output(model_name)
    batch_size_support = model_info.get_model_batchsize_support(model_name)
    
    if batch_size not in batch_size_support:
        print("batch_size:{} not support".format(batch_size))
        exit()
    
    model_lib = model_info.get_model_lib(model_name, batch_size)
    lib=tvm.runtime.load_module(model_lib)
    dev=tvm.cuda(0)
    m=graph_executor.GraphModule(lib["default"](dev))
    data=data_batch(data_generator,model_name,batch_size)
    m.set_input("data",tvm.nd.array((data.astype("float32"))))
    
    #tvm的测量器最少会测量2次，测量次数=1+repeat*number，测试过，即使是两次测量，误差也很小，为了profile获取数据方便，这里测量两次
    # value=m.benchmark(dev, repeat=1, number=10).mean
    # print(value)
    #直接用time.time()计时，不准确，需要大量重复才能得到准确的时间,但是可以用于profile ncu
    m.run()
    time.sleep(1)
    