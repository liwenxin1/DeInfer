#  @Author: Wenxin Li 
#  @Date: 2024-01-09 10:18:31 
#  @Last Modified by:   Wenxin Li 
#  @Last Modified time: 2024-01-09 10:18:31 
#  测试模型在不同mps下的运行时间
import os
from modelDeploy.modelFactory.factoryMethed import Model_Creator
from modelDeploy.dispatch.model_info import Model_Info
from modelDeploy.dispatch.dispatch import DataGenerator
import argparse
import time
import multiprocessing as mp
import numpy as np

#run_2每次测试循环100次
def run_2(model, mps_set, data, write_file):
    os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(mps_set)
    model.load()
    model.activate(0)
    
    start_time = time.time()
    for i in range(100):
        y = model.inference(data)
    end_time = time.time()
        
    # Write data to a text file
    with open(write_file, 'a') as f:
        f.write("{} batch_size:{} mps_set:{} percentage_runtime_cost:{}\n".format(
            model.get_model_name(), model.get_batch_size(), mps_set, (end_time - start_time) / 100
        ))


def run(model, mps_set, data, write_file):
    os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(mps_set)
    model.load()
    model.activate(0)
    
    run_time=model.run_test(data)
        
    # Write data to a text file
    with open(write_file, 'a') as f:
        f.write("{} batch_size:{} mps_set:{} percentage_runtime_cost:{}\n".format(
            model.get_model_name(), model.get_batch_size(), mps_set, run_time
        ))

def data_batch(data_generator:DataGenerator,modelname:str,batch_size:int):
    data = []
    for i in range(batch_size):
        data.append(data_generator.generate_data(modelname))
    return np.array(data)
    
if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--model_name', type=str)
    
    # 模型信息仓库
    model_info = Model_Info("modelDeploy/modelRepository")
    # 数据生成器
    data_generator = DataGenerator(model_info)
    # 模型构建器
    model_creator=Model_Creator()
    # 模型构建
    model_name = argparse.parse_args().model_name
    model_input = model_info.get_model_input(model_name)
    model_output = model_info.get_model_output(model_name)
    batch_size_support = model_info.get_model_batchsize_support(model_name)
    
    # 写入文件
    write_file="modelDeploy/modelRepository/{}/mps_runtime.txt".format(model_name)
    if os.path.exists(write_file):
        os.remove(write_file)
    
    mps_set_list=[10,20,30,40,50,60,70,80,90,100]
    # 对不同的batch_size进行测试
    for batch_size in batch_size_support:
        model_lib = model_info.get_model_lib(model_name, batch_size)
        model = model_creator.create_model(model_name, model_lib, model_input, model_output, batch_size)
        data=data_batch(data_generator,model_name,batch_size)
        for mps_set in mps_set_list:
            process=mp.Process(target=run, args=(model, mps_set, data, write_file))
            process.start()
            process.join()
    


    
 
