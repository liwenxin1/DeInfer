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
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp

def data_batch(data_generator:DataGenerator,modelname:str,batch_size:int):
    data = []
    for i in range(batch_size):
        data.append(data_generator.generate_data(modelname))
    return np.array(data)

def run(model, mps_set, data, write_file):
    os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(mps_set)
    model.load()
    model.activate(0)
    
    latency=model.run_test(data)
        
    # Write data to a text file
    with open(write_file, 'a') as f:
        f.write("{} batch_size:{} mps_set:{} percentage_runtime_cost:{}\n".format(
            model.get_model_name(), model.get_batch_size(), mps_set, latency
        ))
def main():
    # 模型信息仓库
    model_info = Model_Info("modelDeploy/modelRepository")
    # 数据生成器
    data_generator = DataGenerator(model_info)
    # 模型构建器
    model_creator=Model_Creator()
    # 模型构建
    modelname_list=model_info.get_modelname_list()
    
    #mps_set
    mps_set_list=[25,50,75,100]
    #写入file
    write_file="modelDeploy/modelProfile/profiles/model_batchsize_latency.txt"
    if os.path.exists(write_file):
        os.remove(write_file)
    for modelname in modelname_list:
        for mps_set in mps_set_list:
            model_batchsize_support=model_info.get_model_batchsize_support(modelname)
            for batch_size in model_batchsize_support:
                model=model_creator.create_model(modelname,
                                            model_info.get_model_lib(modelname,batch_size),
                                            model_info.get_model_input(modelname),
                                            model_info.get_model_output(modelname),
                                            batch_size)
                data=data_batch(data_generator,modelname,batch_size)
                process=mp.Process(target=run,args=(model,mps_set,data,write_file))
                process.start()
                process.join()
    
            

def draw(draw_modelname:str,flag:str="latency"):
    dictory={}
    with open("modelDeploy/modelProfile/profiles/model_batchsize_latency.txt", 'r') as f:
        mps_set_list=[25,50,75,100]
        for line in f.readlines():
            line=line.strip('\n')
            modelname=line.split(" ")[0]
            batch_size=float(line.split(" ")[1].split(":")[1])
            mps_set=int(line.split(" ")[2].split(":")[1])
            latency=float(line.split(" ")[3].split(":")[1])
            dict_name=modelname+"_"+str(mps_set)
            if dict_name not in dictory.keys():
                dictory[dict_name]=[[],[]]
            if flag=="latency":
                dictory[dict_name][0].append(batch_size)
                dictory[dict_name][1].append(latency)
            elif flag=="throughput":
                dictory[dict_name][0].append(batch_size)
                dictory[dict_name][1].append(batch_size/latency)
    
    
    for i in range(len(mps_set_list)):
        dict_name=draw_modelname+"_"+str(mps_set_list[i])
        plt.plot(dictory[dict_name][0],dictory[dict_name][1],label=dict_name)
        plt.legend(loc='upper right')
        plt.xlabel("batch_size")
        if flag=="latency":
            plt.ylabel("latency(ms)")
        elif flag=="throughput":
            plt.ylabel("throughput(qps)")
        
        plt.savefig("modelDeploy/modelProfile/pic/model_batchsize_{}_{}.png".format(flag,draw_modelname))
        
                
            

if __name__ == "__main__":
    #main()
    draw(draw_modelname="transformer",flag="throughput")
           
        
    
            
   
    
   
    
    


    
 
