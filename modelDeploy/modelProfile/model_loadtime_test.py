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


if __name__ == "__main__":
    # 模型信息仓库
    model_info = Model_Info("modelDeploy/modelRepository")
    # 数据生成器
    data_generator = DataGenerator(model_info)
    # 模型构建器
    model_creator=Model_Creator()
    # 模型构建
    modelname_list=model_info.get_modelname_list()
    
    #写入file
    write_file="modelDeploy/modelRepository/model_loadtime.txt"
    if os.path.exists(write_file):
        os.remove(write_file)
    for modelname in modelname_list:
        model_batchsize_support=model_info.get_model_batchsize_support(modelname)
        for batch_size in model_batchsize_support:
            model=model_creator.create_model(modelname,
                                        model_info.get_model_lib(modelname,batch_size),
                                        model_info.get_model_input(modelname),
                                        model_info.get_model_output(modelname),
                                        batch_size)
            start_time = time.time()
            for i in range(30):
                model.load()
                model.activate(0)
            end_time = time.time()
            with open(write_file, 'a') as f:
                f.write("{} batch_size:{} load_time:{}\n".format(modelname,batch_size,(end_time-start_time)/100))
           
        
    
            
   
    
   
    
    


    
 
