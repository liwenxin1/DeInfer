import socket
import numpy as np
import tvm
import sys
from modelDeploy.modelFactory.modelInterface import ModelFactory
import argparse
import pickle
import time
if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument("--model_name", type=str)
    argparse.add_argument("--lib_file", type=str)
    argparse.add_argument("--batch_size", type=int)
    argparse.add_argument("--GID", type=int)
    argparse.add_argument("--type", type=str, default="float32")
    args = argparse.parse_args()
    model_name=args.model_name
    lib_file=args.lib_file
    batch_size=args.batch_size
    GID=args.GID
    
    #初始化模型
    modelFactory = ModelFactory()
    model = modelFactory.getModel(model_name, lib_file)
    if model ==None:
        print("no model named {}".format(model_name))
        exit(-1)
    model.activeModel(GID)
    while True:
        data=pickle.loads(sys.stdin.buffer.read())
        # 计算需要填充的数量
        num_samples = data.shape[0]
        pad_size = batch_size - num_samples % batch_size if num_samples % batch_size != 0 else 0

        # 使用pad函数来填充数据
        data_padded = np.pad(data, ((0, pad_size), (0, 0), (0, 0), (0, 0)))

        # 使用reshape函数将数据重新整形为每4个为一份的形状
        data_reshaped = data_padded.reshape(-1, batch_size, *data.shape[1:])
        
        for data in data_reshaped:
            y=model.inference(data)
        
            print(y.shape)
            
        
        
    
    