import tvm
from tvm.contrib import graph_executor
import numpy as np
import argparse
import time


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--model_name",type=str,default="resnet-18")
    parser.add_argument("--batch_size",type=int,default=1)
    parser.add_argument("--factor",type=str,default="200")
    args=parser.parse_args()
    model_name=args.model_name
    batch_size=args.batch_size
    factor=args.factor
    if model_name in ["resnet-18","resnet-34","resnet-50","resnet-101","resnet-152","vgg-11","vgg-13",\
        "vgg-16","vgg-19","squeezenet-1.0","squeezenet-1.1","shufflenet","mobilenet"]:
        shape=(batch_size,3,224,224)
        
    elif model_name in ["LSTM-2","LSTM-10"]:
        shape=(batch_size,100,50)
    
    elif model_name in ["transformer"]:
        shape=(batch_size,50,512)
    
    elif model_name in ["other"]:
        pass
    
    file_name="../tune_file/{}-NCHW-B{}-{}-cuda.so".format(model_name,batch_size,factor)
    lib=tvm.runtime.load_module(file_name)
    dev=tvm.cuda(0)
    m=graph_executor.GraphModule(lib["default"](dev))
    m.set_input("data",tvm.nd.array((np.random.uniform(size=shape)).astype("float32")))
    
    #tvm的测量器最少会测量2次，测量次数=1+repeat*number，测试过，即使是两次测量，误差也很小，为了profile获取数据方便，这里测量两次
    # value=m.benchmark(dev, repeat=1, number=10).mean
    # print(value)
    #直接用time.time()计时，不准确，需要大量重复才能得到准确的时间,但是可以用于profile ncu
    
    m.run()
    