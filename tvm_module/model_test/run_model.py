import tvm
from tvm.contrib import graph_executor
import numpy as np
import argparse

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--model_name",type=str,default="resnet-18")
    parser.add_argument("--batch_size",type=int,default=1)
    parser.add_argument("--factor",type=str,default="50")
    
    args=parser.parse_args()
    model_name=args.model_name
    batch_size=args.batch_size
    
    factor=args.factor
    if model_name in ["resnet-18","resnet-34","resnet-50","resnet-101","resnet-152","vgg-11","vgg-13","vgg-16","vgg-19","squeezenet-1.0","squeezenet-1.1","shufflenet","mobilenet"]:
        shape=(batch_size,3,224,224)
    elif model_name in ["LSTM-2","LSTM-10"]:
        shape=(batch_size,100,50)
    elif model_name in ["transformer"]:
        shape=(batch_size,50,512)
    
    file_name="../tune_file/{}-NCHW-B{}-{}-cuda.so".format(model_name,batch_size,factor)
    lib=tvm.runtime.load_module(file_name)
    dev=tvm.cuda(0)
    m=graph_executor.GraphModule(lib["default"](dev))


    m.set_input("data",tvm.nd.array((np.random.uniform(size=shape)).astype("float32")))

    # for i in range(1000):
    #     m.run()
    #干扰线程要保证一直提供干扰，所以运行时间不能比主线程短
    value=m.benchmark(dev,repeat=5,min_repeat_ms=4000).mean
    
    #print("干扰进程运行完成")
    #print("干扰进程运行完成，数据处理平均用时{}ms".format(value*1000))


    
    