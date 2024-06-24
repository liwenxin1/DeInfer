from mpi4py import MPI
import numpy as np
import tvm
from tvm.contrib import graph_executor
import multiprocessing as mp
import os
import argparse
import time
#测试方法通过os.system()调用shell命令，让其他模型进程运行，然后再测模型的参数
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--model_name",type=str,default="resnet-18")
    parser.add_argument("--batch_size",type=int,default=1)
    parser.add_argument("--factor",type=str,default="50")
    parser.add_argument("--mps_set",type=int,default=50)
    
    parser.add_argument("--interference_model_name",type=str,default="resnet-18")
    parser.add_argument("--interference_batch_size",type=int,default=1)
    parser.add_argument("--interference_factor",type=str,default="untuned")
    args=parser.parse_args()
    model_name=args.model_name
    batch_size=args.batch_size
    factor=args.factor
    mps_set=args.mps_set
    
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
    
    #干扰进程运行
    #-------------------------------------------------------------------------------
    if mps_set!=0:
        
        
        model_name_interference=args.interference_model_name
        batch_size_interference=args.interference_batch_size
        factor_interference=args.interference_factor
        
        commend="CUDA_MPS_ACTIVE_THREAD_PERCENTAGE={} python run_model.py --model_name {} --batch_size {} --factor {}".\
            format(mps_set,model_name_interference,batch_size_interference,factor_interference)
        
        p=mp.Process(target=os.system,args=(commend,))
        p.start()
        print("干扰进程启动成功")
    else:
        pass
        #print("不设置干扰进程")
    #-------------------------------------------------------------------------------
    
    
    #测试代码
    
    value=m.benchmark(dev,repeat=6,min_repeat_ms=3000).mean#单位是秒,每一次测试18秒
    print("测试完成,数据处理平均用时   {}    ms".format(value*1000))
    
    if mps_set!=0:
        p.join()
    
    
    
    
