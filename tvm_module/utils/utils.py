import torch
from torchvision import models
from thop import profile
from tvm_module.model_implementation.lstm import LSTMModel
from tvm_module.model_implementation.transformer import TransformerEncoder
# 计算并打印模型的总参数大小(MB)
def model_params_count(model):
    total_params_bytes = sum(p.numel() for p in model.parameters())
    total_params_mb = total_params_bytes / (1024 ** 2)

    #print(f"Total parameters: {total_params_mb:.2f} MB")
    return total_params_mb

# 计算并打印模型的总计算量(FLOPs)以及总参数大小(MB)
def model_gflops_count(model,input_size):
    input = torch.randn(input_size)
    flops, params = profile(model, inputs=(input, ))
    gflops,params = flops/(2**30),params/(1024**2)
    #print("models {} gflops, {} MB params".format(gflops,params))
    return gflops,params

#获取模型参数量的函数
def model_para_get(modelname,batch_size):
    function_name=["resnet18","resnet34","resnet50","resnet101","resnet152","vgg11","vgg13","vgg16",\
        "vgg19","squeezenet1_0","squeezenet1_1","shufflenet_v2_x0_5","mobilenet_v2"]
    
    model_name_list=["resnet-18","resnet-34","resnet-50","resnet-101","resnet-152","vgg-11","vgg-13",\
        "vgg-16","vgg-19","squeezenet","squeezenet-1.1","shufflenet","mobilenet"]
    
    if modelname in model_name_list:
        index=model_name_list.index(modelname)
        function=function_name[index]
        
        model = eval("models.{}(weights=None)".format(function))
    
    elif modelname == "LSTM":
        model = LSTMModel(50,100,2,1)
    elif modelname == "transformer":
        model = TransformerEncoder(6,512,8,2048)
    else:
        print(modelname)
    
    if modelname in ["resnet-18","resnet-34","resnet-50","resnet-101","resnet-152","vgg-11","vgg-13",\
        "vgg-16","vgg-19","squeezenet","squeezenet-1.1","shufflenet","mobilenet"]:
        shape=(batch_size,3,224,224)
    elif modelname in ["LSTM-2","LSTM-10"]:
        shape=(batch_size,100,50)
    elif modelname in ["transformer"]:
        shape=(batch_size,50,512)
    
    return model_gflops_count(model,shape)
if __name__=="__main__":
    import torchvision.models as models
    model=models.resnet18(weights=False)
    model_params_count(model)
    print(model_gflops_count(model,(1,3,224,224)))
    