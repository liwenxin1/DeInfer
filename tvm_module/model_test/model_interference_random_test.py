import os
import random
from itertools import combinations, permutations

from draw import model_mps_interference,model_mps_interference_multi

if __name__=="__main__":
    random.seed(1)
    modelname=["resnet-18","resnet-34","resnet-50","resnet-101","resnet-152","vgg-11","vgg-13",\
        "vgg-16","vgg-19","squeezenet-1.0","squeezenet-1.1","shufflenet","mobilenet","LSTM-2","transformer"]
    
    result=permutations(modelname,2)
    model_list=[]
    for c in result:
        model_list.append(c)
    #print(len(model_list))
    test_set=random.sample(model_list,80)
    test_set=test_set[40:]
    for test_model in test_set:
        batch_com=[]
        factor_com=[]
        for modelname in test_model:
            if modelname =="transformer":
                batch_com.append(random.sample([8,16,32],1)[0])
            elif modelname in ["LSTM-2","LSTM-10"]:
                batch_com.append(100)
            else:
                batch_com.append(random.sample([1,2,4,8,16],1)[0])
                
            if modelname in ["resnet-18","resnet-34","resnet-50","resnet-101","resnet-152"]:
                factor_com.append(200)
            else:
                factor_com.append("untuned")
        
        print(test_model[0],batch_com[0],factor_com[0],test_model[1],batch_com[1],factor_com[1])
        
        commend1="bash model_interference_1.sh {} {} {} {} {} {}".format(test_model[0],batch_com[0],factor_com[0],test_model[1],batch_com[1],factor_com[1])
        
        commend2="bash model_interference_2.sh {} {} {} {} {} {}".format(test_model[0],batch_com[0],factor_com[0],test_model[1],batch_com[1],factor_com[1])
        # #测试两个模型的干扰
        os.system(commend1)
        
        # #测试多个模型的干扰
        os.system(commend2)
        
        filename1="./profiles/exectime_txt/interference_{}-{}-{}_{}-{}-{}.txt".format(test_model[0],batch_com[0],factor_com[0],test_model[1],batch_com[1],factor_com[1])
        
        filename2="./profiles/exectime_txt/multi-interference_{}-{}-{}_{}-{}-{}.txt".format(test_model[0],batch_com[0],factor_com[0],test_model[1],batch_com[1],factor_com[1])
        
        #绘制对应的图像
        model_mps_interference(filename1)
        
        model_mps_interference_multi(filename2)
        
    
    
    
    