import os
from modelDeploy.utils.utils import list_only_directories,load_json_file
import re
import pandas as pd

#auther:liwenxin
#模型信息
class Model_Info:
    def __init__(self, modelrepository):
        self.model_info_dict = {}
        self.modelrepository=modelrepository
        self.model_search_space = pd.read_csv("{}/model_L2_gflops.csv".format(modelrepository))
        self._load_info(modelrepository)
    def _load_info(self, modelrepository):
        model_dir=list_only_directories(modelrepository)
        for filename in model_dir:
            json_path=os.path.join("{}/{}/config.json".format(modelrepository,filename))
            model_inf=load_json_file(json_path)
            self.model_info_dict[model_inf["name"]]=model_inf
            
    def get_model_lib(self, model_name:str,batch_size:int)->str:
        lib_file_list = self.model_info_dict[model_name]["lib_file"]
        # Match strings containing batch size"
        pattern = re.compile("B{}".format(batch_size))
        lib_file = [string for string in lib_file_list if re.search(pattern, string)]
        lib_file="{}/{}/{}".format(self.modelrepository,model_name,lib_file[0])
        return lib_file
    
    #根据模型名称和batch大小获取模型的l2 cache利用率和GFLOPS
    def repositorySearch(self,modelname,batch):
        model_info = self.model_search_space[(self.model_search_space['modelname'] == modelname) & (self.model_search_space['batch'] == batch)]
        model_info_list = model_info.values.tolist()[0]
        return model_info_list[2],model_info_list[3]
    
    def get_modelname_list(self):
        return list(self.model_info_dict.keys())
    def get_model_input(self, model_name):
        return tuple(self.model_info_dict[model_name]["input_shape"])

    def get_model_output(self, model_name):
        return self.model_info_dict[model_name]["output_shape"]

    def get_model_SLO(self, model_name):
        return self.model_info_dict[model_name]["slo"]

    def get_model_batchsize_support(self, model_name):
        return self.model_info_dict[model_name]["batchsize"]
    
    
#模型运行时信息，供调度器使用
class Model_Runtime_Info:
    def __init__(self,modelRepository) -> None:
        self.model_mps_runtime_dict={}
        self.model_loadtime_dict={}
        self._init_MRI(modelRepository)
        
    def _init_MRI(self,modelRepository):
        """
        init model runtime info
        """
        #处理model load time
        model_loadtime_file=os.path.join(modelRepository,"model_loadtime.txt")
        with open(model_loadtime_file,"r") as f:
            for line in f.readlines():
                model_name=line.split(" ")[0]
                batch_size=line.split(" ")[1].split(":")[1]
                load_time=float(line.split(" ")[2].split(":")[1])
                #字典格式：例子：{"resnet-18_1":0.0001}
                self.model_loadtime_dict[model_name+"_"+batch_size]=load_time
                
        #处理mps runtime
        for model_name in list_only_directories(modelRepository):
            model_runtime_file=os.path.join(modelRepository,model_name,"mps_runtime.txt")
            
            with open(model_runtime_file,"r") as f:
                for line in f.readlines():
                    model_name=line.split(" ")[0]
                    batch_size=line.split(" ")[1].split(":")[1]
                    mps_set=float(line.split(" ")[2].split(":")[1])
                    runtime=float(line.split(" ")[3].split(":")[1])
                    
                    #字典格式：例子：{"resnet-18_1":[0.0001,0.0002,0.0003,0.0004,0.0005,0.0006,0.0007,0.0008,0.0009,0.001]}
                    if model_name+"_"+batch_size not in self.model_mps_runtime_dict:
                        self.model_mps_runtime_dict[model_name+"_"+batch_size]=[]
                        self.model_mps_runtime_dict[model_name+"_"+batch_size].append((runtime))
                    else:
                        self.model_mps_runtime_dict[model_name+"_"+batch_size].append((runtime))
                        
    def get_model_loadtime(self,model_name,batch_size)->float:
        return self.model_loadtime_dict[model_name+"_"+batch_size]
    
    def get_model_mps_runtime(self,model_name:str,batch_size:int,mps:int)->float:
        batch_size=str(batch_size)
        #mps should be less than 100
        assert mps<=100,"mps should be less than 100!"
        assert mps>=10,"mps should be greater than 10!"
        #临界值处理
        if mps==100:
            return self.model_mps_runtime_dict[model_name+"_"+batch_size][-1]
        #对mps进行插值
        param=mps/10-1
        index_under=int(param)
        index_upper=index_under+1
        weight=index_upper-param
        runtime=self.model_mps_runtime_dict[model_name+"_"+batch_size][index_under]*weight+\
        self.model_mps_runtime_dict[model_name+"_"+batch_size][index_upper]*(1-weight)
        
        return runtime

if __name__=="__main__":
    MI=Model_Info("modelDeploy/modelRepository")
    print(MI.repositorySearch("resnet-18",1))
    # model_info=Model_Runtime_Info("modelDeploy/modelRepository")
    # print(model_info.get_model_mps_runtime("resnet-18","1",10))
            
        
    
                        
        
    
        
