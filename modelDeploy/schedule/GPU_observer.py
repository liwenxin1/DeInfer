from typing import Dict
from modelDeploy.modelFactory.modelInterface import ModelInterface
from modelDeploy.schedule.tvm_runtime_process import TVM_runtime_process

#GPU_observer负责管理所有的GPU进程,GPU_observer不对索引进行检查，由调用者保证索引的正确性
class GPU_observer:
    def __init__(self,GPU_number:int):
        """TRP: TVM runtime process
        Args:
            GPU_number (int): the number of GPU supported
        """
        #GPU数量必须大于0，否则抛出异常
        if GPU_number<=0:
            raise Exception("GPU number should be greater than 0")

        self.GPU_resource_list = [] #[100,100,..]
        self.TRP_controller = [] #[[TRP1,TRP2,..],[TRP1,TRP2,..],..]
        
        self._init_gpu_observer(GPU_number)
    
    #GPU——observer初始化
    def _init_gpu_observer(self,GPU_number):
        for _ in range(GPU_number):
            self.GPU_resource_list.append(100)
            self.TRP_controller.append([])
    
    #获取某一个GPU上的剩余资源
    def get_gpu_resoure(self, GID:int)->int:
        return self.GPU_resource_list[GID]

    #获取某一个GPU上，所有的GPU进程信息
    def get_process_info_by_GID(self, GID:int)->list:
        info=[]
        if self.TRP_controller[GID]==[]:
            print("There is no TRP on GPU %d"%GID)
            return []
        else:
            for TRP in self.TRP_controller[GID]:
                #类似[resnet-18,1,80,100]
                info.append(TRP.get_model_list()+[TRP.get_mps_set(),TRP.get_threshold()])
        return info
    
    #获取某一个GPU上的某一个进程
    def get_TRP(self,GID:int,TRP_ID:int):
        return self.TRP_controller[GID][TRP_ID]
    
    #获取某个进程的阈值
    def get_TRP_threshold(self,GID:int,TRP_ID:int):
        return self.TRP_controller[GID][TRP_ID].get_threshold()
    
    #设置某个进程的阈值
    def set_TRP_threshold(self,GID:int,TRP_ID:int,threshold):
        self.TRP_controller[GID][TRP_ID].set_threshold(threshold)
    
    #获取某个进程的状态
    def get_TRP_state(self,GID:int,TRP_ID:int):
        return self.TRP_controller[GID][TRP_ID].get_state()
    
    #设置某个进程的状态
    def set_TRP_state(self,GID:int,TRP_ID:int,state:int):
        self.TRP_controller[GID][TRP_ID].set_state(state)

    #创建一个新的GPU进程
    def create_TRP(self,model_dict: Dict[str, ModelInterface],mps_set:int,GID:int,threshold):
        #资源充足才能进行分配
        if self.GPU_resource_list[GID]>=mps_set:
            print("create a new TRP successfully")
            self.GPU_resource_list[GID]-=mps_set
            self.TRP_controller[GID].append(TVM_runtime_process(model_dict,mps_set,GID,threshold))
        else:
            raise Exception("create a new TRP failed! the resource is not enough!")
        
        
    def inference(self,modelname, data,time_out,GID,TRP_ID):
        TRP=self.TRP_controller[GID][TRP_ID]
        TRP.inference(modelname, data,time_out)
    
    def kill_TRP(self,GID:int,TRP_ID:int)->None:
        #释放资源
        TRP=self.TRP_controller[GID][TRP_ID]
        mps_set=TRP.get_mps_set()
        TRP.kill()
        #清除列表
        self.TRP_controller[GID].remove(TRP)
        self.GPU_resource_list[GID]+=mps_set
    
    #清除某一个GPU上的所有进程
    def kill_all_TRP(self,GID:int)->None:
        for TRP in self.TRP_controller[GID]:
            TRP.kill()
        self.TRP_controller[GID]=[]
        self.GPU_resource_list[GID]=100
        
        
if __name__=="__main__":
    gpu_observer=GPU_observer(1)
    from modelDeploy.modelFactory.factoryMethed import Resnet_Factory
    from modelDeploy.dispatch.model_info import Model_Info
    from modelDeploy.dispatch.dispatch import DataGenerator,DataTimeoutAdder
    modelInfo=Model_Info("modelDeploy/modelRepository")
    modelname="resnet-18"
    batchsize=1
    model_input=modelInfo.get_model_input(modelname)
    model_output=modelInfo.get_model_output(modelname)
    model_lib=modelInfo.get_model_lib(modelname,batchsize)
    model=Resnet_Factory().create_model(modelName=modelname,
                                        lib_file=model_lib,
                                        input_shape=model_input,
                                        output_shape=model_output,
                                        batch_size=batchsize)
    data_generator=DataGenerator(modelInfo)
    data_timeout_adder=DataTimeoutAdder(modelInfo)
    data=data_generator.generate_data(modelname)
    time_out=data_timeout_adder.data_add_timeout(modelname)
    model_dict={modelname:model}
    mps_set=100
    gpu_observer.create_tvm_runtime_process(model_dict,mps_set,0)
    gpu_observer.inference(modelname,data,time_out,0,0)
    import time
    time.sleep(5)
    gpu_observer.kill_TRP(0,0)
    time.sleep(10)
    
        
