# -*- coding: utf-8 -*-
# @Time : 2023/11/11 15:32
# @Author : liwenxin
# @File : modelFactory
# @Project : modelDeployment
from abc import ABC,abstractmethod
import tvm
from tvm.contrib import graph_executor
import numpy as np

"""
to register model, you should implement the AbstractModel class,
and append code in ModelFactory class
"""
class ModelInterface(ABC):
    def __init__(self):
        self.model_name=None
        self.lib_file=None
        self.model=None
        self.input_shape=None
        self.output_shape=None
        self.batch_size=None
    
    def get_model_name(self):
        return self.model_name
    
    def get_input_shape(self):
        return self.input_shape
    
    def get_output_shape(self):
        return self.output_shape
    
    def get_batch_size(self):
        return self.batch_size
    
    @abstractmethod
    def load(self):
        """
        Load the model from the specified library file.
        """
        raise NotImplementedError("load method is not implemented")
    @abstractmethod
    def activate(self, GID: int):
        """
        Activate the model on the specified GPU device.
        """
        raise NotImplementedError("activate method is not implemented")
    @abstractmethod
    def inference(self, data):
        """
        Perform inference using the loaded and activated model.
        """
        raise NotImplementedError("inference method is not implemented")



class ResNet18Model(ModelInterface):
    def __init__(self,
                 modelName:str,
                 lib_file:str,
                 input_shape:str,
                 output_shape:str,
                 batch_size:int):
        self.model_name=modelName
        self.lib_file=lib_file
        self.model=None
        self.input_shape=input_shape
        self.output_shape=output_shape
        self.batch_size=batch_size
        
    def inference(self,data):
        """
        :param data: the model input
        :return: the model output
        """
        
        self.model.set_input("data",tvm.nd.array(data.astype("float32")))
        self.model.run()
        output=self.model.get_output(0)
        return output.asnumpy()

    def load(self):
        self.lib=tvm.runtime.load_module(self.lib_file)
        
    
    def activate(self,GID:int):
        """
        GPU资源分配在modelpool进行，这里只需要将模型加载到GPU上
        :param GID: GPU的ID号
        :return: None
        """
        self.GID=GID
        dev=tvm.cuda(self.GID)
        self.model=graph_executor.GraphModule(self.lib["default"](dev))
    
    def run_test(self,data):
        dev=tvm.cuda(self.GID)
        self.model.set_input("data",tvm.nd.array(data.astype("float32")))
        exec_time=self.model.benchmark(dev,repeat=3,min_repeat_ms=5000).mean        
        return exec_time

class ResNet34Model(ModelInterface):
    def __init__(self,
                 modelName:str,
                 lib_file:str,
                 input_shape:str,
                 output_shape:str,
                 batch_size:int):
        self.model_name=modelName
        self.lib_file=lib_file
        self.model=None
        self.input_shape=input_shape
        self.output_shape=output_shape
        self.batch_size=batch_size
        
    def inference(self,data):
        """
        :param data: the model input
        :return: the model output
        """
        
       
        self.model.set_input("data",tvm.nd.array(data.astype("float32")))
        self.model.run()
        output=self.model.get_output(0)
        return output.asnumpy()

    def load(self):
        self.lib=tvm.runtime.load_module(self.lib_file)
        

    def activate(self,GID:int):
        """
        GPU资源分配在modelpool进行，这里只需要将模型加载到GPU上
        :param GID: GPU的ID号
        :return: None
        """
        self.GID=GID
        dev=tvm.cuda(self.GID)
        self.model=graph_executor.GraphModule(self.lib["default"](dev))
        
    def run_test(self,data):
        dev=tvm.cuda(self.GID)
        self.model.set_input("data",tvm.nd.array(data.astype("float32")))
        exec_time=self.model.benchmark(dev,repeat=3,min_repeat_ms=5000).mean        
        return exec_time
    

class ResNet50Model(ModelInterface):
    def __init__(self,
                 modelName:str,
                 lib_file:str,
                 input_shape:str,
                 output_shape:str,
                 batch_size:int):
        self.model_name=modelName
        self.lib_file=lib_file
        self.model=None
        self.input_shape=input_shape
        self.output_shape=output_shape
        self.batch_size=batch_size
        
    def inference(self,data):
        """
        :param data: the model input
        :return: the model output
        """
        
       
        self.model.set_input("data",tvm.nd.array(data.astype("float32")))
        self.model.run()
        output=self.model.get_output(0)
        return output.asnumpy()

    def load(self):
        self.lib=tvm.runtime.load_module(self.lib_file)
        

    def activate(self,GID:int):
        """
        GPU资源分配在modelpool进行，这里只需要将模型加载到GPU上
        :param GID: GPU的ID号
        :return: None
        """
        self.GID=GID
        dev=tvm.cuda(self.GID)
        self.model=graph_executor.GraphModule(self.lib["default"](dev))
        
    def run_test(self,data):
        dev=tvm.cuda(self.GID)
        self.model.set_input("data",tvm.nd.array(data.astype("float32")))
        exec_time=self.model.benchmark(dev,repeat=3,min_repeat_ms=5000).mean        
        return exec_time

class ResNet101Model(ModelInterface):
    def __init__(self,
                 modelName:str,
                 lib_file:str,
                 input_shape:str,
                 output_shape:str,
                 batch_size:int):
        self.model_name=modelName
        self.lib_file=lib_file
        self.model=None
        self.input_shape=input_shape
        self.output_shape=output_shape
        self.batch_size=batch_size
        
    def inference(self,data):
        """
        :param data: the model input
        :return: the model output
        """
        
       
        self.model.set_input("data",tvm.nd.array(data.astype("float32")))
        self.model.run()
        output=self.model.get_output(0)
        return output.asnumpy()

    def load(self):
        self.lib=tvm.runtime.load_module(self.lib_file)
        

    def activate(self,GID:int):
        """
        GPU资源分配在modelpool进行，这里只需要将模型加载到GPU上
        :param GID: GPU的ID号
        :return: None
        """
        self.GID=GID
        dev=tvm.cuda(self.GID)
        self.model=graph_executor.GraphModule(self.lib["default"](dev))
        
    def run_test(self,data):
        dev=tvm.cuda(self.GID)
        self.model.set_input("data",tvm.nd.array(data.astype("float32")))
        exec_time=self.model.benchmark(dev,repeat=3,min_repeat_ms=5000).mean        
        return exec_time


class ResNet152Model(ModelInterface):
    def __init__(self,
                 modelName:str,
                 lib_file:str,
                 input_shape:str,
                 output_shape:str,
                 batch_size:int):
        self.model_name=modelName
        self.lib_file=lib_file
        self.model=None
        self.input_shape=input_shape
        self.output_shape=output_shape
        self.batch_size=batch_size
        
    def inference(self,data):
        """
        :param data: the model input
        :return: the model output
        """
        
       
        self.model.set_input("data",tvm.nd.array(data.astype("float32")))
        self.model.run()
        output=self.model.get_output(0)
        return output.asnumpy()

    def load(self):
        self.lib=tvm.runtime.load_module(self.lib_file)
        

    def activate(self,GID:int):
        """
        GPU资源分配在modelpool进行，这里只需要将模型加载到GPU上
        :param GID: GPU的ID号
        :return: None
        """
        self.GID=GID
        dev=tvm.cuda(self.GID)
        self.model=graph_executor.GraphModule(self.lib["default"](dev))
        
    def run_test(self,data):
        dev=tvm.cuda(self.GID)
        self.model.set_input("data",tvm.nd.array(data.astype("float32")))
        exec_time=self.model.benchmark(dev,repeat=3,min_repeat_ms=5000).mean        
        return exec_time

class TransformerModel(ModelInterface):
    def __init__(self,
                 modelName:str,
                 lib_file:str,
                 input_shape:str,
                 output_shape:str,
                 batch_size:int):
        self.model_name=modelName
        self.lib_file=lib_file
        self.model=None
        self.input_shape=input_shape
        self.output_shape=output_shape
        self.batch_size=batch_size
        
    def inference(self,data):
        """
        :param data: the model input
        :return: the model output
        """
        
       
        self.model.set_input("data",tvm.nd.array(data.astype("float32")))
        self.model.run()
        output=self.model.get_output(0)
        return output.asnumpy()

    def load(self):
        self.lib=tvm.runtime.load_module(self.lib_file)
        

    def activate(self,GID:int):
        """
        GPU资源分配在modelpool进行，这里只需要将模型加载到GPU上
        :param GID: GPU的ID号
        :return: None
        """
        self.GID=GID
        dev=tvm.cuda(self.GID)
        self.model=graph_executor.GraphModule(self.lib["default"](dev))
        
    def run_test(self,data):
        dev=tvm.cuda(self.GID)
        self.model.set_input("data",tvm.nd.array(data.astype("float32")))
        exec_time=self.model.benchmark(dev,repeat=3,min_repeat_ms=5000).mean        
        return exec_time
        

class VGG11Model(ModelInterface):
    def __init__(self,
                 modelName:str,
                 lib_file:str,
                 input_shape:str,
                 output_shape:str,
                 batch_size:int):
        self.model_name=modelName
        self.lib_file=lib_file
        self.model=None
        self.input_shape=input_shape
        self.output_shape=output_shape
        self.batch_size=batch_size
        
    def inference(self,data):
        """
        :param data: the model input
        :return: the model output
        """
        
       
        self.model.set_input("data",tvm.nd.array(data.astype("float32")))
        self.model.run()
        output=self.model.get_output(0)
        return output.asnumpy()

    def load(self):
        self.lib=tvm.runtime.load_module(self.lib_file)
        

    def activate(self,GID:int):
        """
        GPU资源分配在modelpool进行，这里只需要将模型加载到GPU上
        :param GID: GPU的ID号
        :return: None
        """
        self.GID=GID
        dev=tvm.cuda(self.GID)
        self.model=graph_executor.GraphModule(self.lib["default"](dev))
        
    def run_test(self,data):
        dev=tvm.cuda(self.GID)
        self.model.set_input("data",tvm.nd.array(data.astype("float32")))
        exec_time=self.model.benchmark(dev,repeat=3,min_repeat_ms=5000).mean        
        return exec_time

class VGG13Model(ModelInterface):
    def __init__(self,
                 modelName:str,
                 lib_file:str,
                 input_shape:str,
                 output_shape:str,
                 batch_size:int):
        self.model_name=modelName
        self.lib_file=lib_file
        self.model=None
        self.input_shape=input_shape
        self.output_shape=output_shape
        self.batch_size=batch_size
        
    def inference(self,data):
        """
        :param data: the model input
        :return: the model output
        """
        
       
        self.model.set_input("data",tvm.nd.array(data.astype("float32")))
        self.model.run()
        output=self.model.get_output(0)
        return output.asnumpy()

    def load(self):
        self.lib=tvm.runtime.load_module(self.lib_file)
        

    def activate(self,GID:int):
        """
        GPU资源分配在modelpool进行，这里只需要将模型加载到GPU上
        :param GID: GPU的ID号
        :return: None
        """
        self.GID=GID
        dev=tvm.cuda(self.GID)
        self.model=graph_executor.GraphModule(self.lib["default"](dev))
        
    def run_test(self,data):
        dev=tvm.cuda(self.GID)
        self.model.set_input("data",tvm.nd.array(data.astype("float32")))
        exec_time=self.model.benchmark(dev,repeat=3,min_repeat_ms=5000).mean        
        return exec_time

class VGG16Model(ModelInterface):
    def __init__(self,
                 modelName:str,
                 lib_file:str,
                 input_shape:str,
                 output_shape:str,
                 batch_size:int):
        self.model_name=modelName
        self.lib_file=lib_file
        self.model=None
        self.input_shape=input_shape
        self.output_shape=output_shape
        self.batch_size=batch_size
        
    def inference(self,data):
        """
        :param data: the model input
        :return: the model output
        """
        
       
        self.model.set_input("data",tvm.nd.array(data.astype("float32")))
        self.model.run()
        output=self.model.get_output(0)
        return output.asnumpy()

    def load(self):
        self.lib=tvm.runtime.load_module(self.lib_file)
        

    def activate(self,GID:int):
        """
        GPU资源分配在modelpool进行，这里只需要将模型加载到GPU上
        :param GID: GPU的ID号
        :return: None
        """
        self.GID=GID
        dev=tvm.cuda(self.GID)
        self.model=graph_executor.GraphModule(self.lib["default"](dev))
        
    def run_test(self,data):
        dev=tvm.cuda(self.GID)
        self.model.set_input("data",tvm.nd.array(data.astype("float32")))
        exec_time=self.model.benchmark(dev,repeat=3,min_repeat_ms=5000).mean        
        return exec_time

class VGG19Model(ModelInterface):
    def __init__(self,
                 modelName:str,
                 lib_file:str,
                 input_shape:str,
                 output_shape:str,
                 batch_size:int):
        self.model_name=modelName
        self.lib_file=lib_file
        self.model=None
        self.input_shape=input_shape
        self.output_shape=output_shape
        self.batch_size=batch_size
        
    def inference(self,data):
        """
        :param data: the model input
        :return: the model output
        """
        
       
        self.model.set_input("data",tvm.nd.array(data.astype("float32")))
        self.model.run()
        output=self.model.get_output(0)
        return output.asnumpy()

    def load(self):
        self.lib=tvm.runtime.load_module(self.lib_file)
        

    def activate(self,GID:int):
        """
        GPU资源分配在modelpool进行，这里只需要将模型加载到GPU上
        :param GID: GPU的ID号
        :return: None
        """
        self.GID=GID
        dev=tvm.cuda(self.GID)
        self.model=graph_executor.GraphModule(self.lib["default"](dev))
        
    def run_test(self,data):
        dev=tvm.cuda(self.GID)
        self.model.set_input("data",tvm.nd.array(data.astype("float32")))
        exec_time=self.model.benchmark(dev,repeat=3,min_repeat_ms=5000).mean        
        return exec_time

class ShuffleNetModel(ModelInterface):
    def __init__(self,
                 modelName:str,
                 lib_file:str,
                 input_shape:str,
                 output_shape:str,
                 batch_size:int):
        self.model_name=modelName
        self.lib_file=lib_file
        self.model=None
        self.input_shape=input_shape
        self.output_shape=output_shape
        self.batch_size=batch_size
        
    def inference(self,data):
        """
        :param data: the model input
        :return: the model output
        """
        
       
        self.model.set_input("data",tvm.nd.array(data.astype("float32")))
        self.model.run()
        output=self.model.get_output(0)
        return output.asnumpy()

    def load(self):
        self.lib=tvm.runtime.load_module(self.lib_file)
        

    def activate(self,GID:int):
        """
        GPU资源分配在modelpool进行，这里只需要将模型加载到GPU上
        :param GID: GPU的ID号
        :return: None
        """
        self.GID=GID
        dev=tvm.cuda(self.GID)
        self.model=graph_executor.GraphModule(self.lib["default"](dev))
        
    def run_test(self,data):
        dev=tvm.cuda(self.GID)
        self.model.set_input("data",tvm.nd.array(data.astype("float32")))
        exec_time=self.model.benchmark(dev,repeat=3,min_repeat_ms=5000).mean        
        return exec_time
    
class SqueezeNetModel(ModelInterface):
    def __init__(self,
                 modelName:str,
                 lib_file:str,
                 input_shape:str,
                 output_shape:str,
                 batch_size:int):
        self.model_name=modelName
        self.lib_file=lib_file
        self.model=None
        self.input_shape=input_shape
        self.output_shape=output_shape
        self.batch_size=batch_size
        
    def inference(self,data):
        """
        :param data: the model input
        :return: the model output
        """
        
       
        self.model.set_input("data",tvm.nd.array(data.astype("float32")))
        self.model.run()
        output=self.model.get_output(0)
        return output.asnumpy()

    def load(self):
        self.lib=tvm.runtime.load_module(self.lib_file)
        

    def activate(self,GID:int):
        """
        GPU资源分配在modelpool进行，这里只需要将模型加载到GPU上
        :param GID: GPU的ID号
        :return: None
        """
        self.GID=GID
        dev=tvm.cuda(self.GID)
        self.model=graph_executor.GraphModule(self.lib["default"](dev))
        
    def run_test(self,data):
        dev=tvm.cuda(self.GID)
        self.model.set_input("data",tvm.nd.array(data.astype("float32")))
        exec_time=self.model.benchmark(dev,repeat=3,min_repeat_ms=5000).mean        
        return exec_time
    

class MobileNetModel(ModelInterface):
    def __init__(self,
                 modelName:str,
                 lib_file:str,
                 input_shape:str,
                 output_shape:str,
                 batch_size:int):
        self.model_name=modelName
        self.lib_file=lib_file
        self.model=None
        self.input_shape=input_shape
        self.output_shape=output_shape
        self.batch_size=batch_size
        
    def inference(self,data):
        """
        :param data: the model input
        :return: the model output
        """
        
       
        self.model.set_input("data",tvm.nd.array(data.astype("float32")))
        self.model.run()
        output=self.model.get_output(0)
        return output.asnumpy()

    def load(self):
        self.lib=tvm.runtime.load_module(self.lib_file)
        

    def activate(self,GID:int):
        """
        GPU资源分配在modelpool进行，这里只需要将模型加载到GPU上
        :param GID: GPU的ID号
        :return: None
        """
        self.GID=GID
        dev=tvm.cuda(self.GID)
        self.model=graph_executor.GraphModule(self.lib["default"](dev))
        
    def run_test(self,data):
        dev=tvm.cuda(self.GID)
        self.model.set_input("data",tvm.nd.array(data.astype("float32")))
        exec_time=self.model.benchmark(dev,repeat=3,min_repeat_ms=5000).mean        
        return exec_time