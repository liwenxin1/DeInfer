from abc import ABC,abstractmethod
from modelDeploy.modelFactory.modelInterface import *
class ModelFactory(ABC):
    @abstractmethod
    def create_model():
        # TODO: Implement the factory method here
        pass


class Resnet18_Factory(ModelFactory):
    def create_model(self,
                 modelName:str,
                 lib_file:str,
                 input_shape:str,
                 output_shape:str,
                 batch_size:int):
        return ResNet18Model(modelName,lib_file,input_shape,output_shape,batch_size)

class Resnet34_Factory(ModelFactory):
    def create_model(self,
                 modelName:str,
                 lib_file:str,
                 input_shape:str,
                 output_shape:str,
                 batch_size:int):
        return ResNet34Model(modelName,lib_file,input_shape,output_shape,batch_size)

class Resnet50_Factory(ModelFactory):
    def create_model(self,
                 modelName:str,
                 lib_file:str,
                 input_shape:str,
                 output_shape:str,
                 batch_size:int):
        return ResNet50Model(modelName,lib_file,input_shape,output_shape,batch_size)

class Resnet101_Factory(ModelFactory):
    def create_model(self,
                 modelName:str,
                 lib_file:str,
                 input_shape:str,
                 output_shape:str,
                 batch_size:int):
        return ResNet101Model(modelName,lib_file,input_shape,output_shape,batch_size)
    
class Resnet152_Factory(ModelFactory):
    def create_model(self,
                 modelName:str,
                 lib_file:str,
                 input_shape:str,
                 output_shape:str,
                 batch_size:int):
        return ResNet152Model(modelName,lib_file,input_shape,output_shape,batch_size)

class Transformer_Factory(ModelFactory):
    def create_model(self,modelName:str,
                 lib_file:str,
                 input_shape:str,
                 output_shape:str,
                 batch_size:int):
        return TransformerModel(modelName,lib_file,input_shape,output_shape,batch_size)

class VGG11_Factory(ModelFactory):
    def create_model(self,modelName:str,
                 lib_file:str,
                 input_shape:str,
                 output_shape:str,
                 batch_size:int):
        return VGG11Model(modelName,lib_file,input_shape,output_shape,batch_size)

class VGG13_Factory(ModelFactory):
    def create_model(self,modelName:str,
                 lib_file:str,
                 input_shape:str,
                 output_shape:str,
                 batch_size:int):
        return VGG13Model(modelName,lib_file,input_shape,output_shape,batch_size)

class VGG16_Factory(ModelFactory):
    def create_model(self,modelName:str,
                 lib_file:str,
                 input_shape:str,
                 output_shape:str,
                 batch_size:int):
        return VGG16Model(modelName,lib_file,input_shape,output_shape,batch_size)

class VGG19_Factory(ModelFactory):
    def create_model(self,modelName:str,
                 lib_file:str,
                 input_shape:str,
                 output_shape:str,
                 batch_size:int):
        return VGG19Model(modelName,lib_file,input_shape,output_shape,batch_size)

class MobileNet_Factory(ModelFactory):
    def create_model(self,modelName:str,
                 lib_file:str,
                 input_shape:str,
                 output_shape:str,
                 batch_size:int):
        return MobileNetModel(modelName,lib_file,input_shape,output_shape,batch_size)

class ShuffleNet_Factory(ModelFactory):
    def create_model(self,modelName:str,
                 lib_file:str,
                 input_shape:str,
                 output_shape:str,
                 batch_size:int):
        return ShuffleNetModel(modelName,lib_file,input_shape,output_shape,batch_size)

class SqueezeNet_Factory(ModelFactory):
    def create_model(self,modelName:str,
                 lib_file:str,
                 input_shape:str,
                 output_shape:str,
                 batch_size:int):
        return SqueezeNetModel(modelName,lib_file,input_shape,output_shape,batch_size)


class Model_Creator:
    def __init__(self):
        self.model_dict={}
        self.model_dict["resnet-18"]=Resnet18_Factory()
        self.model_dict["resnet-34"]=Resnet34_Factory()
        self.model_dict["resnet-50"]=Resnet50_Factory()
        self.model_dict["resnet-101"]=Resnet101_Factory()
        self.model_dict["resnet-152"]=Resnet152_Factory()
        self.model_dict["transformer"]=Transformer_Factory()
        self.model_dict["vgg-11"]=VGG11_Factory()
        self.model_dict["vgg-13"]=VGG13_Factory()
        self.model_dict["vgg-16"]=VGG16_Factory()
        self.model_dict["vgg-19"]=VGG19_Factory()
        self.model_dict["mobilenet"]=MobileNet_Factory()
        self.model_dict["shufflenet"]=ShuffleNet_Factory()
        self.model_dict["squeezenet"]=SqueezeNet_Factory()
        

    def create_model(self,modelName:str,
                 lib_file:str,
                 input_shape:str,
                 output_shape:str,
                 batch_size:int):
        return self.model_dict[modelName].create_model(modelName,lib_file,input_shape,output_shape,batch_size)
