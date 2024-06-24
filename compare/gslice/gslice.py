# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import subprocess
import math
from numpy.lib.function_base import copy
import numpy as np
import logging



MAX_RESOURCE=100
def get_max_batchsize(openfilepath):
    batchsize_support=[]
    with open(openfilepath, encoding='utf-8') as f:
        for line in f.readlines():
            model_name=line.split(" ")[0]
            batch_size=line.split(" ")[1].split(":")[1]
            mps_set=float(line.split(" ")[2].split(":")[1])
            runtime=float(line.split(" ")[3].split(":")[1])
            if batch_size not in batchsize_support:
                batchsize_support.append(int(batch_size))
    batchsize_support.sort()
    return batchsize_support[-1]
    
def durationps(openfilepath,batch,mps):
    model_mps_runtime_dict={}
    # numbers, inference latency
    batchsize_support=[]
    with open(openfilepath, encoding='utf-8') as f:
        for line in f.readlines():
            model_name=line.split(" ")[0]
            batch_size=line.split(" ")[1].split(":")[1]
            mps_set=float(line.split(" ")[2].split(":")[1])
            runtime=float(line.split(" ")[3].split(":")[1])
            #字典格式：例子：{"resnet-18_1":[0.0001,0.0002,0.0003,0.0004,0.0005,0.0006,0.0007,0.0008,0.0009,0.001]}
            
            if batch_size not in batchsize_support:
                batchsize_support.append(int(batch_size))
                
            if model_name+"_"+batch_size not in model_mps_runtime_dict:
                model_mps_runtime_dict[model_name+"_"+batch_size]=[]
                model_mps_runtime_dict[model_name+"_"+batch_size].append((runtime))
            else:
                model_mps_runtime_dict[model_name+"_"+batch_size].append((runtime))
                
    batchsize_support.sort()
    if batch not in batchsize_support:
        #找到比batch_size大的最小的batch_size的index
        index_b_upper=0
        for i in range(len(batchsize_support)):
            if batch<batchsize_support[i]:
                index_b_upper=i
                break
        index_b_under=index_b_upper-1
        
        runtime_list=[]
        for index in [index_b_under,index_b_upper]:
            batch_size=str(batchsize_support[index])
            #mps should be less than 100
            
            
            #临界值处理,如果mps设置超过100,就返回100的值
            if mps>=100:
                runtime=model_mps_runtime_dict[model_name+"_"+batch_size][-1]
            elif mps<10:
                runtime=model_mps_runtime_dict[model_name+"_"+batch_size][0]/(mps/10)
            else:
                #对mps进行插值
                param=mps/10-1
                index_under=int(param)
                index_upper=index_under+1
                weight=index_upper-param
                runtime=model_mps_runtime_dict[model_name+"_"+batch_size][index_under]*weight+\
                model_mps_runtime_dict[model_name+"_"+batch_size][index_upper]*(1-weight)
            runtime_list.append(runtime)
            
        #对batch_size进行插值
        weight=(batch-batchsize_support[index_b_under])/(batchsize_support[index_b_upper]-batchsize_support[index_b_under])
        
        runtime=runtime_list[0]*(1-weight)+runtime_list[1]*weight
    else:
        batch_size=str(batch)
        
        #mps should be less than 100
        #临界值处理,如果mps设置超过100,就返回100的值
        if mps>=100:
            runtime=model_mps_runtime_dict[model_name+"_"+batch_size][-1]
        elif mps<10:
            runtime=model_mps_runtime_dict[model_name+"_"+batch_size][0]/(mps/10)
        else:
            #对mps进行插值
            param=mps/10-1
            index_under=int(param)
            index_upper=index_under+1
            weight=index_upper-param
            runtime=model_mps_runtime_dict[model_name+"_"+batch_size][index_under]*weight+\
            model_mps_runtime_dict[model_name+"_"+batch_size][index_upper]*(1-weight)
    
    throughput=batch/runtime
    return runtime,throughput
            
        

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def compute_new_gpu_util(current_gpu_util, slo, arrival_rate, avg_latency, avg_throughput):
    """
    :param current_gpu_util: 当前gpu使用量
    :param slo: 服务水平
    :param arrival_rate:  要满足的吞吐量要求
    :param avg_latency: 平均延时
    :param avg_throughput: 平均吞吐量
    :return: 调整后的gpu使用量
    """
    residual_latency = slo - avg_latency  # 剩余延时能力，正常情况avg应该小于slo
    residual_throughput = avg_throughput - arrival_rate  # 剩余吞吐量，正常情况下avg应该大于arrival_rate
    diff_latency = residual_latency * 100 / slo  # 剩余延时比例
    diff_throughput = residual_throughput * 100 / arrival_rate  # 剩余吞吐量比例
    if diff_latency > 0 and diff_latency < 10 and diff_throughput > 0 and diff_throughput < 10:  # 剩余的量不多，就直接返回当前资源量
        return current_gpu_util
    change_factor = max(abs(residual_latency) / avg_latency, abs(residual_throughput) / avg_throughput)  # 调整比例
    change_gpu_util = current_gpu_util * change_factor  # 调整的资源量
    new_gpu_util = 0
    if residual_latency < 0 or residual_throughput < 0:  # 剩余量出现负值，代表资源不够，因此需要增加资源
        new_gpu_util = current_gpu_util + change_gpu_util
    if residual_latency > 0 and residual_throughput > 0:  # 剩余量均为正值，代表资源满足要求，可以适当减少资源
        new_gpu_util = current_gpu_util - change_gpu_util
    if(new_gpu_util>MAX_RESOURCE):
        return MAX_RESOURCE
    elif(new_gpu_util<1):
        return 1
    return new_gpu_util


def is_time_to_adjust(slo, avg_latency, cur_batch):
    return True


def slab(slo, avg_latency, cur_batch,modelname):
    file="compare/data/{}_mps_runtime.txt".format(modelname)
    MAX_BATCH=get_max_batchsize(file)
    residual_latency = slo - avg_latency  # 剩余延时能力，正常情况avg应该小于slo
    diff_latency = residual_latency * 100 / slo  # 剩余延时比例
    if diff_latency > 0 and diff_latency < 10:
        return cur_batch
    change_batch = abs(residual_latency) / avg_latency * cur_batch
    new_batch = 1
    if residual_latency > 0:
        new_batch = cur_batch + change_batch
    elif residual_latency < 0:
        new_batch = cur_batch - change_batch
    #print("batch",cur_batch,new_batch,slo, avg_latency)
    if(new_batch>MAX_BATCH):
        new_batch=MAX_BATCH
    elif(new_batch<1):
        new_batch=1
    return math.ceil(new_batch)


def update_batch(new_batch):
    print("set the new batch as {0}".format(new_batch))


def update_gpu_util(new_gpu_util):
    print("set the new gpu utilization as {0}".format(new_gpu_util))


class Infere:
    batch=[]
    resource=[]
    latency=[]
    throughput=[]
    def printi(self):
        print("batch ",self.batch)
        print("resource ",self.resource)
        print("latency ",self.latency)
        print("throughput ",self.throughput)
        print()
    def printth(self):
        print("batch ",self.batch)
        print("latency ",self.latency)
        th=[]
        for i in range(len(self.throughput)):
            th.append(self.batch[i]*self.throughput[i])
        print("throughput ",th)
        print()
    def __init__(self) -> None:
        self.batch=[]
        self.resource=[]
        self.latency=[]
        self.throughput=[]

def gslice(model_name_list,arrivalrate,slo,index):
    #测试模型的推理性能
    # for model_name in model_name_list:
    #     shell="python modelDeploy/compare/testinference.py --model_name {}".format(model_name)
    #     subprocess.run(shell, shell=True)
        
    l=len(arrivalrate)
    batch=[1]*l
    resource=[30]*l
    #index 100 200
    inferes=[]
    for i in range(l):
        i1=Infere()
        i1.batch.append(1)
        i1.resource.append(30)
        inferes.append(i1)
    #print(inferes)

    for p in range(5):
        for t in range(5):
            
            flag=True
            for i in range(l):
                #durationps返回两个值，一个是模型执行的latency，一个是模型执行的throughput
                observe=durationps("compare/data/{}_mps_runtime.txt".format(model_name_list[i]),batch[i],resource[i])
                inferes[i].latency.append(observe[1])
                inferes[i].throughput.append(observe[0])
                curbatch=batch[i]
                batch[i]=slab(slo[i],observe[0],batch[i],model_name_list[i])
                #logging.info("avg_latency %s",str(observe[1]))
                #logging.info("SLAB %s %s",str(batch[i]),str(resource[i]))
                inferes[i].batch.append(batch[i])
                inferes[i].resource.append(resource[i])
                if(batch[i]!=curbatch):
                    flag=False
            index+=1
            if(flag):
                break
        
        flag=True
        for i in range(l):
            observe=durationps("compare/data/{}_mps_runtime.txt".format(model_name_list[i]),batch[i],resource[i])
            inferes[i].latency.append(observe[1])
            inferes[i].throughput.append(observe[0])
            curresource=resource[i]
            resource[i]=compute_new_gpu_util(resource[i],slo[i],arrivalrate[i],observe[0],observe[1])
            #logging.info("observe %s",str(observe))
            #logging.info("resource %s %s",str(batch[i]),str(resource[i]))
            inferes[i].batch.append(batch[i])
            inferes[i].resource.append(resource[i])
            if(resource[i]!=curresource):
                flag=False
        index+=1
        if(flag):
            break
    for t in range(5):
        
        flag=True
        for i in range(l):
            observe=durationps("compare/data/{}_mps_runtime.txt".format(model_name_list[i]),batch[i],resource[i])
            inferes[i].latency.append(observe[1])
            inferes[i].throughput.append(observe[0])
            curbatch=batch[i]
            batch[i]=slab(slo[i],observe[0],batch[i],model_name_list[i])
            #logging.info("avg_latency %s",str(observe[1]))
            #logging.info("SLAB %s %s",str(batch[i]),str(resource[i]))
            inferes[i].batch.append(batch[i])
            inferes[i].resource.append(resource[i])
            if(batch[i]!=curbatch):
                flag=False
        index+=1
        if(flag):
            break
    
    for i in range(l):
        observe=durationps("compare/data/{}_mps_runtime.txt".format(model_name_list[i]),batch[i],resource[i])
        inferes[i].latency.append(observe[1])
        inferes[i].throughput.append(observe[0])
    # for i in inferes:
    #     for j in range(len(i.batch)):
    #         i.throughput[j]*=i.batch[j]
    #     i.printi()   
    return [batch,resource]

#均值100,噪声50均匀分布
def arrival_rate1(modelname,slo,record_time):
    np.random.seed(0)
    arrivalrate_list=[]
    resource_list=[]
    for _ in range(record_time):
        base_arrivalrate=100
        noise=np.random.randint(-50,50)
        arrivalrate=base_arrivalrate+noise
        g=gslice([modelname],[arrivalrate],[slo],150)
        arrivalrate_list.append(arrivalrate)
        resource_list.append(g[1][0])
    return resource_list,arrivalrate_list

#正常到达率为50，第30秒突然到达1000，然后恢复
def arrival_rate2(modelname,slo,record_time):
    arrivalrate_list=[]
    resource_list=[]
    for i in range(record_time):
        if i==29:
            arrivalrate=1000
        else:
            arrivalrate=50
        g=gslice([modelname],[arrivalrate],[slo],150)
        arrivalrate_list.append(arrivalrate)
        resource_list.append(g[1][0])
    return resource_list,arrivalrate_list
def draw_with_changing_arrivalrate(modelname,slo,record_time,func):
    
    
    resource_list,arrivalrate_list=func(modelname,slo,record_time)
    
    time_list=[i for i in range(record_time)]
    # 创建一个figure对象
    fig = plt.figure()

    # 创建第一个子图
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(time_list, resource_list)
    ax1.set_title('gpu resource variation trend')

    # 创建第二个子图
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(time_list, arrivalrate_list)
    ax2.set_title('{} Arrival Rate'.format(modelname))

    # 保存图形
    fig.tight_layout()
    plt.savefig("compare/pic/{}_with_changing_arrivalrate.png".format(modelname))
    print("finish")
        
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    #slo的单位是秒
    # logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s',
    #                 datefmt='%a %d %b %Y %H:%M:%S', filename='gslice3.log', filemode='a')
    models=["resnet-18","transformer","vgg-11"]
    #g=gslice([models[0],models[1]],[1000,500],[5,20])
    g=gslice([models[2]],[70],[100],150)
    print(g)
    draw_with_changing_arrivalrate("vgg-11",100,60,arrival_rate2)
    # arrivalrate_list=[[100,100,100],[200,200,200],[400,400,400],[800,800,800]]
    # #绘制两个子图
    # plt.figure(figsize=(20,20))
    # #调整子图间距
    # plt.subplots_adjust(hspace=0.4)
    # num=0
    # models=[models[0],models[1],models[2]]
    # for arrivalrate in arrivalrate_list:
    #     batchsize=[]
    #     reousrce=[]
    #     for i in range(1,101):
            
    #         slo=[0.05*i,0.05*i,0.05*i]
    #         g=gslice(models,arrivalrate,slo,150)
    #         batchsize.append(g[0])
    #         reousrce.append(g[1])
    #     slo=[i*0.05 for i in range(1,101)]
    #     batchsize=np.array(batchsize)
    #     batchsize1=batchsize[:,0]
    #     batchsize2=batchsize[:,1]
    #     batchsize3=batchsize[:,2]
    #     reousrce=np.array(reousrce)
    #     reousrce1=reousrce[:,0]
    #     reousrce2=reousrce[:,1]
    #     reousrce3=reousrce[:,2]
        
    #     plt.subplot(4,2,num*2+1)
        
    #     plt.plot(slo,batchsize1,label="resnet-18 arrivalrate={}".format(arrivalrate[0]),color="g")
    #     plt.plot(slo,batchsize2,label="transformer arrivalrate={}".format(arrivalrate[1]),color="r")
    #     plt.plot(slo,batchsize3,label="vgg-11 arrivalrate={}".format(arrivalrate[2]),color="b")
    #     plt.xlabel("slo")
    #     plt.ylabel("batchsize")
    #     plt.legend()
    #     plt.subplot(4,2,num*2+2)
        
    #     plt.plot(slo,reousrce1,label="resnet-18 arrivalrate={}".format(arrivalrate[0]))
    #     plt.plot(slo,reousrce2,label="transformer arrivalrate={}".format(arrivalrate[1]))
    #     plt.plot(slo,reousrce3,label="vgg-11 arrivalrate={}".format(arrivalrate[2]))
        
    #     plt.xlabel("slo")
    #     plt.ylabel("resource")
    #     plt.legend()
    #     num+=1
    # plt.savefig("compare/pic/gslice_vgg-11.png")
    
    #2.
    # slo_list=[[0.1,0.1,0.1],[0.5,0.5,0.5],[1,1,1],[10,10,10]]
    # #绘制两个子图
    # plt.figure(figsize=(20,20))
    # #调整子图间距
    # plt.subplots_adjust(hspace=0.4)
    # num=0
    # models=[models[0],models[1],models[2]]
    # for slo in slo_list:
    #     batchsize=[]
    #     reousrce=[]
    #     for i in range(1,101):
            
    #         arr_rate=[10*i,10*i,10*i]
    #         g=gslice(models,arr_rate,slo,150)
    #         batchsize.append(g[0])
    #         reousrce.append(g[1])
    #     arrval_rate=[i*10 for i in range(1,101)]
    #     batchsize=np.array(batchsize)
    #     batchsize1=batchsize[:,0]
    #     batchsize2=batchsize[:,1]
    #     batchsize3=batchsize[:,2]
    #     reousrce=np.array(reousrce)
    #     reousrce1=reousrce[:,0]
    #     reousrce2=reousrce[:,1]
    #     reousrce3=reousrce[:,2]
        
    #     plt.subplot(4,2,num*2+1)
        
    #     plt.plot(arrval_rate,batchsize1,label="resnet-18 slo={}ms".format(slo[0]*1000),color="g")
    #     plt.plot(arrval_rate,batchsize2,label="transformer slo={}ms".format(slo[1]*1000),color="r")
    #     plt.plot(arrval_rate,batchsize3,label="vgg-11 slo={}ms".format(slo[2]*1000),color="b")
    #     plt.xlabel("arrivalrate")
    #     plt.ylabel("batchsize")
    #     plt.legend()
    #     plt.subplot(4,2,num*2+2)
        
    #     plt.plot(arrval_rate,reousrce1,label="resnet-18 slo={}ms".format(slo[0]*1000))
    #     plt.plot(arrval_rate,reousrce2,label="transformer slo={}ms".format(slo[1]*1000))
    #     plt.plot(arrval_rate,reousrce3,label="vgg-11 slo={}ms".format(slo[2]*1000))
        
    #     plt.xlabel("arrivalrate")
    #     plt.ylabel("resource")
    #     plt.legend()
    #     num+=1
    # plt.savefig("compare/pic/gslice_arrival_rate.png")
    