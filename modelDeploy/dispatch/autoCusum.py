#1.在框架启动的时候，cusum值设置为0，泊松强度的期望值设置为0,设置threshold,设置一个队列长度为n记录每秒的数据到达率
#2.每一秒计算cusum+=当前值-泊松强度期望值，并将当前值放入队列中
#3.if cusum>threshold:
#	根据队列计算泊松强度
#	lam_avg=计算的泊松强度
#	cusum=0
import numpy as np
import math
import matplotlib.pyplot as plt
import random
#这一个队列用来记录每秒的数据到达率，然后计算泊松强度
class AutoQueue:
    def __init__(self,maxsize):
        self.length=0
        self.maxsize=maxsize
        self.queue=[]

    def append(self,x):
        if self.length==self.maxsize:
            self.queue.pop(0)
        else:
            self.length+=1
        self.queue.append(x)

    def find_diff_max(self):
        temp = 0
        index = 0
        for i in range(len(self.queue) - 1):
            if abs(self.queue[i + 1] - self.queue[i]) > temp:
                temp = abs(self.queue[i + 1] - self.queue[i])
                index = i
        return index

    def possion_get(self):
        index=self.find_diff_max()
        return np.mean(self.queue[index:])

class Cusum_Possion_Algorithm:
    def __init__(self,threshold):
        self.cusum=0
        self.threshold=threshold
        self.lam_avg=0
        self.q=AutoQueue(20)

    
    #将数据长度放入队列中，如果超过阈值，则重新计算泊松强度，最后返回当前的泊松强度
    def possion_get(self,x):
        self.cusum+=x-self.lam_avg
        self.q.append(x)
        if abs(self.cusum)>self.threshold:
            self.cusum=0
            self.lam_avg=self.q.possion_get()
        return self.lam_avg
    

if __name__=="__main__":
    #第一种情况，假设数据的泊松强度缓慢变化
    x=[i for i in range(2,100,5)]
    lam_list=[50*math.log(i,2) for i in range(2,100,5)]
    # plt.plot(x,lam)
    # plt.show()

    #构建一个possion序列
    np.random.poisson()
    possion_list=[]
    deal_possion_list=[]
    for lam in lam_list:
        num=random.randint(5,50)
        for j in np.random.poisson(lam,num):
            possion_list.append(j)
            deal_possion_list.append(lam)


    cusum_possion_list=[]
    cusum_possion=Cusum_Possion_Algorithm(50)
    for num in possion_list:
        lam_avg=cusum_possion.possion_get(num)
        cusum_possion_list.append(lam_avg)


    x=[i for i in range(len(deal_possion_list))]

    plt.plot(x,possion_list,label="possion_list")
    plt.plot(x,cusum_possion_list,label="cusum")
    plt.plot(x,deal_possion_list,label="deal")
    plt.legend()

    plt.show()
	
	

