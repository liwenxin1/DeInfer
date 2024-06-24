import numpy as np
import matplotlib.pyplot as plt
import matplotlib
mobilenet=np.load("modelDeploy/data_deal/mobilenet_system3_data.npy",allow_pickle=True)
resnet18=np.load("modelDeploy/data_deal/resnet-18_system3_data.npy",allow_pickle=True)
resnet101=np.load("modelDeploy/data_deal/resnet-101_system3_data.npy",allow_pickle=True)
shufflenet=np.load("modelDeploy/data_deal/shufflenet_system3_data.npy",allow_pickle=True)
transformer=np.load("modelDeploy/data_deal/transformer_system3_data.npy",allow_pickle=True)
vgg11=np.load("modelDeploy/data_deal/vgg-11_system3_data.npy",allow_pickle=True)
vgg13=np.load("modelDeploy/data_deal/vgg-13_system3_data.npy",allow_pickle=True)
vgg16=np.load("modelDeploy/data_deal/vgg-16_system3_data.npy",allow_pickle=True)

time_list=[i for i in range(1,61)]
# 创建一个figure对象
fig = plt.figure(figsize=(9,16))
# 创建第一个子图
ax1 = fig.add_subplot(8, 1, 1)
ax1.plot(time_list, mobilenet[2])
ax1.plot(time_list,[mobilenet[3][0] for _ in range(len(mobilenet[2]))],"r--")
ax1.plot(time_list,[np.mean(mobilenet[2]) for _ in range(len(mobilenet[2]))],"g--")
ax1.set_title('mobilenet queue length')

# 创建第二个子图
ax2 = fig.add_subplot(8, 1, 2)
ax2.plot(time_list, resnet18[2])
ax2.plot(time_list,[resnet18[3][0] for _ in range(len(resnet18[2]))],"r--")
ax2.plot(time_list,[np.mean(resnet18[2]) for _ in range(len(resnet18[2]))],"g--")
ax2.set_title('resnet-18 queue length')

# 创建第三个子图
ax3 = fig.add_subplot(8, 1, 3)
ax3.plot(time_list, resnet101[2])
ax3.plot(time_list,[resnet101[3][0] for _ in range(len(resnet101[2]))],"r--")
ax3.plot(time_list,[np.mean(resnet101[2]) for _ in range(len(resnet101[2]))],"g--")
ax3.set_title('resnet-101 queue length')

# 创建第四个子图
ax4 = fig.add_subplot(8, 1, 4)
ax4.plot(time_list, shufflenet[2])
ax4.plot(time_list,[shufflenet[3][0] for _ in range(len(shufflenet[2]))],"r--")
ax4.plot(time_list,[np.mean(shufflenet[2]) for _ in range(len(shufflenet[2]))],"g--")
ax4.set_title('shufflenet queue length')

# 创建第五个子图
ax5 = fig.add_subplot(8, 1, 5)
ax5.plot(time_list, transformer[2])
ax5.plot(time_list,[transformer[3][0] for _ in range(len(transformer[2]))],"r--")
ax5.plot(time_list,[np.mean(transformer[2]) for _ in range(len(transformer[2]))],"g--")
ax5.set_title('transformer queue length')

# 创建第六个子图
ax6 = fig.add_subplot(8, 1, 6)
ax6.plot(time_list, vgg11[2])
ax6.plot(time_list,[vgg11[3][0] for _ in range(len(vgg11[2]))],"r--")
ax6.plot(time_list,[np.mean(vgg11[2]) for _ in range(len(vgg11[2]))],"g--")
ax6.set_title('vgg11 queue length')

# 创建第七个子图
ax7 = fig.add_subplot(8, 1, 7)
ax7.plot(time_list, vgg13[2])
ax7.plot(time_list,[vgg13[3][0] for _ in range(len(vgg13[2]))],"r--")
ax7.plot(time_list,[np.mean(vgg13[2]) for _ in range(len(vgg13[2]))],"g--")
ax7.set_title('vgg-13 queue length')

# 创建第八个子图
ax8 = fig.add_subplot(8, 1, 8)
ax8.plot(time_list, vgg16[2])
ax8.plot(time_list,[vgg16[3][0] for _ in range(len(vgg16[2]))],"r--")
ax8.plot(time_list,[np.mean(vgg16[2]) for _ in range(len(vgg16[2]))],"g--")
ax8.set_title('vgg-16 queue length')

fig.tight_layout()
plt.savefig("modelDeploy/pic/system3_queue_length.png")


