from itertools import combinations
#对所有测试的组合进行计算
def find_combinations():
    def find_combinations_with_repetition(data, target_sum):
        def backtrack(start, target, path):
            if target == 0:
                result.append(path)
                return
            for i in range(start, len(data)):
                if target >= data[i]:
                    # 递归调用，允许重复使用元素
                    backtrack(i, target - data[i], path + [data[i]])

        result = []
        data.sort()  # 可选，对数据进行排序可以帮助优化
        backtrack(0, target_sum, [])
        return result

    data = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    target_sum = 100

    combinations_list = find_combinations_with_repetition(data, target_sum)

    # 打印结果
    combo_5_list=[]
    combo_4_list=[]
    combo_3_list=[]
    combo_2_list=[]
    for combo in combinations_list:
        if len(combo) == 5:
            combo_5_list.append(combo)
        elif len(combo) == 4:
            combo_4_list.append(combo)
        elif len(combo) == 3:
            combo_3_list.append(combo)
        elif len(combo) == 2:
            combo_2_list.append(combo)
        print(combo)
    print(len(combo_5_list))
    print(len(combo_4_list))
    print(len(combo_3_list))
    print(len(combo_2_list))
    import math
    num=5
    result=math.comb(num,2)*math.perm(num,2)+math.comb(num,3)*math.perm(8,3)+math.comb(num,4)*math.perm(9,4)+math.comb(num,5)*math.perm(7,5)
    print(result)
    print((result*10)/60/24/365)#20个模型需要1000年才能完成计算

import os


def find_name(prefix):
    folder_path = "./profiles/exectime_txt/"  # Replace with the actual folder path
    file_names = []
    for file_name in os.listdir(folder_path):
        if file_name.startswith(prefix):
            file_names.append(file_name)
    return file_names

def get_modelname(filename):
        index=len(filename)-1
        count=0
        while index>=0:
            if filename[index]=="-":
                count+=1
            if count==2:
                break
            index-=1
        return filename[:index]
if __name__ == "__main__":
    x = find_name("interference")
    for name in x:
        name1=name.split("_")[-2]
        name2=name.split("_")[-1][:-4]
        filename="./profiles/exectime_txt/mps_{}.txt".format(name1)
        sh_batch,sh_factor=name1.split("-")[-2:]
        sh_name=get_modelname(name1)
        shell_commend=("bash ./model_mps_no_interference.sh {} {} {}").format(sh_name,sh_batch,sh_factor)
        os.system(shell_commend)
        
        filename="./profiles/exectime_txt/mps_{}.txt".format(name2)
        sh_batch,sh_factor=name2.split("-")[-2:]
        sh_name=get_modelname(name2)
        shell_commend=("bash ./model_mps_no_interference.sh {} {} {}").format(sh_name,sh_batch,sh_factor)
        os.system(shell_commend)
        
        
        
        


