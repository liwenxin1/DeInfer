import os
import json
def list_only_directories(path):
    # 使用列表推导式过滤结果，仅保留文件夹
    directories = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    return directories

def list_so_files(path):
    # 使用列表推导式过滤结果，仅保留以 ".so" 为后缀的文件
    so_files = [f for f in os.listdir(path) if f.endswith('.so') and os.path.isfile(os.path.join(path, f))]
    return so_files

def load_json_file(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in file '{file_path}': {e}")
        return None

def rename_tuneFile(file_dir):
    directories=list_only_directories(file_dir)
    for directory in directories:
        files=list_so_files(os.path.join(file_dir,directory))
        for file in files:
            if len(file.split("-"))>=3:
                batch_str=file.split("-")[-3]
                filename=directory+"_"+batch_str+".so"
                
                os.rename(os.path.join(file_dir,directory,file),os.path.join(file_dir,directory,filename))

if __name__=="__main__":
    pass
                