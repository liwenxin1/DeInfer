import os
import shutil

source_folder = 'tvm_module/tune_file'
destination_folder = 'experiment/lib_file'

for root, dirs, files in os.walk(source_folder):
    for file in files:
        if 'mobilenet' in file:
            source_file = os.path.join(root, file)
            destination_file = os.path.join(destination_folder, file)
            shutil.copy(source_file, destination_file)
 
