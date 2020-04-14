import os
import pathlib
import random
import shutil
from os import path

def get_file_list_from_dir(datadir):
    all_files = os.listdir(os.path.abspath(datadir))
    data_files = list(filter(lambda file: file.endswith('.txt'), all_files))
    return data_files

def copy_files(filenames, datatype, output,input):
    """Copies the files from the input folder to the output folder
    """
    # get the last part within the file
    full_path = path.join(output, datatype)
    pathlib.Path(full_path).mkdir(parents=True, exist_ok=True)
    for files in filenames:
        print(files)
        files = input + files
        shutil.copy2(files, full_path)
        base = os.path.splitext(files)[0]
        shutil.copy2(base + '.ann', full_path)

filenames = get_file_list_from_dir("./COMPLETE")
filenames.sort()  # make sure that the filenames have a fixed order before shuffling
random.seed(230)
random.shuffle(filenames) # shuffles the ordering of filenames (deterministic given the chosen seed)
#print(filenames)
split_1 = int(0.7 * len(filenames))
split_2 = int(0.8 * len(filenames))
train_filenames = filenames[:split_2]
dev_filenames = filenames[split_1:split_2]
test_filenames = filenames[split_2:]
#copy_files(train_filenames, "train", "./COMPLETE/","./COMPLETE/")
copy_files(test_filenames, "test", "./COMPLETE/","./COMPLETE/")
copy_files(dev_filenames, "dev", "./COMPLETE/","./COMPLETE/")