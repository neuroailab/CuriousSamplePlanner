import subprocess
import os 
import shutil


max_num = 16
num_workers = 4
prefix  = "./data_collection/"
exp_name = "first_attempt"


# First, we need to create a folder to hold all of the solutions
if (os.path.isdir(prefix+exp_name)):
    shutil.rmtree(prefix+exp_name)

os.mkdir(prefix+exp_name)

for _ in range(num_workers):
	subprocess.Popen(["python", "data_collection/behavioral_cloning.py", str(exp_name), str(max_num)])


