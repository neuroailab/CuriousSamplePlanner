import subprocess
import os 
import shutil


max_num = 2048
num_workers = 10
prefix  = "./data_collection/solution_data/"
exp_name = "trajs_threeblocks"
cuda = "-1"


# First, we need to create a folder to hold all of the solutions
if (not os.path.isdir(prefix+exp_name)):
	os.mkdir(prefix+exp_name)



for _ in range(num_workers):
	subprocess.Popen(["python", "data_collection/behavioral_cloning.py", str(exp_name), cuda, str(max_num)])


