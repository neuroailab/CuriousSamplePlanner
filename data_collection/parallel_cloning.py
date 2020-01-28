import subprocess

size = 4096
split = 8
exp_name = "abc"
for index in range(split):
	subprocess.Popen(["python", "data_collection/behavioral_cloning.py", str(exp_name), str(size), str(split), str(index)])