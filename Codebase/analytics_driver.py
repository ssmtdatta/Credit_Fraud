import subprocess

import config as _config

'''
parameterize main
use mlflow to keep track for resampling process and features.
'''

num = 4

command_str = f"python test.py -num {num}"
print(command_str)

subprocess.run(command_str, 
	           shell=True)