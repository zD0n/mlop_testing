import subprocess
import re
import os

#Run Under Conda enviroment from the follow requirements.txt with python version 3.10.18
#How to run this Example you have to run like this C:\Coding\ML\Mini_project>python Run_1_4.py

folder_path = "./pipeline/scripts"
model_name = "emotion-classifier"
epoch = 10


files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
print("Files in folder:", files[0:4])

subprocess.run(["python", f"{folder_path}/{files[0]}"],check=True)

result = subprocess.run(
    ["python", f"{folder_path}/{files[1]}"],
    capture_output=True, text=True, check=True
)

subprocess.run(
    ["python", f"{folder_path}/{files[2]}",re.search(r"Preprocessing Run ID:\s*(\w+)", result.stdout).group(1),str(epoch),model_name],
    check=True
)

subprocess.run(
    ["python", f"{folder_path}/{files[3]}",model_name,"Staging"],
    check=True
)
