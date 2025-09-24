import subprocess
import re
import os

#Run Under Conda enviroment from the follow requirements.txt with python version 3.10.18

folder_path = "./scripts"

files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
print("Files in folder:", files[0:3])

subprocess.run(["python", f"{folder_path}/{files[0]}"],check=True)

result = subprocess.run(
    ["python", f"{folder_path}/{files[1]}"],
    capture_output=True, text=True, check=True
)
stdout = result.stdout

match = re.search(r"Preprocessing Run ID:\s*(\w+)", result.stdout)
run_id = match.group(1) if match else None

subprocess.run(
    ["python", f"{folder_path}/{files[2]}",run_id],
    check=True
)

