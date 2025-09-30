import subprocess
import re
import os
import sys
#Run Under Conda enviroment from the follow requirements.txt with python version 3.10.18
#How to run this Example you have to run like this C:\Coding\ML\Mini_project>python Run_1_4.py

def run(model_name="emotion-classifier",epoch=10):

    folder_path = "./pipeline/scripts"
    print("Finding Files")
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    print("Found in folder:", files[0:4])

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

if __name__ == "__main__":
    try:
        if len(sys.argv) < 2:
            print("Usage: ./pipeline/scripts/Run_1_4.py <Model_name> <Epochs>")
            print("Or you just run everything being set default")
            sys.exit(1)
        
        model_name = sys.argv[1]
        epoch = sys.argv[2]

        run(model_name,epoch)
    except:
        run()