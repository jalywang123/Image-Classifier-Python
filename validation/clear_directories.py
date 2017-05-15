import os 

#path to perfect 
path_to_perfect="./perfect"
labels=os.listdir(path_to_perfect)
for f in labels:
    files=os.listdir(path_to_perfect+"/"+f)
    for file in files:
        if file.endswith(".tiff"):
            os.remove(path_to_perfect+"/"+f+"/"+file)
        
path_to_errors="./errors"
labels=os.listdir(path_to_errors)
for f in labels:
    files=os.listdir(path_to_errors+"/"+f)
    for file in files:
        if file.endswith(".tiff"):
            os.remove(path_to_errors+"/"+f+"/"+file)