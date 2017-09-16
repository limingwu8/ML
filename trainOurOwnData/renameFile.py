import os

path = "F:\\datasets\\dogs"
i = 0
for file in os.listdir(path):
    if os.path.isfile(os.path.join(path,file))==True:
        split = file.split(".")
        newName = "dog"+str(i)+"."+split[1]
        os.rename(os.path.join(path, file), os.path.join(path, newName))
    i += 1