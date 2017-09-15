import random
import urllib.request
import os
print(os.curdir)

def download_web_image(url,name):
    full_name = "C://Users//wu1114//Desktop//imgData//cats//" + name + ".jpg"
    urllib.request.urlretrieve(url,full_name)


# download cats images

with open("C://Users//wu1114//Desktop//imgData//imagenet.synset.txt") as f:
    urls = [line.rstrip("\n") for line in f.readlines()]

for i in range(len(urls)):
    print(i)
    try:
        name = "cat" + str(i)
        download_web_image(urls[i], name)
    except ():
        print("Fail in downloading " + str(i) + "image")



