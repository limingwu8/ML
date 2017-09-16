# download image from imageNet
# multi-thread downloading

import threading
import urllib.request
import time

class DownloadThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        while len(urls)!=0:
            try:
                threadLock.acquire()
                url = urls.pop()
                threadLock.release()
                download_web_image(url,str(time.time()))

            except Exception as e:
                print("Fail in downloading image" + url)


def download_web_image(url,name):
    full_name = "F://datasets//cats//" + name + ".jpg"
    # urllib.request.urlretrieve(url,full_name)
    request = urllib.request.Request(url, None)
    timeout = 5
    response = urllib.request.urlopen(request, None,timeout=timeout)
    str = response.read()
    foo = open(full_name, "wb")
    foo.write(str)
    foo.close()

# download cats images
with open("F://datasets//urls.txt") as f:
    urls = [line.rstrip("\n") for line in f.readlines()]

threadLock = threading.Lock()
header = {
    'User-Agent': 'Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US; rv:1.9.1.6) Gecko/20091201 Firefox/3.5.6'
}
threads = []
if __name__ == "__main__":
    for i in range(10):
        thread = DownloadThread()
        thread.start()
        threads.append(thread)
