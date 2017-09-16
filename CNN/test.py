import urllib.request

url="http://farm2.static.flickr.com/1369/1162408779_6eb2d41a35.jpg"
header = {
    'User-Agent': 'Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US; rv:1.9.1.6) Gecko/20091201 Firefox/3.5.6'
}
request = urllib.request.Request(url,None,headers=header)
response = urllib.request.urlopen(request,None)
str = response.read()
foo = open("F://datasets//cats//" + "cat1" + ".jpg","wb")
foo.write(str)
foo.close()