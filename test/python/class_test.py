class AAA(object):

    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

    def show(self):
        print(self.a)
        print(self.b)
        print(self.c)


a = AAA(3,4,5)
print(a.a)  # public is the default