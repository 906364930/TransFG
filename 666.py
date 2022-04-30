class A():
    def __init__(self, a=1):
        self.a = a

    def func(self):
        print(self.a)


class B(A):
    def __init__(self, a=1):
        super().__init__(a)
        self.a = 6

    def func(self):
        print(self.a)


cc = B()
cc.func()