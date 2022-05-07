class MyList(list):
    def __init__(self, *args):
        super().__init__()

        self.l = list(args)

    def __repr__(self):
        return str(list(self.l))

    def __str__(self):
        return str(list(self.l))


l1 = MyList(1, 2, 3)
print(l1)


