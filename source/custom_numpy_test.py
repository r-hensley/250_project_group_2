import numpy as np


class Test(np.ndarray):
    def __new__(subtype, *args):
        # Create the ndarray instance of our type, given the usual
        # ndarray input arguments.  This will call the standard
        # ndarray constructor, but return an object of our type.
        # It also triggers a call to InfoArray.__array_finalize__
        print('args', args)
        print('args as np.array', np.array(args), type(np.array(args)))
        l = np.array(args)
        print(l.shape)
        print(l, len(l))
        obj = super().__new__(subtype, shape=(4,), dtype=float,
                              buffer=np.array(args), offset=0, strides=None, order=None)

        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # ``self`` is a new object resulting from
        # ndarray.__new__(InfoArray, ...), therefore it only has
        # attributes that the ndarray.__new__ constructor gave it -
        # i.e. those of a standard ndarray.
        #
        # We could have got to the ndarray.__new__ call in 3 ways:
        # From an explicit constructor - e.g. InfoArray():
        #    obj is None
        #    (we're in the middle of the InfoArray.__new__
        #    constructor, and self.info will be set when we return to
        #    InfoArray.__new__)
        if obj is None: return
        # From view casting - e.g arr.view(InfoArray):
        #    obj is arr
        #    (type(obj) can be InfoArray)
        # From new-from-template - e.g infoarr[:3]
        #    type(obj) is InfoArray
        #

        # We do not need to return anything


x = np.array([1., 2., 3., 4.])
t1 = Test(1., 2., 3., 4.)

print(f"{t1=}")

a1 = np.array(t1)
print(a1, type(a1), type(a1[0]))

a2 = np.asarray(t1)
print(a2, type(a2), type(a2[0]))

a3 = np.asanyarray(t1)
print(a3, type(a3), type(a3[0]))

print('now with ([t1])')

a1 = np.array(Test(x))
print(a1, type(a1), type(a1[0]))

a2 = np.asarray(Test(x))
print(a2, type(a2), type(a2[0]))

a3 = np.asanyarray(Test(x))
print(a3, type(a3), type(a3[0]))

a3 = np.vstack([a3, t1])
print(a3)
for i in a3:
    print(i, type(i))

print('sum')
s = np.sum(t1)
print(s, type(s))
