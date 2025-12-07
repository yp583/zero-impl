from torch.optim import Optimizer

class SuperClass:
    def __init_subclass__(cls) -> None:
        print(cls.__name__)

class SubClassBefore(SuperClass):
    def __init__(self):
        print("hi before")

def new_init(self):
    print("hi from super")

@classmethod
def new_init_subclass(cls):
    cls.__init__ = new_init

def set_all_subclass_inits(cls):
    for scls in cls.__subclasses__():
        scls.__init__ = new_init

SuperClass.__init_subclass__ = new_init_subclass
set_all_subclass_inits(SuperClass)


class SubClassAfter(SuperClass):
    def __init__(self):
        print("hi after")
    

SubClassBefore()
SubClassAfter()

