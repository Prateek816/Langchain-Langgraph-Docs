from langchain_core.runnables import RunnableLambda

class Runnable:
    def __init__(self,func):
        self.func = func
    
    def __or__(self,other):
        def chained_func(*args,**kwargs):
            return other.func(self.func(*args,**kwargs))
        return Runnable(chained_func)
    
    def invoke(self,*args,**kwargs):
        return self.func(*args,**kwargs)



def add_five(x):
    return x+5

def sub_five(x):
    return x-5

def mul_five(x):
    return x*5

def div_five(x):
    return x/5

add_five_runnable = RunnableLambda(add_five)
sub_five_runnale = RunnableLambda(sub_five)
mul_five_runnable = RunnableLambda(mul_five)
div_five_runnable = RunnableLambda(div_five)

chain  = (add_five_runnable).__or__(sub_five_runnale).__or__(mul_five_runnable).__or__(div_five_runnable)
#same as -
chain = add_five_runnable | sub_five_runnale | mul_five_runnable | div_five_runnable


