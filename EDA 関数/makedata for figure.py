def makedata(func,xmin,xmax,step,**args):
    x = np.arange(xmin,xmax,step)
    y = func(x,**args)
    return x,y