import time

def measureTime(func):
    def wrapper(*args, **kwargs):
        starttime = time.perf_counter()
        temp = func(*args, **kwargs)
        endtime = time.perf_counter()
        print(f"Time needed to run {func.__name__}: {time.strftime('%H:%M:%S', time.gmtime(endtime - starttime))}")
        return(temp)
    return wrapper
