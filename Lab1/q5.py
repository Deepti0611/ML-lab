import numpy as np
list=np.random.randint(1,10,25)# random integers generated 
print(list)
n=len(list)
def mean(list,n):# function to find mean
    sum=0
    for i in range(n):
        sum+=list[i]
        mean=sum/n
    return mean
def median(list,n):# function to find median
    list.sort()
    print(list)
    median=list[(n)//2]
    return median 
def mode(list,n):# function to find mode
    list.sort()
    max_count=0
    count=0
    for i in range(n):
        count=1
        for j in range(i+1,n):
            if list[i]==list[j]:
                count+=1
            else:
                break 
        if count>max_count:
            max_count=count
            mode=list[i]
    return mode
print("mean is", mean(list,n))
print("median is",median(list,n))
print("mode is",mode(list,n))

        
            


