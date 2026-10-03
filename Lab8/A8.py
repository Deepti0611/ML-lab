import math
def summation(inputs,w,b):
    weighted_sum = sum(i * w for i, w in zip(inputs, w)) + b
    return weighted_sum
def step(y):
    return 1 if y >= 0 else 0
def bipolar(y):
    return 1 if y >= 0 else -1
def sigmoid(y):
    return 1 / (1 + math.exp(-y))
def tanh(y): 
    return math.tanh(y)
def relu(y):
    return max(0, y)
def leaky_relu(y, alpha=0.01):
    return y if y >= 0 else alpha * y
def comparator(target,output):
    return target - output
def backpropagation(A,B,y,v11,v12,v21,v22,w1,w2,alpha):
    err=[]
    epochs=[]
    for e in range(1000):
        epoch_errors=[]
        for i in range(len(A)):
            net1=A[i]*v11+B[i]*v21
            h1=sigmoid(net1)
            net2=A[i]*v12+B[i]*v22
            h2=sigmoid(net2)
            sumoutput=h1*w1+h2*w2
            output=sigmoid(sumoutput)
            target=y[i]
            error=comparator(target,output)
            delta_last=output*(1-output)*error
            delta_h1=h1*(1-h1)*delta_last*w1
            delta_h2=h2*(1-h2)*delta_last*w2
            w1=w1+alpha*delta_last*h1
            w2=w2+alpha*delta_last*h2
            v11=v11+alpha*delta_h1*A[i]
            v21=v21+alpha*delta_h1*B[i]
            v12=v12+alpha*delta_h2*A[i]
            v22=v22+alpha*delta_h2*B[i]
            epoch_errors.append(error**2)
        sse=sum(epoch_errors)
        err.append(sse)
        epochs.append(e+1)
        if sse<=0.002:
            break

    return v11,v12,v21,v22,w1,w2,err,epochs
A=[0,0,1,1]
B=[0,1,0,1]
y=[0,0,0,1]
n=0.05
v11 = 0.1
v12 = 0.2
v21 = 0.3
v22 = 0.4
w1 = 0.5
w2 = 0.6
v11, v12, v21, v22, w1, w2, err, epochs = backpropagation(A, B, y,v11, v12, v21, v22,w1, w2,n)
print(f"Final weights:")
print(f"V11 = {v11}")
print(f"V12 = {v12}")
print(f"V21 = {v21}")
print(f"V22 = {v22}")
print(f"W1  = {w1}")
print(f"W2  = {w2}")

print(f"Number of epochs = {epochs[-1]}")
print(f"Final SSE = {err[-1]}")



    


