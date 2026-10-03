
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
def backpropagation(A,B,y,v11,v12,v21,v22,w11,w12,w21,w22,alpha):
    err=[]
    epochs=[]
    for e in range(1000):
        epoch_errors=[]
        for i in range(len(A)):
            net1=A[i]*v11+B[i]*v21
            h1=step(net1)
            net2=A[i]*v12+B[i]*v22
            h2=step(net2)
            net_01=h1*w11+h2*w21
            o1=step(net_01)
            net_02=h1*w12+h2*w22
            o2=step(net_02)
            if y[i]==0:
                target1=1
                target2=0
            else:
                target1=0
                target2=1
            error1=comparator(target1,o1)
            error2=comparator(target2,o2)
            epoch_errors.append(error1**2+error2**2)
            delta_o1=o1*(1-o1)*error1
            delta_o2=o2*(1-o2)*error2
            delta_h1=h1*(1-h1)*(delta_o1*w11+delta_o2*w12)
            delta_h2=h2*(1-h2)*(delta_o1*w21+delta_o2*w22)
            w11=w11+alpha*delta_o1*h1
            w12=w12+alpha*delta_o2*h1
            w21=w21+alpha*delta_o1*h2
            w22=w22+alpha*delta_o2*h2
            v11=v11+alpha*delta_h1*A[i]
            v21=v21+alpha*delta_h1*B[i]
            v12=v12+alpha*delta_h2*A[i]
            v22=v22+alpha*delta_h2*B[i]

        sse=sum(epoch_errors)
        err.append(sse)
        epochs.append(e+1)
        if sse<=0.002:
            break

    return v11,v12,v21,v22,w11,w12,w21,w22,err,epochs
A=[0,0,1,1]
B=[0,1,0,1]
y=[0,0,0,1]
n=0.05
v11 = 0.1
v12 = 0.2
v21 = 0.3
v22 = 0.4
w11 = 0.5
w12 = 0.6
w21 = 0.7
w22 = 0.8
v11, v12, v21, v22, w11, w12, w21, w22, err, epochs = backpropagation(A, B, y,v11, v12, v21, v22,w11, w12, w21, w22,n)
print(f"Final weights:")
print(f"V11 = {v11}")
print(f"V12 = {v12}")
print(f"V21 = {v21}")
print(f"V22 = {v22}")
print(f"W11  = {w11}")
print(f"W12  = {w12}")
print(f"W21  = {w21}")
print(f"W22  = {w22}")

print(f"Number of epochs = {epochs[-1]}")
print(f"Final SSE = {err[-1]}")



    


