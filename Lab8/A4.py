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
def perceptron(A,B,y,w0,w1,w2,n):
    err=[]
    epochs=[]
    for e in range(1000):
        epoch_errors=[]
        for i in range(len(A)):
            target=y[i]
            yin=summation([A[i],B[i]],[w1,w2],w0)
            output=step(yin)
            error=comparator(target,output)
            epoch_errors.append(error**2)
            
            w0=w0+n*error
            w1=w1+n*error*A[i]
            w2=w2+n*error*B[i]
        
        sse=sum(epoch_errors)
        err.append(sse)
        epochs.append(e+1)
        if sse<=0.002:
            break
    return w0,w1,w2,err,epochs
'''def plot(err,epochs):
    import matplotlib.pyplot as plt
    plt.plot(epochs, err)
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.title('Error vs Epochs')
    plt.grid()
    plt.show()'''
print("Perceptron Learning Algorithm")

n=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.9,1]
for i in n:
    w0,w1,w2=10,0.2,-0.75
    A=[0,0,1,1]
    B=[0,1,0,1]
    y=[0,0,0,1]
    w0,w1,w2,err,epochs=perceptron(A,B,y,w0,w1,w2,i)
    print(f"Final weights: w0={w0}, w1={w1}, w2={w2}".format(w0, w1, w2))
    print (f"Learning rate: {i}, Number of epochs: {epochs[-1]}")

   




    


