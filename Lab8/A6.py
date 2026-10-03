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
def perceptron(A,B,C,D,y,w0,w1,w2,w3,w4,n):
    err=[]
    epochs=[]
    for e in range(1000):
        epoch_errors=[]
        for i in range(len(A)):
            target=y[i]
            yin=summation([A[i],B[i],C[i],D[i]],[w1,w2,w3,w4],w0)
            output=step(yin)
            error=comparator(target,output)
            epoch_errors.append(error**2)
            
            w0=w0+n*error
            w1=w1+n*error*A[i]
            w2=w2+n*error*B[i]
            w3=w3+n*error*C[i]
            w4=w4+n*error*D[i]
        
        sse=sum(epoch_errors)
        err.append(sse)
        epochs.append(e+1)
        if sse<=0.002:
            break
    return w0,w1,w2,w3,w4,err,epochs
'''def plot(err,epochs):
    import matplotlib.pyplot as plt
    plt.plot(epochs, err)
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.title('Error vs Epochs')
    plt.grid()
    plt.show()'''
print("Perceptron Learning Algorithm")
w0,w1,w2,w3,w4=10,0.2,-0.75,0.1,-0.2
candies=[20,16,27,19,24,22,15,18,21,16]
mangoes=[6,3,6,1,4,1,4,4,1,2]
milk=[2,6,2,2,2,5,2,2,4,4]
payment=[386,289,393,110,280,167,271,274,148,198]

y=[1,1,1,0,1,0,1,1,0,0]
n=0.05
w0,w1,w2,w3,w4,err,epochs=perceptron(candies,mangoes,milk,payment,y,w0,w1,w2,w3,w4,n)
print(f"Final weights: w0={w0}, w1={w1}, w2={w2}, w3={w3}, w4={w4}".format(w0, w1, w2, w3, w4))
print(f"Number of epochs: {epochs[-1]}")
#plot(err,epochs)



    


