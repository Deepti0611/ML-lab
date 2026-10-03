
from sklearn.neural_network import MLPClassifier 
def and_gate(X,Y):
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(15,), random_state=1)
    clf.fit(X, Y)
    predictions = clf.predict(X)
    return clf,predictions

X=[[0, 0], [0, 1], [1, 0], [1, 1]]
y=[0, 0, 0, 1]
clf,predictions=and_gate(X,y)
print(f"Predictions: {predictions}")
print("actual", y)
y=[0,1,1,0]
clf,predictions=and_gate(X,y)
print(f"Predictions: {predictions}")
print("actual", y)
