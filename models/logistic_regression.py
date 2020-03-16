import numpy as np

class LogisticRegression:
    def __init__(self, input_dim, num_classes):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.W = np.random.randn(self.num_classes, self.input_dim)
        self.b = np.random.randn(self.num_classes, 1)


    def softmax(self, z):
        return np.exp(z) / np.sum(np.exp(z), axis=0)


    def train(self, X, Y, num_epochs=100, alpha=0.05, print_delta=100):
        m = X.shape[0]
        for i in range(num_epochs):
            # forward propagation
            Z = np.matmul(self.W, X) + self.b
            probs = self.softmax(Z)

            # compute loss
            L = -(1. / m) * np.sum(np.multiply(Y, np.log(probs)))
            if i % print_delta == 0:
                print("Loss at epoch {}: {}".format(i, L))

            # backpropagation (compute gradients)
            dZ = probs - Y
            dW = (1. / m) * np.matmul(dZ, X.T)
            db = (1. / m) * np.sum(dZ, axis=1, keepdims=True)

            # update weights
            self.W = self.W - alpha * dW
            self.b = self.b - alpha * db


    def predict(self, X):
        Z = np.matmul(self.W, X) + self.b
        probs = self.softmax(Z)
        preds = np.argmax(probs, axis=0)
        return preds





