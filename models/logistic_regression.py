import numpy as np
from .data_utils import get_minibatches
from sklearn.metrics import accuracy_score

class LogisticRegression:
    def __init__(self, input_dim, num_classes):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.W = np.random.randn(self.num_classes, self.input_dim)
        self.b = np.random.randn(self.num_classes, 1)


    def softmax(self, z):
        return np.exp(z) / np.sum(np.exp(z), axis=0)


    def train(self, X_train, Y_train, X_dev, Y_dev, num_epochs=100, alpha=0.05, minibatch_size=32, print_interval=100):
        minibatches = get_minibatches(X_train, Y_train, minibatch_size)
        for i in range(num_epochs):
            loss = -1
            for minibatch_X, minibatch_Y in minibatches:
                m = minibatch_X.shape[1]

                # forward propagation
                Z = np.matmul(self.W, minibatch_X) + self.b
                probs = self.softmax(Z)

                # compute loss
                loss = -(1. / m) * np.sum(np.multiply(minibatch_Y, np.log(probs)))

                # backpropagation (compute gradients)
                dZ = probs - minibatch_Y
                dW = (1. / m) * np.matmul(dZ, minibatch_X.T)
                db = (1. / m) * np.sum(dZ, axis=1, keepdims=True)

                # update weights
                self.W = self.W - alpha * dW
                self.b = self.b - alpha * db

            if i % print_interval == 0:
                y_pred = self.predict(X_dev)
                y_actual = np.argmax(Y_dev, axis=0)
                accuracy = accuracy_score(y_actual, y_pred)

                print("Loss at epoch {}: {}; Accuracy: {}".format(i, loss, accuracy))


    def predict(self, X):
        Z = np.matmul(self.W, X) + self.b
        probs = self.softmax(Z)
        preds = np.argmax(probs, axis=0)
        return preds


