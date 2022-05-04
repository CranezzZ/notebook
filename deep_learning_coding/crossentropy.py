import numpy as np
from sklearn.metrics import log_loss
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

def cross_entropy(x, y):
    x_softmax = [softmax(x[i]) for i in range(len(x))]
    x_log = [np.log(x_softmax[i][y[i]]) for i in range(len(y))]
    loss = - np.sum(x_log) / len(y)
    return loss

def main():
    x = np.array([[0.093, 0.1939, -1.0649, 0.4476, -2.0769],
                  [-1.8024, 0.3696, 0.7796, -1.0346, 0.473],
                  [0.5593, -2.5067, -2.1275, 0.5548, -1.6639]])
    y = np.array([1, 2, 3])
    xx = [[0.093, 0.1939, -1.0649, 0.4476, -2.0769],
                  [-1.8024, 0.3696, 0.7796, -1.0346, 0.473],
                  [0.5593, -2.5067, -2.1275, 0.5548, -1.6639]]
    yy = ['1','2','3']
    print(cross_entropy(x, y))
    labels = ['0','1','2','3','4']
    print(log_loss(yy,xx,labels))
    
if __name__ == '__main__':
    main()