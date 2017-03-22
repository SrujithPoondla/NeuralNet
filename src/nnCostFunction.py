import numpy as np

from sigmoid import sigmoid


def vectorize_y(y, num_labels):
    new_y = np.array([(yi - 1) for yi in y])
    b = np.zeros((new_y.size, num_labels))
    b[np.arange(new_y.size), new_y] = 1
    return b


def nnCostFunction(Theta1, Theta2, X, y, input_bias, hidden_bias):
    m = X.shape[0]
    X = np.hstack((input_bias, X))

    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)

    z2 = np.dot(Theta1, X.transpose())
    a2 = sigmoid(z2)

    # Feedforward to output layer.
    a2 = np.vstack((hidden_bias, a2))
    z3 = np.dot(Theta2, a2)
    a3 = sigmoid(z3)

    Y = y

    term1 = sum(np.multiply(-Y, np.log(a3)))
    term2 = sum(np.multiply((1 - Y), np.log(1 - a3)))

    J = sum(term1 - term2) / m

    # Gradient for hidden unit
    delta3 = a3 - Y
    Theta2_grad = (Theta2_grad + np.dot(delta3, a2.transpose())) / m

    activation = np.multiply(a2, (1 - a2))
    temp = np.dot(Theta2.transpose(), delta3)
    delta2 = np.multiply(temp, activation)

    temp = np.dot(delta2, X)
    temp = np.delete(temp, 0, 0)

    Theta1_grad = (Theta1_grad + temp) / m
    return J, Theta1_grad, Theta2_grad


def perceptronCostFunction(Theta1, X, y):
    m = X.shape[0]
    X = np.hstack((np.ones((m, 1)), X))

    Theta1_grad = np.zeros(Theta1.shape)

    z2 = np.dot(Theta1, X.transpose())
    a2 = sigmoid(z2)

    Y = y

    term1 = sum(np.multiply(-Y, np.log(a2)))
    term2 = sum(np.multiply((1 - Y), np.log(1 - a2)))

    J = sum(term1 - term2) / m

    # Gradient for hidden unit
    delta3 = a2 - Y
    Theta1_grad = (Theta1_grad + np.dot(delta3, X)) / m

    return J, Theta1_grad
