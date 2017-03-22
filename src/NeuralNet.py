import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm, datasets, metrics


import stratifiednfold
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold
from nnCostFunction import nnCostFunction, perceptronCostFunction
from predict import predict, encode_and_scale
from preprocess import preprocess


def initialize_weights(input_units, hidden_units):
    if hidden_units == 0:
        Theta1 = np.random.uniform(-0.01, 0.01, (1, input_units + 1))
    else:
        Theta1 = np.random.uniform(-0.01, 0.01, (hidden_units, input_units + 1))
    Theta2 = np.random.uniform(-0.01, 0.01, (1, hidden_units + 1))
    return Theta1, Theta2


def print_matches(activation, predicted_class, test_y, class_dict):
    test_index=test_y.index
    length = len(test_y)
    actual_predicted = zip(predicted_class, test_y)
    for idx, tuple in enumerate(actual_predicted):
        print(
            "fold_of_instance {3}\t\t\tActivation {0}\t\t\t\tPredicted class : {1}\t\tActual class : {2}".format(activation[idx],
                                                                                       class_dict[tuple[0]],
                                                                                       class_dict[tuple[1]], fold_num[idx]))

    accuracy = accuracy_score(predicted_class, test_y)
    print(
        "Number of correctly classisfied instances {0} : Number of Incorrectly classisfied instances {1}"
            .format(round(accuracy * length), length - round(accuracy * length)))


if __name__ == '__main__':
    features,metadata, stratifiedfolds = preprocess(sys.argv[1],sys.argv[2])
    learning_rate = float(sys.argv[3])
    epoch = int(sys.argv[4])
    class_dict = {
        0: metadata['Class'][1][0],
        1: metadata['Class'][1][1],
    }


    input_units = len(features)-1
    hidden_units = input_units
    Theta1, Theta2 = initialize_weights(input_units, hidden_units)
    input_bias = np.random.uniform(-0.01, 0.01, (1, 1))
    hidden_bias = np.random.uniform(-0.01, 0.01, (1, 1))

    num=0

    final_results = pd.DataFrame()
    train_data=pd.DataFrame()
    for i in stratifiedfolds:
        test_fold= i
        test_y = test_fold['Class']
        test_y = (test_y.apply(lambda x: 0 if (x == metadata['Class'][1][0]) else 1))
        test_X = test_fold.drop('Class', axis=1)
        for j in range(len(stratifiedfolds)):
            if j is not num:
                train_data =pd.concat([train_data,stratifiedfolds[j]])
        training_data_size = len(train_data)
        train_y= train_data['Class']
        train_y = (train_y.apply(lambda x: 0 if (x == metadata['Class'][1][0]) else 1)).values
        train_X = train_data.drop('Class', axis=1)

        J = 0
        for x in xrange(epoch):
            cross_entropy = 0
            for index in range(training_data_size):
                cross_entropy += J
                if hidden_units == 0:
                    [J, Theta1_grad] = perceptronCostFunction(Theta1, train_X[index:index + 1], train_y[index])
                    Theta1 -= (learning_rate * Theta1_grad)
                    cross_entropy += J
                else:
                    [J, Theta1_grad, Theta2_grad] = nnCostFunction(Theta1, Theta2, train_X[index:index + 1], train_y[index],input_bias, hidden_bias)
                    Theta1 -= (learning_rate * Theta1_grad)
                    Theta2 -= (learning_rate * Theta2_grad)
                    cross_entropy += J
            if hidden_units == 0:
                Theta = [Theta1]
            else:
                Theta = [Theta1, Theta2]

            activation, predicted_class = predict(Theta, train_X, input_bias, hidden_bias)
            accuracy = accuracy_score(predicted_class, train_y)

        activation, predicted_class = predict(Theta, test_X, input_bias, hidden_bias)

        fold_num=[]
        for k in range(len(test_y)):
            fold_num.append(num)
        # results = pd.DataFrame(test_y)
        results=pd.DataFrame()
        results['fold_of_instance'] = fold_num
        results['predicted_class'] = predicted_class
        results['actual_class'] = test_y.values
        results['confidence_of_prediction'] = activation
        final_results = pd.concat([final_results, results])

        # print_matches(activation, predicted_class, test_y, class_dict)
        num = num + 1


    final_results.sort_index(inplace=True)
    print final_results.to_string(index=False)
    accuracy = accuracy_score(final_results['predicted_class'], final_results['actual_class'])
    print accuracy


    # fpr, tpr, thresholds = metrics.roc_curve(final_results['actual_class'], final_results['confidence_of_prediction'], pos_label=1)
    # roc_auc = metrics.auc(fpr, tpr)
    #
    # plt.figure()
    # lw = 2
    # plt.plot(fpr, tpr, color='darkred',
    #          lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic example')
    # plt.legend(loc="lower right")
    # plt.show()
