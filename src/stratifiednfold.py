import random
import numpy as np
import pandas as pd
import sklearn.utils

class create_stratified_folds:
    def __init__(self, num_folds):
        self.num_folds = float(num_folds)
        self.stratified_data = []

    def create_stratified_data(self, data, classes):

        num_instances_fold = len(data) / self.num_folds

        class1 = data[data['Class']==classes[0]].index
        class1_shuffle=sklearn.utils.shuffle(class1)
        class2 = data[data['Class']==classes[1]].index
        class2_shuffle = sklearn.utils.shuffle(class2)
        class1_ratio=round(num_instances_fold * len(class1)/float(len(data)))
        class2_ratio=round(num_instances_fold * len(class2)/float(len(data)))
        for i in range(int(self.num_folds)):
            class1_index = []
            for k in range(i * int(class1_ratio), (i + 1) * int(class1_ratio)):
                if (k < len(class1)):
                    class1_index.append(class1_shuffle[k])

            class2_index = []
            for k in range(i * int(class2_ratio), (i + 1) * int(class2_ratio)):
                if(k < len(class2)):
                    class2_index.append(class2_shuffle[k])

            df1 = data.ix[class1_index]
            df2 = data.ix[class2_index]
            self.stratified_data.append(pd.concat([df1, df2]))
        return self.stratified_data
