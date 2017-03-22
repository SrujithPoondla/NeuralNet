import pandas as pd
from scipy.io import arff
import stratifiednfold

def df_train(args):
    pass


def preprocess(input_file,num_folds):
    loaded_arff = arff.loadarff(open(input_file, 'rb'))
    (df, metadata) = loaded_arff
    features = metadata.names()

    df = pd.DataFrame(df)
    print "Entered PreProcess"
    create_train = stratifiednfold.create_stratified_folds(num_folds)
    folds = create_train.create_stratified_data(df,metadata['Class'][1])



    return  features, metadata,folds
