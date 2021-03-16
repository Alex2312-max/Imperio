'''
Created with love by Sigmoid
@Author - Clefos Alexandru - clefos.alexandru@isa.utm.md
'''

# Importing all libraries
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class FrequencyImputationTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, categorical_index='auto', min_int_freq=5):
        '''
            Setting up the algorithm
        :param categorical_index: list, default = 'auto'
            A parameter that specifies the list of indexes of categorical columns that should be transformed.
        :param min_int_freq: int, default = 5
            A parameter that indicates the number of minimal values in a categorical column for the transformer
            to be applied.
        '''
        self.__categorical_index = categorical_index
        self.__min_int_freq = min_int_freq

    def fit(self, X, y=None, **fit_params):
        '''
            Fit function
        :param X: 2-d numpy array
            A parameter that stores the data set without the target vector.
        :param y: 1-d numpy array, default = None
            A parameter that stores the target vector.
        :param fit_params: dict
            Additional fit parameters.
        :return: FrequencyImputationTransformer
            The fitted transformer.
        '''
        # Creating a dictionary with all mappers.
        self.__mappers = dict()

        # Defining the list with categorical data types.
        categorical = [str]

        # Checking the value of __categorical_index.
        if self.__categorical_index == 'auto':

            # If __categorical_index is set as default, transformer finds all the categorical columns
            # for which mapper should be created.
            for i in range(len(X[0])):
                if (type(X[0, i]) in categorical or type(X[0, i]) == int) and len(set(X[:, i])) < self.__min_int_freq and len(set(X[:, i])) != 2:
                    self.__mappers[i] = dict()

                    # Mapping out all the categorical values in the selected column.
                    for element in np.unique(X[:, i]):
                        self.__mappers[i][element] = len(X[:, i][np.where(X[:, i] == element)]) / len(X)
        else:

            # If __categorical_index is set by the user, transformer iterates through the passed list of indexes.
            for i in self.__categorical_index:
                if i >= len(X[0]):
                    raise ValueError('Passed index list contains a invalid index!')
                self.__mappers[i] = dict()

                # Mapping out all the categorical values in the selected column.
                for element in np.unique(X[:, i]):
                    self.__mappers[i][element] = len(X[:, i][np.where(X[:, i] == element)]) / len(X)

        # Returning the fitted instance of FrequencyImputationTransformer class.
        return self

    def transform(self, X, **fit_transform):
        '''
            Function that transforms the new given data
        :param X: 2-d numpy array
            A parameter that stores the data set without the target vector.
        :param fit_transform: dict
            Additional fit parameters.
        :return: 2-d numpy array
            The transformed 2-d numpy array.
        '''
        for key in self.__mappers:

            # If a new value is passed than it is replaced with 0.
            X[:, key] = [self.__mappers[key][value] if value in self.__mappers[key] else 0 for value in X[:, key]]
        return X

    def fit_transform(self, X, y=None, **fit_params):
        '''
            Function that fits and transform the data
        :param X: 2-d numpy array
            A parameter that stores the data set without the target vector.
        :param y: 1-d numpy array
            A parameter that stores the target vector.
        :param fit_params: dict
            Additional fit parameters.
        :return: 2-d numpy array
            The transformed 2-d numpy array.
        '''
        return self.fit(X, y).transform(X)

    def apply(self, df, target):
        '''
            Function that apply the transformer on the pandas data frame.
        :param df: pandas data frame
            The pandas data frame on which the transformer should be applied.
        :param target: str
            The column name of the value that we want to predict.
        :return: pandas data frame
            The pandas data frame on which the transformer was applied.
        '''
        # Separating the feature matrix from the target vector and transforming them in numpy arrays.
        feature_columns = [column for column in df.columns if column != target]
        X = df[feature_columns].values
        y = df[target].values

        # Transforming the passed data.
        new_x = self.fit_transform(X, y)

        # Creating a new data frame with the processed categorical columns.
        new_df = pd.DataFrame(new_x, columns=feature_columns)
        new_df[target] = y

        # Returning the new data frame.
        return new_df

df = pd.read_csv('stroke.csv')
df = df.fillna(0)
print(df.dtypes)

X, y = df.drop(['stroke', 'id'], axis=1).values, df['stroke'].values

tgi = FrequencyImputationTransformer()
print()
print(X)
print(tgi.fit_transform(X, y))
print(tgi._FrequencyImputationTransformer__mappers)
print(X[:, 1].dtype.char)
print(tgi.apply(df, 'stroke'))
