import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np

"""
create dataframe with different types of data
"""
df = pd.DataFrame(
    {
            'num1': [1, 8, 2, 3, 4, 12, 23, 55],
            'num2': [-1, 9, 0, 33, np.nan, 5, 42, 69],
            'enum1': list('abcabcde'),
            'enum2': list('qretwret'),
            'free_text':
                ['cats are purple cute', 'dogs are good', 'blue is a color', 'i am hung,ry',
                 'glue sticks to stuff', 'flipping is fun', "i can't see in the dark?", 'space is  very big'],
            'output': [0, 0, 0, 1, 1, 0, 1, 1]
    }
)

print(df)


def clean(col):
    """
    :param col: name of column to do cleaning on
    :param df: dataframe
    :return: cleaned dataframe
    """
    col.replace('[^\w\s]', '')  # remove punctuation
    col.lower()  # lowercase text
    col.join('')  # join text back together
    return col


df['free_text'] = df['free_text'].apply(lambda x: clean(x))

"""
Build feature transformers for each of the input columns based on their type or how you want to handle them
For example, can create different numeric transformers if you want different imputation methods for different cols
"""
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # impute missing values with the mean of the column
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),  # fill unknown  w 'missing'
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # OHE the enums
])

free_text_transformer = Pipeline(steps=[
    ('cv', CountVectorizer(stop_words='english')),  # create count vectorizer and then normalize it with tfidf transform
    ('tfidf', TfidfTransformer())
])


"""
define your column types
"""
numeric_features = ['num1', 'num2']
categorical_features = ['enum1', 'enum2']
free_text_features = ['free_text']

"""
create your preprocessor
"""
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
        ('txt', free_text_transformer, free_text_features[0])
    ])

"""
select a model and pass it into your pipeline with the preprocessor
"""
rfc = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=0)
clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', rfc)
])

"""
Split the output column from the rest of the data, split train and test data, check scores and heat maps
"""
X = df.drop('output', axis=1)
y = df['output']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

clf.fit(X_train, y_train)
print("model score: %.3f" % clf.score(X_test, y_test))
prediction_probs = clf.predict_proba(X_test)
y_pred = clf.predict(X_test)
print(prediction_probs)
X_test['prediction0'] = prediction_probs[:, 0]
X_test['prediction1'] = prediction_probs[:, 1]
X_test['Actual'] = y_test.values
print(X_test)
