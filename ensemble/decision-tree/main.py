import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

if __name__ == '__main__':
    df = pd.read_csv('ad.data', header=None)
    explanatory_variable_columns = set(df.columns.values)
    response_variable_column = df[len(df.columns.values) - 1]

    # The last column describes the targets
    explanatory_variable_columns.remove(len(df.columns.values) - 1)

    Y = [1 if e == 'ad.' else 0 for e in response_variable_column]
    X = df[list(explanatory_variable_columns)]

    X.replace(to_replace=' *\?', value=-1, regex=True, inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(X, Y)

    pipeline = Pipeline([
        ('clf', DecisionTreeClassifier(criterion='entropy'))
    ])

    parameters = {
        'clf__max_depth': (150, 155, 160),
        'clf__min_samples_split': (1, 2, 3),
        'clf__min_samples_leaf': (1, 2, 3)
    }

    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1,
                               verbose=1, scoring='f1')
    grid_search.fit(X_train, y_train)