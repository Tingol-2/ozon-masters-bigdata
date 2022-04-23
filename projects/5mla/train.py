#!/opt/conda/envs/dsenv/bin/python
import os, sys
import logging

import pandas as pd
import argparse, sys

# from model import model, fields

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer

import mlflow

def get_model():
    numeric_features = ["if" + str(i) for i in range(1, 14)]
    categorical_features = ["cf" + str(i) for i in range(1, 27)] + ["day_number"]

    fields = ["id", "label"] + numeric_features + categorical_features
    fill_fields = numeric_features

    categorical_features_used_for_encodings = \
        ['cf6', 'cf9', 'cf13', 'cf16', 'cf17', 'cf19', 'cf25', 'cf26', 'day_number']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features_used_for_encodings)
        ]
    )

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('logisticregression', LogisticRegression(random_state=42, n_jobs=-1, max_iter=1000))
    ])

    return model, fields, fill_fields

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('train_path', type=str, help='Path to train dataset')
    parser.add_argument('model_param1', type=int, default=1000, help='max_iter parameter for model')
    args = parser.parse_args()

    model, fields, fill_fields = get_model()

    table_opts = dict(sep="\t", names=fields, index_col=False, nrows=10000)
    data = pd.read_csv(args.train_path, **table_opts)

    data[fill_fields] = data[fill_fields].fillna(0)

    data = data.astype({col: 'int' for col in fill_fields})

    X_train, y_train = data.iloc[:, 2:], data.iloc[:, 1]
    mlflow.sklearn.autolog()
    with mlflow.start_run():
        model.set_params(logisticregression__max_iter=args.model_param1)
        model.fit(X_train, y_train)


if __name__ == "__main__":
    main()
