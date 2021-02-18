from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV


numeric_features = ["if"+str(i) for i in range(1,14)]
categorical_features = ["cf"+str(i) for i in range(1,27)] + ["day_number"]

fields = ["id", "label"] + numeric_features + categorical_features

categorical_to_transform = ['cf6', 'cf9', 'cf13', 'cf16', 'cf17', 'cf19', 'cf25', 'cf26']   #cf8,cf18 

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_to_transform)
    ]
)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('linearregression', LogisticRegression())
])
