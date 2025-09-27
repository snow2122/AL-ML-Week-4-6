import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Load the Telco Churn dataset
# (Replace with your actual path or dataframe)
data = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Target variable
y = data["Churn"].map({"Yes": 1, "No": 0})  # encode target as 0/1
X = data.drop("Churn", axis=1)

# Separate numerical and categorical features
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object"]).columns

# Preprocessing: scale numbers, encode categories
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

# Choose models
log_reg = LogisticRegression(max_iter=1000)
rf = RandomForestClassifier(random_state=42)

# Build pipeline (preprocessing + model)
pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", log_reg)])

# Define hyperparameters to search
param_grid = [
    {
        "classifier": [log_reg],
        "classifier__C": [0.01, 0.1, 1, 10],
        "classifier__solver": ["liblinear", "lbfgs"],
    },
    {
        "classifier": [rf],
        "classifier__n_estimators": [50, 100],
        "classifier__max_depth": [None, 10, 20],
    },
]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Grid search with cross-validation
grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, scoring="accuracy")
grid_search.fit(X_train, y_train)

# Evaluate on test set
y_pred = grid_search.predict(X_test)
print("Best parameters:", grid_search.best_params_)
print(classification_report(y_test, y_pred))

# Save the trained pipeline
joblib.dump(grid_search.best_estimator_, "churn_pipeline.joblib")
