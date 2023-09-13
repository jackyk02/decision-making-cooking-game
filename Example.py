import xgboost as xgb  # import XGBoost
from sklearn.model_selection import train_test_split # import sklearn

# Spliting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.1, random_state = 67)

clf_xgb = xgb.XGBClassifier(objective='multi:softmax',
                            eval_metric="logloss",
                            seed=42,
                            tree_method="gpu_hist",
                            predictor='gpu_predictor',
                            use_label_encoder=False)
clf_xgb.fit(X_train,
            y_train,
            verbose=True,
            early_stopping_rounds=10,
            eval_metric='mlogloss',
            eval_set=[(X_test, y_test)])

