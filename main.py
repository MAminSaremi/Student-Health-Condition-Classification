from src.preprocessing import *
from src.feature_engineering import *
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report
from sklearn.metrics import balanced_accuracy_score
train = pd.read_csv('data/data_clean.csv')
test = pd.read_csv('data/test.csv')

#? This section contains the final feature engineering steps used before model training.
#? Throughout the development process, many engineered features were created and evaluated
#? (some of which are no longer present in `feature_engineering.py` because they did not
#? improve model performance).
#?
#? Only the feature engineering functions that consistently improved the validation
#? score were retained. Therefore, the active (non-commented) feature functions
#? represent the final feature set used for training the model.
train = Stress_exercise(train)
# train = sleepDU_cate(train)
# train = sleepL_stress_flag(train)

test["stress_missing"] = (test["stress_level"] == "Missing").astype(int)
test["stress_level"] = test["stress_level"].map(stress_map)
test = fill_physicalact(test)
test = fill_sleepqu(test)

test = Stress_exercise(test)
# test= sleepDU_cate(test)
# test= sleepL_stress_flag(test)

x = train.drop(["id","health_condition"] , axis=1)
y = train["health_condition"]


train_x , test_x , train_y , test_y = train_test_split(x , y , test_size=0.3 , random_state=42 ,  stratify=y)



model = CatBoostClassifier(
    iterations=800,
    learning_rate=0.05,
    depth=9,
    loss_function="MultiClass",
    eval_metric="TotalF1",
    random_seed=42,
    l2_leaf_reg=5,
    bootstrap_type="Bernoulli",
    subsample=0.8,
    auto_class_weights="Balanced",
    verbose=False
)

cat_features = [
    "diet_type",
    "smoking_alcohol",
    "gender",
]
model.fit(train_x , train_y , cat_features=cat_features)
pred = model.predict(test_x)


print(classification_report(test_y,pred))
print(balanced_accuracy_score(test_y,pred))


importance = pd.DataFrame({
    "feature": train_x.columns,
    "importance": model.get_feature_importance()
})

importance = importance.sort_values(
    by="importance",
    ascending=False
)

print(importance)

# import joblib
# joblib.dump(model, "cat_model2.pkl")

#     precision    recall  f1-score   support

#      at-risk       0.99      0.94      0.96    177769
#          fit       0.74      0.95      0.83     11941
#    unhealthy       0.70      0.96      0.81     17317

#     accuracy                           0.94    207027
#    macro avg       0.81      0.95      0.87    207027
# weighted avg       0.95      0.94      0.94    207027

# 0.9491777928817863
#                     feature  importance
# 0            sleep_duration   21.581411
# 8              stress_level   16.528541
# 2                       bmi   10.243161
# 10  physical_activity_level    9.605405
# 14             stress_sleep    7.406135
# 4                step_count    5.372013
# 6              water_intake    4.796374
# 1                heart_rate    4.643653
# 11          smoking_alcohol    4.434527
# 3       calorie_expenditure    4.005120
# 5         exercise_duration    3.836301
# 9             sleep_quality    3.329915
# 12                   gender    1.640585
# 7                 diet_type    1.463639
# 13           stress_missing    1.113218

