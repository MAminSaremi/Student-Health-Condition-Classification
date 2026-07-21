import pandas as pd
from preprocessing import *
from feature_engineering import *
def make_submission(model):
    test = pd.read_csv("data/test.csv")

    

    # ---------- preprocessing ----------
    test = fill_smok(test)
    test = fill_gender(test)
    test = fill_step_count(test)
    test = fill_heart_rate(test)
    test = fill_exercise_duration(test)
    test = fill_diet_type(test)
    test = fill_bmi(test)
    test = fill_stress(test)
    test = fill_sleepqu(test)
    test = fill_physicalact(test)
    test["stress_missing"] = (test["stress_level"] == "Missing").astype(int)
    test["stress_level"] = test["stress_level"].map(stress_map)
    test = Stress_exercise(test)
    


    ids = test["id"]

    test = test.drop(columns=["id"])

    pred = model.predict(test)
    pred = pred.ravel()

    submission = pd.DataFrame({
        "id": ids,
        "health_condition": pred
    })

    submission.to_csv("submission.csv", index=False)

    print("submission.csv ساخته شد.")

import joblib

model = joblib.load("cat_model2.pkl")

make_submission(model)