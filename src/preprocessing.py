import pandas as pd
from sklearn.impute import SimpleImputer
df = pd.read_csv('data/train.csv')

# print(df.isnull().sum())
# id                             0
# health_condition               0
# sleep_duration             75999
# heart_rate                  7833
# bmi                        13898
# calorie_expenditure        52853
# step_count                 13916
# exercise_duration           6901
# water_intake               43477
# diet_type                   6901
# stress_level               82811
# sleep_quality              58331
# physical_activity_level    36621
# smoking_alcohol            28582
# gender                     21373
# dtype: int64

#? The analysis below shows that the feature means are very similar across the three target classes.
#? This suggests that manually imputing missing values based on class-specific patterns is unlikely
#? to provide a meaningful performance improvement, even if such patterns can be identified.
# numeric_cols=df.select_dtypes(include=['int64','float64']).columns
# print(df.groupby('health_condition')[numeric_cols].mean())

#? As these features contain only about 5% missing values, SimpleImputer was
#? considered an appropriate and efficient imputation method.
#? Numerical features (e.g., heart rate) were imputed with the mean because
#? their distributions showed little difference between the mean and median.
#? Categorical features were imputed using the most frequent category.
def fill_smok(df):
    imp_smok = SimpleImputer(strategy='most_frequent')
    df[['smoking_alcohol']] = imp_smok.fit_transform(df[['smoking_alcohol']])
    return df
df = fill_smok(df)
def fill_gender(df):
    imp_gender = SimpleImputer(strategy='most_frequent')
    df[['gender']] = imp_gender.fit_transform(df[['gender']])
    return df
df = fill_gender(df)

def fill_step_count(df):
    imp_step_count = SimpleImputer(strategy='median')
    df[['step_count']] = imp_step_count.fit_transform(df[['step_count']])
    return df
df = fill_step_count(df)

def fill_heart_rate  (df):
    imp_heart_rate   = SimpleImputer(strategy='mean')
    df[['heart_rate']] = imp_heart_rate  .fit_transform(df[['heart_rate']])
    return df
df = fill_heart_rate  (df)

def fill_exercise_duration (df):
    imp_exercise_duration  = SimpleImputer(strategy='mean')
    df[['exercise_duration']] = imp_exercise_duration .fit_transform(df[['exercise_duration']])
    return df
df = fill_exercise_duration (df)

def fill_diet_type(df):
    imp_diet_type = SimpleImputer(strategy='most_frequent')
    df[['diet_type']] = imp_diet_type.fit_transform(df[['diet_type']])
    return df
df = fill_diet_type(df)

def fill_bmi(df):
    imp_bmi = SimpleImputer(strategy='mean')
    df[['bmi']] = imp_bmi.fit_transform(df[['bmi']])
    return df
df = fill_bmi(df)

#? An exception was made for the stress_level feature. Since exploratory analysis revealed
#? a meaningful relationship between its missing values and the target variable, missing entries
#? were preserved as a separate "Missing" category instead of being imputed with the mode.
def fill_stress(df):
    df["stress_level"] = df["stress_level"].fillna("Missing")
    return df
df = fill_stress(df)

#? Features with naturally ordered categories were encoded using ordinal mapping.
#? Preserving the ordinal relationship enables the model to leverage the relative
#? ranking between categories during training.
def fill_sleepqu(df):
    sleep_map = {'poor':0 , 'average':1 , 'good' : 2}
    df['sleep_quality'] = df['sleep_quality'].map(sleep_map)
    return df
df = fill_sleepqu(df)
def fill_physicalact(df):
    physical_map = {'sedentary':0 , 'moderate' : 1 , 'active' : 2}
    df['physical_activity_level'] = df['physical_activity_level'].map(physical_map)
    return df
df = fill_physicalact(df)

from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
#? Features with more than 5% missing values were imputed using KNNImputer,
#? as this method provides more accurate and context-aware estimates by leveraging
#? the similarity between samples.
#?
#? Categorical features with an inherent order were first encoded using ordinal
#? mapping. This preserves their ordinal relationship and allows them to be
#? treated as numerical features, making them compatible with KNN-based imputation.
scaler = StandardScaler()
num_cols = ['sleep_duration', 'sleep_quality','physical_activity_level','calorie_expenditure' , 'water_intake']
df[num_cols]=scaler.fit_transform(df[num_cols])
imp = KNNImputer(n_neighbors=5)
df[num_cols] = imp.fit_transform(df[num_cols])
df[num_cols] = scaler.inverse_transform(df[num_cols])

df["sleep_quality"] = df["sleep_quality"].round().astype(int)
df["physical_activity_level"] = (
    df["physical_activity_level"]
      .round()
      .astype(int)
)


assert df.isnull().sum().sum() == 0

#? Missing values in the stress_level feature were first isolated into a separate
#? binary indicator (stress_missing) to explicitly preserve missingness as a source
#? of information.
#?
#? The original stress_level categories were then ordinally encoded to preserve
#? their natural ordering. After extracting the missingness information, missing
#? entries in stress_level become NaN.
#?
#? Since CatBoost natively handles missing numerical values, these NaN values are
#? intentionally retained. This allows the model to exploit both the ordinal
#? relationship of the observed categories and the informative missingness pattern,
#? which may improve predictive performance.
df['stress_missing'] = (df['stress_level'] =='Missing').astype(int)
stress_map = {
    "low": 0,
    "medium" :1,
    "high" :2
}
df['stress_level'] = df['stress_level'].map(stress_map)

df.to_csv(
    "data/data_clean.csv",
    index=False
)

