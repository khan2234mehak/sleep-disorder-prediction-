"""
train_model.py — Sleep Disorder ML Model Training (v2)
=======================================================
Priority:
  1. Real Kaggle CSV  →  data/sleep_data.csv
                      OR Sleep_health_and_lifestyle_dataset.csv
  2. Synthetic data   →  fallback (1200 samples)

Run:
  python train_model.py

Outputs:
  model/sleep_model.pkl
  model/scaler.pkl
  model/encoders.pkl
  model/feature_cols.pkl

Kaggle dataset:
  https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset
  → save as  data/sleep_data.csv  then re-run
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle, os

np.random.seed(42)

# ─────────────────────────────────────────
# SYNTHETIC FALLBACK DATASET
# ─────────────────────────────────────────
def make_synthetic_dataset(n=1200):
    print(f'  Generating {n} synthetic samples...')
    ages        = np.random.randint(18, 70, n)
    genders     = np.random.choice(['Male', 'Female'], n)
    occupations = np.random.choice([
        'Engineer', 'Doctor', 'Teacher', 'Nurse', 'Accountant',
        'Lawyer', 'Sales Representative', 'Software Engineer', 'Scientist', 'Manager'
    ], n)
    sleep_dur   = np.round(np.random.uniform(4.5, 9.5, n), 1)
    stress      = np.random.randint(1, 11, n)
    bmi_cat     = np.random.choice(
        ['Underweight', 'Normal', 'Overweight', 'Obese'], n,
        p=[0.07, 0.45, 0.30, 0.18]
    )
    heart_rate  = np.random.randint(55, 100, n)
    daily_steps = np.random.randint(1500, 15000, n)
    phys_act    = np.random.randint(10, 90, n)
    sys_bp      = np.random.randint(100, 160, n)
    dia_bp      = np.random.randint(65, 105, n)

    labels = []
    for i in range(n):
        s = 0
        if sleep_dur[i] < 6:                        s += 2
        if stress[i] >= 7:                          s += 2
        if heart_rate[i] > 85:                      s += 1
        if bmi_cat[i] in ['Overweight', 'Obese']:   s += 1
        if daily_steps[i] < 4000:                   s += 1
        if phys_act[i] < 30:                        s += 1
        if sys_bp[i] > 140:                         s += 2
        labels.append(
            'Sleep Apnea' if s >= 6 else ('Insomnia' if s >= 3 else 'None')
        )

    return pd.DataFrame({
        'Age': ages, 'Gender': genders, 'Occupation': occupations,
        'Sleep Duration': sleep_dur, 'Stress Level': stress, 'BMI Category': bmi_cat,
        'Heart Rate': heart_rate, 'Daily Steps': daily_steps,
        'Physical Activity Level': phys_act,
        'Systolic BP': sys_bp, 'Diastolic BP': dia_bp,
        'Sleep Disorder': labels
    })


# ─────────────────────────────────────────
# REAL KAGGLE DATASET LOADER
# ─────────────────────────────────────────
REAL_CSV_CANDIDATES = [
    'data/sleep_data.csv',
    'Sleep_health_and_lifestyle_dataset.csv',
    'data/Sleep_health_and_lifestyle_dataset.csv',
    'sleep_data.csv',
]

def load_kaggle_dataset(path):
    """
    Loads and normalises the Kaggle Sleep Health & Lifestyle dataset.
    Handles:
      - Blood Pressure  "120/80"  →  Systolic BP + Diastolic BP
      - BMI Category    "Normal Weight"  →  "Normal"
      - Sleep Disorder  NaN  →  "None"
    """
    print(f'  Loading real dataset: {path}')
    df = pd.read_csv(path)
    print(f'  Raw shape: {df.shape}')
    print(f'  Columns: {list(df.columns)}')

    # ── Rename common Kaggle column variants ──
    rename_map = {
        'Quality of Sleep':        'Quality of Sleep',   # informational, not used as feature
        'Physical Activity Level': 'Physical Activity Level',
        'Stress Level':            'Stress Level',
        'BMI Category':            'BMI Category',
        'Blood Pressure':          'Blood Pressure',
        'Heart Rate':              'Heart Rate',
        'Daily Steps':             'Daily Steps',
        'Sleep Disorder':          'Sleep Disorder',
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    # ── Split Blood Pressure "systolic/diastolic" ──
    if 'Blood Pressure' in df.columns:
        bp = df['Blood Pressure'].astype(str).str.split('/', expand=True)
        df['Systolic BP']  = pd.to_numeric(bp[0], errors='coerce').fillna(120).astype(int)
        df['Diastolic BP'] = pd.to_numeric(bp[1], errors='coerce').fillna(80).astype(int)
    else:
        if 'Systolic BP'  not in df.columns: df['Systolic BP']  = 120
        if 'Diastolic BP' not in df.columns: df['Diastolic BP'] = 80

    # ── Normalise BMI labels ──
    if 'BMI Category' in df.columns:
        df['BMI Category'] = df['BMI Category'].replace({
            'Normal Weight': 'Normal',
            'normal':        'Normal',
            'overweight':    'Overweight',
            'obese':         'Obese',
        })

    # ── Fill missing Sleep Disorder ──
    if 'Sleep Disorder' in df.columns:
        df['Sleep Disorder'] = df['Sleep Disorder'].fillna('None').astype(str).str.strip()
    else:
        raise ValueError("Column 'Sleep Disorder' not found in CSV.")

    # ── Ensure required feature columns exist ──
    required = [
        'Age', 'Gender', 'Occupation', 'Sleep Duration', 'Stress Level',
        'BMI Category', 'Heart Rate', 'Daily Steps', 'Physical Activity Level',
        'Systolic BP', 'Diastolic BP', 'Sleep Disorder'
    ]
    for col in required:
        if col not in df.columns:
            print(f'  ⚠  Missing column "{col}" — filling with 0')
            df[col] = 0

    df = df[required].dropna(subset=['Sleep Disorder'])
    print(f'  Clean shape: {df.shape}')
    print(f'  Disorder distribution:\n{df["Sleep Disorder"].value_counts().to_string()}')
    return df


# ─────────────────────────────────────────
# LOAD DATASET (real or synthetic)
# ─────────────────────────────────────────
df = None
for csv_path in REAL_CSV_CANDIDATES:
    if os.path.exists(csv_path):
        try:
            df = load_kaggle_dataset(csv_path)
            print(f'\n✅ Using REAL Kaggle dataset ({len(df)} rows)\n')
            break
        except Exception as e:
            print(f'  ❌ Failed to load {csv_path}: {e}')

if df is None:
    print('\n⚠️  No real CSV found. Falling back to synthetic data.')
    print('   To use real data:')
    print('   1. Download: https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset')
    print('   2. Save as:  data/sleep_data.csv')
    print('   3. Re-run:   python train_model.py\n')
    df = make_synthetic_dataset(1200)

print(f'Dataset shape: {df.shape}')
print(df['Sleep Disorder'].value_counts())

# ─────────────────────────────────────────
# ENCODE FEATURES
# ─────────────────────────────────────────
encoders = {}
for col in ['Gender', 'Occupation', 'BMI Category']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le
    print(f'  {col} classes: {list(le.classes_)}')

target_le = LabelEncoder()
df['Sleep Disorder'] = target_le.fit_transform(df['Sleep Disorder'].astype(str))
encoders['Sleep Disorder'] = target_le
print(f'\nTarget classes: {list(target_le.classes_)}')

feature_cols = [
    'Age', 'Gender', 'Occupation', 'Sleep Duration', 'Stress Level',
    'BMI Category', 'Heart Rate', 'Daily Steps', 'Physical Activity Level',
    'Systolic BP', 'Diastolic BP'
]

X = df[feature_cols].astype(float)
y = df['Sleep Disorder']

# ─────────────────────────────────────────
# SCALE + SPLIT
# ─────────────────────────────────────────
scaler  = StandardScaler()
X_sc    = scaler.fit_transform(X)
X_tr, X_te, y_tr, y_te = train_test_split(
    X_sc, y, test_size=0.2, random_state=42, stratify=y
)
print(f'\nTrain: {len(X_tr)}  |  Test: {len(X_te)}')

# ─────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────
print('\nTraining Random Forest (200 trees, max_depth=12)...')
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_split=4,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)
model.fit(X_tr, y_tr)

# ─────────────────────────────────────────
# EVALUATE
# ─────────────────────────────────────────
y_pred = model.predict(X_te)
acc    = accuracy_score(y_te, y_pred) * 100

print(f'\n{"="*50}')
print(f'  Test Accuracy : {acc:.2f}%')
print(f'{"="*50}')
print(classification_report(y_te, y_pred, target_names=target_le.classes_))
print('Confusion Matrix:')
print(confusion_matrix(y_te, y_pred))

cv = cross_val_score(model, X_sc, y, cv=5, scoring='accuracy')
print(f'\n5-Fold CV : {cv.mean()*100:.2f}% ± {cv.std()*100:.2f}%')

print('\nFeature Importances:')
for feat, imp in sorted(
    zip(feature_cols, model.feature_importances_), key=lambda x: -x[1]
):
    bar = '█' * int(imp * 200)
    print(f'  {feat:<28} {imp*100:5.1f}%  {bar}')

# ─────────────────────────────────────────
# SAVE MODEL ARTIFACTS
# ─────────────────────────────────────────
os.makedirs('model', exist_ok=True)
artifacts = [
    ('sleep_model',  model),
    ('scaler',       scaler),
    ('encoders',     encoders),
    ('feature_cols', feature_cols),
]
for name, obj in artifacts:
    path = f'model/{name}.pkl'
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    print(f'  Saved → {path}')

print(f'\n✅ All artifacts saved to model/')
print(f'   Classes : {list(target_le.classes_)}')
print(f'   Accuracy: {acc:.2f}%')
print('\nNext step: python app.py')
