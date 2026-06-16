import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import mean_absolute_error, r2_score

# =====================================
# LOAD DATA
# =====================================

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("Jumlah Data :", df.shape)
print(df.head())

# =====================================
# PREPROCESSING
# =====================================

# Menghapus kolom ID
if 'customerID' in df.columns:
    df.drop('customerID', axis=1, inplace=True)

# Mengubah TotalCharges menjadi numerik
if 'TotalCharges' in df.columns:
    df['TotalCharges'] = pd.to_numeric(
        df['TotalCharges'],
        errors='coerce'
    )

    df['TotalCharges'].fillna(
        df['TotalCharges'].median(),
        inplace=True
    )

# Encoding data kategorikal
encoder = LabelEncoder()

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = encoder.fit_transform(df[col])

# =====================================
# MODEL 1 : SVC (KLASIFIKASI CHURN)
# =====================================

X_cls = df.drop('Churn', axis=1)
y_cls = df['Churn']

X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
    X_cls,
    y_cls,
    test_size=0.2,
    random_state=42
)

svc_model = SVC(kernel='rbf')

svc_model.fit(X_train_cls, y_train_cls)

y_pred_cls = svc_model.predict(X_test_cls)

print("\n")
print("=" * 50)
print("HASIL KLASIFIKASI SVC")
print("=" * 50)

print("Accuracy :",
      accuracy_score(y_test_cls, y_pred_cls))

print("\nClassification Report:")
print(classification_report(y_test_cls,
                            y_pred_cls))

# =====================================
# MODEL 2 : LINEAR REGRESSION
# PREDIKSI MONTHLY CHARGES
# =====================================

X_reg = df.drop('MonthlyCharges', axis=1)
y_reg = df['MonthlyCharges']

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg,
    y_reg,
    test_size=0.2,
    random_state=42
)

reg_model = LinearRegression()

reg_model.fit(X_train_reg, y_train_reg)

y_pred_reg = reg_model.predict(X_test_reg)

print("\n")
print("=" * 50)
print("HASIL LINEAR REGRESSION")
print("=" * 50)

print("MAE :",
      mean_absolute_error(y_test_reg,
                          y_pred_reg))

print("R2 Score :",
      r2_score(y_test_reg,
               y_pred_reg))