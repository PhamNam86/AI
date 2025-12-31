import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Xử lý dữ liệu
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Mô hình Logistic Regression
from sklearn.linear_model import LogisticRegression

# Đánh giá mô hình
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_curve, auc, confusion_matrix
)
# 1. Load dữ liệu

df = pd.read_csv("data/dataset.csv")
df.columns = df.columns.str.strip()  # Remove leading/trailing spaces
print("Head of dataset:\n", df.head(3))
# 2. Kiểm tra missing values

plt.figure(figsize=(10,4))
plt.hist(df.isna().sum())
plt.title('Columns with Null Values')
plt.xlabel('Feature')
plt.ylabel('Number of features')
plt.show()
# 3. Xử lý missing values và inf

data_f = df.copy()
data_f.replace([np.inf, -np.inf], np.nan, inplace=True)  # inf → NaN
data_f.dropna(inplace=True)  # loại bỏ các hàng có Na

print("Hoàn thành!!!")
# 4. Chuyển Label sang 0/1

# Map giá trị Label và loại bỏ các giá trị lạ
data_f['Label'] = data_f['Label'].map({'BENIGN': 0, 'DDoS': 1})
data_f.dropna(subset=['Label'], inplace=True)  # nếu có NaN sau map
data_f['Label'] = data_f['Label'].astype(int)

# Histogram các lớp
plt.hist(data_f['Label'], bins=[-0.5, 0.5, 1.5], edgecolor='black')
plt.xticks([0, 1], labels=['BENIGN=0', 'DDoS=1'])
plt.xlabel("Classes")
plt.ylabel("Count")
plt.show()
# 5. Tách feature và target
# --------------------------
X = data_f.drop('Label', axis=1)
y = data_f['Label']

print("Hoàn thành!!!")

# 6. Split train/test
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
print("Train dataset size =", X_train.shape)
print("Test dataset size =", X_test.shape)

# 7. Chuẩn hóa dữ liệu
# --------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Hoàn thành!!!")

# 8. Train Logistic Regression
# --------------------------
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)
lr_prob = lr_model.predict_proba(X_test_scaled)[:, 1]  # xác suất lớp 1

print("Hoàn thành!!!")

# 9. Đánh giá mô hình
# --------------------------
lr_accuracy = accuracy_score(y_test, lr_pred)
lr_f1 = f1_score(y_test, lr_pred)
lr_precision = precision_score(y_test, lr_pred)
lr_recall = recall_score(y_test, lr_pred)

print('\nLogistic Regression Metrics:')
print(f'Accuracy: {lr_accuracy:.4f}')
print(f'F1 Score: {lr_f1:.4f}')
print(f'Precision: {lr_precision:.4f}')
print(f'Recall: {lr_recall:.4f}')

# 10. Confusion Matrix
# --------------------------
cm = confusion_matrix(y_test, lr_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

plt.xlabel("Giá trị dự đoán")
plt.ylabel("Giá trị thực tế")
plt.title("Confusion Matrix")
plt.show()

# --------------------------
# 1. Load dữ liệu test
# --------------------------
df_test = pd.read_csv("datatest.csv")
df_test.columns = df_test.columns.str.strip()  # loại bỏ khoảng trắng

# --------------------------
# 2. Xử lý missing values và inf
# --------------------------
df_test.replace([np.inf, -np.inf], np.nan, inplace=True)
df_test.dropna(inplace=True)

# --------------------------
# 3. Kiểm tra Label nếu có
# --------------------------
if 'Label' in df_test.columns:
    df_test['Label'] = df_test['Label'].map({'BENIGN': 0, 'DDoS': 1})
    df_test.dropna(subset=['Label'], inplace=True)
    df_test['Label'] = df_test['Label'].astype(int)

# --------------------------
# 4. Tách features
# --------------------------
X_new = df_test.drop('Label', axis=1, errors='ignore')

# --------------------------
# 5. Chuẩn hóa dữ liệu test
# --------------------------
X_new_scaled = scaler.transform(X_new)

# --------------------------
# 6. Dự đoán nhãn
# --------------------------
y_new_pred = lr_model.predict(X_new_scaled)
y_new_prob = lr_model.predict_proba(X_new_scaled)[:, 1]

# --------------------------
# 7. Đánh giá nếu có nhãn gốc
# --------------------------
if 'Label' in df_test.columns:
    y_new_true = df_test['Label']
    acc = accuracy_score(y_new_true, y_new_pred)
    f1 = f1_score(y_new_true, y_new_pred)
    prec = precision_score(y_new_true, y_new_pred)
    rec = recall_score(y_new_true, y_new_pred)

    print('\nTest Metrics:')
    print(f'Accuracy: {acc:.4f}, F1: {f1:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}')

    # Confusion Matrix
    cm_new = confusion_matrix(y_new_true, y_new_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm_new, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix on Test Data")
    plt.show()

# --------------------------
# 8. Thêm kết quả và xuất file
# --------------------------
df_test['Predicted_Label'] = y_new_pred
df_test['Predicted_Prob'] = y_new_prob

# Thay số bằng tên lớp
df_test['Predicted_Label'] = df_test['Predicted_Label'].map({0: 'BENIGN', 1: 'DDoS'})

# Nếu có Label gốc thì thêm cột so sánh True/False
if 'Label' in df_test.columns:
    df_test['True_Label'] = df_test['Label'].map({0: 'BENIGN', 1: 'DDoS'})
    df_test['Correct'] = df_test['True_Label'] == df_test['Predicted_Label']

# Xuất file CSV
output_path = "datatest_predicted.csv"
df_test.to_csv(output_path, index=False)
print(f"\n✅ Kết quả dự đoán đã được lưu tại: {output_path}")
