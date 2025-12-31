from datetime import datetime
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


class MachineLearning:

    def __init__(self):
        print("[+] Loading dataset ...")

        # =========================
        # 1. LOAD DATASET
        # =========================
        self.dataset = pd.read_csv("FlowStatsfile.csv")

        # =========================
        # 2. SELECT 10 FEATURES
        # =========================
        self.feature_columns = [
            'flow_duration_sec',
            'flow_duration_nsec',
            'ip_src',
            'ip_dst',
            'tp_src',
            'tp_dst',
            'ip_proto',
            'packet_count',
            'byte_count',
            'packet_count_per_second'
        ]

        self.label_column = 'label'

        self.dataset = self.dataset[self.feature_columns + [self.label_column]]

        # =========================
        # 3. PREPROCESSING
        # =========================
        self.preprocessing()

        # =========================
        # 4. TRAIN / TEST SPLIT
        # =========================
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y,
            test_size=0.25,
            random_state=42,
            stratify=self.y
        )

        print("[+] Dataset ready")
        print("    Training samples:", len(self.X_train))
        print("    Testing samples :", len(self.X_test))

    def preprocessing(self):
        print("[+] Preprocessing data ...")

        # ---- IP address: remove dots ----
        self.dataset['ip_src'] = self.dataset['ip_src'].astype(str).str.replace('.', '', regex=False)
        self.dataset['ip_dst'] = self.dataset['ip_dst'].astype(str).str.replace('.', '', regex=False)

        # ---- Convert to numeric ----
        for col in self.feature_columns:
            self.dataset[col] = pd.to_numeric(self.dataset[col], errors='coerce')

        # ---- Handle NaN & inf ----
        self.dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.dataset.fillna(0, inplace=True)

        # ---- Encode label ----
        encoder = LabelEncoder()
        self.dataset[self.label_column] = encoder.fit_transform(self.dataset[self.label_column])
        # Normal = 0, Attack = 1

        # ---- Feature scaling ----
        self.X = self.dataset[self.feature_columns].values
        self.y = self.dataset[self.label_column].values

        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X)

    # ==================================================
    # EVALUATION
    # ==================================================
    def evaluate(self, model, name):
        print("\n" + "=" * 70)
        print("[+] Model:", name)

        start = datetime.now()
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        end = datetime.now()

        print("Confusion Matrix:")
        print(confusion_matrix(self.y_test, y_pred))

        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred, target_names=["Normal", "Attack"]))

        acc = accuracy_score(self.y_test, y_pred)
        print(f"Accuracy: {acc * 100:.2f}%")
        print("Time:", end - start)
        print("=" * 70)

    # ==================================================
    # MODELS
    # ==================================================
    def LR(self):
        self.evaluate(LogisticRegression(solver='liblinear'), "Logistic Regression")

    def KNN(self):
        self.evaluate(KNeighborsClassifier(n_neighbors=5), "KNN")

    def NB(self):
        self.evaluate(GaussianNB(), "Naive Bayes")

    def DT(self):
        self.evaluate(DecisionTreeClassifier(criterion='entropy'), "Decision Tree")

    def RF(self):
        self.evaluate(RandomForestClassifier(n_estimators=100, n_jobs=-1), "Random Forest")


# ==================================================
# MAIN
# ==================================================
def main():
    start = datetime.now()

    ml = MachineLearning()
    ml.LR()
    ml.KNN()
    ml.NB()
    ml.DT()
    ml.RF()

    print("\n[+] Total time:", datetime.now() - start)


if __name__ == "__main__":
    main()
