import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import joblib
from custom_knn import CustomKNN  # ← import from the new file

# 1. Load dataset
df = pd.read_csv('train_data.csv')

# 2. Convert 'class' column to numeric
df['class'] = pd.to_numeric(df['class'].map({'normal': 0, 'anomaly': 1}), errors='coerce')
df.dropna(subset=['class'], inplace=True)

# 3. Encode categorical columns
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].astype('category').cat.codes

# 4. Separate features and labels
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
feature_names = df.columns[:-1].tolist()

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Normalize using MinMaxScaler
scaler = MinMaxScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)
joblib.dump(scaler, "scaler.pkl")

# 7. Mandatory features to always include
mandatory_features = ['duration', 'src_bytes', 'protocol_type', 'service', 'flag']
mandatory_indices = [feature_names.index(f) for f in mandatory_features]

# 8. Compute variance for all features (excluding mandatory ones)
all_indices = set(range(len(feature_names)))
non_mandatory_indices = list(all_indices - set(mandatory_indices))
variances = np.var(X_train_norm, axis=0)

# 9. Select top 20 from non-mandatory
sorted_non_mandatory = sorted(non_mandatory_indices, key=lambda i: variances[i], reverse=True)
top_20_indices = sorted_non_mandatory[:20]

# 10. Combine with mandatory
selected_indices = sorted(mandatory_indices + top_20_indices)
X_train_sel = X_train_norm[:, selected_indices]
X_test_sel = X_test_norm[:, selected_indices]
joblib.dump(selected_indices, "selected_feature_indices.pkl")

# 11. Train Custom KNN
k_values = range(1, 2)
accuracies = []

for k in k_values:
    knn = CustomKNN(k=k)
    knn.fit(X_train_sel, y_train)
    y_pred = knn.predict(X_test_sel)
    joblib.dump(knn, 'knn_model.pkl')  # Now this will work!
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    print(f"K={k}, Accuracy={acc:.4f}")

# 12. Accuracy Plot
plt.figure(figsize=(10, 5))
plt.plot(k_values, accuracies, marker='o')
plt.title('Accuracy vs K (Mandatory + Top 20 Variance Features)')
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()

# 13. Best K
best_k = k_values[np.argmax(accuracies)]
print(f"\nBest K: {best_k}, Accuracy: {max(accuracies):.4f}")

# 14. Map labels
label_map = {0: 'Normal', 1: 'Anomaly'}
y_test_labels = np.array([label_map[label] for label in y_test])
from collections import Counter
label_counts = Counter(y_test_labels)

print("\nTest Data Label Counts:")
for label, count in label_counts.items():
    print(f"{label}: {count}")
