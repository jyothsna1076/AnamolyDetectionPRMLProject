import pandas as pd
import numpy as np
import joblib

encoders = joblib.load('/Users/pradeepikanori/PRML_project/models/label_encoders.pkl')
target_encoder = joblib.load('/Users/pradeepikanori/PRML_project/models/target_encoder.pkl')
scaler = joblib.load('/Users/pradeepikanori/PRML_project/models/scaler.pkl')
model = joblib.load('/Users/pradeepikanori/PRML_project/models/SVM_best_model.pkl')
feature_order = joblib.load('/Users/pradeepikanori/PRML_project/models/feature_order.pkl')

def safe_label_transform(le, col_values):
    known_classes = set(le.classes_)
    unknowns = set(col_values.unique()) - known_classes
    if unknowns:
        print(f"[Warning] Unknown categories in column: {unknowns}")
        col_values = col_values.apply(lambda x: x if x in known_classes else "unknown")
        if "unknown" not in le.classes_:
            le.classes_ = np.append(le.classes_, "unknown")
    return le.transform(col_values)

def check_and_predict(test_csv='real_time_nids_features.csv', output_csv='predictions.csv'):
    # Load test data
    test_data = pd.read_csv(test_csv)
    test_data.drop(['num_outbound_cmds'], axis=1, errors='ignore', inplace=True)

    # Apply label encoding
    for col in encoders:
        if col in test_data.columns:
            test_data[col] = safe_label_transform(encoders[col], test_data[col].astype(str))

    # Reorder columns and fill missing
    for col in feature_order:
        if col not in test_data.columns:
            test_data[col] = 0
    test_data = test_data[feature_order]

    # Scale and predict
    X_scaled = scaler.transform(test_data)
    predictions = model.predict(X_scaled)
    predicted_labels = target_encoder.inverse_transform(predictions)

    # Save predictions
    pd.DataFrame(predicted_labels, columns=["Prediction"]).to_csv(output_csv, index=False)

    # Print summary
    unique, counts = np.unique(predicted_labels, return_counts=True)
    print("=== Prediction Summary ===")
    for label, count in zip(unique, counts):
        print(f"{label}: {count}")

    if 'anomaly' in predicted_labels or np.sum(predictions == 0) > 150:
        print(">>> Anomaly Detected!")
    else:
        print(">>> Traffic appears normal.")

if __name__ == "__main__":
    check_and_predict()