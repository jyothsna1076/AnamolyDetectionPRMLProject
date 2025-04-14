import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import PolynomialFeatures
import SVM_best_model
import LogisticRegression
from LogisticRegression import My_Logistic_Regression
import os
encoders = joblib.load('models/SVM_models/label_encoders.pkl')
target_encoder = joblib.load('models/SVM_models/target_encoder.pkl')
scaler = joblib.load('models/SVM_models/scaler.pkl')
model = joblib.load('models/SVM_models/SVM_best_model.pkl')
feature_order = joblib.load('models/SVM_models/feature_order.pkl')

def safe_label_transform(le, col_values):
    known_classes = set(le.classes_)
    unknowns = set(col_values.unique()) - known_classes
    if unknowns:
        print(f"[Warning] Unknown categories in column: {unknowns}")
        col_values = col_values.apply(lambda x: x if x in known_classes else "unknown")
        if "unknown" not in le.classes_:
            le.classes_ = np.append(le.classes_, "unknown")
    return le.transform(col_values)

def check_and_predict_SVM(test_csv='real_time_nids_features.csv', output_csv='predictions.csv'):
    test_data = pd.read_csv(test_csv)
    test_data.drop(['num_outbound_cmds'], axis=1, errors='ignore', inplace=True)

    for col in encoders:
        if col in test_data.columns:
            test_data[col] = safe_label_transform(encoders[col], test_data[col].astype(str))

    for col in feature_order:
        if col not in test_data.columns:
            test_data[col] = 0
    test_data = test_data[feature_order]

    X_scaled = scaler.transform(test_data)
    predictions = model.predict(X_scaled)
    predicted_labels = target_encoder.inverse_transform(predictions.astype(int))

    # Save SVM predictions
    pd.DataFrame(predicted_labels, columns=["svm"]).to_csv(output_csv, index=False)

    unique, counts = np.unique(predicted_labels, return_counts=True)
    print("=== Prediction Summary from SVM===")
    for label, count in zip(unique, counts):
        print(f"{label}: {count}")

    if 'anomaly' in predicted_labels or np.sum(predictions == 0) > 150:
        print(">>> Anomaly Detected!")
    else:
        print(">>> Traffic appears normal.")


def check_and_predict_LR(output_csv='predictions.csv'):
    clf = joblib.load("models/LR_models/classifier.pkl")
    le_protocol = joblib.load("models/LR_models/le_protocol.pkl")
    le_service = joblib.load("models/LR_models/le_service.pkl")
    le_flag = joblib.load("models/LR_models/le_flag.pkl")
    scaler = joblib.load("models/LR_models/scaler.pkl")

    try:
        df = pd.read_csv("real_time_nids_features.csv")
        print("✅ Data read")

        df["protocol_type"] = safe_label_transform(le_protocol, df["protocol_type"])
        df["service"] = safe_label_transform(le_service, df["service"])
        df["flag"] = safe_label_transform(le_flag, df["flag"])

        feature_cols = [
            "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
            "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
            "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
            "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
            "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
            "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
            "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
            "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
            "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate"
        ]

        X = df[feature_cols]
        X_scaled = scaler.transform(X)
        X_trf = PolynomialFeatures(degree=1).fit_transform(X_scaled)
        preds = clf.predict(X_trf)

        pred_labels = ["normal" if p == 0 else "anomaly" for p in preds]

        # Append to existing predictions.csv
        if os.path.exists(output_csv):
            combined = pd.read_csv(output_csv)
            combined["lr"] = pred_labels
            combined.to_csv(output_csv, index=False)
        else:
            pd.DataFrame(pred_labels, columns=["lr"]).to_csv(output_csv, index=False)

        normal_count = pred_labels.count("normal")
        anomaly_count = pred_labels.count("anomaly")

        print("✅ Predictions complete from Logistic Regression")
        print("Normal : ", normal_count)
        print("Anomaly: ", anomaly_count)
        return df, normal_count, anomaly_count

    except Exception as e:
        print("❌ Error during prediction:", e)
        raise

if __name__ == "__main__":
    check_and_predict_SVM()
    check_and_predict_LR()