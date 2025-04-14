import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder
import SVM_best_model
import LogisticRegression
from LogisticRegression import My_Logistic_Regression
from my_models import MyGaussianNaiveBayes
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




def preprocess_and_predict_NV(output_csv='predictions.csv'):
    df = pd.read_csv('real_time_nids_features.csv')
    
    nb_model = joblib.load('models/Navie_bayes_Model/naive_bayes_model.pkl')
    scaler = joblib.load('models/Navie_bayes_Model/scaler.pkl')
    selected_features = joblib.load('models/Navie_bayes_Model/selected_features.pkl')
    encoding_dict = joblib.load('models/Navie_bayes_Model/all_label_encoders.pkl')

    # Apply safe label transform to handle unseen categories
    for col, encoder in encoding_dict.items():
        if col in df.columns:
            df[col] = safe_label_transform(encoder, df[col].astype(str))
        else:
            print(f"[Warning] Column {col} missing from input data")

    # Select and scale features
    df = df[selected_features]
    df_scaled = scaler.transform(df)

    # Predict
    predictions = nb_model.predict(df_scaled)

    # Save to CSV
    if os.path.exists(output_csv):
        combined = pd.read_csv(output_csv)
        combined["nv"] = predictions
        combined.to_csv(output_csv, index=False)
    else:
        pd.DataFrame(predictions, columns=["nv"]).to_csv(output_csv, index=False)

    # Summary
    normal = np.sum(predictions == 0)
    anomaly = np.sum(predictions == 1)
    print("✅ Predictions complete from Naive Bayes")
    print("Normal :", normal)
    print("Anomaly:", anomaly)

    return normal, anomaly
def predict_knn(output_csv='predictions.csv'):
    real_data = pd.read_csv("real_time_nids_features.csv")

    cat_cols = ['protocol_type', 'service', 'flag']

    def load_or_fit_encoder(file_name, column_name):
        if os.path.exists(file_name):
            return joblib.load(file_name)
        else:
            print(f"[Warning] '{file_name}' not found. Creating and fitting a new LabelEncoder on column '{column_name}'")
            le = LabelEncoder()
            real_data[column_name] = le.fit_transform(real_data[column_name])
            joblib.dump(le, file_name)
            return le

    def safe_label_transform(le, series):
        valid_labels = set(le.classes_)
        return series.apply(lambda x: le.transform([x])[0] if x in valid_labels else -1)

    # Encode and mark invalid rows
    for col in cat_cols:
        le = load_or_fit_encoder(f"{col}_le.pkl", col)
        real_data[col] = safe_label_transform(le, real_data[col])

    # Keep mask of valid rows (to also filter predictions.csv if needed)
    valid_mask = (real_data[cat_cols] != -1).all(axis=1)
    real_data = real_data[valid_mask]

    scaler = joblib.load("models/Knn_model/scaler.pkl")
    selected_indices = joblib.load("models/Knn_model/selected_feature_indices.pkl")

    real_data_scaled = scaler.transform(real_data.values)
    real_data_selected = real_data_scaled[:, selected_indices]

    knn_model = joblib.load("models/Knn_model/knn_model.pkl")
    predictions = knn_model.predict(real_data_selected)

    # Append predictions to output_csv if lengths match
    if os.path.exists(output_csv):
        combined = pd.read_csv(output_csv)

        # Apply the same mask to combined so rows match
        combined = combined[valid_mask.values]

        if len(combined) == len(predictions):
            combined["knn"] = predictions
            combined.to_csv(output_csv, index=False)
        else:
            print(f"❌ Length mismatch after filtering: combined={len(combined)}, predictions={len(predictions)}")
    else:
        pd.DataFrame(predictions, columns=["knn"]).to_csv(output_csv, index=False)

    label_map = {0: 'Normal', 1: 'Anomaly'}
    predicted_labels = [label_map[pred] for pred in predictions]

    print("\nPredictions:")
    print(predicted_labels)

    unique_labels, counts = np.unique(predicted_labels, return_counts=True)
    print("\nPrediction Summary:")
    for label, count in zip(unique_labels, counts):
        print(f"{label}: {count}")

if __name__ == "__main__":
    check_and_predict_SVM()
    check_and_predict_LR()
    preprocess_and_predict_NV()
    predict_knn()
