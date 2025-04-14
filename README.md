# Anamolyse
# 🚨 Network Anomaly Detection using Machine Learning

A Comparative Study of Supervised and Unsupervised Models for Intrusion Detection  
**Course Project | Pattern Recognition and Machine Learning (CSL2050)**  
**IIT Jodhpur**

## 📌 Overview

This project aims to detect anomalies in network traffic using various machine learning models. We explore and compare six popular algorithms—Logistic Regression, SVM, Random Forest, KNN, Gaussian Mixture Model (GMM), and Naive Bayes—on a benchmark intrusion detection dataset.

Our goal: **Build an accurate, robust, and scalable Intrusion Detection System (IDS)**.

## ✨ Key Highlights

- ✅ Binary classification: **Normal vs Malicious**
- 📊 Performance evaluated using **Accuracy, Precision, Recall, F1-score**
- 🚀 Best performing model: **Random Forest (99.5% Accuracy)**
- 📁 Dataset: Kaggle Network Intrusion Detection Dataset  
  [Link](https://www.kaggle.com/datasets/sampadab17/network-intrusion-detection)

---

## 🧠 Models Implemented

| Model                | Accuracy | Precision | Recall | F1-Score | Training Time |
|---------------------|----------|-----------|--------|----------|----------------|
| Logistic Regression | 92.75%   | 95%       | 91%    | 93%      | 4.79 sec       |
| SVM (RBF Kernel)    | 91.2%    | 90.5%     | 89.8%  | 90.1%    | Moderate       |
| Random Forest       | 99.5%    | 99.5%     | 99.5%  | 99.5%    | 0.84 sec       |
| KNN (k=5)           | 88.4%    | 87.2%     | 86.7%  | 86.9%    | Slow           |
| GMM                 | 83.1%    | 81.4%     | 82.0%  | 81.7%    | Moderate       |
| Naive Bayes         | 90.63%   | 88%       | 96%    | 92%      | 0.02 sec       |

---

## 🧪 Dataset Details

- 📦 **Total Samples**: 25,192  
- 🔢 **Features**: 42 (3 categorical, 38 numerical, 1 target)
- 🎯 **Target Classes**: `normal`, `anomaly`

### 🧹 Preprocessing Steps:
- Label encoding for categorical features
- StandardScaler for feature scaling
- Stratified Train-Test Split (80:20 and 70:30)

---

## 🛠️ Setup Instructions

```bash
git clone https://github.com/jyothsna1076/AnamolyDetectionPRMLProject.git
cd AnamolyDetectionPRMLProject
pip install -r requirements.txt
```

> 📌 Note: Ensure Python 3.7+ is installed.

---

## 📂 Project Structure

```bash
AnamolyDetectionPRMLProject/
├── data/                   # Dataset files
├── notebooks/              # Jupyter notebooks per model
│   ├── logistic_regression.ipynb
│   ├── svm_model.ipynb
│   ├── random_forest.ipynb
│   ├── knn_model.ipynb
│   ├── gmm_model.ipynb
│   └── naive_bayes.ipynb
├── preprocessing.py        # Data preprocessing pipeline
├── utils.py                # Helper functions
├── results/                # Graphs, reports
├── README.md               # Project overview
└── requirements.txt        # Package dependencies
```

---

## 👥 Team Members & Contributions

| Name                | Contribution                                |
|---------------------|---------------------------------------------|
| Vadlamudi Jyothsna | Random Forest, project documentation        |
| Pradeepika Nori    | SVM implementation, UI integration          |
| Nishu Verma         | Logistic Regression                         |
| BhagyaShree         | KNN implementation                          |
| Reshma Maurya       | Naive Bayes implementation                  |
| Nagma Saj           | Gaussian Mixture Model implementation       |

---

## 📈 Results & Conclusion

Random Forest consistently outperformed other models in both accuracy and generalization. Despite its simplicity, Logistic Regression also performed competitively. Models like KNN and GMM lagged behind in inference time and assumptions.

> ✅ **Takeaway**: Ensemble methods like Random Forest are powerful for IDS tasks due to their robustness and low overfitting tendencies.

---

## 📜 License

This project is for academic use under the **MIT License**.

---

## 📞 Contact

For queries or collaborations, reach out to:  
📧 b23cs1076@iitj.ac.in  
📧 b23cs1007@iitj.ac.in  
📧 b23cs1034@iitj.ac.in  
📧 b23cs1045@iitj.ac.in
📧 b23cs1047@iitj.ac.in
📧 b23ee1043@iitj.ac.in  


