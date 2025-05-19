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


🛠️ Setup & Usage Instructions (Windows)


1.✅ Prerequisites
Python 3.7+ must be installed.

For Windows, ensure:

Npcap is installed.

Sudo Mode and Developer Mode are enabled:

Go to Settings → System → For Developers

Enable Developer Mode and Sudo Mode

2.📦 Environment Setup
Create a virtual environment and install dependencies:

```bash
# Clone the repository
git clone https://github.com/jyothsna1076/AnamolyDetectionPRMLProject.git
cd AnamolyDetectionPRMLProject
```

For Windows :
```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate
```

For Linux :
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate
```

```bash
# Install required packages
pip install -r requirements.txt
```

3.🚀 Running the Application

For Windows :
```bash
# Run the app
python app.py
```

For Linux :
```bash
# Run the app(sudo mode is required)
sudo python3 app.py
```

- Open your browser and go to: http://127.0.0.1:5000/
- The web interface will load.

4. 🌐 Using the Web Interface
   
Option A: 📡 Real-Time Traffic Capture
Click the "Capture Real Traffic" button on the webpage.

Wait approximately 1-1.5 minute for real-time anomaly detection results to appear.

Option B: 📁 Upload CSV Test Data
Upload the file Test_data.csv provided in the repository.

This file contains a large dataset — please wait up to 5 minutes for results to process.

⚠️ Do not modify any backend files (Python, HTML, etc.) while the app is running in the browser — doing so may interrupt processing.

📎 Notes
For Windows: Make sure Npcap is installed before running the app.

Ensure all developer settings are correctly enabled if you're using Windows.

The app uses live network traffic, so admin/sudo privileges are necessary.

---

## 📂 Project Structure

```bash
AnamolyDetectionPRMLProject/
├── jupyter_files/              
│   ├── BGMM_model.ipynb
│   ├── gaussian_naive_bayes.ipynb
│   ├── NIDS_regression.ipynb
│   ├── knn_model.ipynb
│   ├── gmm_model.ipynb
│   ├── naive_bayes.ipynb
├── MidProjectReport
|   ├── MidSemReportPRML.pdf
|   ├── prml_mid_project.ipynb     
├── python_models
|   ├── BGMM_model.py
|   ├── LogisticRegression.py
|   ├── model_knn.py
|   ├── Model.py
|   ├── randomforestprml.py
|   ├── SVM_best_model.py      
├── app.py
├── capture_script.py
├── index.html
├── predict1.py                
├── README.md
└── requirements.txt
├── Train_data.csv
├── Test_data.csv
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


