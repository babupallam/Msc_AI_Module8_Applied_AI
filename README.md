# **MSc AI - Module 8: Applied AI (DMU)** 🚀  

## **📌 Overview**  
This repository is part of the **MSc Artificial Intelligence program at De Montfort University (DMU)** and contains all coursework, research, and practical implementations related to **Applied AI (Module 8)**.  

The primary focus of this project is **Email Spam Detection**, where we explore the **effectiveness of traditional Machine Learning (ML) models** (Logistic Regression, Naive Bayes, SVM) compared to **Deep Learning architectures** (CNN, RNN, LSTM, GRU) in accurately classifying spam emails.  

---

## **📂 Repository Structure**  

```
📁 Final Work - Email Spam Detection/
│── 📁 01_Guidelines/ → Coursework briefs, LaTeX templates, research papers
│── 📁 02_Dataset Used/ → Emails dataset (CSV files)
│── 📁 03_Code_Implemented/ → Machine Learning & Deep Learning models
│   │── 📁 ESD_CNN/ → CNN-based spam detection models
│   │── 📁 ESD_ML/ → Traditional ML models (Logistic Regression, Naive Bayes, SVM)
│── 📁 04_Report Submitted/ → Final project reports, research papers
│── 📁 LEC 01 - LEC 07/ → Lecture slides & practical works
│── 📄 README.md → This documentation file
```

---

## **📚 Course Breakdown**  

### 📂 **Lecture Materials & Practical Work**  
Each lecture folder (LEC 01 - LEC 07) contains:  
✔ **Lecture slides (PDFs) covering key concepts**  
✔ **Practical exercises & solutions** related to spam detection models  

### 📂 **Final Work - Email Spam Detection**  
This section contains all project-related files:  

#### **📖 01_Guidelines/**  
- 📄 `CSIP5403_CEM_coursework_brief.pdf` - Coursework details  
- 📜 `conference-latex-template.zip` - Research paper template  

#### **📊 02_Dataset Used/**  
- 📜 `emails.csv` - Dataset used for training spam detection models  
- 📜 `spam.csv` - Additional dataset  

#### **🛠 03_Code_Implemented/**  
- 📁 `ESD_CNN/` → CNN-based models (`CNN_01.ipynb`, `CNN_02.ipynb`, `CNN_03.ipynb`)  
- 📁 `ESD_ML/` → Traditional ML models (`Logistic Regression.ipynb`, `NaiveBayes.ipynb`, `SVM.ipynb`)  

#### **📑 04_Report Submitted/**  
- 📄 `FINAL_REPORT_LIVE.pdf` - Final research report  
- 📄 `FINAL_REPORT_TURNITIN.pdf` - Plagiarism-checked report  
- 📊 `PERFORMANCE_ANALYSIS.png` - Model performance comparison  

---

## **🛠 Implementation & Usage**  

### **1️⃣ Clone the Repository**  
```sh
git clone https://github.com/babupallam/Msc_AI_Module8_Applied_AI.git

```

### **2️⃣ Install Dependencies**  
```sh
pip install -r requirements.txt
```

### **3️⃣ Run ML Models**  
```sh
python models/logistic_regression.py
```

### **4️⃣ Run Deep Learning Models**  
```sh
jupyter notebook
```
Open and execute the required `.ipynb` notebook.

---

## **📊 Performance Analysis**  
Different ML and DL models were evaluated on **accuracy, precision, recall, and F1-score**.  

**📌 Best Model:**  
✔ **GRU-based RNN achieved the highest accuracy (0.99)**  
✔ **CNN with Bidirectional LSTM** performed best among CNN models  

![Performance Analysis](Final%20Work%20-%20Email%20Spam%20Detection/PERFORMANCE%20ANALYSIS.png)

---

## **📌 Key Findings & Future Enhancements**  
✅ **Deep Learning models outperformed Traditional ML models in spam classification**  
✅ **GRU and CNN with Bidirectional LSTM achieved the best recall and precision**  
✅ **Hyperparameter tuning improved model performance significantly**  

🚀 **Future Work:**  
🔹 Implement **BERT for email spam detection**  
🔹 Deploy **real-time spam detection API**  
🔹 Train models on **larger and multilingual datasets**  

---

## **📢 Contributors**  
👤 **Babu Pallam** (P2849288) - Deep Learning & CNN Research  
👤 **Fatima P.** (P2833125) - Machine Learning & Performance Analysis  
👤 **Jonathan Atiene** (P2839161) - RNN Model Development  
