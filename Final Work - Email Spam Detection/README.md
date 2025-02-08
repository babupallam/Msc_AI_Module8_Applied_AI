# **Email Spam Detection Using Machine Learning & Deep Learning**

## **Project Overview**
This project investigates the effectiveness of **machine learning** and **deep learning** models for **email spam detection**. The research compares **traditional machine learning models** (Logistic Regression, Naive Bayes, SVM) and **deep learning models** (CNN, RNN, LSTM, GRU) to determine their efficiency in filtering spam emails.

## **Repository Structure**
This repository is organized into the following main sections:

### **1. Guidelines**
Contains important reference materials and coursework guidelines.
- 📄 `CSIP5403_CEM_coursework_brief_2324.pdf` - The coursework brief.
- 📂 `conference-latex-template_10-17-19.zip` - LaTeX template for writing reports.
- 📂 `LabPlayGround/` - Contains sample Jupyter notebooks:
  - 📜 `plot_keras_digits.ipynb` - Example on Keras digits dataset.
  - 📜 `plot_ols.ipynb` - Ordinary Least Squares Regression.
  - 📜 `Q-Learning_Solution.ipynb` - Q-learning implementation.

### **2. Dataset Used**
This folder contains the datasets used for training and evaluation.
- 📜 `emails.csv` - The main dataset containing email messages and their classification (spam or not spam).
- 📂 `emails.csv.zip` - Compressed dataset.
- 📜 `spam.csv` - Additional dataset used for training.

### **3. Code Implemented**
This section contains the implementations of different **machine learning (ML)** and **deep learning (DL)** models.

#### **(A) CNN-based Models (Deep Learning)**
- 📜 `CNN_01.ipynb` - First iteration of CNN-based spam detection.
- 📜 `CNN_02.ipynb` - Improved version of CNN with additional layers.
- 📜 `CNN_03.ipynb` - Final CNN model for email spam detection.
- 📜 `README.md` - Explanation of CNN models and their evaluation.
- 📂 `solutionForThePaper.zip` - Model implementation and results.

#### **(B) ML-based Models (Traditional Algorithms)**
- 📜 `ML1-LogisticRegression.ipynb` - Logistic Regression model.
- 📜 `ML2-NaiveBayesClassifier.ipynb` - Naive Bayes model.
- 📜 `ML3-SupportVectorMachine.ipynb` - Support Vector Machine model.
- 📜 `README.md` - Documentation on traditional ML models.

#### **(C) RNN-based Models (Deep Learning)**
- 📜 `test_data.csv` - Test dataset for models.
- 📜 `ML1.png` - Performance graph for ML1 model.
- 📜 `ML2.png` - Performance graph for ML2 model.
- 📜 `ML3.png` - Performance graph for ML3 model.

### **4. Reports Submitted**
This folder contains the submitted reports related to the project.
- 📜 `01_COMMON.pdf` - Common report on spam detection.
- 📜 `02_FATIMA_ML.pdf` - Report on ML experiments by Fatima.
- 📜 `03_JOE_RNN.pdf` - Research paper on RNN models.
- 📜 `04_BABU_CNN.pdf` - Analysis of CNN models for spam detection.
- 📜 `FINAL_REPORT_TURNITIN.pdf` - Final Turnitin-submitted report.
- 📜 `FINAL_REPORT_LIVE.pdf` - Live final report submission.
- 📜 `PERFORMANCE_ANALYSIS.png` - Performance analysis results.
- 📜 `README.md` - Summary of submitted reports.

---

## **Project Details**
### **Objective**
The project focuses on **enhancing email spam detection** by comparing traditional **Machine Learning algorithms** (Logistic Regression, Naive Bayes, SVM) with **Deep Learning approaches** (CNN, RNN, LSTM, GRU). The aim is to build a **robust spam classification model** that can effectively **differentiate between spam and non-spam emails**.

### **Methodology**
1. **Data Collection & Preprocessing**
   - Datasets were obtained from **Kaggle and other sources**.
   - Text preprocessing techniques like **tokenization, TF-IDF, and embeddings** were used.
2. **Model Training & Evaluation**
   - Three **Machine Learning** models: **Logistic Regression, Naive Bayes, and SVM**.
   - Three **Recurrent Neural Network (RNN) models**: **Simple RNN, LSTM, and GRU**.
   - Three **Convolutional Neural Network (CNN) models** with different architectures.
3. **Performance Metrics**
   - **Accuracy, Precision, Recall, and F1-score** were used for model evaluation.
   - **Confusion matrices** were analyzed to understand false positives and false negatives.
4. **Best Model Selection**
   - **GRU (Recurrent Neural Network)** achieved **99% accuracy** and was found to be the most effective model.

### **Results Summary**
- **Naive Bayes** performed well in speed but had lower accuracy.
- **SVM** provided **high accuracy** but was computationally expensive.
- **GRU-based RNN** had **the best balance of accuracy and computational efficiency**.
- **CNN with Bidirectional LSTM** gave the **highest accuracy** but required **significant computational power**.

---

### **📊 Performance Analysis**
The performance of different **Machine Learning (ML), Recurrent Neural Networks (RNN), and Convolutional Neural Networks (CNN)** models was evaluated using **accuracy, precision, recall, and F1-score**.

#### **Performance Comparison Table**
Below is a comparison of the models' performance:

![Performance Analysis](PERFORMANCE%20ANALYSIS.png)

#### **Key Observations:**
- **Traditional ML Models:**  
  - **Support Vector Machine (SVM)** had the highest accuracy (**0.98**) among traditional ML models.  
  - **Naive Bayes and Logistic Regression** performed well but had slightly lower precision.  

- **Recurrent Neural Networks (RNN):**  
  - **GRU-based model achieved the highest accuracy (0.99)** with a strong balance between recall and F1-score.  
  - **LSTM performed similarly to GRU** but had a slightly lower recall.  

- **Convolutional Neural Networks (CNN):**  
  - **CNN with Bidirectional LSTM and Dropout** provided **0.986 accuracy**, indicating robust learning.  
  - **CNN with GlobalMaxPooling1D** performed well but had a **lower recall score (0.90)**.  

#### **Conclusion:**
- **Deep Learning models (CNN + LSTM, GRU) outperformed traditional ML models in spam classification.**  
- **GRU and CNN with Bidirectional LSTM achieved the highest recall and precision, making them the best models for email spam detection.**  

---

## **Key Findings**
1. **Deep Learning models outperformed Traditional ML models** in spam classification.
2. **Hybrid models (CNN + LSTM)** provided **the most robust performance**.
3. **Hyperparameter tuning (Batch Size, Learning Rate, Dropout) significantly improved model performance**.
4. **Preprocessing techniques like TF-IDF, Stopword Removal, and Tokenization impacted model accuracy**.

---

## **Future Enhancements**
- **Implementing Transfer Learning** with pre-trained NLP models like **BERT**.
- **Exploring Federated Learning** for **privacy-preserving spam detection**.
- **Real-time email spam detection system** using cloud deployment.
- **Enhancing dataset diversity** to improve generalization.

---

## **Contributors**
- **Fatima P.** (P2833125) - **Machine Learning Lead**
- **Jonathan Atiene** (P2839161) - **RNN Research**
- **Babu Pallam** (P2849288) - **CNN and Deep Learning Expert**

---

## **References**
- Kaggle Dataset: [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- TensorFlow: [https://www.tensorflow.org](https://www.tensorflow.org)
- Scikit-Learn: [https://scikit-learn.org](https://scikit-learn.org)
