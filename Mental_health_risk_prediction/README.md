#  Early Mental Health Risk Prediction System

##  Description

This project builds a Machine Learning system to predict early mental health risk in students based on academic, social, and psychological factors.  
The model classifies students into **Low Risk** or **High Risk** categories and provides probability-based predictions through a Streamlit web application.

---

##  What This Project Does

- Performs data preprocessing and feature engineering
- Creates a binary risk target using psychological indicators
- Trains and compares Logistic Regression and Random Forest models
- Selects the best model based on F1-score
- Deploys the trained model using Streamlit for real-time prediction
- Displays prediction probability and feature importance

---

##  Technologies Used

- Python
- Pandas
- Scikit-learn
- Streamlit
- Matplotlib / Seaborn
- Pickle (for saving the trained model)

---

##  Model Performance

Best Model: **Logistic Regression**

- Accuracy: ~80%
- F1-score: ~0.73
- Precision: ~0.80
- Recall: ~0.67

Model selected based on highest F1-score using stratified train-test split.

---

##  Project Structure

- `train_model.py` → Model training and evaluation  
- `app.py` → Streamlit deployment app  
- `preprocessing.py` → Data preprocessing functions  
- `eda.py` → Exploratory data analysis  
- `model.pkl` → Saved trained model  
- `metrics.json` → Saved model comparison results  

---

## ▶ How to Run

1. Train the model:  python train_model.py
2. Run the Streamlit app: streamlit run app.py
