# 🛌 SleepSense AI – Sleep Disorder Prediction System

A full-stack Machine Learning web application that predicts sleep disorders using clinical and lifestyle data. The application leverages a **Random Forest Classifier** trained on the **Sleep Health and Lifestyle Dataset** to provide accurate predictions through a user-friendly web interface.

The system features **secure user authentication**, **role-based access control (Admin, Doctor, and Patient)**, **prediction history tracking**, **dashboard analytics**, and **RESTful APIs**. Built with **Flask**, **Scikit-learn**, and **SQLite**, SleepSense AI demonstrates an end-to-end machine learning workflow—from data preprocessing and model training to deployment in a production-ready web application.

## 🚀 Features

### 🤖 Machine Learning
- Predicts sleep disorders using a trained **Random Forest Classifier**
- Performs data preprocessing with **Label Encoding** and **StandardScaler**
- Uses serialized ML models (`.pkl`) for fast and consistent predictions

### 👥 User Management
- Secure user registration and login
- Password hashing using **Flask-Bcrypt**
- Role-based access control for **Admin, Doctor, and Patient**

### 📊 Dashboard & Analytics
- Interactive admin dashboard
- View prediction statistics and user information
- Monitor system activity and prediction history

### 🩺 Prediction System
- Predicts sleep disorders from clinical and lifestyle data
- Displays prediction results through an intuitive web interface
- Stores prediction history for future reference

### 🌐 Web Application
- Flask-based backend
- Responsive HTML, CSS, and JavaScript frontend
- RESTful API architecture
- SQLite database integration using SQLAlchemy

### 📁 Additional Features
- User profile management
- Prediction history tracking
- CSV data export
- Clean and modular project structure

## 🛠 Tech Stack

### Programming Language
- Python

### Machine Learning
- Scikit-learn
- Random Forest Classifier
- Pandas
- NumPy
- Joblib

### Backend
- Flask
- Flask-Bcrypt
- SQLAlchemy
- Flask-CORS

### Database
- SQLite

### Frontend
- HTML5
- CSS3
- JavaScript

### Development Tools
- Git & GitHub
- VS Code
- Jupyter Notebook
- Postman

 ## 🤖 Machine Learning Workflow

1. **Dataset Collection**
   - Used the Sleep Health and Lifestyle Dataset containing clinical and lifestyle attributes.

2. **Data Preprocessing**
   - Handled missing values
   - Encoded categorical features using Label Encoding
   - Scaled numerical features using StandardScaler

3. **Model Training**
   - Split the dataset into training and testing sets
   - Trained a Random Forest Classifier for sleep disorder prediction

4. **Model Evaluation**
   - Evaluated model performance using accuracy and classification metrics

5. **Model Deployment**
   - Serialized the trained model using Joblib (`.pkl`)
   - Integrated the model into a Flask web application for real-time predictions

## 📂 Project Structure

```text
SleepSense-AI/
│
├── app.py                     # Main Flask application
├── train_model.py             # Machine learning model training script
├── requirements.txt           # Project dependencies
├── schema.sql                 # Database schema
├── sleep_disorder.db          # SQLite database
├── README.md                  # Project documentation
│
├── model/
│   ├── sleep_model.pkl        # Trained Random Forest model
│   ├── scaler.pkl             # StandardScaler object
│   ├── encoders.pkl           # Label encoders
│   └── feature_cols.pkl       # Feature column information
│
├── templates/
│   ├── index.html
│   └── admin_panel.html
│
├── static/
│
└── data/
```
## 📊 Dataset

This project uses the **Sleep Health and Lifestyle Dataset** from Kaggle to predict sleep disorders based on clinical and lifestyle factors.

### Dataset Features

- Gender
- Age
- Occupation
- Sleep Duration
- Quality of Sleep
- Physical Activity Level
- Stress Level
- BMI Category
- Blood Pressure
- Heart Rate
- Daily Steps
- Sleep Disorder (Target)

> **Note:** The dataset is not included in this repository. You can download it from Kaggle and place it inside the `data/` folder before training the model.

## ⚙️ Installation

### Clone the Repository

```bash
git clone https://github.com/khan2234mehak/sleep-disorder-prediction.git

cd sleep-disorder-prediction
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
python app.py
```

The application will start on:

```text
http://127.0.0.1:5000/
```
## 👩‍💻 Author

**Mehak Khan**

📧 Email: mehakkhan020503@gmail.com

🔗 LinkedIn: https://linkedin.com/in/mehak-khan-a08965354

💻 GitHub: https://github.com/khan2234mehak
