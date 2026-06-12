# naive_bayes_project

🌸 Iris Flower Classification using Naive Bayes
📌 Project Overview
This project demonstrates the implementation of a Gaussian Naive Bayes Classification Model using the famous Iris Dataset. The model is trained to classify iris flowers into one of the following species:

Iris Setosa

Iris Versicolor

Iris Virginica

The project includes:

Data Preprocessing

Feature Selection

Model Training

Model Evaluation

User Prediction

Model Serialization using Pickle

🚀 Technologies Used
Python

NumPy

Pandas

Scikit-Learn

Pickle

📂 Project Structure
├── Iris.csv
├── navie_bayes.py
├── navie_bayes.pkl
├── requirements.txt
└── README.md
📊 Dataset Information
The project uses the Iris Dataset, which contains 150 flower samples.

Features
Feature	Description
Sepal Length	Length of Sepal (cm)
Sepal Width	Width of Sepal (cm)
Petal Length	Length of Petal (cm)
Petal Width	Width of Petal (cm)
Target Variable
Species	Encoded Value
Iris-setosa	0
Iris-versicolor	1
Iris-virginica	2
⚙️ Data Preprocessing
The following preprocessing steps are performed:

1. Missing Value Check
self.df.isnull().sum()
2. Remove Unnecessary Column
self.df.drop(['Id'], axis=1)
3. Label Encoding
{
    'Iris-setosa': 0,
    'Iris-versicolor': 1,
    'Iris-virginica': 2
}
🧠 Model Training
The model used is:

GaussianNB()
Train-Test Split
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=3
)
Training Data: 80%

Testing Data: 20%

📈 Model Evaluation
The model performance is evaluated using:

Confusion Matrix
confusion_matrix()
Classification Report
classification_report()
Metrics included:

Precision

Recall

F1-Score

Accuracy

🔍 Sample Prediction
The model predicts flower species based on the following sample values:

Sepal Length = 7.8
Sepal Width  = 6.7
Petal Length = 5.3
Petal Width  = 5.2
Prediction:

self.reg.predict([[7.8, 6.7, 5.3, 5.2]])
💾 Model Serialization
The trained model is saved using Pickle.

with open("navie_bayes.pkl", "wb") as f:
    pickle.dump(self.reg, f)
This allows the model to be reused later without retraining.

▶️ How to Run the Project
Step 1: Clone Repository
git clone https://github.com/yourusername/iris-naive-bayes.git
Step 2: Navigate to Project Folder
cd iris-naive-bayes
Step 3: Install Dependencies
pip install -r requirements.txt
Step 4: Run the Script
python navie_bayes.py
📦 Requirements
numpy
pandas
scikit-learn
Install using:

pip install numpy pandas scikit-learn
🎯 Project Workflow
Load Dataset
      ↓
Check Missing Values
      ↓
Drop ID Column
      ↓
Encode Target Labels
      ↓
Split Dataset
      ↓
Train Gaussian Naive Bayes Model
      ↓
Evaluate Model
      ↓
Predict New Data
      ↓
Save Model (.pkl)
📚 Learning Outcomes
By completing this project, you will understand:

Classification Problems

Naive Bayes Algorithm

Gaussian Naive Bayes

Data Preprocessing

Train-Test Splitting

Confusion Matrix Analysis

Classification Reports

Model Serialization with Pickle

Machine Learning Workflow

🌟 Future Enhancements
Build a Flask Web Application

Deploy on Render
**https://naive-bayes-project.onrender.com**

Add Interactive UI

Accept User Inputs through Web Forms

Visualize Results using Matplotlib

👨‍💻 Author
Bharathi Vennela

Machine Learning & Python Enthusiast

⭐ If you found this project useful, don't forget to star the repository! ⭐

