# 🩺 Diabetes Prediction Project  

### 📚 Overview  
The **Diabetes Prediction Project** utilizes Python and Support Vector Machine (SVM) to predict whether a person is diabetic or non-diabetic based on medical input parameters. <br>  
This simple model demonstrates how machine learning can assist in early detection of diabetes.  

---

## 🚀 Features  
- **Support Vector Machine Classifier:** A robust algorithm with a linear kernel for prediction. <br>  
- **Standardized Input Data:** Ensures consistent performance by standardizing input features. <br>  
- **User-Friendly Prediction Process:** Input medical data and get instant predictions.  

---

## 🛠️ Tech Stack  
- **Language:** Python <br>  
- **Libraries Used:** <br>  
  - `NumPy` for numerical operations <br>  
  - `Pandas` for data manipulation <br>  
  - `Scikit-learn` for preprocessing, model training, and evaluation <br>  

---

## 📂 Project Structure  
diabetes-prediction/ <br>
├── diabetes.csv               (Dataset used for training and testing) <br>
├── diabetes_prediction.py     (Main script for training and prediction) <br>
└── README.md                  (Project overview) <br>


---

## 🔍 How It Works  
1. **Dataset:** The model uses the `diabetes.csv` dataset. <br>  
   - Features include glucose level, BMI, age, and more. <br>  
   - The target variable is `Outcome` (0: Non-Diabetic, 1: Diabetic). <br>  

2. **Data Preprocessing:** <br>  
   - Features are standardized using `StandardScaler`. <br>  
   - Data is split into training (80%) and test (20%) sets using `train_test_split`. <br>  

3. **Model Training:** <br>  
   - A Support Vector Machine (SVM) with a linear kernel is trained on the dataset. <br>  

4. **Prediction:** <br>  
   - The trained model predicts whether the person is diabetic or non-diabetic based on medical inputs. <br>  

---

## 🧑‍💻 Getting Started  

### Prerequisites  
Ensure Python 3.7+ is installed along with the following libraries: <br>  
  
pip install numpy pandas scikit-learn  

Usage
1. Clone this repository: <br>

git clone https://github.com/yourusername/diabetes-prediction.git  
cd diabetes-prediction  

2. Place your dataset (diabetes.csv) in the project directory. <br>

3. Run the script: <br>

python diabetes_prediction.py  

4. Input data examples: <br>

Non-Diabetic: (4, 110, 92, 0, 0, 37.6, 0.191, 30) <br>
Diabetic: (5, 166, 72, 19, 175, 25.8, 0.587, 51) <br>

📜 License
This project is licensed under the MIT License.


You can copy-paste the above directly into your `README.md` file!

<br>
Author: Shourjya Chakraborty
