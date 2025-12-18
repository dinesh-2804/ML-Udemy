## 🧠 Feature Engineering — Day 1

# Handling Missing Values
## 🎯 Introduction
In this first part of the Feature Engineering series, we focus on Handling Missing Values — one of the most important steps in preparing data for machine learning models. In real-world datasets, missing values are extremely common, and how you handle them directly impacts the accuracy and reliability of your model.

❓ Why Do Missing Values Occur?
Missing values appear when certain information is not recorded or available. For example, in a survey, a participant might skip a question about salary, leading to a missing value. These gaps can occur due to data entry errors, system failures, or respondents not providing complete information.

# ⚙️ Mechanisms of Missing Data

## Missing Completely at Random (MCAR)
The missingness is completely random and unrelated to any feature.

Example: accidental data entry or technical errors.

## Missing at Random (MAR)
The missingness depends only on other observed variables.

Example: men may avoid reporting their income, while women may avoid reporting their age.

## Missing Not at Random (MNAR)
The missingness depends on unobserved variables or the missing value itself.

Example: employees dissatisfied with their salaries may choose not to disclose them.

## 🚢 Example Dataset
The Titanic dataset is a popular example for understanding missing values, as it contains a mix of numerical and categorical variables with missing data. It helps illustrate how different handling techniques affect real-world datasets.

🔍 Identifying Missing Values
Before handling missing data, it is essential to identify where the gaps exist. You can explore the dataset to see which columns contain missing values and how many are missing in each.

## 🗑️ Deleting Missing Values
One approach is to remove rows or columns containing missing values. However, this method can cause significant data loss and is generally used only when the missing data proportion is small or the dataset is very large. Columns with too many missing values may also be dropped if they provide little information.

## 🧮 Imputation Techniques
Instead of deleting data, missing values can be replaced using various imputation methods.

Mean Imputation: Replace missing values with the average of that column. Works best for normally distributed numerical data.

Median Imputation: Replace missing values with the median. Best for data with outliers or skewed distributions.

Mode Imputation: Used for categorical data. Replace missing values with the most frequently occurring category.

Random Sample Imputation: Replace missing values with randomly selected existing values from the same column to preserve data variability.

## 🔁 Forward Fill and Backward Fill
These methods are simple yet effective for sequential or time-based data.

Forward Fill (ffill): Fills missing values using the previous available value.

Backward Fill (bfill): Fills missing values using the next available value.

They work well for maintaining continuity in ordered datasets.

## Technique	Suitable For	Description

Mean Imputation	Normal distributions	Replace missing values with mean
Median Imputation	Skewed or outlier-heavy data	Replace missing values with median
Mode Imputation	Categorical data	Replace missing values with mode
Random Sample Imputation	Numeric variables	Replace missing values randomly
Forward/Backward Fill	Sequential or time-series data	Fill using previous or next value
Deletion	Large datasets	Remove missing data when impact is minimal
🧠 Key Takeaways
Missing data is categorized into MCAR, MAR, and MNAR.

Always identify the cause of missingness before applying a solution.

Avoid deleting data unless the dataset is large enough to absorb the loss.

Use imputation methods to maintain dataset completeness.

Forward and backward fill are efficient for time-series or ordered data.



# 📊 Feature Engineering — Day 2  
## Outlier Treatment & Feature Scaling

This repository documents **Day 2** of my Feature Engineering learning series, where I focused on **handling outliers** and **scaling numerical features** — two critical steps in preparing data for machine learning models.

---

## 🎯 Topics Covered

### 1️⃣ Outlier Detection & Handling

Outliers can significantly impact model performance, especially for distance-based and statistical models. In this session, I explored multiple techniques to identify and treat outliers using **NumPy** and **Pandas**.

#### 🔹 Techniques Used:
- **IQR (Interquartile Range) Method**
- **Z-Score Method**
- **Capping / Clipping**
- **Winsorization**

#### 🔹 Libraries:
- NumPy
- Pandas

---

### 2️⃣ Feature Scaling

Feature scaling ensures that all numerical features contribute equally to the model. Different algorithms behave differently depending on feature magnitude, so choosing the right scaling technique is important.

#### 🔹 Scaling Techniques Implemented:
- **Standardization (StandardScaler)**  
  Scales data to have mean = 0 and standard deviation = 1.
  
- **Normalization (MinMaxScaler)**  
  Scales data into a fixed range (usually 0 to 1).
  
- **Robust Scaling (RobustScaler)**  
  Uses median and IQR, making it robust to outliers.

#### 🔹 Library Used:
- `sklearn.preprocessing`

---

## 🧠 Why This Matters

- Prevents model bias caused by extreme values
- Improves convergence speed of ML algorithms
- Essential for algorithms like:
  - Linear Regression
  - Logistic Regression
  - KNN
  - SVM
  - K-Means

---

## 🛠️ Tools & Technologies

- Python 🐍
- NumPy
- Pandas
- Scikit-learn
- Jupyter Notebook

---

## 📌 Learning Outcome

By the end of this session, I gained hands-on experience in:
- Identifying and treating outliers
- Applying appropriate feature scaling techniques
- Understanding when to use which scaler based on data distribution and model requirements

---

## 🚀 Next Steps

Moving forward, I’ll be diving deeper into:
- Feature transformation
- Encoding categorical variables
- Advanced feature selection techniques

---

# 📘 Feature Engineering – Day 3  
## Handling Categorical Variables for Machine Learning

---

### 📌 Overview
On Day 3 of my Machine Learning learning journey, I focused on **handling categorical variables** and converting them into numerical formats that machine learning models can understand.

Since most ML algorithms work only with numerical data, choosing the **right encoding technique at the right time** is critical for model performance.

---

### 🎯 Objectives
- Understand different types of categorical data
- Learn **when and why** to use each encoding technique
- Apply encoders correctly on a real-world dataset
- Prepare an ML-ready dataset

---

### 📊 Dataset Used
- **Loan Dataset**
- Contains numerical, ordinal categorical, and nominal categorical features
- Includes missing values and multiple category levels

---

## 🧠 Encoding Techniques: What & When to Use

---

### 1️⃣ Label Encoding

**What it does:**  
Converts categorical values into integer labels (e.g., Yes → 1, No → 0).

**When to use:**  
- Binary categorical variables  
- Target variables  
- Tree-based models (Decision Tree, Random Forest, XGBoost)

**When NOT to use:**  
- Nominal features with no order (can mislead distance-based models)

**Example Use Case:**  
Gender, Loan_Status

---

### 2️⃣ Ordinal Encoding

**What it does:**  
Assigns ordered numerical values based on the **natural ranking** of categories.

**When to use:**  
- Categorical features with a **clear order**
- When order carries meaning

**Why it matters:**  
Preserves ranking information that the model can learn from.

**Example Use Case:**  
Poor < Average < Good < Excellent  
Low < Medium < High

---

### 3️⃣ One-Hot Encoding

**What it does:**  
Creates separate binary columns for each category.

**When to use:**  
- Nominal categorical variables  
- No natural order exists among categories  
- Linear models, Logistic Regression, KNN, SVM

**Advantages:**  
- Prevents false ordinal relationships  
- Safe and widely used

**Disadvantages:**  
- Increases dimensionality for high-cardinality features

**Example Use Case:**  
Property_Area, Education, Marital_Status

---

### 4️⃣ Frequency Encoding

**What it does:**  
Encodes categories based on how frequently they appear in the dataset.

**When to use:**  
- High-cardinality categorical variables  
- When One-Hot Encoding causes too many columns

**Advantages:**  
- Reduces dimensionality  
- Useful in large datasets

**Caution:**  
- Frequency may not always correlate with importance

**Example Use Case:**  
City names, Product IDs

---

## ⚙️ Implementation Approach

- Identified categorical columns based on type:
  - Ordinal
  - Nominal
- Applied appropriate encoders using **scikit-learn**
- Combined numerical scaling and categorical encoding using **ColumnTransformer**
- Converted all categorical values into numerical format safely

---

### 🛠️ Tools & Libraries Used
- Python
- Pandas
- NumPy
- Scikit-learn

---

### ✅ Outcome
- Clear understanding of **which encoding to use and when**
- Successfully transformed categorical variables into numerical features
- Built a reusable preprocessing pipeline
- Dataset is fully ready for machine learning models

---

### 🚀 Next Steps
- Feature scaling
- Model training
- Model evaluation and optimization

---

📌 *This repository is part of my daily Machine Learning learning series.*


📢 **Follow my LinkedIn series for daily hands-on learning in Data Science & Machine Learning.**

