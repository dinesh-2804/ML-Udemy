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



# 📘 Machine Learning Learning Series — Day 4
  # Categorical Encoding & Feature Scaling
   --  📌 Overview
   -  The focus of this day is on data preprocessing, specifically:
   -  Converting categorical features into numerical form
   - Applying and comparing feature scaling techniques
    These steps are essential to make data suitable for machine learning models and to improve model performance.
# 📂 Datasets Used
## 1️⃣ Loan Approval Dataset
- Used to practice categorical encoding techniques
- Contains multiple categorical and numerical features
## 2️⃣ Diabetes Dataset
- Used to practice and compare feature scaling techniques
- Primarily numerical features
# 🔁 Techniques Covered
- 🔹 Categorical Encoding
  - Applied on the Loan Approval dataset:
- One-Hot Encoding
   -  Used for nominal categorical features
   -  Prevents unintended ordinal relationships
- Ordinal Encoding
  - Used when categories have a meaningful order
  - Preserves ranking information
  - These encoders help transform categorical data into a machine-readable numerical format.
# 🔹 Feature Scaling
- Applied on the Diabetes dataset:
# StandardScaler
 -  Centers data to mean = 0 and standard deviation = 1
 -  Best suited for normally distributed data
# MinMaxScaler
  - Scales values between 0 and 1
  - Useful when features have fixed bounds
# RobustScaler
 -  Uses median and IQR
 - Effective when the dataset contains outliers
 -  Understanding when to use each scaler is critical for model accuracy.
# 📁 Project Files
- HandlingCategoricalMissingValues.ipynb
- → Categorical encoding techniques on Loan Approval dataset
# DailyTask1.ipynb
→ Feature scaling techniques on Diabetes dataset
Each notebook contains step-by-step preprocessing with explanations.
# 🧠 Key Learnings
- Machine learning models cannot work directly with categorical data
- Choosing the right encoding method depends on the nature of the feature
- Feature scaling is essential for distance-based and gradient-based algorithms
- Different scalers behave differently depending on data distribution and outliers

 ## Day 5 (Machine Learning Learning Series)

 # # 📘 ColumnTransformer 
 
# 📌 Overview
  - This repository contains my Day 5 learning work from the Machine Learning learning series.
  - The main focus is on ColumnTransformer, a powerful preprocessing tool from scikit-learn used to handle datasets with mixed feature types.
  - Most real-world datasets include both numerical and categorical features, and each requires different preprocessing techniques. ColumnTransformer allows us to apply the right transformation to the right column  efficiently.
  
## 🎯 Objectives
  - Understand the purpose of ColumnTransformer
  - Learn column-wise preprocessing
  - Apply appropriate transformations to numerical and categorical data
  - Practice the fit() and transform() workflow
  - Prepare data for machine learning models
    
## 🧠 Concepts Covered
  - Numerical vs Categorical features
  - Feature-wise preprocessing
  - StandardScaler for numerical columns
  - OneHotEncoder for nominal categorical columns
  - OrdinalEncoder for ordered categorical columns
  - Role of ColumnTransformer in ML pipelines

    
## 📂 Notebooks Included
  - ColumnTransformers.ipynb
  - Focuses on conceptual understanding
  - Explains how ColumnTransformer works
  - practice_ColumnTransformer.ipynb
  - Hands-on implementation
  - Applies transformations on a dataset
  - Outputs model-ready transformed data\

    
## ⚙️ Tools & Libraries Used
  - Python
  - NumPy
  - Pandas
  - Scikit-learn

    
##🚀 Key Takeaways
  - ColumnTransformer enables clean and structured preprocessing
  - Different data types require different transformations
  - Proper preprocessing is critical for model performance
  - ColumnTransformer is essential for scalable ML workflows

    
##📌 Next Steps
  - Integrate ColumnTransformer with Pipeline
  - Combine preprocessing and model training
  - Apply the approach to real-world datasets



# Feature Selection (Numerical) — Day 6  
Machine Learning Learning Series

## 📌 Overview
This repository contains the **Day 6** work of my Machine Learning learning series, focused on **Feature Selection techniques for numerical features**.  
Feature selection helps improve model performance by removing irrelevant or redundant features, reducing overfitting, and improving interpretability.

In this notebook, I applied multiple statistical and information-based methods to identify the most important numerical features.

---

## 📂 Files Included
- `FeatureSelection(Numerical).ipynb` — Jupyter Notebook implementing numerical feature selection techniques

---

## 🔍 Feature Selection Techniques Covered

### 1️⃣ Correlation Method
- Measures the linear relationship between numerical features and the target variable.
- Highly correlated features with the target are retained.
- Features with very low correlation are candidates for removal.

**When to use:**
- Linear models
- Quick baseline feature filtering
- When relationships are expected to be linear

---

### 2️⃣ Variance Threshold
- Removes features with very low variance.
- Features with near-constant values carry little information for prediction.

**When to use:**
- High-dimensional datasets
- Removing constant or near-constant numerical features
- As a preprocessing step before modeling

---

### 3️⃣ Mutual Information
- Measures the dependency between features and the target variable.
- Captures both linear and non-linear relationships.
- Higher mutual information score indicates more predictive power.

**When to use:**
- Non-linear models
- Complex datasets
- When feature–target relationships are unknown

---

## 🧠 Key Learnings
- Feature selection improves model efficiency and generalization.
- Different methods capture different types of relationships.
- Combining multiple techniques leads to more reliable feature selection.
- Numerical feature selection is a critical step before model training.

---

## 🛠️ Tools & Libraries Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib / Seaborn (for analysis & visualization)

---

## 🚀 Next Steps
- Apply selected features to machine learning models
- Compare model performance before and after feature selection
- Extend feature selection techniques to categorical data

---

## Day 7 (Machine Learning Learning Series)

# Feature Selection — Categorical Variables (Machine Learning)

## 📌 Overview
This project focuses on **Feature Selection techniques for categorical variables** in Machine Learning.  
Feature selection is a critical preprocessing step that helps improve model performance, reduce overfitting, and enhance interpretability by selecting only the most relevant features.

In this notebook, different **statistical and information-theoretic methods** are applied to identify important categorical features.

---

## 🎯 Objectives
- Understand why feature selection is important
- Apply feature selection techniques specifically for categorical variables
- Identify the most relevant features for model training

---

## 🧠 Techniques Covered

### 1️⃣ Mutual Information (MI)
- Measures the **dependency between input features and the target variable**
- Captures **both linear and non-linear relationships**
- Higher MI score → higher relevance

**When to use:**
- When relationships between features and target may be non-linear
- Works well with both classification and regression problems

---

### 2️⃣ Chi-Square Test (χ²)
- Statistical test to check **association between categorical features and target**
- Compares observed vs expected frequencies

**When to use:**
- Categorical input features
- Categorical target variable
- Data must be **non-negative**

---

### 3️⃣ SelectKBest
- Used to select the **top K most important features**
- Works with scoring functions like:
  - `chi2`
  - `mutual_info_classif`

**When to use:**
- When you want a fixed number of best features
- Helpful for dimensionality reduction

---

## 🛠️ Tools & Libraries
- Python
- Pandas
- NumPy
- Scikit-learn

---

## 📂 Files Included
- `Feature Selection (Categorical).ipynb` — Notebook with implementation and explanations

---

## 🚀 Key Learnings
- Feature selection improves model efficiency and accuracy
- Different techniques serve different data assumptions
- Choosing the right method depends on feature type and problem statement

---

## 📎 Conclusion
This project demonstrates practical implementation of **categorical feature selection techniques**, which are essential for building reliable and scalable machine learning models.

---

# Day 8 (Machine Learning Learning Series)

# Image Data Preprocessing using OpenCV

## 📌 Overview
This project is part of my LinkedIn learning series focused on **Image Data Preprocessing**.  
In this notebook, I explored how to read, transform, and manipulate image data using **OpenCV (cv2)** — a foundational step for computer vision and deep learning applications.

The goal is to understand how raw image data can be converted into different formats suitable for analysis and modeling.

---

## 🛠️ Tools & Libraries Used
- Python
- OpenCV (cv2)
- NumPy
- Matplotlib

---

## 📂 Key Concepts Covered

### 1. Image Importing
- Loading images using `cv2.imread()`
- Understanding image shape and pixel structure

### 2. Color Space Conversions
- **BGR to Grayscale**
- **BGR to RGB**
- Black & White image representation
- Visualization using Matplotlib

### 3. Image Transformations
- **Image Rotation (45°)**
- **Image Flipping**
  - Horizontal flip
  - Vertical flip

### 4. Visualization
- Displaying images using `matplotlib.pyplot`
- Comparing original vs transformed images

---

## 🎯 Why Image Preprocessing Matters
- Improves model performance
- Reduces noise and unnecessary information
- Converts images into model-friendly formats
- Essential for Computer Vision & Deep Learning pipelines

---

## 📁 File Structure
- `Image_Data_Preprocessing.ipynb` – Main notebook containing all preprocessing steps

---

## 🚀 Next Steps
- Image resizing & normalization
- Edge detection
- Feature extraction
- Applying preprocessing for CNN models

---



📌 *This project is part of my continuous Machine Learning learning journey.*


📢 **Follow my LinkedIn series for daily hands-on learning in Data Science & Machine Learning.**

