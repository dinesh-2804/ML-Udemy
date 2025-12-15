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
