# 📘 Lecture 9: Linear Regression & Supervised Learning Process

## 🎯 Objective
Learn how **supervised learning** works through the example of **linear regression**, one of the simplest and most widely-used machine learning algorithms.

---

## 🏡 Problem Example: Predicting House Prices

You're a real estate agent trying to predict the **price of a house** based on its **size**.

- **Dataset**: Houses in *Portland, USA*
- **Input (x)**: Size in square feet  
- **Output (y)**: Price in thousands of dollars

Each data point is a house:
(x = size in sqft, y = price in $1000s)

---

## 📐 Linear Regression

Linear regression fits a **straight line** to data.

### Example:
- Given a house of **1250 sqft**, find the predicted price.
- Use the learned **best-fit line** from the training data.
- Predict ≈ **$220,000**

> 🔍 Regression predicts **numbers**. Classification predicts **categories**.

---

## 🧠 Why It's Called Supervised Learning

- You're training the model with **input-output pairs**.
- Each example has a **correct label (supervision)**:  
  `x = size`, `y = price`
- The model **learns from these correct answers**.

---

## 📏 Regression vs Classification

| Type               | Output              | Examples                                     |
|--------------------|---------------------|----------------------------------------------|
| **Regression**     | Continuous numbers  | Price prediction, temperature, age           |
| **Classification** | Discrete categories | Cat vs Dog, Disease: Yes/No, Spam/Not Spam   |

---

## 📊 Representing the Data: Tables and Notation

### Data Table
| x (size in sqft) | y (price in $1000s) |
|------------------|---------------------|
| 2104             | 400                 |
| 1600             | 330                 |
| ...              | ...                 |

### Plot
- **X-axis**: House Size  
- **Y-axis**: Price  
- Each point: One house `(x, y)`

---

## 🔤 Machine Learning Notation

| Symbol              | Meaning                                | Example                     |
|---------------------|----------------------------------------|-----------------------------|
| `x`                 | Input feature (e.g., house size)       | `x = 2104`                  |
| `y`                 | Output/target value (e.g., price)      | `y = 400`                   |
| `m`                 | Number of training examples            | `m = 47`                    |
| `(x, y)`            | A training example pair                | `(2104, 400)`               |
| `x⁽ⁱ⁾`, `y⁽ⁱ⁾`      | i-th training example                  | `x⁽¹⁾ = 2104`, `y⁽¹⁾ = 400` |

> 🔸 Note: Superscript `(i)` is **not exponentiation** — it just refers to the i-th example in the dataset.

---

## 🧪 Training Set

- All input-output pairs used to **train the model**
- Used to **learn patterns**
- Does **not** include the data for the house you want to predict

---

## ▶️ Coming Up Next

You'll learn how to **train a model** on the dataset and **use it to make predictions**.

🎥 See you in the next video!
