# 📘 Lecture 5: Classification (Supervised Learning)

---

## 🔍 What is Classification?

Classification is the **second major type** of supervised learning, after regression.

While **regression** predicts **continuous numbers**,  
**classification** predicts **discrete categories** (a small, finite set of outputs).

---

## 💡 Example: Breast Cancer Detection

Imagine you're building an AI tool to help doctors detect breast cancer.

- **Input (x)**: Patient's tumor features (e.g. size, age, shape)
- **Output (y)**: Tumor is
  - 🟢 **Benign** (not dangerous) → Label = 0  
  - 🔴 **Malignant** (cancerous) → Label = 1

### 📊 Data Plot (1 Input Feature: Tumor Size)

- **x-axis**: Tumor Size  
- **y-axis**: Class (0 = benign, 1 = malignant)

> 🧠 Since we only predict **two possible outputs**, this is called **binary classification**

---

## 🔄 Comparison: Regression vs. Classification

| Feature                  | Regression                      | Classification                      |
|--------------------------|----------------------------------|--------------------------------------|
| Output Type              | Continuous (real numbers)       | Discrete (finite categories)         |
| Example Task             | Predict house price             | Predict tumor is benign/malignant    |
| Output Possibilities     | Infinite (e.g. 150.75, 232.5)   | Limited (e.g. 0 or 1)                |
| Output Examples          | $123,456                        | Class 0 or 1                         |

> 🧩 Classification is about choosing the **right class**, not a precise number.

---

## 📈 Extending Inputs: Multi-Feature Classification

Classification works with **multiple input features**, not just one.

### 🧪 Example: Add Patient Age

- **Inputs (x)**:
  - Tumor size
  - Patient's age  
- **Output (y)**: 0 (benign) or 1 (malignant)

### 🧭 The Algorithm’s Job:

- Learn to draw a **decision boundary** that separates the classes:
  - 🔵 Circles = benign
  - ❌ Crosses = malignant

> The boundary helps predict whether a new patient's tumor is likely to be benign or malignant.

---

## 🎯 More than Two Classes

Classification isn't limited to two outputs.

Example:
- 0 → Benign  
- 1 → Cancer Type 1  
- 2 → Cancer Type 2

This is called **multiclass classification**.

> 🧠 Classification predicts from a **small set** of categories — whether they're numbers or labels like "cat", "dog", etc.

---

## 📝 Summary

| Concept                    | Description |
|----------------------------|-------------|
| **Supervised Learning**    | Learn from labeled examples (x → y) |
| **Regression**             | Predict a number from infinite possibilities |
| **Classification**         | Predict a class/category from a finite set |

📌 Classification = Predicting **which class** an example belongs to.

---

## 🔜 Coming Up Next

Next, you’ll learn about the **second major branch of machine learning**:

> **Unsupervised Learning** — where the algorithm finds patterns without being told the "right answers."

Let’s go explore that together!
