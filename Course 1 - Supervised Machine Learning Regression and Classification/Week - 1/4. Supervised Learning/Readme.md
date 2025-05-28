# 📘 Lecture 4: Supervised Learning

---

## 💡 Why Supervised Learning Matters

Machine learning is driving **tremendous economic value** today.

> 🔢 **~99% of that value comes from just one type of ML: _Supervised Learning_**

---

## 🧭 What is Supervised Learning?

Supervised learning is the branch of machine learning where:

- You teach the algorithm by providing **input-output pairs** (x → y)
- The algorithm **learns a mapping** from inputs **x** to outputs **y**
- Once trained, it can **predict y** for **new, unseen x**

### 📌 Key Features

- Labeled data: input **x** and correct output **y**
- Learns by example
- Goal: accurately predict **y** for future **x**

---

## 🔍 Examples of Supervised Learning

| Input (x)                         | Output (y)                      | Application              |
|----------------------------------|----------------------------------|---------------------------|
| Email                            | Spam or Not Spam                 | Spam Filter               |
| Audio Clip                       | Text Transcript                  | Speech Recognition        |
| English Sentence                 | Sentence in Spanish              | Machine Translation       |
| Ad & User Info                   | Click or Not Click               | Online Advertising        |
| Image + Sensor Data              | Object Positions                 | Self-Driving Cars         |
| Image of Manufactured Product    | Defect or No Defect              | Visual Inspection (Industry) |

> 🤑 Online ad click prediction is **one of the most profitable ML applications** in the world.

---

## 🏠 Regression Example: Predicting Housing Prices

### 🏡 Problem

Predict house prices based on the **size of the house** (in square feet).

### 🔢 Data

- **x-axis**: House size (sq ft)
- **y-axis**: Price (in thousands of dollars)
- Example:
  - For a **750 sq ft house**, what price should we expect?

### 🔍 Model Fitting Options

1. **Linear Regression**
   - Fit a **straight line** to the data
   - Estimate: ~$150,000

2. **Polynomial Regression**
   - Fit a **curve** (more complex function)
   - Estimate: ~$200,000

🧠 Later in this course, you’ll learn **how to decide**:
- Whether to fit a line, a curve, or a more complex function
- How to **choose the best model** for the problem

---

## 🧠 Regression vs. Classification

| Type           | Description                                    | Example                          |
|----------------|------------------------------------------------|----------------------------------|
| **Regression** | Predict **a number** from infinite values       | House Price: $123,456            |
| **Classification** | Predict **a class** or category               | Spam / Not Spam                  |

In this lecture, we focused on **regression**. You'll explore **classification** in the next lecture.

---

## ✅ Summary

- Supervised learning = **input-output learning**
- Most economically valuable form of ML today
- Two types:
  - **Regression**: Predict continuous values
  - **Classification**: Predict discrete labels (coming next!)

---

## 🔜 Coming Up Next

🎯 Learn about **Classification** — the second key form of supervised learning.

- What it is
- How it differs from regression
- Where it’s used

See you in the next lecture!
