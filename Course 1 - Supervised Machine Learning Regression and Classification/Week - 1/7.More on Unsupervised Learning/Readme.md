# 📘 Lecture 7: More on Unsupervised Learning

---

## 🧠 What Is Unsupervised Learning (Formally)?

### ✅ Recap:
- **Supervised Learning**: You get inputs `x` and labels `y`
- **Unsupervised Learning**: You only get inputs `x` — **no labels!**

📌 The goal is to:
- Discover structure
- Find patterns
- Detect interesting trends — **without explicit answers**

---

## 🔍 Types of Unsupervised Learning

So far, we’ve seen:

### 1. **Clustering**:
- Groups similar data points into clusters
- Example: Grouping news articles, customer segmentation, DNA patterns

But unsupervised learning includes more than just clustering!

---

### 2. **Anomaly Detection**:
- Detects data points that **don't fit** usual patterns
- ⚠️ Used heavily in:
  - **Fraud detection** in finance
  - Monitoring **system failures**
  - Identifying **outliers** in health metrics or industrial machines

> 🧪 Example: A sudden, unusual bank transaction might signal fraud

---

### 3. **Dimensionality Reduction**:
- Compresses high-dimensional data into a smaller set of features
- 🔍 Useful for:
  - **Visualization**
  - **Speeding up learning algorithms**
  - **Removing noise or redundancy**
  
> 💡 It’s like summarizing a book while keeping its core message

We'll explore **both** anomaly detection and dimensionality reduction in more detail later in the course.

---

## ❓ Quiz: Supervised or Unsupervised?

Which of these are unsupervised learning problems?

| Example                              | Type                 | Why                                                   |
|--------------------------------------|----------------------|--------------------------------------------------------|
| 1. Spam detection                    | ❌ Supervised         | Labels: "spam" vs "not spam"                          |
| 2. News article grouping (Google News) | ✅ Unsupervised       | No labels — just text; uses clustering                |
| 3. Market segmentation               | ✅ Unsupervised       | Find customer groups from raw data                    |
| 4. Diagnosing diabetes               | ❌ Supervised         | Labels: "has diabetes" or "does not have diabetes"    |

> 🎯 If you picked 2 and 3 as unsupervised — you’re absolutely right!

---

## 💡 Why This Matters

Unsupervised learning gives us tools to:
- **Discover hidden structure** in data
- Make sense of unlabeled, real-world datasets
- **Enable smarter analytics** with less human supervision

---

## 🛠 Coming Up: Jupyter Notebooks

Next, we’ll explore a practical tool used by machine learning engineers and data scientists every day:

> 🧰 **Jupyter Notebooks** — a powerful environment for code, text, and visualizations.

Let’s dive into that next.
