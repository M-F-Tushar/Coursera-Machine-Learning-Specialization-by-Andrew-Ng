# 📘 Lecture 6: Unsupervised Learning — Clustering

---

## 🧠 What is Unsupervised Learning?

Unsupervised learning is the second major category of machine learning.

Unlike **supervised learning**, where each data point comes with a label (like "malignant" or "benign")...

> 🔍 **In unsupervised learning**, **there are no labels**.

You're simply given a bunch of input data, and the algorithm is asked to **find structure** in it — patterns, groups, or insights — all **without guidance**.

---

## 🔄 Supervised vs. Unsupervised Learning

| Feature                   | Supervised Learning               | Unsupervised Learning                 |
|---------------------------|-----------------------------------|---------------------------------------|
| Has Labels?               | ✅ Yes (x → y)                     | ❌ No labels (just x)                 |
| Goal                      | Predict output (y)                | Find patterns / structure in data     |
| Examples                  | Regression, Classification        | Clustering, Dimensionality Reduction  |
| Analogy                   | Teaching with answer key          | Exploring without instructions        |

---

## 🔍 Clustering: A Key Unsupervised Learning Task

Clustering is one of the most widely used forms of unsupervised learning.

### ✨ Goal:
Group similar data points together into **clusters**.

---

## 🧪 Example 1: Tumor Data (Without Labels)

Imagine you're given this dataset:

- **Inputs**: Tumor size and patient age
- **No labels** (no info on whether tumors are benign or malignant)

📌 The algorithm might automatically discover:
- One cluster of younger patients with smaller tumors
- Another cluster of older patients with larger tumors

> 🔗 This is clustering: finding natural groupings in data.

---

## 🗞 Example 2: Google News

Google News processes **hundreds of thousands of news articles daily**.

📰 Clustering helps group similar articles:
- Articles that mention **"panda"**, **"twins"**, and **"zoo"** get grouped together — all about the same news event.

> 🧠 No human tells the algorithm what topics to look for — it discovers topics by itself!

---

## 🧬 Example 3: DNA Microarray Data

Each column = 1 person  
Each row = 1 gene  
Each color = level of gene expression

📊 Clustering groups people by genetic patterns:

- 🧬 Type 1: One set of gene expression
- 🧬 Type 2: Another gene pattern
- 🧬 Type 3: A third distinct group

> ⚡️ No prior labels or types were provided — the algorithm **discovers** them.

---

## 🛍️ Example 4: Customer Segmentation

Companies can use clustering to:

- Analyze customer behavior data
- Automatically group users into **market segments**

🎯 Example groups from DeepLearning.AI:
1. Learners seeking knowledge and skill growth
2. Learners aiming for career development
3. Professionals wanting to stay updated on AI in their field

> 🧭 This insight helps companies **personalize content**, courses, or services.

---

## 📌 Summary

| Concept                 | Description |
|--------------------------|-------------|
| **Unsupervised Learning** | Learns patterns from **unlabeled data** |
| **Clustering**           | Groups similar examples into clusters |
| **Key Applications**     | News aggregation, genetics, customer segmentation |

🧠 The goal is **not prediction**, but **discovery**.

---

## 🐼 Panda Reminder

Next time you see a panda, think of clustering — because finding hidden groupings in chaos is a *superpower* of unsupervised learning.

---

## 🔜 Coming Up

Clustering is just one type of unsupervised learning.

> Next, we’ll explore **other kinds** of unsupervised algorithms beyond clustering.

Let’s keep going!
