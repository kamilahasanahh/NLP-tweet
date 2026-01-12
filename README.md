# Tweet Sentiment & Classification Analysis

This project explores social media data (tweets) to perform sentiment analysis, topic modeling, and classification. It utilizes a variety of techniques ranging from traditional machine learning algorithms like SVM to deep learning architectures like LSTM and LinearSVC.

## 📊 Project Overview

The pipeline is divided into several key stages:

### **1. Exploratory Data Analysis (EDA)**

* **Visualizations:** Generates word clouds, pie charts for location distribution, and interactive histograms for tweet length using `Plotly`.
* **Content Analysis:** Detects emoji frequency and identifies common patterns in the dataset.
* **Metadata Insights:** Analyzes location-based trends and tweet statistics.

### **2. Text Preprocessing**

To prepare raw tweets for modeling, the system performs:

* **Cleaning:** Removal of emojis, punctuation, and non-ASCII characters.
* **Normalization:** Tokenization, Stemming (Porter), and Lemmatization (WordNet).
* **Feature Engineering:** TF-IDF Vectorization for machine learning and Sequence Padding for deep learning.

### **3. Sentiment & Topic Analysis**

* **VADER Sentiment:** Automatically labels tweets as *Positive*, *Negative*, or *Neutral* based on linguistic intensity.
* **LDA Topic Modeling:** Discovers latent themes across the dataset using Latent Dirichlet Allocation to group similar discussions.

---

## 🤖 Models & Algorithms

The project compares multiple classification strategies:

* **Machine Learning:**
  * Naïve Bayes
  * **Linear Support Vector Classifier (LinearSVC)**
  * Random Forest

* **Deep Learning:**
  * **Artificial Neural Networks (ANN):** Multi-layer dense networks using Keras.
  * **LSTM (Long Short-Term Memory):** Recurrent networks designed to capture sequential context in text.

---

## 🛠️ Requirements

To run this notebook, you will need:

* **Libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `nltk`, `scikit-learn`, `tensorflow`, `plotly`, `wordcloud`.
* **NLTK Data:** `vader_lexicon`, `wordnet`.

## 🚀 How to Use

1. **Environment:** The notebook is optimized for **Google Colab**.
2. **Dataset:** Ensure `tweets.csv` is uploaded to your Google Drive in the specified path (`/MyDrive/`).
3. **Execution:** Run the cells sequentially. The notebook will handle library installations (like `segmentation-models-pytorch` or `nltk` downloads) automatically.
4. **Evaluation:** Check the **Classification Report** at the end of each model section to compare Precision, Recall, and F1-scores.

---

### **Model Comparison: Traditional Machine Learning vs. Deep Learning**

In this project, we compared the **Linear Support Vector Classifier (LinearSVC)**—a representative of traditional statistical learning—against **Long Short-Term Memory (LSTM)**—a specialized architecture of Deep Learning for sequential data.

| Feature | LinearSVC (Machine Learning) | LSTM (Deep Learning) |
| --- | --- | --- |
| **Data Requirements** | Performs well even with smaller datasets. | Requires a large volume of data to avoid overfitting. |
| **Text Representation** | Uses **TF-IDF** (Frequency-based); loses word order. | Uses **Word Embeddings**; preserves word sequence and context. |
| **Training Speed** | Very fast; can be trained in seconds on a standard CPU. | Slow; requires significant computation (GPUs preferred). |
| **Context Awareness** | Limited; treats "not good" as two separate features. | High; understands that "not" flips the sentiment of "good." |
| **Interpretability** | High; easy to see which words (features) impact the score. | "Black box"; difficult to trace why a specific decision was made. |
| **Complexity** | Simple; requires less hyperparameter tuning. | Complex; requires tuning of layers, dropout, and learning rates. |

---

### **Which one performed better?**

**1. LinearSVC (The Efficient Baseline):**
LinearSVC is excellent for social media text when you need a fast, lightweight model. Because tweets are short (limited characters), the "bag-of-words" approach (TF-IDF) often captures the most important sentiment-carrying words (e.g., "love," "hate," "disaster") effectively without needing to understand complex grammar.

**2. LSTM (The Context Expert):**
The LSTM model shines when the sentiment is hidden in the structure of the sentence or involves sarcasm and negation. By using a hidden state to "remember" previous words in the tweet, it can catch nuances that the LinearSVC might miss. However, it requires more preprocessing (tokenization and padding) and significantly more time to train.

### **Conclusion for README**

> *"While **LinearSVC** provided a highly competitive accuracy with minimal computational cost, the **LSTM** model demonstrated superior potential in capturing the linguistic nuances of social media slang and complex sentence structures."*
