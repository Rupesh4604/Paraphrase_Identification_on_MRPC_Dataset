# Semantic Paraphrase Identification with a Siamese LSTM Network

This project implements a deep learning model to determine if two sentences are semantically equivalent (i.e., if they are paraphrases of each other). It utilizes a Siamese Long Short-Term Memory (LSTM) network with pre-trained GloVe word embeddings to capture the contextual meaning of sentences.

---

## 📋 Table of Contents

- [Project Description](#-project-description)
- [Features](#-features)
- [Model Architecture](#-model-architecture)
- [Dataset](#-dataset)
- [Setup and Installation](#-setup-and-installation)
- [Usage](#-usage)
- [Results](#-results)
- [Future Improvements](#-future-improvements)

---

### 📖 Project Description

Paraphrase identification is a fundamental task in Natural Language Processing (NLP). This project moves beyond traditional methods that rely on word counts (like TF-IDF) and instead builds a model that learns to _understand_ sentence meaning. The Siamese network architecture is specifically designed for comparison tasks, making it a powerful choice for determining semantic similarity.

**Technologies Used:**

- Python 3.x
- TensorFlow & Keras
- Scikit-learn
- Pandas & NumPy
- NLTK
- GloVe (Global Vectors for Word Representation)

---

### ✨ Features

- **Deep Learning Core:** Employs a Siamese LSTM network to learn sentence representations optimized for similarity comparison.
- **Transfer Learning:** Leverages pre-trained GloVe word embeddings to inject rich semantic knowledge into the model from the start.
- **Automated Setup:** Automatically downloads the required GloVe embedding files if they are not found locally.
- **Robust Training:** Uses an Early Stopping mechanism to prevent overfitting and find the best model weights efficiently.
- **Standardized Evaluation:** Trained and evaluated on the benchmark Microsoft Research Paraphrase Corpus (MRPC).

---

### 🧠 Model Architecture

The model is a **Siamese Network** with two identical LSTM "towers" that share weights.

1.  **Input:** Two sentences are fed into the network separately.
2.  **Embedding:** Each sentence's words are converted into dense vectors using the shared, pre-trained GloVe `Embedding` layer.
3.  **Encoding:** The sequence of word vectors for each sentence is processed by a shared `LSTM` layer, which outputs a single vector representing the sentence's contextual meaning.
4.  **Comparison:** The two sentence vectors are compared using the **Manhattan distance** to calculate a final similarity score between 0 and 1.

---

### 📊 Dataset

This project uses the **Microsoft Research Paraphrase Corpus (MRPC)**. You will need to download the training and test sets and place them in the project's root directory.

- `msr-paraphrase-train.txt`
- `msr-paraphrase-test.txt`

You can typically find the dataset as part of the GLUE benchmark datasets or from various academic sources online.

---

### ⚙️ Setup and Installation

Follow these steps to set up the project environment.

**1. Prerequisites:**

- Python 3.8 or higher
- pip (Python package installer)

**2. Clone the Repository:**

```bash
git clone https://github.com/Rupesh4604/Paraphrase_Identification_on_MRPC_Dataset
cd Paraphrase_Identification_on_MRPC_Dataset

```

**3. Install Dependencies:**
It's recommended to use a virtual environment.

# Create and activate a virtual environment (optional but recommended)

python -m venv venv
source venv/bin/activate # On Windows, use `venv\Scripts\activate`

# Install the required packages

```bash
pip install tensorflow scikit-learn pandas numpy nltk requests
```

**4. Place Dataset Files:**
Ensure msr-paraphrase-train.txt and msr-paraphrase-test.txt are in the root directory of the project.

▶️ Usage
Once the setup is complete, you can run the main script to train and evaluate the model. The script will automatically download the GloVe embeddings on its first run (this is a large file and may take some time).

Run the colab notebook

The script will output the following:

The model summary.

Training progress for each epoch.

Final evaluation metrics (Accuracy, Classification Report, Confusion Matrix) on the unseen test set.

## 📈 Results

After running the model on the full MRPC dataset, the system achieves a robust performance in identifying paraphrases.

Final Test Accuracy: 0.9142

Classification Report:

              precision    recall  f1-score   support

           0       0.92      0.82      0.86       578
           1       0.91      0.96      0.94      1146

    accuracy                           0.91      1724

macro avg 0.92 0.89 0.90 1724
weighted avg 0.91 0.91 0.91 1724
Confusion Matrix:

[[472  106]
 [  42 1104]]

🚀 Future Improvements
Bidirectional LSTMs: Enhance context understanding by processing text in both forward and backward directions.

Attention Mechanisms: Allow the model to weigh the importance of different words when comparing sentences.

Transformer Models: For state-of-the-art results, fine-tune a pre-trained Transformer model like BERT or RoBERTa.

Hyperparameter Tuning: Systematically search for the optimal model parameters to improve performance.
