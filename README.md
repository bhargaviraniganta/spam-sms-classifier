 📨 **Spam SMS Detector (BiLSTM Model)**

This repository contains a **Spam SMS Detection Model** built using **Deep Learning (BiLSTM)**.
The model classifies text messages into **Spam** or **Not Spam (Ham)** using natural language processing techniques.

The project is implemented inside the Jupyter Notebook:

```
Final_Spam_Detector.ipynb
```

---

## 🚀 **Project Overview**

Spam SMS messages are one of the most common communication threats.
This project uses:

* Text preprocessing
* Tokenization
* Padding
* Embedding layer
* Bidirectional LSTM neural network
* Training & evaluation
* Spam prediction

The model is built using **TensorFlow / Keras** and trained on a labeled spam dataset.

---

## 🧠 **Model Architecture**

The classifier uses:

* **Embedding Layer (word embeddings)**
* **Bidirectional LSTM Layer**
* **Dense Layer (ReLU)**
* **Sigmoid Output Layer (binary classification)**

This architecture helps capture contextual meaning and improves classification accuracy.

---

## 📁 **Repository Structure**


📦 spam-sms-classifier
 ┣ 📜 Final_Spam_Detector.ipynb
 ┣ 📜 README.md
 ┗ 📄 requirements.txt


---

## 🔧 **Technologies Used**

* Python
* TensorFlow / Keras
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* NLTK

---

## 📊 **Dataset**

The notebook expects a dataset containing at least the following columns:

| label      | message             |
| ---------- | ------------------- |
| ham / spam | SMS message content |

You can use publicly available datasets such as the **UCI SMS Spam Collection**.

---

## ▶️ **How to Run the Notebook**

1. Install required dependencies:

```
pip install tensorflow pandas numpy scikit-learn nltk
```

2. Open the notebook:

```
jupyter notebook Final_Spam_Detector.ipynb
```

3. Run all cells in order.

---

## 📈 **Results**

The model achieves:

* High accuracy
* Strong recall for spam messages
* Reliable performance on unseen messages

(Exact metrics depend on dataset and training parameters.)

---



## 📜 **License**

This project is open-source and free to use for educational purposes.


