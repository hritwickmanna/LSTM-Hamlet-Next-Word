# 📚 Hamlet Next Word Prediction using LSTM

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Deployed App](https://img.shields.io/badge/Live%20Demo-Streamlit-brightgreen?logo=streamlit)](https://hamlet-nextword.streamlit.app/)

---

## 📌 Project Overview

This project trains an **LSTM (Long Short-Term Memory)** neural network to predict the **next word** in Shakespeare's *Hamlet*. The model learns patterns in text sequences and generates word suggestions for creative writing or text autocompletion.

- **Current PoC deployment**: [Streamlit Web App](https://hamlet-nextword.streamlit.app/)
- **Model Performance**: Achieved ~60% accuracy after 100 epochs.  
- **Training Dataset**: Tokenized text from *Hamlet*.  

---

## 🧠 Model & Training

### Model Architecture

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_sequence_length - 1))
model.add(LSTM(150, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dense(total_words, activation='softmax'))
```

### Compilation

```python
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
model.summary()
```

### Training

```python
history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=64,
    validation_data=(X_test, y_test)
)
```

---

## ⚡ Training Highlights

- **Max Epochs**: 100  
- **Accuracy Achieved**: ~60%  
- **Loss Function**: `categorical_crossentropy`  
- **Optimizer**: `Adam`  

---

## 🚀 How to Run Locally

1️⃣ **Clone the repository**
```bash
git clone https://github.com/hritwickmanna/Hamlet-Next-Word-LSTM.git
cd Hamlet-Next-Word-LSTM
```

2️⃣ **Install dependencies**
```bash
pip install -r requirements.txt
```

3️⃣ **Train the model in jupyter notebook**
```bash
experiments.ipynb
```

4️⃣ **Run the prediction script**
```bash
python app.py
```

---

## 📦 Project Structure

```plaintext
├── hamlet.txt                  # Dataset
├── experiments.ipynb           # Model training notebook
├── app.py                      # Prediction script
├── hamlet_lstm_model.h5        # Saved trained model
├── tokenizer.pkl               # Saved tokenizer
├── requirements.txt            # Dependencies
├── README.md                   # This README file
└── LICENSE                     # License file
```

---

## 📌 Next Steps

- 🔄 **Increase Epochs** for more training time.  
- 🔀 **Use Bidirectional LSTM** for richer context understanding.  
- 🏗 **Implement Encoder-Decoder architecture** for sequence generation.  
 
---

## 📜 License

This project is licensed under the AGPL-3.0 License.
