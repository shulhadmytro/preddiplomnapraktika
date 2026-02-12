import json
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- 1. ЗАВАНТАЖЕННЯ ДАНИХ ---
with open('intents.json', encoding='utf-8') as f:
    data = json.load(f)

patterns = []
tags = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern.lower())
        tags.append(intent['tag'])

# --- 2. ПІДГОТОВКА ТОКЕНІЗАТОРА ---
vocab_size = 1000 
embedding_dim = 16 

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(patterns)
sequences = tokenizer.texts_to_sequences(patterns)

max_len = max([len(x) for x in sequences])
X = pad_sequences(sequences, maxlen=max_len, padding='post')

# --- 3. КОДУВАННЯ МІТОК ---
classes = sorted(list(set(tags)))
label_map = {tag: i for i, tag in enumerate(classes)}
y = np.array([label_map[t] for t in tags])
y = tf.keras.utils.to_categorical(y, num_classes=len(classes))

# --- 4. АРХІТЕКТУРА МОДЕЛІ ---
model = Sequential([
    # Шар вбудовування (Рисунок 5.6 буде базуватися на навчанні цих ваг)
    Embedding(vocab_size, embedding_dim, input_length=max_len),
    GlobalAveragePooling1D(),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(len(classes), activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# --- 5. НАВЧАННЯ ---
# Зберігаємо результат навчання в змінну history
epochs = 300
history = model.fit(X, y, epochs=epochs, batch_size=8, verbose=1)

# --- 6. ВІЗУАЛІЗАЦІЯ (ГЕНЕРАЦІЯ РИСУНКУ 5.6) ---
print("Генерація графіків навчання...")
plt.figure(figsize=(12, 5))

# Графік точності (Accuracy)
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Точність', color='#2ecc71', linewidth=2)
plt.title('Точність моделі (Accuracy)')
plt.xlabel('Епоха')
plt.ylabel('Значення')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# Графік втрат (Loss)
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Втрати', color='#e74c3c', linewidth=2)
plt.title('Втрати моделі (Loss)')
plt.xlabel('Епоха')
plt.ylabel('Значення')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# Збереження для звіту
plt.tight_layout()
plt.savefig('learning_curves.png', dpi=300)
print("Рисунок 5.6 збережено як 'learning_curves.png'")
plt.show()

# --- 7. ЗБЕРЕЖЕННЯ МОДЕЛІ ТА РЕСУРСІВ ---
model.save('chatbot_model.h5')
pickle.dump(tokenizer, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))
pickle.dump(max_len, open('max_len.pkl', 'wb'))

print("Модель та словники успішно оновлені!")