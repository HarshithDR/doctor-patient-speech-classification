import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

df = pd.read_csv('Downloads/data_cleaned.csv')
statements = df['statement'].values  
labels = df['speaker'].values  

# Checked and working run on entire dataset
#df=df.head(10)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)  


tfidf = TfidfVectorizer()  
X = tfidf.fit_transform(statements).toarray()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),  
    Dense(64, activation='relu'),  
    Dense(1, activation='sigmoid')  
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, validation_split=0.2, epochs=30, batch_size=32)


test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test accuracy is : {:.2f}%".format(test_accuracy * 100))

plt.figure(figsize=(12, 5))


plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.tight_layout()
plt.show()
