import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import messagebox
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import string

# Load the dataset
df = pd.read_csv('spam_ham_dataset.csv', encoding='latin-1')
df = df[['v1', 'v2']]  # Keep only the relevant columns
df.columns = ['label', 'message']

# Preprocessing
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
X = df['message']
y = df['label']

# Function to clean the text data
def clean_text(text):
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    text = " ".join([word for word in text.split() if word not in stopwords.words('english')])
    return text

# Apply text cleaning
X = X.apply(clean_text)

# Vectorize the text data using CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(X)

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Check the accuracy
y_pred = model.predict(X_test)
print("Model Accuracy: ", accuracy_score(y_test, y_pred))

# Function to predict spam or not
def predict_message(message):
    cleaned_message = clean_text(message)
    data = cv.transform([cleaned_message])
    prediction = model.predict(data)
    return "Spam" if prediction[0] == 1 else "Not Spam"

# GUI Setup
def send_email():
    email_text = text_area.get("1.0", "end-1c")  # Get the text from the text area
    if email_text.strip():
        prediction = predict_message(email_text)
        messagebox.showinfo("Prediction", f"The email is: {prediction}")
    else:
        messagebox.showwarning("Warning", "Please write an email before sending!")

# Create main window
root = tk.Tk()
root.title("Email Writer & Spam Detector")
root.geometry("500x400")

# Add a label
label = tk.Label(root, text="Write your email below:", font=("Helvetica", 14))
label.pack(pady=10)

# Create a text area for email writing
text_area = tk.Text(root, height=10, width=50, font=("Helvetica", 12))
text_area.pack(pady=20)

# Add a send button
send_button = tk.Button(root, text="Send", command=send_email, font=("Helvetica", 12), bg="lightblue")
send_button.pack(pady=20)

# Run the Tkinter loop
root.mainloop()
