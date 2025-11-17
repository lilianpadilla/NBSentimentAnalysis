from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample data
documents = [
    "This is a positive review of the product.",
    "The product is excellent and I love it.",
    "I am very disappointed with this item.",
    "Terrible quality, do not recommend.",
    "Great value for money, highly satisfied."
]
labels = ["positive", "positive", "negative", "negative", "positive"]

# Feature Extraction (Convert text to numerical features)
# Use CountVectorizer to convert text into word count vectors
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)

# Initialize and train the Multinomial Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")

# Example of predicting a new document
new_review = ["This movie is amazing!"]
new_review_vectorized = vectorizer.transform(new_review)
predicted_label = model.predict(new_review_vectorized)
print(f"Predicted label for '{new_review[0]}': {predicted_label[0]}")