print("Script started")

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


# -----------------------------
# Step 1: Create dataset
# -----------------------------
data = {
    "ticket_text": [
        "Internet is not working",
        "Slow connection issue",
        "App crashes frequently",
        "Server timeout error",
        "Network keeps disconnecting",
        "Cancel my subscription",
        "Please cancel my plan",
        "I want to stop my service",
        "End my membership",
        "Close my account",
        "Incorrect billing amount",
        "Billing issue again",
        "Refund not received",
        "Charged twice this month",
        "Wrong invoice sent",
        "Cannot login to account",
        "Password reset not working",
        "Account locked out",
        "Username not recognized",
        "Unable to update profile",
        "Internet very slow today",
        "App not loading properly",
        "Connection unstable",
        "Technical error occurred",
        "Service down completely",
        "Stop recurring payment",
        "Remove my subscription",
        "Deactivate account",
        "Terminate service now",
        "Subscription cancellation request",
        "Payment failed issue",
        "Refund request pending",
        "Overcharged on bill",
        "Invoice mismatch problem",
        "Billing dispute request",
        "Login failure problem",
        "Authentication error",
        "Account recovery issue",
        "Cannot access dashboard",
        "Profile update error"
    ],
    "category": [
        "Technical","Technical","Technical","Technical","Technical",
        "Cancellation","Cancellation","Cancellation","Cancellation","Cancellation",
        "Billing","Billing","Billing","Billing","Billing",
        "Account","Account","Account","Account","Account",
        "Technical","Technical","Technical","Technical","Technical",
        "Cancellation","Cancellation","Cancellation","Cancellation","Cancellation",
        "Billing","Billing","Billing","Billing","Billing",
        "Account","Account","Account","Account","Account"
    ]
}

df = pd.DataFrame(data)


# -----------------------------
# Step 2: Split data
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df["ticket_text"],
    df["category"],
    test_size=0.3,
    random_state=42
)


# -----------------------------
# Step 3: Vectorize text
# -----------------------------
vectorizer = TfidfVectorizer()

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# -----------------------------
# Step 4: Train model
# -----------------------------
model = LogisticRegression()
model.fit(X_train_vec, y_train)


# -----------------------------
# Step 5: Evaluate model
# -----------------------------
predictions = model.predict(X_test_vec)

print("\n=== Classification Report ===\n")
print(classification_report(y_test, predictions))

print("\n=== Confusion Matrix ===\n")
print(confusion_matrix(y_test, predictions))


# -----------------------------
# Step 6: Test new prediction
# -----------------------------
sample = ["Internet connection very slow"]
sample_vector = vectorizer.transform(sample)
prediction = model.predict(sample_vector)

print("\n=== Sample Prediction ===")
print("Input:", sample[0])
print("Predicted Category:", prediction[0])
