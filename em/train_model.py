import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import joblib

# ✅ Step 1: Prepare realistic dataset
data = {
    "subject": [
        "50% OFF only today on Myntra!",
        "Your RummyCircle prize is waiting!",
        "New LinkedIn message from recruiter",
        "You're selected for the Google Hackathon",
        "TechCrunch Weekly Update",
        "Instagram password change alert",
        "Win ₹10,000 by clicking here!",
        "Internship confirmation letter attached",
        "Medium: New post from HackerNoon",
        "Bank KYC: Update PAN details"
    ],
    "body": [
        "Exclusive deals on shoes and gadgets",
        "Claim your winnings now by logging in",
        "Check out this opportunity!",
        "Congratulations! Your idea made it!",
        "Top stories of the week from TechCrunch",
        "Your account was accessed from a new device",
        "Don't miss your reward",
        "Please find your offer letter and next steps",
        "New article: Why AI will shape 2025",
        "Please update your documents to avoid suspension"
    ],
    "label": [
        "Spam",
        "Spam",
        "Social",
        "Important",
        "Newsletter",
        "Social",
        "Spam",
        "Important",
        "Newsletter",
        "Important"
    ]
}

# ✅ Step 2: Combine into training set
df = pd.DataFrame(data)
df["text"] = df["subject"] + " " + df["body"]

# ✅ Step 3: Train pipeline
model = make_pipeline(TfidfVectorizer(), LogisticRegression())
model.fit(df["text"], df["label"])

# ✅ Step 4: Save model
joblib.dump(model, "model.pkl")

print("✅ Model trained and saved as model.pkl")
