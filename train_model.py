import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
import joblib

# ✅ Custom realistic dataset for each category based on your rules
data = {
    "subject": [
        # Important (green)
        "Devfolio Submission Confirmed",
        "Amazon Order Delivered",
        "Login Alert from Google",
        "Internship Offer Accepted",
        "Password Reset Request",

        # Social (blue)
        "You have a new message from LinkedIn",
        "New follower on Instagram",
        "LinkedIn: Job Alert from Recruiter",

        # Promotional (mustard)
        "Flat 60% OFF on Myntra Today Only",
        "Amazon Deal of the Day",
        "Unstop Hackathon Sale Now Live",
        "Lovable Lingerie Offer Inside",

        # Newsletter (grey)
        "TechCrunch Weekly Roundup",
        "Your NewsDigest: World Headlines",
        "HackerNoon Newsletter: AI Special",

        # Spam (red)
        "Win $10,000 by Clicking This Link!",
        "You have been selected for a lucky draw",
        "Free Recharge Now!",
        "Bulk Message Offer - 5k WhatsApp Leads",
        "Earn Money Fast - No Work Needed!",

        # Other (fallback)
        "Test email from my own address",
        "Meeting notes attached",
        "Your account summary"
    ],
    "body": [
        # Important
        "Your Devfolio hackathon project was successfully submitted.",
        "Your Amazon order has been delivered to your address.",
        "Suspicious login detected. If this was you, ignore this.",
        "Congratulations, your internship application is accepted!",
        "Click here to reset your password securely.",

        # Social
        "John Doe has sent you a message on LinkedIn.",
        "Alice started following you on Instagram.",
        "Check out new job opportunities from recruiters.",

        # Promotional
        "Shop now with huge discounts on Myntra products.",
        "Grab exciting deals on electronics and fashion.",
        "Unstop is hosting discounts on hackathons.",
        "Buy 2 get 1 free on all Lovable items.",

        # Newsletter
        "Here are the top tech stories this week from TechCrunch.",
        "Read today’s global headlines and updates.",
        "Explore the future of AI in this HackerNoon issue.",

        # Spam
        "Click this link to win prize money instantly.",
        "Congratulations! You're selected. Claim your gift.",
        "Recharge your phone for free. Limited time offer!",
        "Mass email offer for marketers to reach 5k users instantly.",
        "Earn without investment. Click here now.",

        # Other
        "This is a test email sent from my account.",
        "Here are the notes from today's team meeting.",
        "Here’s a quick summary of your recent activity."
    ],
    "label": [
        "Important", "Important", "Important", "Important", "Important",
        "Social", "Social", "Social",
        "Promotional", "Promotional", "Promotional", "Promotional",
        "Newsletter", "Newsletter", "Newsletter",
        "Spam", "Spam", "Spam", "Spam", "Spam",
        "Other", "Other", "Other"
    ]
}

# ✅ Combine subject and body into training set
df = pd.DataFrame(data)
df["text"] = df["subject"] + " " + df["body"]

# ✅ Use SGDClassifier with adaptive learning support
model = make_pipeline(
    TfidfVectorizer(),
    SGDClassifier(loss="log_loss")
)

# ✅ Initial training with class awareness
classes = ["Important", "Social", "Promotional", "Newsletter", "Spam", "Other"]
model.named_steps['sgdclassifier'].partial_fit(
    model.named_steps['tfidfvectorizer'].fit_transform(df["text"]),
    df["label"],
    classes=classes
)

# ✅ Save model
joblib.dump(model, "model.pkl")

print("✅ Custom adaptive model trained and saved as model.pkl")
