from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import imaplib
import email
from email.header import decode_header
import joblib
import db_helper

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Needed for session management
model = joblib.load("model.pkl")

import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

def classify_email(subject, body, sender=None):
    text = subject + " " + body
    # Custom rules for specific senders
    if sender:
        sender_lower = sender.lower()
        if "linkedin" in sender_lower:
            return "Social"
        if "unstop" in sender_lower:
            return "Promotional"
    # Check sender preference
    preferred_category = None
    if sender:
        preferred_category = db_helper.get_sender_preference(sender)
    # If sender is marked as spam, override category to Spam
    if preferred_category == "Spam":
        return "Spam"
    if preferred_category:
        return preferred_category
    prediction = model.predict([text])[0]
    # Fill empty category with Other
    if not prediction or prediction.strip() == "":
        return "Other"
    return prediction

def fetch_gmail_emails(user_email, app_password, max_emails=15):
    imap_host = "imap.gmail.com"
    mail = imaplib.IMAP4_SSL(imap_host)
    try:
        mail.login(user_email, app_password)
        mail.select("inbox")

        status, messages = mail.search(None, "ALL")
        email_ids = messages[0].split()
        latest_ids = email_ids[-max_emails:]
        results = []

        for num in latest_ids:
            status, msg_data = mail.fetch(num, "(RFC822 FLAGS)")
            flags = None
            email_size = 0
            for response_part in msg_data:
                if isinstance(response_part, tuple):
                    raw_email = response_part[1]
                    email_size = len(raw_email)  # size in bytes
                    msg = email.message_from_bytes(raw_email)

                    subject, encoding = decode_header(msg["Subject"])[0]
                    if isinstance(subject, bytes):
                        subject = subject.decode(encoding or "utf-8")
                    from_ = msg.get("From")

                    body = ""
                    if msg.is_multipart():
                        for part in msg.walk():
                            content_type = part.get_content_type()
                            content_dispo = str(part.get("Content-Disposition"))

                            if content_type == "text/plain" and "attachment" not in content_dispo:
                                body = part.get_payload(decode=True).decode(errors="ignore")
                                break
                    else:
                        body = msg.get_payload(decode=True).decode(errors="ignore")

                if isinstance(response_part, bytes) or isinstance(response_part, str):
                    continue
                elif isinstance(response_part, tuple):
                    continue
                else:
                    # flags info sometimes comes as separate response in imaplib
                    if isinstance(response_part, bytes):
                        continue
                    if isinstance(response_part, str):
                        continue
                    if isinstance(response_part, list):
                        continue
            # Get flags separately (safe fallback)
            status, flag_data = mail.fetch(num, '(FLAGS)')
            flags = flag_data[0].decode() if flag_data else ""

            # Determine unread and starred
            is_unread = b'\\Seen' not in mail.fetch(num, '(FLAGS)')[1][0]
            # safer way: if '\\Seen' in flags means read
            flags_str = flag_data[0].decode() if flag_data else ""
            is_unread = '\\Seen' not in flags_str
            is_starred = '\\Flagged' in flags_str

            # Calculate CO2 contribution based on email size (example factor: 0.000001 kg CO2 per byte)
            co2_contribution = email_size * 0.000001

            results.append({
                "id": num.decode(),
                "from": from_,
                "subject": subject,
                "body": body[:300],
                "unread": is_unread,
                "starred": is_starred,
                "co2": co2_contribution
            })

        return results

    except Exception as e:
        return str(e)

    finally:
        mail.logout()

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/connect', methods=['POST'])
def connect():
    user_email = request.form.get('email')
    app_password = request.form.get('password')
    if not user_email or not app_password:
        return redirect(url_for('index'))

    # Store credentials in session (for demo only; in production, handle securely)
    session['user_email'] = user_email
    session['app_password'] = app_password

    # Optionally, test login here or fetch emails
    emails = fetch_gmail_emails(user_email, app_password)
    session['emails'] = emails if isinstance(emails, list) else []

    return redirect(url_for('inbox'))

@app.route('/inbox')
def inbox():
    user_email = session.get('user_email')
    if not user_email:
        return redirect(url_for('index'))
    emails = session.get('emails', [])

    # Increment visit count for senders of emails in inbox
    for mail in emails:
        sender = mail.get("from")
        if sender:
            db_helper.increment_sender_visit_count(sender)

    # Reorder emails to prioritize frequent senders and starred emails
    # Get visit counts for senders
    sender_visit_counts = {}
    for mail in emails:
        sender = mail.get("from")
        if sender:
            count = db_helper.get_sender_visit_count(sender)
            sender_visit_counts[sender] = count

    # Define threshold for frequent sender (e.g., 3 visits)
    FREQUENT_VISIT_THRESHOLD = 3

    # Separate emails into frequent sender and others
    frequent_sender_emails = [e for e in emails if sender_visit_counts.get(e.get("from"), 0) >= FREQUENT_VISIT_THRESHOLD]
    other_emails = [e for e in emails if sender_visit_counts.get(e.get("from"), 0) < FREQUENT_VISIT_THRESHOLD]

    # Update categories of all emails from frequent senders based on sender preference
    for mail in emails:
        sender = mail.get("from")
        if sender and sender_visit_counts.get(sender, 0) >= FREQUENT_VISIT_THRESHOLD:
            preferred_category = db_helper.get_sender_preference(sender)
            if preferred_category:
                mail["category"] = preferred_category

    # Further reorder frequent sender emails to put starred on top
    frequent_sender_emails.sort(key=lambda x: x.get("starred", False), reverse=True)
    # Reorder other emails to put starred on top
    other_emails.sort(key=lambda x: x.get("starred", False), reverse=True)

    # Combine lists with frequent sender emails first
    emails = frequent_sender_emails + other_emails

    session['emails'] = emails

    return render_template('inbox.html', emails=emails, user_email=user_email)

@app.route('/fetch-emails', methods=['POST'])
def fetch_emails():
    user_email = request.form['email']
    app_password = request.form['password']
    emails = fetch_gmail_emails(user_email, app_password)

    # Increment visit count for senders of emails
    for mail in emails:
        sender = mail.get("from")
        if sender:
            db_helper.increment_sender_visit_count(sender)

    # Get visit counts for senders
    sender_visit_counts = {}
    for mail in emails:
        sender = mail.get("from")
        if sender:
            count = db_helper.get_sender_visit_count(sender)
            sender_visit_counts[sender] = count

    # Define threshold for frequent sender (e.g., 3 visits)
    FREQUENT_VISIT_THRESHOLD = 3

    # Update categories of all emails from frequent senders based on sender preference
    for mail in emails:
        sender = mail.get("from")
        if sender and sender_visit_counts.get(sender, 0) >= FREQUENT_VISIT_THRESHOLD:
            preferred_category = db_helper.get_sender_preference(sender)
            if preferred_category:
                mail["category"] = preferred_category

    return jsonify(emails)

@app.route('/train-model', methods=['POST'])
def train_model():
    # Retrain model using feedback data from SQLite
    import train_model
    import joblib

    feedbacks = db_helper.fetch_all_feedback()
    if not feedbacks:
        return jsonify({"message": "No feedback data available for training."})

    texts, labels = zip(*feedbacks)

    # Load existing model pipeline
    model = joblib.load("model.pkl")

    vectorizer = model.named_steps["tfidfvectorizer"]
    classifier = model.named_steps["sgdclassifier"]

    # Transform texts
    X = vectorizer.transform(texts)

    # Partial fit with feedback data
    known_labels = ["Spam", "Important", "Social", "Promotional", "Newsletter", "Other"]
    classifier.partial_fit(X, labels, classes=known_labels)

    # Save updated model
    joblib.dump(model, "model.pkl")

    return jsonify({"message": "Model retrained successfully with feedback data."})

@app.route('/classify-emails', methods=['POST'])
def classify_emails():
    emails = request.json.get("emails", [])
    for mail in emails:
        mail["category"] = classify_email(mail["subject"], mail["body"])
    return jsonify(emails)

@app.route('/delete-emails', methods=['POST'])
def delete_emails():
    data = request.get_json()
    email_ids = data.get("emailIds", [])
    user_email = data.get("email") or session.get('user_email')
    app_password = data.get("password") or session.get('app_password')

    if not email_ids or not user_email or not app_password:
        return jsonify({"message": "Missing parameters"}), 400

    try:
        mail = imaplib.IMAP4_SSL("imap.gmail.com")
        mail.login(user_email, app_password)
        mail.select("inbox")

        for eid in email_ids:
            mail.store(eid, '+FLAGS', '\\Deleted')
        mail.expunge()
        mail.logout()

        return jsonify({"message": f"Deleted {len(email_ids)} spam emails successfully."})

    except Exception as e:
        # Log the error for debugging (optional)
        # print(f"Error deleting emails: {e}")
        return jsonify({"message": "Failed to delete emails. Please check your credentials and try again."}), 500

@app.route('/disconnect')
def disconnect():
    session.clear()
    return redirect(url_for('index'))


from flask import request, jsonify
import joblib

known_labels = ["Spam", "Important", "Social", "Newsletter", "Other"]

@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.get_json()
    text = data.get("text")
    correct_label = data.get("correct_label")
    sender_email = data.get("sender_email")
    starred = data.get("starred", False)

    try:
        # Store feedback in SQLite
        db_helper.insert_feedback(text, correct_label)

        # Update sender preference if sender_email provided
        if sender_email:
            db_helper.upsert_sender_preference(sender_email, correct_label)

            # If category is Spam, update sender preference to Spam and update session emails
            if correct_label == "Spam":
                db_helper.upsert_sender_preference(sender_email, "Spam")
                emails = session.get('emails', [])
                for mail in emails:
                    if mail.get("from", "").strip().lower() == sender_email.strip().lower():
                        mail["category"] = "Spam"
                session['emails'] = emails

            # If category changed, update sender preference and session emails
            else:
                db_helper.upsert_sender_preference(sender_email, correct_label)
                emails = session.get('emails', [])
                for mail in emails:
                    if mail.get("from", "").strip().lower() == sender_email.strip().lower():
                        mail["category"] = correct_label
                session['emails'] = emails

            # Update session emails to reflect new category for all emails from sender
            emails = session.get('emails', [])
            for mail in emails:
                # Normalize sender email strings for comparison
                if mail.get("from", "").strip().lower() == sender_email.strip().lower():
                    mail["category"] = correct_label
            session['emails'] = emails

        # Update model incrementally
        vectorizer = model.named_steps["tfidfvectorizer"]
        classifier = model.named_steps["sgdclassifier"]
        classifier.partial_fit(vectorizer.transform([text]), [correct_label], classes=known_labels)
        joblib.dump(model, "model.pkl")

        # If starred, update sender preference to Important and reorder emails in session
        if starred and sender_email:
            db_helper.upsert_sender_preference(sender_email, "Important")
            # Reorder emails in session to put starred sender emails on top
            emails = session.get('emails', [])
            starred_emails = [e for e in emails if e.get("from") == sender_email]
            other_emails = [e for e in emails if e.get("from") != sender_email]
            session['emails'] = starred_emails + other_emails

        return jsonify({"status": "updated", "new_label": correct_label})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/adaptive-status')
def adaptive_status():
    # Generate adaptive learning status summary based on sender preferences and visit counts
    sender_prefs = []
    conn = db_helper.create_connection()
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT sender_email, preferred_category FROM sender_preferences")
            sender_prefs = cursor.fetchall()
        except Exception as e:
            sender_prefs = []
        finally:
            conn.close()

    sender_visits = []
    conn = db_helper.create_connection()
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT sender_email, visit_count FROM sender_visit_counts")
            sender_visits = cursor.fetchall()
        except Exception as e:
            sender_visits = []
        finally:
            conn.close()

    # Compose status message
    messages = []
    if sender_prefs:
        messages.append(f"Personalized categories set for {len(sender_prefs)} senders.")
    if sender_visits:
        frequent_senders = [s for s, c in sender_visits if c >= 3]
        messages.append(f"{len(frequent_senders)} senders frequently visited.")
    if not messages:
        messages.append("No adaptive learning data available yet.")

    status_message = " ".join(messages)
    return jsonify({"status": status_message})

@app.route('/home')
def home():
    return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)
