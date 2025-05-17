from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import imaplib
import email
from email.header import decode_header
import joblib

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Needed for session management
model = joblib.load("model.pkl")

def classify_email(subject, body):
    text = subject + " " + body
    prediction = model.predict([text])[0]
    return prediction

def fetch_gmail_emails(user_email, app_password, max_emails=5):
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
            for response_part in msg_data:
                if isinstance(response_part, tuple):
                    raw_email = response_part[1]
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

            results.append({
                "id": num.decode(),
                "from": from_,
                "subject": subject,
                "body": body[:300],
                "unread": is_unread,
                "starred": is_starred
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
    return render_template('inbox.html', emails=emails, user_email=user_email)

@app.route('/fetch-emails', methods=['POST'])
def fetch_emails():
    user_email = request.form['email']
    app_password = request.form['password']
    emails = fetch_gmail_emails(user_email, app_password)
    # Return emails without categorization
    return jsonify(emails)

@app.route('/train-model', methods=['POST'])
def train_model():
    # Run the training logic from train_model.py
    import train_model
    # The train_model.py script trains and saves the model on import
    return jsonify({"message": "Model trained successfully."})

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
    user_email = data.get("email")
    app_password = data.get("password")

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
        return jsonify({"message": str(e)}), 500

@app.route('/disconnect')
def disconnect():
    session.clear()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
