import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.utils import formatdate

from tool.mail_config import MAIL_SENDER, MAIL_RECEIVER, MAIL_AUTH_CODE


SMTP_SERVER = "smtp.qq.com"
SMTP_PORT = 465


def send_email(subject, body, attachment_path=None):
    if not MAIL_AUTH_CODE:
        print(f"[Mail] Skipped (MAIL_AUTH_CODE not set in tool/mail_config.py). Subject: {subject}")
        print(f"[Mail] Body preview: {body[:200]}")
        return False

    msg = MIMEMultipart()
    msg["From"] = MAIL_SENDER
    msg["To"] = MAIL_RECEIVER
    msg["Subject"] = subject
    msg["Date"] = formatdate(localtime=True)

    msg.attach(MIMEText(body, "plain", "utf-8"))

    if attachment_path and os.path.exists(attachment_path):
        with open(attachment_path, "r", encoding="utf-8") as f:
            attachment = MIMEText(f.read(), "plain", "utf-8")
        attachment.add_header("Content-Disposition", "attachment", filename=os.path.basename(attachment_path))
        msg.attach(attachment)

    try:
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
            server.login(MAIL_SENDER, MAIL_AUTH_CODE)
            server.sendmail(MAIL_SENDER, MAIL_RECEIVER, msg.as_string())
        print(f"[Mail] Sent: {subject}")
        return True
    except Exception as e:
        print(f"[Mail] Failed: {e}")
        return False


def notify_experiment_done(algorithm, dataset, result_path=None, extra_info=""):
    subject = f"[FL Experiment Done] {algorithm} / {dataset}"
    body = f"Algorithm: {algorithm}\nDataset: {dataset}\n\n{extra_info}"
    if result_path and os.path.exists(result_path):
        with open(result_path, "r", encoding="utf-8") as f:
            body += f"\n\n--- Result ---\n{f.read()}"
    send_email(subject, body, attachment_path=result_path)


def notify_experiment_error(algorithm, dataset, error_msg):
    subject = f"[FL Experiment ERROR] {algorithm} / {dataset}"
    body = f"Algorithm: {algorithm}\nDataset: {dataset}\n\nError:\n{error_msg}"
    send_email(subject, body)


def notify_batch_done(summary_text):
    subject = "[FL Batch] All experiments completed"
    send_email(subject, summary_text)
