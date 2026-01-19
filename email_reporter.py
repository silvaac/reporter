"""
Email functionality for sending reports with attachments.
This module wraps mail_it.py to add attachment support via Mailgun API.
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()


def send_report_email(to, subject, summary_text, html_content, attachment_filename="report.html"):
    """
    Send trading report email with HTML attachment and text summary.
    
    Args:
        to: Recipient email address
        subject: Email subject line
        summary_text: Plain text summary for email body
        html_content: Full HTML report content to attach
        attachment_filename: Name for the HTML attachment
    
    Returns:
        0 on success, 1 on failure
    """
    
    # Get credentials from environment
    api_key = os.getenv('MAILGUN_API_KEY')
    domain = os.getenv('MAILGUN_DOMAIN')
    from_email = os.getenv('EMAIL_FROM', 'igncultura@gmail.com')
    
    if not api_key or not domain:
        print("Error: MAILGUN_API_KEY or MAILGUN_DOMAIN not set in .env file")
        return 1
    
    try:
        # Mailgun API endpoint
        url = f"https://api.mailgun.net/v3/{domain}/messages"
        
        # Prepare email data
        email_data = {
            "from": f"Trading Reporter <{from_email}>",
            "to": [to],
            "subject": subject,
            "text": summary_text
        }
        
        # Prepare HTML attachment
        files = [
            ("attachment", (attachment_filename, html_content.encode('utf-8'), "text/html"))
        ]
        
        # Send the email
        response = requests.post(
            url,
            auth=("api", api_key),
            data=email_data,
            files=files
        )
        
        # Check response
        if response.status_code == 200:
            print(f"✓ Email sent successfully!")
            print(f"  To: {to}")
            print(f"  Subject: {subject}")
            print(f"  Attachment: {attachment_filename}")
            print(f"  Message ID: {response.json().get('id')}")
            return 0
        else:
            print(f"✗ Error: {response.status_code}")
            print(f"  Response: {response.text}")
            return 1
            
    except Exception as e:
        print(f"✗ Error sending email: {e}")
        return 1
