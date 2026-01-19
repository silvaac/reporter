# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 09:14:07 2024

@author: 14152
"""

import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def mail_it(to = "silvaac@yahoo.com", subj = "Mailgun Defualt Run", body = "Sending email using Mailgun API through python. Success!"):
    """
    Send email using Mailgun API
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
        
        # Send the email
        response = requests.post(
            url,
            auth=("api", api_key),
            data={
                "from": f"Python Script <{from_email}>",
                "to": [to],
                "subject": subj,
                "text": body
            }
        )
        
        # Check response
        if response.status_code == 200:
            print(f"✓ Email sent successfully!")
            print(f"  Message ID: {response.json().get('id')}")
            return 0
        else:
            print(f"✗ Error: {response.status_code}")
            print(f"  Response: {response.text}")
            return 1
            
    except Exception as e:
        print(f"✗ Error sending email: {e}")
        return 1


if __name__ == "__main__":
    # Test sending email
    print("Testing Mailgun email...")
    test_result = mail_it(
        to="silvaac@yahoo.com",  # Must be authorized in sandbox
        subj="Mailgun Test - Jan 2026", 
        body="If you receive this, Mailgun is working perfectly!"
    )
    
    if test_result == 0:
        print("\n✓ Test passed! Check your inbox.")
    else:
        print("\n✗ Test failed. Check your API key and domain.")