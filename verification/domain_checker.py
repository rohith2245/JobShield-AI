SUSPICIOUS_DOMAINS = [
    "gmail.com",
    "yahoo.com",
    "hotmail.com",
    "outlook.com"
]

def check_email_domain(email):

    if not email:
        return False

    domain = email.split("@")[-1].lower()

    if domain in SUSPICIOUS_DOMAINS:
        return False

    return True