from email_validator import validate_email, EmailNotValidError

def validate_user_email(email):

    try:
        valid = validate_email(email)
        return valid.email

    except EmailNotValidError:
        return None