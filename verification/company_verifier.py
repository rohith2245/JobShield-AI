import requests

from verification.domain_checker import check_email_domain

from verification.trust_score import calculate_trust_score

# =====================================================
# WEBSITE VALIDATION
# =====================================================

def validate_website(url):

    if not url.strip():

        return False

    try:

        response = requests.get(
            url,
            timeout=5
        )

        if response.status_code == 200:

            return True

        return False

    except:

        return False

# =====================================================
# MAIN COMPANY VERIFIER
# =====================================================

def verify_company(
    recruiter_email,
    company_website,
    linkedin_url,
    has_salary,
    description_length
):

    signals = []

    # ==========================================
    # EMAIL CHECK
    # ==========================================

    official_email = check_email_domain(
        recruiter_email
    )

    # ==========================================
    # WEBSITE VALIDATION
    # ==========================================

    website_exists = validate_website(
        company_website
    )

    # ==========================================
    # LINKEDIN CHECK
    # ==========================================

    linkedin_exists = bool(
        linkedin_url.strip()
    )

    detailed_description = (
        description_length > 200
    )

    # ==========================================
    # SIGNALS
    # ==========================================

    if official_email:

        signals.append(
            "Official company email detected"
        )

    else:

        signals.append(
            "Suspicious public email domain detected"
        )

    if website_exists:

        signals.append(
            "Company website is reachable"
        )

    else:

        signals.append(
            "Company website unreachable or invalid"
        )

    if linkedin_exists:

        signals.append(
            "LinkedIn company profile available"
        )

    else:

        signals.append(
            "LinkedIn company profile missing"
        )

    # ==========================================
    # HTTPS CHECK
    # ==========================================

    if company_website.startswith("https://"):

        signals.append(
            "Secure HTTPS website detected"
        )

    # ==========================================
    # TRUST SCORE
    # ==========================================

    trust_score = calculate_trust_score(
        official_email,
        website_exists,
        linkedin_exists,
        has_salary,
        detailed_description
    )

    # ==========================================
    # VERIFICATION STATUS
    # ==========================================

    if trust_score >= 75:

        verification_status = "VERIFIED"

    elif trust_score >= 45:

        verification_status = "PARTIALLY VERIFIED"

    else:

        verification_status = "HIGH RISK"

    return {

        "trust_score": trust_score,

        "verification_status": verification_status,

        "verification_signals": signals
    }