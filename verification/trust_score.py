def calculate_trust_score(
    official_email,
    website_exists,
    linkedin_exists,
    has_salary,
    detailed_description
):

    score = 0

    if official_email:
        score += 30

    if website_exists:
        score += 25

    if linkedin_exists:
        score += 20

    if has_salary:
        score += 10

    if detailed_description:
        score += 15

    return score