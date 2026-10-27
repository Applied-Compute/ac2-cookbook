"""Fictional persona ingredients for the optional synthetic-data generator."""

import random

from .data import Persona

SELLERS = [
    ("Pipeline IQ", "an AI sales-forecasting platform that syncs with Salesforce and flags at-risk deals"),
    ("Shipwell Labs", "a freight-visibility API that gives shippers real-time container tracking"),
    (
        "Kestrel Security",
        "an attack-surface-management tool that finds exposed cloud assets before attackers do",
    ),
    ("Ledgerline", "an accounts-receivable automation suite that cuts DSO for mid-market finance teams"),
    ("Hirelight", "a structured-interview platform that reduces time-to-hire for hourly workforces"),
    ("Cropsense", "a soil-moisture analytics service for large row-crop operations"),
    ("Faststand", "a modular trade-show booth system that ships in two crates and assembles in an hour"),
    ("Relayform", "a HIPAA-compliant patient-intake platform for specialty clinics"),
    ("Gridmetric", "an energy-usage analytics platform for commercial real-estate portfolios"),
    ("Stackwatch", "a cloud-cost observability tool that attributes spend to teams and features"),
    ("Brightloop", "a customer-onboarding automation platform for B2B SaaS companies"),
    ("Portside AI", "a customs-documentation copilot for freight forwarders"),
]

PROSPECT_TITLES = [
    ("VP of Sales", "B2B software"),
    ("Head of Revenue Operations", "fintech"),
    ("Director of Supply Chain", "consumer goods"),
    ("CISO", "regional banking"),
    ("Controller", "industrial distribution"),
    ("VP of Talent Acquisition", "quick-service restaurants"),
    ("Director of Agronomy", "agricultural cooperative"),
    ("Head of Events", "enterprise software"),
    ("Practice Administrator", "dermatology clinics"),
    ("Director of Facilities", "commercial real estate"),
    ("VP of Engineering", "developer tools"),
    ("Head of Customer Success", "HR tech"),
]

FIRST_NAMES = [
    "Dana",
    "Marcus",
    "Priya",
    "Tom",
    "Elena",
    "Jordan",
    "Kwame",
    "Sofia",
    "Liam",
    "Grace",
    "Hiro",
    "Nadia",
]

LAST_NAMES = [
    "Whitfield",
    "Okafor",
    "Ramanathan",
    "Beck",
    "Moreau",
    "Castillo",
    "Novak",
    "Lindqvist",
    "Tanaka",
    "Osei",
]

PROSPECT_COMPANIES = [
    "Northgate Group",
    "Halcyon Foods",
    "Meridian Bancorp",
    "TruWest Distribution",
    "Bluepeak Software",
    "Cascadia Health Partners",
    "Ironwood Properties",
    "Summit Grain Co-op",
    "Beacon Hospitality",
    "Arcfield Systems",
    "Lakeshore Robotics",
    "Redbud Retail Group",
]

TRIGGERS = [
    "they just raised a Series B",
    "they posted three job openings on the team you sell into",
    "their CEO mentioned margin pressure on a podcast",
    "they expanded into two new regions last quarter",
    "a mutual connection suggested reaching out",
    "they were hit by an incident your product would have prevented",
    "their annual report flagged the exact problem you solve",
    "you met their colleague at a trade show last month",
    "they switched ERPs this year",
    "a competitor of theirs is already your customer",
]

GOALS = [
    "book a 20-minute intro call next week",
    "get a reply agreeing to see a short demo",
    "get forwarded to the right owner of this problem",
    "invite them to a small executive roundtable you're hosting",
]

TONES = [
    "direct and numbers-first",
    "warm and conversational",
    "brief and no-nonsense",
    "curious and consultative",
]

ITERATION_MOVES = [
    "make it shorter",
    "make the subject line punchier",
    "sound less salesy",
    "add a customer proof point or case study",
    "personalize the opening line to the trigger",
    "make the call to action softer",
    "make the call to action more specific",
    "add a short PS line",
    "make it more formal",
    "make it more casual",
    "lead with the pain point instead of the product",
    "cut the buzzwords",
    "try a completely different angle",
    "make it feel less like a template",
]


def build_personas(n: int, rng: random.Random) -> list[Persona]:
    personas = []
    for _ in range(n):
        segment = rng.randrange(len(SELLERS))
        seller_company, product = SELLERS[segment]
        prospect_title, industry = PROSPECT_TITLES[segment]
        persona = {
            "seller_name": f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}",
            "seller_role": "SDR" if rng.random() < 0.7 else "Account Executive",
            "seller_company": seller_company,
            "product": product,
            "prospect_name": f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}",
            "prospect_title": prospect_title,
            "prospect_company": rng.choice(PROSPECT_COMPANIES),
            "prospect_industry": industry,
            "trigger": rng.choice(TRIGGERS),
            "goal": rng.choice(GOALS),
            "tone": rng.choice(TONES),
            "iteration_moves": rng.sample(ITERATION_MOVES, k=4),
            "n_user_turns": rng.randint(3, 5),
        }
        personas.append(Persona.model_validate(persona))
    return personas
