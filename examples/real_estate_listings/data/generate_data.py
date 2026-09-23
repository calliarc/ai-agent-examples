"""Regenerate the SYNTHETIC real estate listings (deterministic, stdlib only).

    python examples/real_estate_listings/data/generate_data.py

Neighborhoods, addresses, coordinates and prices are all made up. Coordinates
are placed in an arbitrary grid and do not refer to real properties.
"""

from __future__ import annotations

import datetime as dt
import json
import random
from pathlib import Path

HERE = Path(__file__).parent

# name, center lat, center lon, $/sqft base
NEIGHBORHOODS = [
    ("Maple Hollow", 39.9612, -82.9988, 245),
    ("Cedar Heights", 39.9780, -83.0205, 310),
    ("Riverside Flats", 39.9505, -82.9770, 205),
]
STREETS = ["Birch", "Willow", "Lantern", "Harbor", "Juniper", "Summit", "Orchard", "Fox Run",
           "Aspen", "Quarry", "Linden", "Meadow"]
SUFFIX = ["St", "Ave", "Ln", "Ct", "Dr", "Way"]
FEATURES = ["hardwood floors", "updated kitchen", "quartz countertops", "finished basement",
            "two-car garage", "fenced yard", "solar panels", "primary suite", "open floor plan",
            "new roof (2023)", "covered patio", "walk-in closets", "energy-efficient windows",
            "fireplace", "home office", "EV charger"]
TYPES = [("single_family", 0.6), ("townhouse", 0.25), ("condo", 0.15)]


def pick_type(rng: random.Random) -> str:
    r, acc = rng.random(), 0.0
    for t, p in TYPES:
        acc += p
        if r < acc:
            return t
    return TYPES[-1][0]


def main() -> None:
    rng = random.Random(7)
    listings = []
    for i in range(1, 37):
        hood, lat0, lon0, ppsf = NEIGHBORHOODS[(i - 1) % 3]
        ptype = pick_type(rng)
        beds = {"condo": rng.choice([1, 2, 2, 3]), "townhouse": rng.choice([2, 3, 3]),
                "single_family": rng.choice([3, 3, 4, 4, 5])}[ptype]
        baths = max(1.0, beds - rng.choice([0, 1, 1.5]))
        sqft = int(rng.gauss(520 + beds * 430, 180) // 10 * 10)
        year = rng.randint(1958, 2022)
        lot_mean = 6500 if ptype == "single_family" else 2200
        lot = None if ptype == "condo" else int(rng.gauss(lot_mean, 900) // 50 * 50)
        feats = rng.sample(FEATURES, rng.randint(3, 6))
        age_factor = 1 - (2026 - year) * 0.002
        price = round(sqft * ppsf * age_factor * rng.uniform(0.9, 1.12) + 5000 * len(feats), -3)
        sold = rng.random() < 0.55
        listed = dt.date(2026, 9, 1) - dt.timedelta(days=rng.randint(5, 200))
        rec = {
            "listing_id": f"L-{2000 + i}",
            "address": f"{rng.randint(100, 9899)} {rng.choice(STREETS)} {rng.choice(SUFFIX)}",
            "neighborhood": hood,
            "city": "Fairhaven",  # fictional
            "lat": round(lat0 + rng.uniform(-0.012, 0.012), 5),
            "lon": round(lon0 + rng.uniform(-0.015, 0.015), 5),
            "property_type": ptype,
            "status": "sold" if sold else "active",
            "list_price": int(price),
            "sold_price": int(round(price * rng.uniform(0.95, 1.04), -3)) if sold else None,
            "list_date": listed.isoformat(),
            "sold_date": (listed + dt.timedelta(days=rng.randint(7, 60))).isoformat()
            if sold else None,
            "beds": beds,
            "baths": baths,
            "sqft": sqft,
            "lot_sqft": lot,
            "year_built": year,
            "hoa_monthly": rng.choice([0, 0, 150, 240, 310]) if ptype != "single_family" else 0,
            "features": feats,
            "description": (
                f"{'Charming' if year < 1990 else 'Modern'} {beds}-bed "
                f"{ptype.replace('_', ' ')} in {hood} with {feats[0]} and {feats[1]}. "
                f"Close to parks, schools and the Fairhaven riverfront trail."
            ),
        }
        listings.append(rec)
    (HERE / "listings.json").write_text(json.dumps(listings, indent=2) + "\n")


if __name__ == "__main__":
    main()
