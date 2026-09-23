"""Real estate tools: listing search, structured summaries and comparable properties.

Comparable scoring
------------------
Each candidate gets a similarity score in [0, 100]:

    score = 100 * (1 - sum(w_i * penalty_i) / sum(w_i))

with penalties in [0, 1] for distance (vs. search radius), living area,
bedrooms, bathrooms, age and lot size (skipped when either side has no lot,
e.g. condos). Candidates outside the radius or of a different property type
(when ``same_type``) are excluded. The value estimate is the median comp
price per sqft times the subject's living area.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from common.agent import ToolRegistry

DATA_DIR = Path(__file__).parent / "data"
PropertyType = Literal["single_family", "townhouse", "condo"]

WEIGHTS = {"distance": 0.30, "sqft": 0.25, "beds": 0.15, "baths": 0.10, "age": 0.10, "lot": 0.10}


class Listing(BaseModel):
    listing_id: str
    address: str
    neighborhood: str
    city: str
    lat: float
    lon: float
    property_type: PropertyType
    status: Literal["active", "sold"]
    list_price: int
    sold_price: int | None = None
    list_date: str
    sold_date: str | None = None
    beds: int
    baths: float
    sqft: int
    lot_sqft: int | None = None
    year_built: int
    hoa_monthly: int = 0
    features: list[str] = []
    description: str = ""

    @property
    def price(self) -> int:
        """Sold price when sold, otherwise asking price."""
        return self.sold_price or self.list_price

    @property
    def price_per_sqft(self) -> float:
        return self.price / self.sqft


def load_listings(data_dir: Path | None = None) -> dict[str, Listing]:
    raw = json.loads((Path(data_dir or DATA_DIR) / "listings.json").read_text())
    return {r["listing_id"]: Listing.model_validate(r) for r in raw}


def get(listings: dict[str, Listing], listing_id: str) -> Listing:
    key = listing_id.strip().upper()
    if key not in listings:
        raise KeyError(f"unknown listing {listing_id!r}")
    return listings[key]


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
def summarize(listings: dict[str, Listing], listing_id: str, as_of_year: int = 2026) -> dict:
    s = get(listings, listing_id)
    peers = [x for x in listings.values()
             if x.neighborhood == s.neighborhood and x.property_type == s.property_type]
    median_ppsf = statistics.median(x.price_per_sqft for x in peers)
    diff = (s.price_per_sqft / median_ppsf - 1) * 100
    if diff > 5:
        position = f"{diff:.0f}% above"
    elif diff < -5:
        position = f"{-diff:.0f}% below"
    else:
        position = "in line with"
    headline = (f"{s.beds} bd / {s.baths:g} ba {s.property_type.replace('_', ' ')}, "
                f"{s.sqft:,} sqft, built {s.year_built}, {s.neighborhood}")
    return {
        "listing_id": s.listing_id,
        "address": f"{s.address}, {s.city}",
        "headline": headline,
        "status": s.status,
        "price": s.price,
        "price_label": "sold price" if s.sold_price else "list price",
        "price_per_sqft": round(s.price_per_sqft, 1),
        "neighborhood_median_ppsf": round(median_ppsf, 1),
        "pricing_position": f"{position} the {s.neighborhood} median for this property type "
                            f"(n={len(peers)})",
        "age_years": as_of_year - s.year_built,
        "lot_sqft": s.lot_sqft,
        "hoa_monthly": s.hoa_monthly,
        "key_features": s.features,
        "description": s.description,
    }


# --------------------------------------------------------------------------- #
# Comparables
# --------------------------------------------------------------------------- #
class Comparable(BaseModel):
    listing_id: str
    address: str
    status: str
    price: int
    price_per_sqft: float
    beds: int
    baths: float
    sqft: int
    year_built: int
    distance_km: float
    similarity: float = Field(description="0-100, higher is more similar")
    differences: list[str]


def score_pair(subject: Listing, c: Listing, radius_km: float) -> tuple[float, float]:
    """Return (similarity 0-100, distance km)."""
    dist = haversine_km(subject.lat, subject.lon, c.lat, c.lon)
    pen = {
        "distance": min(dist / radius_km, 1.0),
        "sqft": min(abs(c.sqft - subject.sqft) / subject.sqft, 1.0),
        "beds": min(abs(c.beds - subject.beds) / 3, 1.0),
        "baths": min(abs(c.baths - subject.baths) / 2, 1.0),
        "age": min(abs(c.year_built - subject.year_built) / 40, 1.0),
    }
    if subject.lot_sqft and c.lot_sqft:
        pen["lot"] = min(abs(c.lot_sqft - subject.lot_sqft) / subject.lot_sqft, 1.0)
    wsum = sum(WEIGHTS[k] for k in pen)
    score = 100 * (1 - sum(WEIGHTS[k] * v for k, v in pen.items()) / wsum)
    return round(score, 1), round(dist, 2)


def _differences(s: Listing, c: Listing) -> list[str]:
    out = []
    if c.beds != s.beds:
        out.append(f"{c.beds - s.beds:+d} bed")
    if c.baths != s.baths:
        out.append(f"{c.baths - s.baths:+g} bath")
    if abs(c.sqft - s.sqft) >= 100:
        out.append(f"{c.sqft - s.sqft:+,} sqft")
    if abs(c.year_built - s.year_built) >= 5:
        out.append(f"built {c.year_built}")
    return out


def comparables(listings: dict[str, Listing], listing_id: str, radius_km: float = 3.0,
                max_results: int = 5, status: Literal["sold", "active", "any"] = "sold",
                same_type: bool = True) -> dict:
    s = get(listings, listing_id)
    comps: list[Comparable] = []
    for c in listings.values():
        if c.listing_id == s.listing_id:
            continue
        if status != "any" and c.status != status:
            continue
        if same_type and c.property_type != s.property_type:
            continue
        score, dist = score_pair(s, c, radius_km)
        if dist > radius_km:
            continue
        comps.append(Comparable(
            listing_id=c.listing_id, address=c.address, status=c.status, price=c.price,
            price_per_sqft=round(c.price_per_sqft, 1), beds=c.beds, baths=c.baths, sqft=c.sqft,
            year_built=c.year_built, distance_km=dist, similarity=score,
            differences=_differences(s, c),
        ))
    comps.sort(key=lambda x: (-x.similarity, x.distance_km))
    comps = comps[:max_results]

    estimate = None
    if comps:
        ppsf = [c.price_per_sqft for c in comps]
        mid = statistics.median(ppsf)
        estimate = {
            "method": "median comp $/sqft x subject sqft",
            "estimated_value": int(round(mid * s.sqft, -3)),
            "low": int(round(min(ppsf) * s.sqft, -3)),
            "high": int(round(max(ppsf) * s.sqft, -3)),
            "subject_price": s.price,
            "subject_vs_estimate_pct": round((s.price / (mid * s.sqft) - 1) * 100, 1),
        }
    return {
        "subject": {"listing_id": s.listing_id, "address": s.address, "price": s.price,
                    "sqft": s.sqft, "beds": s.beds, "baths": s.baths,
                    "property_type": s.property_type},
        "criteria": {"radius_km": radius_km, "status": status, "same_type": same_type},
        "comparables": [c.model_dump() for c in comps],
        "estimate": estimate,
    }


# --------------------------------------------------------------------------- #
# Agent tools
# --------------------------------------------------------------------------- #
ListingId = Annotated[str, Field(description="listing id, e.g. L-2001")]


def build_registry(data_dir: Path | None = None,
                   listings: dict[str, Listing] | None = None) -> ToolRegistry:
    listings = listings or load_listings(data_dir)
    reg = ToolRegistry()

    @reg.tool
    def search_listings(
        neighborhood: str | None = None,
        property_type: PropertyType | None = None,
        status: Literal["active", "sold", "any"] = "active",
        min_beds: Annotated[int, Field(ge=0)] = 0,
        max_price: Annotated[int | None, Field(ge=0)] = None,
        limit: Annotated[int, Field(ge=1, le=50)] = 10,
    ) -> list[dict]:
        """Search listings by neighborhood, type, status, bedrooms and max price."""
        out = []
        for x in listings.values():
            if neighborhood and x.neighborhood.lower() != neighborhood.lower():
                continue
            if property_type and x.property_type != property_type:
                continue
            if status != "any" and x.status != status:
                continue
            if x.beds < min_beds or (max_price is not None and x.price > max_price):
                continue
            out.append({"listing_id": x.listing_id, "address": x.address,
                        "neighborhood": x.neighborhood, "property_type": x.property_type,
                        "status": x.status, "price": x.price, "beds": x.beds,
                        "baths": x.baths, "sqft": x.sqft})
        return sorted(out, key=lambda r: r["price"])[:limit]

    @reg.tool
    def get_listing(listing_id: ListingId) -> Listing:
        """Return the full record for one listing."""
        return get(listings, listing_id)

    @reg.tool
    def summarize_listing(listing_id: ListingId) -> dict:
        """Structured summary of a listing: headline facts, $/sqft vs. neighborhood, features."""
        return summarize(listings, listing_id)

    @reg.tool
    def find_comparables(
        listing_id: ListingId,
        radius_km: Annotated[float, Field(gt=0, le=25)] = 3.0,
        max_results: Annotated[int, Field(ge=1, le=20)] = 5,
        status: Literal["sold", "active", "any"] = "sold",
        same_type: bool = True,
    ) -> dict:
        """Find the most similar nearby properties (distance + attributes) and estimate value."""
        return comparables(listings, listing_id, radius_km, max_results, status, same_type)

    return reg
