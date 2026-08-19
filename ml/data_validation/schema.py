"""Canonical schema for the repository's flight-price dataset."""
from __future__ import annotations

EXPECTED_COLUMNS = [
    "Unnamed: 0", "airline", "flight", "source_city", "departure_time",
    "stops", "arrival_time", "destination_city", "class", "duration",
    "days_left", "price",
]
TARGET = "price"
IDENTIFIER_COLUMNS = ["Unnamed: 0"]
CATEGORICAL_FEATURES = [
    "airline", "flight", "source_city", "departure_time", "stops",
    "arrival_time", "destination_city", "class",
]
NUMERICAL_FEATURES = ["duration", "days_left"]
FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
SCHEMA_VERSION = "1.0.0"

# These are observed/semantic constraints, not a license to delete outliers.
DOMAIN_CONSTRAINTS = {
    "duration": {"min": 0.0, "description": "Flight duration in hours must be non-negative."},
    "days_left": {"min": 0, "description": "Booking horizon in days must be non-negative."},
    "price": {"min": 0.0, "description": "Flight price must be non-negative."},
}

CATEGORY_FIELDS = {
    "airline": ["AirAsia", "Air_India", "GO_FIRST", "Indigo", "SpiceJet", "Vistara"],
    "source_city": ["Bangalore", "Chennai", "Delhi", "Hyderabad", "Kolkata", "Mumbai"],
    "destination_city": ["Bangalore", "Chennai", "Delhi", "Hyderabad", "Kolkata", "Mumbai"],
    "departure_time": ["Early_Morning", "Morning", "Afternoon", "Evening", "Night", "Late_Night"],
    "arrival_time": ["Early_Morning", "Morning", "Afternoon", "Evening", "Night", "Late_Night"],
    "stops": ["zero", "one", "two_or_more"],
    "class": ["Economy", "Business"],
}
