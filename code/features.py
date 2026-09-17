# column names and feature engineering
# imported by both training and the API, so the formulas cannot differ

TARGET = "median_house_value"

NUMERIC = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
]

CATEGORICAL = ["ocean_proximity"]

DERIVED = ["rooms_per_household", "bedrooms_per_room", "population_per_household"]

# what the model finally sees
MODEL_NUMERIC = NUMERIC + DERIVED


def add_features(df):
    df = df.copy()
    df["rooms_per_household"] = df["total_rooms"] / df["households"]
    df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"]
    df["population_per_household"] = df["population"] / df["households"]
    return df
