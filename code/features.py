# описание колонок и генерация признаков
# импортируется и обучением, и API, чтобы формулы не разъехались

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

# то, что в итоге видит модель
MODEL_NUMERIC = NUMERIC + DERIVED


def add_features(df):
    df = df.copy()
    df["rooms_per_household"] = df["total_rooms"] / df["households"]
    df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"]
    df["population_per_household"] = df["population"] / df["households"]
    return df
