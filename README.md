# Albany Airbnb Price Prediction (Linear Regression + PyTorch)

This project predicts nightly Airbnb prices for **Albany, New York** using data from [Inside Airbnb](http://insideairbnb.com/).

It covers:
- **Feature engineering** (distance to city center, amenities, reviews)
- **Preprocessing pipelines** (scaling + one-hot encoding via `ColumnTransformer`)
- **Linear models**: Ordinary Least Squares, Ridge, Lasso
- **From-scratch PyTorch** linear regression (Ridge-like with L2 regularization)
- **Human-friendly price predictions** using a `predict_price()` helper

---

## Dataset

- **Source:** Inside Airbnb → `listings.csv` for Albany, NY (download manually)
- **Target variable:** `price` (modeled as `log_price = log1p(price)` for stability)
- **File location:** Place `listings.csv` in `data/` folder (not tracked in GitHub)

---

## Features Used

### Location
- `dist_city_km`: Haversine distance from listing to Albany center `(42.6499, -73.7573)`

### Property Attributes
- `accommodates`, `bedrooms`, `bathrooms`, `minimum_nights`

### Social Proof
- `number_of_reviews`, `review_scores_rating`

### Amenities
- `amenities_count` + binary flags:
  - `amen_wifi`, `amen_kitchen`, `amen_air_conditioning`
  - `amen_heating`, `amen_washer`, `amen_elevator`, `amen_tv`

### Derived
- `reviews_per_night_min = number_of_reviews / (minimum_nights + 1)`

---

## Results

**5-Fold CV R² (log-price)**:  
```
[0.8541, 0.8913, 0.8728, 0.8020, 0.8485]
Mean: ~0.854
```

- **R² (log):** ≈ 0.85 → model explains ~85% of variance in log-price on unseen data  
- **RMSE / MAE:** computed on original scale after inverse-transform

---

## Example Prediction

Once the model is trained, you can pass in details for a new listing and instantly get a predicted nightly price.

```python
example_details = {
    "latitude": 42.6526, "longitude": -73.7562,
    "room_type": "Entire home/apt", "neighbourhood_cleansed": "Central Avenue",
    "accommodates": 4, "bedrooms": 2, "bathrooms": 1.0, "minimum_nights": 2,
    "number_of_reviews": 35, "review_scores_rating": 95,
    "amen_wifi": 1, "amen_kitchen": 1, "amen_air_conditioning": 1,
    "amen_heating": 1, "amen_washer": 0, "amen_elevator": 0, "amen_tv": 1,
    "amenities_count": 7
}

predict_price(example_details)
# Output: Estimated nightly price: $85
```

---

## How to Run

### Google Colab
1. Open `notebooks/albany_airbnb_price.ipynb` in Colab.
2. Upload `data/listings.csv` when prompted (or mount Google Drive).
3. Run all cells to train and evaluate the model.

### Local
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab  # or jupyter notebook
```

---

## Project Structure

```
airbnb-albany-price/
├─ notebooks/
│  └─ albany_airbnb_price.ipynb     # main notebook
├─ data/
│  └─ README.md                     # explains dataset download
```

---

## Requirements

```
numpy
pandas
scikit-learn
matplotlib
torch
joblib
```

---

## Notes

- Train/val/test split uses fixed random state for reproducibility
- Preprocessing (scaling + encoding) fit on **train only**
- Avoided target leakage by not including price-derived features
- Seeds set for NumPy and PyTorch

---
