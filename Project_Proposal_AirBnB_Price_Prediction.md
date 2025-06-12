# Project Proposal: Airbnb NYC Price Prediction App

## Problem Statement

I would like to help short-term rental hosts competitively price their properties in New York City. Pricing correctly can improve booking rates and maximize earnings. The goal is to develop a predictive model that estimates the nightly price of a listing based on its attributes (e.g., location, property type, number of bedrooms, review scores).

Stretch Goal: I will also like to deploy a simple app that allows users to enter listing details and receive a price estimate from the trained model.

## Hypothesis / Assumptions

- Listings with more bedrooms, higher review scores, and better location (e.g., Manhattan) are likely to command higher prices.
- Hosts with superhost status or more reviews may earn more per night.
- Some amenities may have a measurable effect on pricing (e.g., pool, hot tub).

## Goals & Success Metrics

- Build a regression model to predict price with **Root Mean Square Error (RMSE) < $50**.
- Use feature importance to identify which variables drive price most significantly.
- Stretch: Deploy a simple app (e.g., Streamlit or React/Flask) where users input listing features and see predicted prices.

## Risks / Limitations

- **Missing or noisy data**: Some listings lack reviews or pricing fields (e.g., only ~21,800 out of ~37,000 entries have price values).
- **Price formatting**: The price field is stored as a string (e.g., "$150.00") and will require cleaning.
- **Outliers**: Extremely high-priced listings may skew the model.
- **Temporal factors** (like seasonality or current demand) are not present in this static dataset.

## Data Sources

- **Primary Dataset**: `listings.csv` from [Inside Airbnb’s NYC dataset](http://insideairbnb.com/get-the-data.html) (37018 entries, 79 columns)
  - Fields include: `latitude`, `longitude`, `room_type`, `bedrooms`, `number_of_reviews`, `review_scores_rating`, `availability_365`, and `price`.
- **Permissions**: Data is publicly available and used for educational purposes.

## Data Cleaning & Feature Engineering

- Remove columns irrelevant to pricing (e.g., URLs, IDs, free-text descriptions).
- Convert `price` to numeric, handle missing prices.
- Encode categorical variables (`room_type`, `neighbourhood_group_cleansed`, etc.).
- Normalize or scale numerical features.
- Create new features like:
  - `price_per_accommodate`
  - `has_reviews` (binary)
  - `is_superhost` (binary)
  - `average_review_score` (aggregated from components)

## Planned ML Techniques

- Linear Regression (baseline)
- Decision Tree Regressor
- Random Forest Regressor
- Model comparison using RMSE and R² on a test set

## Deliverables

- 📈 Trained ML model predicting price of listings
- 📊 Summary of feature importance
- 🧼 Cleaned dataset with selected features
- 💻 **Interactive App**: User enters listing info → model returns a predicted price
- 🗂️ Project write-up (Jupyter notebook) with EDA, modeling, evaluation, and deployment summary
