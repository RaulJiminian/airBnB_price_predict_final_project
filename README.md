# Airbnb NYC Price Prediction Project

## Project Overview

This project aims to predict Airbnb listing prices in New York City using machine learning techniques. The goal is to help hosts competitively price their properties and maximize their earnings.

## Dataset

- Source: Inside Airbnb's NYC dataset
- File: listings.csv
- Size: 37,018 entries with 79 columns
- Last Updated: May 2025

## Project Structure

```
airBnB_price_predict_final_project/
├── data/
│   ├── listings.csv
│   ├── cleaned_listings.csv
│   ├── cleaned_listings_final.csv
│   └── model_ready_listings.csv
├── notebook-analysis/
│   └── AirBnB_Price_Prediction_Project.ipynb
├── README.md
└── Project_Proposal_AirBnB_Price_Prediction.md
```

## Implementation Steps

### Step 1: Data Cleaning ✅

- [x] Load listings.csv into pandas DataFrame
- [x] Drop irrelevant columns
- [x] Convert price to numeric format
- [x] Handle missing values
- [x] Encode categorical variables
- [x] Create new features
- [x] Second pass: Complete missing value elimination
- [x] Final optimization pass for modeling performance

**Key Data Cleaning Steps Completed:**

**First Pass:**

1. **Import additional libraries** - Added numpy, matplotlib, seaborn, and sklearn's LabelEncoder
2. **Drop irrelevant columns** - Removed URLs, IDs, descriptions, and other non-predictive features (reduced from 79 to ~60 columns)
3. **Convert price to numeric** - Cleaned price column from string format (e.g., "$150.00") to float values
4. **Handle missing values** - Filled missing values in key columns (bedrooms, bathrooms, beds, review_scores_rating) with median values
5. **Encode categorical variables** - Converted categorical features (room_type, neighbourhood_group_cleansed, property_type) to numeric using label encoding
6. **Create new features** - Added three new predictive features:
   - `price_per_accommodate`: Price divided by number of guests accommodated
   - `has_reviews`: Binary indicator if listing has any reviews
   - `review_score_ratio`: Normalized review score (0-1 scale)
7. **Final data check** - Verified cleaned dataset and saved to `cleaned_listings.csv`

**Second Pass - Complete Missing Value Elimination:** 8. **Drop low-value columns** - Removed additional columns with high missing values or low predictive power:

- `neighbourhood` (redundant with neighbourhood_group_cleansed)
- `first_review`, `last_review` (time-series complexity)
- `host_location`, `host_since` (high missing values, low predictive value)

9. **Handle review-related missing values** - Filled all review score columns with 0 (indicating no reviews)
10. **Handle host-related missing values** - Filled host attributes with appropriate defaults:
    - `host_is_superhost`: filled with mode
    - `has_availability`: converted to binary and filled with mode
    - Host response columns: filled with median or 'unknown'
11. **Handle remaining missing values** - Systematic approach:
    - Numerical columns: filled with median values
    - Categorical columns: filled with mode or appropriate defaults
12. **Encode remaining categorical variables** - Automatically detected and encoded any remaining categorical columns
13. **Final verification** - Ensured ALL columns have exactly 21,833 non-null values (no missing data)
14. **Save final dataset** - Exported to `cleaned_listings_final.csv`

**Final Optimization Pass - Modeling Performance Enhancement:** 15. **Log transformation of target variable** - Created `log_price = np.log(price + 1)` to address severe skewness: - Original price skewness: ~23.19 (heavily skewed) - Log-transformed skewness: significantly reduced for better model performance - Conversion formula: `np.exp(log_price_prediction) - 1` to get actual price 16. **Redundant feature removal** - Systematically dropped highly correlated features to reduce multicollinearity: - Review redundancy: Dropped `review_score_ratio`, kept `review_scores_rating` - Host redundancy: Dropped `host_total_listings_count`, kept `host_listings_count` - Property redundancy: Dropped `property_type`, kept `room_type` - Review count redundancy: Dropped `number_of_reviews_ltm`, `number_of_reviews_ly`, `reviews_per_month` - Availability redundancy: Dropped `availability_eoy`, kept `availability_365` - Nights redundancy: Dropped all `_minimum`, `_maximum`, `_avg_ntm` variants - Host listings redundancy: Dropped room-type-specific counts, kept main count - Occupancy redundancy: Dropped `estimated_occupancy_l365d` 17. **Scaling preparation** - Identified numerical features requiring standardization for linear models: - Tree-based models (Random Forest, XGBoost): No scaling needed - Linear models (Ridge, Lasso, KNN): StandardScaler required for numerical features 18. **Final dataset optimization** - Exported model-ready dataset as `model_ready_listings.csv`: - Optimized feature set with reduced multicollinearity - Log-transformed target variable for better model performance - All preprocessing complete and ready for feature selection

### Step 2: Exploratory Data Analysis (EDA) ✅

- [x] Basic statistical analysis
- [x] Price distribution visualization
- [x] Feature correlation analysis
- [x] Outlier detection

**Key EDA Analysis Completed:**

**2.1 Target Distribution Analysis:**

- **Price statistics and outlier detection** using IQR method with bounds calculation
- **Skewness analysis** with automated recommendations for log transformation
- **Comprehensive visualizations**: histogram, box plot, log-transformed distribution, and Q-Q plot
- **Distribution assessment** to determine if price transformation is needed for modeling

**2.2 Categorical Feature Analysis:**

- **Decoded categorical variables** for readable visualizations using label encoders
- **Price analysis by room type and borough** with bar charts showing mean prices and sample counts
- **Box plot distributions** comparing price ranges across different categories
- **Summary statistics** for each categorical level (mean, median, count)

**2.3 Numerical Feature Analysis:**

- **Scatter plots with trend lines** for 8 key numerical features vs price
- **Correlation coefficients** calculated and displayed for each feature relationship
- **Feature relationship assessment** for accommodates, bedrooms, bathrooms, beds, reviews, etc.
- **Visual identification** of linear and non-linear relationships

**2.4 Correlation Heatmap:**

- **Comprehensive correlation matrix** of all numerical features with masked upper triangle
- **High correlation pairs identification** (|r| > 0.7) to detect multicollinearity
- **Top features correlated with price** ranking for feature importance insights
- **Visual correlation patterns** using color-coded heatmap

**2.5 Data Quality & Leakage Check:**

- **Encoding quality verification** for binary (0/1) and categorical variables
- **Data leakage detection** for features like estimated_revenue_l365d that could "cheat"
- **Final missing values confirmation** ensuring 21,833 non-null values across all columns
- **Sanity checks** for all data transformations and encodings

**2.6 EDA Summary & Recommendations:**

- **Comprehensive findings summary** with dataset overview and key insights
- **Modeling recommendations** based on price distribution, skewness, and feature relationships
- **Data quality status report** confirming readiness for feature selection
- **Next steps preparation** with specific guidance for Step 3

### Step 3: Feature Selection 🔍

- [ ] Split features and target
- [ ] Apply feature selection techniques
- [ ] Document feature importance

### Step 4: Model Selection & Training 🤖

- [ ] Implement baseline models
- [ ] Train advanced models
- [ ] Model evaluation and comparison

### Step 5: Final Model & Deployment 🎯

- [ ] Select best performing model
- [ ] Export model
- [ ] Create prediction interface

## Dependencies

- Python 3.12.7
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- scipy (for statistical analysis)

## Progress Log

- [2024-03-19] Project initialization
- [2024-03-19] Completed Step 1: Data Cleaning (First Pass)
  - Successfully cleaned and preprocessed the dataset
  - Reduced dataset from 79 to ~60 relevant columns
  - Converted price column to numeric format
  - Handled missing values using median imputation
  - Encoded categorical variables for machine learning
  - Created new engineered features for better prediction
  - Saved cleaned dataset for future analysis
- [2024-03-19] Completed Step 1: Data Cleaning (Second Pass)
  - Eliminated ALL missing values from the dataset
  - Dropped additional low-value columns (neighbourhood, time-series columns, etc.)
  - Comprehensive missing value handling for review and host-related features
  - Final dataset: 21,833 rows with NO missing values
  - All categorical variables properly encoded
  - Saved final cleaned dataset as `cleaned_listings_final.csv`
- [2024-03-19] Completed Step 2: Exploratory Data Analysis (EDA)
  - Comprehensive price distribution analysis with skewness assessment
  - Detailed categorical feature analysis (room types, boroughs) with visualizations
  - Numerical feature correlation analysis with scatter plots and trend lines
  - Created correlation heatmap identifying feature relationships and multicollinearity
  - Performed data quality checks and data leakage detection
  - Generated modeling recommendations based on data characteristics
  - Confirmed dataset readiness for feature selection and modeling
- [2024-03-19] Completed Step 1: Final Optimization Pass
  - Applied log transformation to target variable (log_price) to address severe skewness
  - Removed redundant and highly correlated features to reduce multicollinearity
  - Identified features requiring scaling for linear models vs tree-based models
  - Created final optimized dataset saved as `model_ready_listings.csv`
  - Dataset now optimally prepared for feature selection and model training
