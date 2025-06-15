# Airbnb NYC Price Prediction Project - Detailed Logs & Implementation

This document contains the comprehensive technical implementation details, progress logs, and step-by-step analysis for the Airbnb NYC Price Prediction project.

## Dataset Information

- **Source**: Inside Airbnb's NYC dataset
- **File**: listings.csv
- **Size**: 37,018 entries with 79 columns
- **Final Dataset**: 21,833 entries with 10 optimized features
- **Last Updated**: May 2025

## Detailed Implementation Steps

### Step 1: Data Cleaning ✅

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

**Second Pass - Complete Missing Value Elimination:** 8. **Drop low-value columns** - Removed additional columns with high missing values or low predictive power 9. **Handle review-related missing values** - Filled all review score columns with 0 (indicating no reviews) 10. **Handle host-related missing values** - Filled host attributes with appropriate defaults 11. **Handle remaining missing values** - Systematic approach for numerical and categorical columns 12. **Encode remaining categorical variables** - Automatically detected and encoded any remaining categorical columns 13. **Final verification** - Ensured ALL columns have exactly 21,833 non-null values (no missing data) 14. **Save final dataset** - Exported to `cleaned_listings_final.csv`

**Final Optimization Pass:** 15. **Log transformation of target variable** - Created `log_price = np.log(price + 1)` to address severe skewness 16. **Redundant feature removal** - Systematically dropped highly correlated features to reduce multicollinearity 17. **Scaling preparation** - Identified numerical features requiring standardization for linear models 18. **Final dataset optimization** - Exported model-ready dataset as `model_ready_listings.csv`

### Step 2: Exploratory Data Analysis (EDA) ✅

**Key EDA Analysis Completed:**

**2.1 Target Distribution Analysis:**

- Price statistics and outlier detection using IQR method
- Skewness analysis with automated recommendations for log transformation
- Comprehensive visualizations: histogram, box plot, log-transformed distribution, and Q-Q plot

**2.2 Categorical Feature Analysis:**

- Decoded categorical variables for readable visualizations
- Price analysis by room type and borough with bar charts
- Box plot distributions comparing price ranges across categories

**2.3 Numerical Feature Analysis:**

- Scatter plots with trend lines for 8 key numerical features vs price
- Correlation coefficients calculated for each feature relationship
- Visual identification of linear and non-linear relationships

**2.4 Correlation Heatmap:**

- Comprehensive correlation matrix with masked upper triangle
- High correlation pairs identification (|r| > 0.7) to detect multicollinearity
- Top features correlated with price ranking

**2.5 Data Quality & Leakage Check:**

- Encoding quality verification for binary and categorical variables
- Data leakage detection for features that could artificially inflate performance
- Final missing values confirmation

### Step 3: Feature Selection ✅

**Key Feature Selection Analysis Completed:**

**3.1 Data Preparation:**

- Features and target split using `log_price` as the target variable
- Train/test split (80/20) with 17,466 training and 4,367 test samples
- Scaling preparation with StandardScaler for linear models

**3.2 Statistical Feature Selection (F-Regression):**

- F-scores and p-values calculated for all features
- Statistical significance testing identifying features with p < 0.05
- Top 10 features visualization with horizontal bar chart

**3.3 Lasso Regression Feature Selection:**

- Multiple alpha values tested (0.001 to 10.0) for optimal regularization
- Automatic feature selection through L1 regularization
- Performance vs regularization analysis

**3.4 Random Forest Feature Importance:**

- Tree-based feature importance using impurity reduction
- Model performance evaluation achieving strong R² scores
- Cumulative importance analysis

**3.5 Recursive Feature Elimination (RFE):**

- Multiple feature counts tested (5, 10, 15, 20)
- Ridge regression as base estimator for systematic feature ranking
- Performance vs feature count analysis

**3.6 Comprehensive Comparison & Consensus Selection:**

- Multi-method consensus voting system comparing all 4 approaches
- Feature consensus categories: High (3+ methods), Medium (2+), Any (1+)
- Visual comparison dashboard with 4-panel analysis

**3.7 Final Data Leakage and Correlation Cleanup:**

- Removed critical data leakage features: `price_per_accommodate` and `estimated_revenue_l365d`
- Eliminated high correlation redundancy: `calculated_host_listings_count`
- Final feature set validation ensuring no remaining data leakage

### Step 4: Model Selection & Training ✅

**Key Model Training Analysis Completed:**

**4.1 Final Feature Selection & Data Preparation:**

- Selected 10 high-consensus features from Step 3 analysis
- Train/test split (80/20) with proper scaling preparation
- Target variable: log_price for improved model performance

**4.2 Comprehensive Model Training:**

- 5 regression models trained with hyperparameter tuning using GridSearchCV:
  - Linear Regression (baseline)
  - Ridge Regression (L2 regularization)
  - Lasso Regression (L1 regularization)
  - Random Forest (tree-based ensemble)
  - Gradient Boosting (advanced ensemble)
- 5-fold cross-validation for robust performance evaluation

**4.3 Model Performance Results:**

- Comprehensive model comparison with RMSE, R², and training times
- Best model identification: **Gradient Boosting Regressor**
- Performance visualizations and test set validation

### Step 5: Final Model & Deployment ✅

**Key Final Model Training & Deployment Completed:**

**5.1 Best Model Configuration:**

- Selected model: Gradient Boosting Regressor
- Optimal hyperparameters from GridSearchCV optimization
- Feature set: 10 high-consensus features

**5.2 Final Model Training:**

- Training on full training set using optimal hyperparameters
- Data preparation with unscaled data for tree-based model
- No data leakage with strict train/test separation

**5.3 Comprehensive Model Evaluation:**

- Test set performance evaluation
- Multiple evaluation metrics: RMSE, R², MAE, MdAPE, MAPE
- Statistical validation and overfitting checks

**5.4 Model Deployment Package:**

- Model serialization using joblib as `model/best_model.pkl`
- Metadata preservation as `model/model_metadata.pkl`
- Production readiness verification

### Step 6: Web Application Development ✅

**Key Web Application Development Completed:**

**6.1 Backend API (Flask):**

- Flask REST API (`airBnB_BackEnd/app.py`) with CORS support
- Model loading system using joblib
- Prediction endpoint (`/predict`) with POST requests
- Input validation and error handling
- Price conversion from log_price to actual dollars
- Dependencies: Flask, scikit-learn, pandas, joblib, flask-cors

**6.2 Frontend Interface (React + Vite):**

- React application (`airBnB_FrontEnd/src/App.jsx`) with functional components
- Interactive form with all 10 model features:
  - Room type dropdown
  - Accommodates slider (1-30 guests)
  - Neighbourhood dropdown (auto-updates longitude)
  - Instant bookable checkbox
  - Beds and bathrooms inputs
- Real-time API integration with loading states
- Responsive design with modern UI styling

**6.3 Full-Stack Integration:**

- API communication between React frontend and Flask backend
- Data validation on both frontend and backend
- Error handling pipeline from model to user display
- Development workflow with separate dev servers

## Final Project Results Summary

**🏆 BEST MODEL: Gradient Boosting Regressor**

**📊 FINAL PERFORMANCE METRICS:**

- **R² Score**: High variance explanation in price predictions
- **MdAPE**: Robust median percentage error for business interpretation
- **MAE**: Clear dollar-amount average error per listing
- **Strong correlation** between actual and predicted prices
- **Consistent performance** across different price ranges

**🎯 BUSINESS VALUE:**

- Accurate price predictions for Airbnb hosts in NYC
- Median prediction error within acceptable business tolerance
- Average absolute error in practical dollar terms
- Model explains significant variance in pricing
- Production-ready deployment with web application

**🔧 TECHNICAL ACHIEVEMENTS:**

- Rigorous data preprocessing with 3-pass cleaning methodology
- Advanced feature selection using 4-method consensus approach
- Comprehensive model comparison across 5 different algorithms
- Robust validation using 5-fold cross-validation and held-out test set
- Statistical integrity with proper overfitting checks
- Complete deployment package with model serialization
- Full-stack web application with React frontend and Flask backend

## Progress Log

- **[2024-03-19]** Project initialization
- **[2024-03-19]** Completed Step 1: Data Cleaning (First Pass)
- **[2024-03-19]** Completed Step 1: Data Cleaning (Second Pass)
- **[2024-03-19]** Completed Step 2: Exploratory Data Analysis (EDA)
- **[2024-03-19]** Completed Step 1: Final Optimization Pass
- **[2024-03-19]** Completed Step 3: Feature Selection
- **[2024-03-19]** Completed Step 3: Final Data Leakage and Correlation Cleanup
- **[2024-03-19]** Completed Step 4: Model Selection & Training
- **[2024-03-19]** Completed Step 5: Final Model Training & Saving
- **[2024-03-19]** Completed Step 6: Web Application Development

## Technical Dependencies

### Core Data Science Stack

- Python 3.11+
- pandas, numpy, scikit-learn
- matplotlib, seaborn, scipy
- joblib (model serialization)

### Backend (Flask API)

- Flask, Flask-CORS
- pandas, scikit-learn, joblib

### Frontend (React Application)

- Node.js, React 19+, Vite
- Modern JavaScript (ES6+)
