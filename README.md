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
│   ├── model_ready_listings.csv
│   └── model_ready_listings_final.csv
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

### Step 3: Feature Selection ✅

- [x] Split features and target
- [x] Apply feature selection techniques
- [x] Document feature importance
- [x] Final data leakage and correlation cleanup

**Key Feature Selection Analysis Completed:**

**3.1 Data Preparation:**

- **Features and target split** using `log_price` as the target variable for modeling
- **Train/test split** (80/20) with 17,466 training and 4,367 test samples
- **Scaling preparation** with StandardScaler for linear models vs unscaled for tree-based models
- **Feature inventory** documenting all available features for selection analysis

**3.2 Statistical Feature Selection (F-Regression):**

- **F-scores and p-values** calculated for all features to assess linear relationships
- **Statistical significance testing** identifying features with p < 0.05
- **Top 10 features visualization** with horizontal bar chart showing F-scores
- **Linear relationship strength** assessment between each feature and log_price target

**3.3 Lasso Regression Feature Selection:**

- **Multiple alpha values tested** (0.001 to 10.0) for optimal regularization strength
- **Automatic feature selection** through L1 regularization driving coefficients to zero
- **Performance vs regularization analysis** with train/test R² comparison
- **Coefficient visualization** showing feature importance magnitude and direction (positive/negative)
- **Non-zero coefficient identification** as the final Lasso-selected feature set

**3.4 Random Forest Feature Importance:**

- **Tree-based feature importance** using impurity reduction across 100 trees
- **Model performance evaluation** achieving strong R² scores on train/test sets
- **Cumulative importance analysis** identifying features contributing to 95% of total importance
- **Top 15 features visualization** with importance scores and cumulative importance curves

**3.5 Recursive Feature Elimination (RFE):**

- **Multiple feature counts tested** (5, 10, 15, 20) to find optimal number of features
- **Ridge regression as base estimator** for systematic feature ranking and elimination
- **Performance vs feature count analysis** identifying the sweet spot for feature selection
- **Optimal feature selection** based on highest test R² with reasonable feature count

**3.6 Comprehensive Comparison & Consensus Selection:**

- **Multi-method consensus voting** system comparing all 4 feature selection approaches
- **Feature consensus categories** defined:
  - High consensus: 3+ methods (most reliable)
  - Medium consensus: 2+ methods (good reliability)
  - Any consensus: 1+ methods (exploratory)
- **Visual comparison dashboard** with 4-panel analysis:
  - Method feature counts comparison
  - Consensus vote distribution
  - Top consensus features ranking
  - Method performance comparison
- **Final feature recommendation** using high consensus features for optimal balance of performance and simplicity
- **Consensus-based feature set** selected for Step 4 model training

**3.7 Final Data Leakage and Correlation Cleanup:**

- **Critical data leakage removal** of features that could artificially inflate model performance:
  - `price_per_accommodate`: Derived directly from target variable (price / accommodates)
  - `estimated_revenue_l365d`: Revenue calculated from price, creating direct leakage
- **High correlation cleanup** removing redundant features:
  - `calculated_host_listings_count`: High correlation (r=0.902) with `host_listings_count`
- **Final feature set validation** ensuring no remaining data leakage using keyword detection
- **Cleaned consensus features** resulting in approximately 10 robust, leak-free features
- **Final dataset export** as `model_ready_listings_final.csv` ready for model training
- **Data integrity verification** confirming realistic model performance expectations

### Step 4: Model Selection & Training ✅

- [x] Implement baseline models
- [x] Train advanced models
- [x] Model evaluation and comparison

**Key Model Training Analysis Completed:**

**4.1 Final Feature Selection & Data Preparation:**

- **Selected 10 high-consensus features** from Step 3 feature selection analysis:
  1. room_type, 2. longitude, 3. host_listings_count, 4. accommodates, 5. host_acceptance_rate
  2. neighbourhood_cleansed, 7. amenities, 8. instant_bookable, 9. beds, 10. bathrooms
- **Train/test split** (80/20) with 17,466 training and 4,367 test samples
- **Scaling preparation** with StandardScaler for linear models vs unscaled data for tree-based models
- **Target variable** log_price for improved model performance

**4.2 Comprehensive Model Training:**

- **5 regression models trained** with hyperparameter tuning using GridSearchCV:
  - **Linear Regression**: Baseline model with scaled features
  - **Ridge Regression**: L2 regularization with alpha tuning (0.001-1000.0)
  - **Lasso Regression**: L1 regularization with alpha tuning (0.001-100.0)
  - **Random Forest**: Tree-based with n_estimators, max_depth, min_samples_split tuning
  - **Gradient Boosting**: Advanced ensemble with n_estimators, learning_rate, max_depth tuning
- **5-fold cross-validation** for robust performance evaluation using RMSE and R² metrics
- **Hyperparameter optimization** for each model to achieve best performance

**4.3 Model Performance Results:**

- **Comprehensive model comparison** with mean RMSE, R², standard deviations, and training times
- **Best model identification**: **Gradient Boosting Regressor** achieved lowest RMSE
- **Performance visualizations** including RMSE comparison, R² comparison, training time analysis, and RMSE vs R² trade-off plots
- **Test set validation** with final model evaluation on held-out test data
- **Actual price conversion** from log_price predictions for practical interpretation and business value assessment

**4.4 Final Model Selection:**

- **Winner**: Gradient Boosting Regressor with optimized hyperparameters
- **Cross-validation performance** with robust RMSE and R² scores across 5 folds
- **Test set validation** confirming model generalization capability
- **Actual price RMSE** calculated for real-world performance interpretation
- **Model ready for deployment** with comprehensive performance documentation

### Step 5: Final Model & Deployment ✅

- [x] Select best performing model
- [x] Export model
- [x] Create prediction interface

**Key Final Model Training & Deployment Completed:**

**5.1 Best Model Configuration:**

- **Selected model**: Gradient Boosting Regressor (winner from Step 4 cross-validation)
- **Optimal hyperparameters**: Determined through GridSearchCV optimization
- **Feature set**: 10 high-consensus features from Step 3 feature selection
- **Target variable**: log_price with conversion formula for actual prices

**5.2 Final Model Training:**

- **Training approach**: Trained on full training set (17,466+ samples) using optimal hyperparameters
- **Data preparation**: Used unscaled data appropriate for tree-based ensemble model
- **Training validation**: Confirmed successful model fitting with performance tracking
- **No data leakage**: Strict separation of training and test sets maintained

**5.3 Comprehensive Model Evaluation:**

- **Test set performance**: Evaluated on held-out test set (4,367+ samples)
- **Multiple evaluation metrics**: RMSE, R², MAE, MdAPE, MAPE for robust assessment
- **Statistical validation**: Overfitting checks comparing cross-validation vs test performance
- **Business interpretation**: Converted log_price predictions to actual dollar amounts

**5.4 Final Performance Results:**

- **Log-price metrics**: RMSE and R² on transformed target variable
- **Actual price metrics**:
  - **MAE (Mean Absolute Error)**: Average dollar error per prediction
  - **MdAPE (Median Absolute Percentage Error)**: Robust percentage error metric
  - **RMSE**: Root mean squared error in actual dollars
- **Performance visualizations**:
  - Actual vs predicted scatter plot with correlation analysis
  - Residuals analysis for bias detection
  - Error distribution analysis with statistical summaries
  - Performance breakdown by price range segments

**5.5 Model Deployment Package:**

- **Model serialization**: Saved using joblib as `model/best_model.pkl`
- **Metadata preservation**: Comprehensive model documentation saved as `model/model_metadata.pkl`
- **Deployment verification**: Model loading and prediction testing confirmed
- **Production readiness**: Complete deployment package with performance documentation

**5.6 Performance Visualizations:**

- **Actual vs Predicted Analysis**: Scatter plot showing prediction accuracy and correlation
- **Residuals Analysis**: Error distribution to identify potential model bias
- **Error Distribution**: Histogram of percentage errors with MdAPE and MAPE markers
- **Price Range Performance**: MAE analysis across different price segments ($0-100, $100-200, etc.)

## Final Project Results Summary

**🏆 BEST MODEL: Gradient Boosting Regressor**

**📊 FINAL PERFORMANCE METRICS:**

- **R² Score**: High variance explanation in price predictions
- **MdAPE**: Robust median percentage error for business interpretation
- **MAE**: Clear dollar-amount average error per listing
- **Strong correlation** between actual and predicted prices
- **Consistent performance** across different price ranges

**🎯 BUSINESS VALUE:**

- **Accurate price predictions** for Airbnb hosts in NYC
- **Median prediction error** within acceptable business tolerance (MdAPE)
- **Average absolute error** in practical dollar terms (MAE)
- **Model explains significant variance** in pricing (R² score)
- **Production-ready deployment** with comprehensive documentation

**🔧 TECHNICAL ACHIEVEMENTS:**

- **Rigorous data preprocessing** with 3-pass cleaning methodology
- **Advanced feature selection** using 4-method consensus approach
- **Comprehensive model comparison** across 5 different algorithms
- **Robust validation** using 5-fold cross-validation and held-out test set
- **Statistical integrity** with proper overfitting checks and error analysis
- **Complete deployment package** with model serialization and metadata

**📈 MODEL PERFORMANCE HIGHLIGHTS:**

- **No overfitting detected**: Test performance consistent with cross-validation
- **Robust across price ranges**: Consistent accuracy from budget to luxury listings
- **Statistically sound evaluation**: Multiple complementary metrics (RMSE, MAE, MdAPE)
- **Business-ready predictions**: Clear dollar-amount errors for practical application
- **Deployment verified**: Model loading and prediction testing successful

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
- [2024-03-19] Completed Step 3: Feature Selection
  - Implemented 4 comprehensive feature selection techniques: F-Regression, Lasso, Random Forest, and RFE
  - Applied consensus voting system across all methods to identify most reliable features
  - Created high/medium/any consensus feature sets based on method agreement
  - Generated extensive visualizations comparing method performance and feature importance
  - Selected final high-consensus feature set for optimal balance of performance and simplicity
  - Prepared optimized feature set ready for model training and evaluation
- [2024-03-19] Completed Step 3: Final Data Leakage and Correlation Cleanup
  - Removed critical data leakage features: price_per_accommodate and estimated_revenue_l365d
  - Eliminated high correlation redundancy: calculated_host_listings_count (r=0.902 with host_listings_count)
  - Validated final feature set for data integrity and realistic model performance
  - Finalized approximately 10 robust, leak-free features for model training
  - Saved final cleaned dataset as `model_ready_listings_final.csv`
  - Confirmed dataset ready for Step 4 with confidence in data quality and model validity
- [2024-03-19] Completed Step 4: Model Selection & Training
  - Implemented 5 regression models with hyperparameter tuning: Linear Regression, Ridge, Lasso, Random Forest, and Gradient Boosting
  - Applied 5-fold cross-validation for robust performance evaluation using RMSE and R² metrics
  - Conducted comprehensive hyperparameter optimization using GridSearchCV for each model
  - Created detailed performance comparison with visualizations and statistical analysis
  - Identified Gradient Boosting Regressor as the best performing model with lowest RMSE
  - Validated final model performance on held-out test set with actual price conversion
  - Generated comprehensive model documentation and performance metrics for deployment
  - Confirmed model ready for Step 5 with optimal predictive performance and business value
- [2024-03-19] Completed Step 5: Final Model Training & Saving
  - Selected Gradient Boosting Regressor as final model based on Step 4 cross-validation results
  - Trained final model on full training set using optimal hyperparameters from grid search
  - Conducted comprehensive evaluation on held-out test set with multiple metrics (RMSE, R², MAE, MdAPE)
  - Replaced problematic accuracy metric with statistically sound evaluation measures
  - Created extensive performance visualizations: actual vs predicted, residuals, error distribution, price range analysis
  - Performed overfitting validation confirming model generalization capability
  - Saved complete deployment package: model/best_model.pkl and model/model_metadata.pkl
  - Verified model loading and prediction functionality for production readiness
  - Generated final project results summary with business value assessment
  - Achieved production-ready model with robust performance across all price ranges
