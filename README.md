# Airbnb NYC Price Prediction Project

## 🏠 Project Overview

This project helps Airbnb hosts in New York City competitively price their properties using machine learning. By analyzing over 37,000 NYC listings, we built a predictive model that estimates nightly prices based on property characteristics like location, room type, amenities, and host attributes.

**Goal**: Provide accurate price predictions to help hosts maximize their earnings while staying competitive in the NYC short-term rental market.

## 🎯 Problem Statement

Pricing Airbnb properties correctly is crucial for hosts to:

- Maximize booking rates and revenue
- Stay competitive in their neighborhood
- Avoid underpricing (lost revenue) or overpricing (low occupancy)

Our solution provides data-driven price recommendations based on similar successful listings in NYC.

## 🚀 Key Features

- **Accurate Price Predictions**: Machine learning model trained on 21,000+ NYC listings
- **User-Friendly Web Interface**: Simple form to input property details
- **Real-Time Results**: Instant price predictions via REST API
- **Data-Driven Insights**: Based on comprehensive analysis of NYC Airbnb market

## 📊 Project Methodology

This project follows a comprehensive 6-step data science workflow:

1. **Data Cleaning & Preprocessing** - Cleaned 37K listings down to 21K high-quality records
2. **Exploratory Data Analysis** - Analyzed price distributions, correlations, and market trends
3. **Feature Selection** - Used 4 different methods to identify the 10 most predictive features
4. **Model Training & Selection** - Compared 5 ML algorithms, selected Gradient Boosting as best performer
5. **Model Evaluation & Validation** - Rigorous testing with cross-validation and held-out test sets
6. **Web Application Development** - Built full-stack app with React frontend and Flask backend

_For detailed technical implementation, see [logs_overview.md](logs_overview.md)_

## 🏆 Final Model Performance

**Best Model**: Gradient Boosting Regressor

- **High accuracy** across all price ranges ($50-$500+ per night)
- **Robust performance** with comprehensive validation
- **Business-ready predictions** with clear error metrics
- **No overfitting** - consistent performance on unseen data

## 🛠️ Tech Stack

### Machine Learning & Data Science

- **Python 3.11+** - Core programming language
- **pandas & numpy** - Data manipulation and analysis
- **scikit-learn** - Machine learning algorithms and evaluation
- **matplotlib & seaborn** - Data visualization

### Backend API

- **Flask** - REST API framework
- **Flask-CORS** - Cross-origin resource sharing
- **joblib** - Model serialization and loading

### Frontend Application

- **React 19** - Modern UI framework
- **Vite** - Fast build tool and dev server
- **JavaScript (ES6+)** - Frontend logic and API integration

### Development Tools

- **Pipenv** - Python dependency management
- **npm** - Node.js package management
- **Git** - Version control

## 📦 Installation & Setup

### Prerequisites

- Python 3.11+
- Node.js 16+
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/RaulJiminian/airBnB_price_predict_final_project
cd airBnB_price_predict_final_project
```

### 2. Backend Setup (Flask API)

```bash
cd airBnB_BackEnd
pipenv shell                    # Enter virtual environment
pipenv install                 # Install Python dependencies
python app.py                  # Start Flask server (runs on http://127.0.0.1:5000)
```

### 3. Frontend Setup (React App)

```bash
cd airBnB_FrontEnd
npm install                     # Install Node.js dependencies
npm run dev                     # Start development server (runs on http://localhost:5173)
```

### 4. Access the Application

- **Frontend**: Open http://localhost:5173 in your browser
- **Backend API**: Available at http://127.0.0.1:5000
- **API Endpoint**: POST to http://127.0.0.1:5000/predict with listing data

## 🎥 Demo Video

_[Video overview placeholder - to be added]_

## 📈 Model Details & Performance

Our final Gradient Boosting model uses 10 carefully selected features:

- **Location**: Longitude, Neighbourhood (borough)
- **Property**: Room type, Accommodates, Beds, Bathrooms
- **Host**: Listings count, Acceptance rate
- **Booking**: Instant bookable option
- **Amenities**: Total amenity count

**Key Performance Metrics**:

- Strong R² score indicating high variance explanation
- Low median absolute percentage error (MdAPE) for business reliability
- Consistent accuracy across budget ($50-100) to luxury ($300+) price ranges
- Robust validation with no overfitting detected

## 🗂️ Project Structure

```
airBnB_price_predict_final_project/
├── data/                           # Dataset files (original and processed)
├── notebook-analysis/              # Jupyter notebook with full analysis
├── airBnB_BackEnd/                # Flask API backend
│   ├── app.py                     # Main Flask application
│   ├── model/                     # Trained model files
│   └── Pipfile                    # Python dependencies
├── airBnB_FrontEnd/               # React frontend application
│   ├── src/App.jsx               # Main React component
│   └── package.json              # Node.js dependencies
├── README.md                      # Project overview (this file)
├── logs_overview.md              # Detailed technical implementation
└── Project_Proposal_AirBnB_Price_Prediction.md
```

## 🤝 Contributing

This project was developed as a final project for an introductory data science course. The implementation demonstrates end-to-end machine learning workflow from data cleaning through model deployment.

## 📄 License

This project is for educational purposes. Dataset sourced from [Inside Airbnb](http://insideairbnb.com/get-the-data.html) under their terms of use.
