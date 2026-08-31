#  House Price Prediction — Linear Regression

A machine learning project that predicts house prices using the **King County, USA** housing dataset. Built with Python and Scikit-learn.

---

##  What This Project Does

- Loads and cleans real housing data
- Removes outliers (top 1% expensive houses)
- Trains a **Linear Regression** model on key features
- Evaluates model with MSE and R² Score
- Saves the trained model and scaler for reuse
- Accepts **live user input** to predict a house price

---

##  Project Structure

```
house-price-prediction/
│
├── linear.py                 # Main script — train, evaluate, predict
├── kc_house_data[1].csv      # Dataset (King County house sales)
├── house_price_model.pkl     # Saved trained model
├── scaler.pkl                # Saved StandardScaler
└── README.md
```

---

##  Dataset

- **Source:** King County, USA House Sales dataset
- **Rows:** ~21,000+ records (after cleaning)
- **Target variable:** `price`

### Features Used

| Feature | Description |
|---|---|
| `bedrooms` | Number of bedrooms |
| `bathrooms` | Number of bathrooms |
| `sqft_living` | Interior living space (sq ft) |
| `floors` | Number of floors |
| `sqft_lot` | Total lot area (sq ft) |
| `grade` | Overall grade given by KC grading system (1–13) |
| `condition` | Condition of the house (1–5) |

---

##  How to Run

### 1. Clone the repo

```bash
git clone https://github.com/your-username/house-price-prediction.git
cd house-price-prediction
```

### 2. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

### 3. Run the script

```bash
python linear.py
```

You will be prompted to enter house details for a live price prediction.

---

##  Model Performance

| Metric | Value |
|---|---|
| Algorithm | Linear Regression |
| R² Score | ~0.65–0.70 (approx.) |
| Evaluation | MSE + Residual Plot |

> **Note:** R² may vary slightly depending on the train-test split.

---

##  Visualizations

The script generates two plots:

1. **Actual vs Predicted Prices** — How close predictions are to real values
2. **Residual Plot** — Shows error distribution across predictions

---

## Sample User Input

```
Bedrooms (>1): 3
Bathrooms (>1): 2.0
Sqft Living (>0): 1800
Floors (>0): 1.0
Sqft Lot (>0): 5000
Grade (1-13): 7
Condition (1-5): 3

Predicted Price: ₹285,432.00
```

---

## Tech Stack

- Python 3.x
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Joblib

---

## Future Improvements

- [ ] Try Ridge / Lasso / XGBoost for better accuracy
- [ ] Add more features like `waterfront`, `view`, `yr_built`
- [ ] Build a web UI using Flask or Streamlit
- [ ] Deploy on Render / Hugging Face Spaces

---

## Author

**Varun** — B.Tech Student, IIIT Bhagalpur  
Building AI/ML projects | [GitHub](https://github.com/your-username)

---

## 📄 License

This project is open source under the [MIT License](LICENSE).
