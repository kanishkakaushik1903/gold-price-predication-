# 📈 Gold Price Prediction using Machine Learning

This project implements a Machine Learning model to predict the price of Gold (**GLD**) based on various financial indicators. It utilizes the **Random Forest Regressor** algorithm to analyze historical data and forecast future prices.

## 📝 Project Overview

The goal of this project is to build a predictive model that can accurately estimate gold prices. The system processes historical financial data, trains a regression model, and evaluates its performance using the R-squared error metric. Finally, it visualizes the difference between actual and predicted values.

### Key Features:
* **Data Collection & Processing:** Loads data from CSV and checks for missing values.
* **Data Analysis:** Performs statistical analysis and correlation checks on the dataset.
* **Model Training:** Uses `RandomForestRegressor` from Scikit-Learn.
* **Evaluation:** Calculates the R-squared error to measure model accuracy.
* **Visualization:** Plots a graph comparing Actual Prices vs. Predicted Prices.

## 🛠️ Technologies Used

* **Python 3.x**
* **Pandas:** For data manipulation and analysis.
* **NumPy:** For numerical operations.
* **Matplotlib & Seaborn:** For data visualization and plotting graphs.
* **Scikit-Learn:** For machine learning algorithms (Random Forest) and evaluation metrics.

## 📂 Dataset

The project requires a dataset named `gld_price_data.csv`.
* **Target Variable:** `GLD` (Gold Price)
* **Features:** The model uses other columns (excluding Date) as features to predict the price. Typically, this dataset includes correlations with SPX (S&P 500), USO (United States Oil Fund), SLV (Silver), and EUR/USD exchange rates.

## 🚀 How to Run

1.  **Clone the repository** (if applicable) or download the script.
2.  **Install the required dependencies**:
    ```bash
    pip install numpy pandas matplotlib seaborn scikit-learn
    ```
3.  **Place the dataset**: Ensure `gld_price_data.csv` is in the same directory as your Python script.
4.  **Run the script**:
    ```bash
    python your_script_name.py
    ```

## 📊 Model Performance

The model uses the **R-squared (R2) Error** to evaluate accuracy.
* An R2 score closer to `1.0` indicates a highly accurate model.
* The script prints the R2 score to the console after execution.

### Visualization
After training, the script generates a line plot:
* <span style="color:red">**Red Line:**</span> Represents the **Actual** Gold values.
* <span style="color:green">**Green Line:**</span> Represents the **Predicted** Gold values.

This visual comparison helps in understanding how well the model generalizes to the test data.

## 📜 Code Structure

1.  **Import Libraries:** Loads necessary Python packages.
2.  **Data Loading:** Reads the CSV file into a Pandas DataFrame.
3.  **Preprocessing:** Checks for null values and describes statistical properties.
4.  **Feature Selection:** Separates the target (`GLD`) from the features (`X`).
5.  **Train-Test Split:** Splits data into 80% training and 20% testing sets.
6.  **Model Training:** Fits the Random Forest Regressor to the training data.
7.  **Prediction & Evaluation:** Predicts values for the test set and calculates the error score.
8.  **Plotting:** Displays the results graph.

## 🤝 Contributing

Feel free to fork this project, open issues, or submit pull requests if you have suggestions for improving the model's accuracy or visualization.
