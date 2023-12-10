# Prediction

Overview:

This Python script uses a Decision Tree Classifier to predict stock volume based on historical data. The dataset is loaded from a CSV file (data.csv) containing stock information, and the machine learning model is trained to predict stock volumes. The predictions are then compared with the actual values, and the results are visualized using a bar chart.

Dependencies:

Make sure you have the following Python libraries installed:

pandas
scikit-learn
joblib
graphviz
yfinance
matplotlib
You can install the required dependencies using the following:

bash
Copy code
pip install pandas scikit-learn joblib graphviz yfinance matplotlib
Usage:

Clone the repository:
bash
Copy code
git clone https://github.com/blueraymusic/Prediction.git
Install dependencies:
pip install -r requirements.txt
Run the script:
python Prediction.ipynb

Code Explanation:

Data Loading:
The script reads stock data from a CSV file (data.csv) using the pandas library.
python
Copy code
file = 'data.csv'
data = pd.read_csv(file)
Data Preprocessing:
The 'date' column is converted to a numeric format.
python
Copy code
data['date'] = data['date'].apply(lambda x: float(x.split()[0].replace('-', '')))
Model Training:
The Decision Tree Classifier is used to train the model on the stock data.
python
Copy code
x_values = data.drop(columns=['volume', 'Name', 'date'])
y = data['volume']
x_train, x_test, y_train, y_test = train_test_split(x_values, y, test_size=0.0005)
model = DecisionTreeClassifier()
model.fit(x_train, y_train)
Prediction and Evaluation:
The model predicts stock volumes and calculates the accuracy score.
python
Copy code
predictions = model.predict(x_test)
error = 0.10 * predictions
score = accuracy_score(y_test, predictions)
Visualization:
The results are visualized using a bar chart.
python
Copy code
results_df = pd.DataFrame({
    'Real Numbers': y_test,
    'Predictions': predictions - error
})

try:
    # Visualization using matplotlib
    ax = results_df.plot(kind='bar', figsize=(10, 6), legend=True)
    plt.title('Real Numbers vs Predictions')
    plt.xlabel('Volumes')
    plt.ylabel('Values')
    plt.xticks([])  # Remove x-axis tick labels
    plt.legend(loc='upper right')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.2f}'.format(x)))
    plt.show()
except:
    print('<-----Could not Visualize the data----->')
Model Saving and Loading:
The trained model can be saved using joblib and loaded for future use.
python
Copy code
# Save the trained model
# joblib.dump(model, 'model_trained.joblib')

# Load the model
# model = joblib.load('model_trained.joblib')
Stock Data Retrieval:
Historical stock data is downloaded using the yfinance library and plotted.
python
Copy code
# Get historical stock data
data = yf.download(ticker, '1980-12-12', '2023-11-17')

# Plot adjusted close price data
data['Volume'].plot()
plt.show()
