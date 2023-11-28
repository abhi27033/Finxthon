# import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go

# # Read the dataset
# df = pd.read_csv('startup_funding-checkpoint.csv')  # Replace 'your_dataset.csv' with the actual file path
# df = df.drop('Remarks', axis=1)
# df = df.dropna()

# # Remove non-numeric characters from 'Amount in USD' column
# df['Amount in USD'] = df['Amount in USD'].str.replace('[^\d.]', '', regex=True)

# # Convert 'Amount in USD' to numeric values
# df['Amount in USD'] = pd.to_numeric(df['Amount in USD'], errors='coerce')

# # Group by 'Startup Name' and aggregate the sum of 'Amount in USD'
# top_startups = df.groupby('Startup Name')['Amount in USD'].sum()

# # Sort the result in descending order
# top_startups = top_startups.sort_values(ascending=False).head(50)

# blooming_verticals = df['Industry Vertical'].value_counts().sort_values(ascending=False).head(5)
# popular_investors = df['Investorsxe2x80x99 Name'].value_counts().sort_values(ascending=False)
# popular_investors = popular_investors[~popular_investors.index.str.lower().str.contains('undisclosed')]

# popular_investors = popular_investors.head(5)

# # Assign scores based on ranking
# df['Blooming Vertical Score'] = df['Industry Vertical'].map(df['Industry Vertical'].value_counts().rank(ascending=False, method='min'))
# df['Popular Investors Score'] = df['Investorsxe2x80x99 Name'].map(df['Investorsxe2x80x99 Name'].value_counts().rank(ascending=False, method='min'))
# df['Top Startups Score'] = df['Startup Name'].map(top_startups.rank(ascending=False, method='min'))

# # Normalize scores to be between 0 and 1
# df['Blooming Vertical Score'] = df['Blooming Vertical Score'] / df['Blooming Vertical Score'].max()
# df['Popular Investors Score'] = df['Popular Investors Score'] / df['Popular Investors Score'].max()
# df['Top Startups Score'] = df['Top Startups Score'] / df['Top Startups Score'].max()

# # Calculate the hybrid score
# weight_blooming_verticals = 0.4
# weight_popular_investors = 0.3
# weight_top_startups = 0.3

# df['Hybrid Score'] = (
#    1-( weight_blooming_verticals * df['Blooming Vertical Score'] +
#     weight_popular_investors * df['Popular Investors Score'] +
#     weight_top_startups * df['Top Startups Score'])
# )

# # Sort DataFrame by Hybrid Score
# df = df.sort_values(by=['Hybrid Score', 'Amount in USD'], ascending=[False, False])

# # Remove duplicates and select top 12 startups
# df_unique = df.drop_duplicates(subset='Startup Name', keep='first').head(25)

# recommended_startups = df_unique[['Startup Name', 'Hybrid Score']]
# print('Recommended Startups:\n', recommended_startups)

# # Draw graphs similar to earlier code
# fig1 = px.bar(blooming_verticals, x=blooming_verticals.index, y=blooming_verticals.values, labels={'y': 'Count', 'x': 'Industry Vertical'}, title='Top 5 Blooming Verticals')
# fig1.show()

# fig2 = px.bar(popular_investors, x=popular_investors.index, y=popular_investors.values, labels={'y': 'Count', 'x': "Investors' Name"}, title='Top 5 Popular Investors')
# fig2.show()

# fig3 = px.bar(top_startups, x=top_startups.index, y=top_startups.values, labels={'y': 'Total Funding (USD)', 'x': 'Startup Name'}, title='Top 50 Startups by Funding')
# fig3.show()

# # Create a bar chart for recommended startups and their hybrid scores
# fig4 = go.Figure(data=[go.Bar(x=df_unique['Startup Name'], y=df['Hybrid Score'])])
# fig4.update_layout(title='Hybrid Scores for Recommended Startups', xaxis_title='Startup Name', yaxis_title='Hybrid Score')
# fig4.show()

from flask import Flask, render_template
import subprocess
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from flask import Flask, render_template, request
import subprocess
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
import io
import base64
app = Flask(__name__, static_url_path='/static')
# app = Flask(__name__)

# Read the dataset
df = pd.read_csv('startup_funding-checkpoint.csv')  # Replace with your actual file path
df = df.drop('Remarks', axis=1)
df = df.dropna()
df['Amount in USD'] = df['Amount in USD'].str.replace('[^\d.]', '', regex=True)
df['Amount in USD'] = pd.to_numeric(df['Amount in USD'], errors='coerce')

# Group by 'Startup Name' and aggregate the sum of 'Amount in USD'
top_startups = df.groupby('Startup Name')['Amount in USD'].sum()

# Sort the result in descending order
top_startups = top_startups.sort_values(ascending=False).head(50)

blooming_verticals = df['Industry Vertical'].value_counts().sort_values(ascending=False).head(5)
popular_investors = df['Investorsxe2x80x99 Name'].value_counts().sort_values(ascending=False)
popular_investors = popular_investors[~popular_investors.index.str.lower().str.contains('undisclosed')]
popular_investors = popular_investors.head(5)

# Assign scores based on ranking
df['Blooming Vertical Score'] = df['Industry Vertical'].map(df['Industry Vertical'].value_counts().rank(ascending=False, method='min'))
df['Popular Investors Score'] = df['Investorsxe2x80x99 Name'].map(df['Investorsxe2x80x99 Name'].value_counts().rank(ascending=False, method='min'))
df['Top Startups Score'] = df['Startup Name'].map(top_startups.rank(ascending=False, method='min'))

# Normalize scores to be between 0 and 1
df['Blooming Vertical Score'] = df['Blooming Vertical Score'] / df['Blooming Vertical Score'].max()
df['Popular Investors Score'] = df['Popular Investors Score'] / df['Popular Investors Score'].max()
df['Top Startups Score'] = df['Top Startups Score'] / df['Top Startups Score'].max()

# Calculate the hybrid score
weight_blooming_verticals = 0.4
weight_popular_investors = 0.3
weight_top_startups = 0.3

df['Hybrid Score'] = (
   1-( weight_blooming_verticals * df['Blooming Vertical Score'] +
    weight_popular_investors * df['Popular Investors Score'] +
    weight_top_startups * df['Top Startups Score'])
)

# Sort DataFrame by Hybrid Score
df = df.sort_values(by=['Hybrid Score', 'Amount in USD'], ascending=[False, False])

# Remove duplicates and select top 12 startups
df_unique = df.drop_duplicates(subset='Startup Name', keep='first').head(25)

recommended_startups = df_unique[['Startup Name', 'Hybrid Score']]


# Function to fetch historical stock data
def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data['Adj Close']
def calculate_risk_return(stock_prices):
    # Calculate daily returns
    daily_returns = stock_prices.pct_change().dropna().values.reshape(-1, 1)

    # Scale the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(daily_returns)

    # Prepare training data
    X_train = scaled_data[:-1]
    y_train = scaled_data[1:].ravel()  # Flatten y_train

    # Build and train the neural network
    model = MLPRegressor(hidden_layer_sizes=(10, 10), max_iter=1000)
    model.fit(X_train, y_train)

    # Predict returns for the next day
    predicted_return = model.predict(scaled_data[-1].reshape(1, -1))

    # Calculate risk as the standard deviation of historical returns
    risk = np.std(daily_returns)

    # Calculate expected return based on the neural network prediction
    expected_return = predicted_return[0] if predicted_return.ndim == 1 else predicted_return[0][0]

    return risk, expected_return

# Function to plot a pie chart for sectors
def plot_pie_chart(sectors, labels):
    plt.figure(figsize=(8, 8))
    plt.pie(sectors, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title('Best Performing Sector Based on Risk and Return')

    # Save the plot to a BytesIO object
    img_data = io.BytesIO()
    plt.savefig(img_data, format='png')
    img_data.seek(0)

    # Encode the image as base64
    img_base64 = base64.b64encode(img_data.read()).decode()

    plt.close()

    return img_base64

@app.route('/stock_analysis.html', methods=['GET', 'POST'])
def stock_analysis():
    if request.method == 'GET':
        # Display the initial stock_analysis.html page
        return render_template('stock_analysis.html')
    elif request.method == 'POST':
        # Handle form submission and process data
        selected_sectors = request.form.getlist('sectors')
        print(selected_sectors)
        tickers = {
            'IT': ['TCS.BO', 'INFY.BO', 'WIPRO.BO'],
            'Metals': ['VEDL.BO', 'TATASTEEL.BO', 'HINDALCO.BO'],
            'Pharma': ['SUNPHARMA.BO', 'DRREDDY.BO', 'CIPLA.BO']
            # Add more tickers for each sector
        }

        start_date = '2023-10-01'
        end_date = '2023-11-01'

        risk_return_data = {}

        for sector, stocks in tickers.items():
            if sector in selected_sectors:
                sector_data = []
                for stock in stocks:
                    stock_prices = get_stock_data(stock, start_date, end_date)
                    risk, expected_return = calculate_risk_return(stock_prices)
                    sector_data.append((risk, expected_return))
                # Calculate the overall risk and return for the sector
                avg_risk = np.mean([data[0] for data in sector_data])
                avg_return = np.mean([data[1] for data in sector_data])
                risk_return_data[sector] = (avg_risk, avg_return)

        # Plot a pie chart
        pie_chart_base64 = plot_pie_chart([risk_return_data[sector][1] - risk_return_data[sector][0] for sector in selected_sectors], selected_sectors)

        return render_template('stock_analysis_result.html', pie_chart_base64=pie_chart_base64)

# Index route
@app.route('/')
def index():
    return render_template('index.html')

# # Route to handle startup recommendations
@app.route('/star.html')
def star():
    return render_template('star.html')

@app.route('/equity.html')
def equity():
    return render_template('equity.html')

@app.route('/stock_forecast')
def stock_forecast():
    # Run Streamlit app as a subprocess
    subprocess.run(['streamlit', 'run', 'streamlit_app.py'])
    return "Streamlit app is running..."

@app.route('/recommendations')
def recommendations():
    # Draw graphs similar to earlier code
    fig1 = px.bar(blooming_verticals, x=blooming_verticals.index, y=blooming_verticals.values, labels={'y': 'Count', 'x': 'Industry Vertical'}, title='Top 5 Blooming Verticals')
    fig2 = px.bar(popular_investors, x=popular_investors.index, y=popular_investors.values, labels={'y': 'Count', 'x': "Investors' Name"}, title='Top 5 Popular Investors')
    fig3 = px.bar(top_startups, x=top_startups.index, y=top_startups.values, labels={'y': 'Total Funding (USD)', 'x': 'Startup Name'}, title='Top 50 Startups by Funding')
    fig4 = go.Figure(data=[go.Bar(x=df_unique['Startup Name'], y=df['Hybrid Score'])])
    fig4.update_layout(title='Hybrid Scores for Recommended Startups', xaxis_title='Startup Name', yaxis_title='Hybrid Score')

    # Convert Plotly figures to HTML
    graph1 = fig1.to_html(full_html=False)
    graph2 = fig2.to_html(full_html=False)
    graph3 = fig3.to_html(full_html=False)
    graph4 = fig4.to_html(full_html=False)

    return render_template('recommendations.html', graph1=graph1, graph2=graph2, graph3=graph3, graph4=graph4, recommended_startups=recommended_startups)

if __name__ == '__main__':
    app.run(debug=True)
