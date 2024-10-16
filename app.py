import streamlit as st
from importnb import Notebook

with Notebook():
    import Stock_Market_Analysis as analysis

# Streamlit UI
st.title("Stock Market Prediction")

# Get user input for stock symbol
stock_name = st.text_input("Enter Stock Symbol", "AAPL")

# Fetch and display stock data
if stock_name:
    st.write(f"Stock Data for {stock_name}")
    stock_data = analysis.fetch_stock_data(stock_name)

    st.line_chart(stock_data['Close'])
    st.write(stock_data[['Close', 'Volume']].tail(10))

# Predict future prices
if st.button('Predict Future Prices'):
    future_price, future_dates = analysis.predict_future_prices(stock_name)
    st.write(f"Predicted Price for {stock_name}: {future_price[0][0]}")
    st.line_chart(future_price)

    st.write("**We do not guarantee this price, it can go both up and down. Invest at your own risk.**")

# Competitors section
st.subheader("Competitors")
competitors = analysis.fetch_competitors(stock_name)
selected_competitor = st.selectbox("Select Competitor", competitors)

if selected_competitor:
    st.write(f"Showing data for {selected_competitor}")
    competitor_data = analysis.fetch_stock_data(selected_competitor)
    st.line_chart(competitor_data['Close'])
    st.write(competitor_data[['Close', 'Volume']].tail(10))

    if st.button('Predict Competitor Prices'):
        competitor_future_price, _ = analysis.predict_future_prices(selected_competitor)
        st.write(f"Predicted Price for {selected_competitor}: {competitor_future_price[0][0]}")
        st.line_chart(competitor_future_price)

# Sentiment analysis
if st.button('Run Sentiment Analysis'):
    sentiment = analysis.sentiment_analysis(stock_name)
    st.write(f"Sentiment for {stock_name}: {sentiment[0]}")
