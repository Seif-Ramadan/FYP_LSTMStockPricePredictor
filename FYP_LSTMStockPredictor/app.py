import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Import modules from src directory
from src.data import fetch_data, preprocess_daily_data, create_sequences   
from src.model import (
    build_lstm_model, 
    calculate_metrics, 
    predict_future_prices,
    inverse_transform_series,
    walk_forward_validation,  
    plot_walk_forward_results,
    calculate_permutation_importance, 
    plot_feature_importance  
)
from src.sentiment import (
    fetch_reddit_comments_about_ticker, 
    analyze_sentiment_ratio, 
    get_sentiment_by_subreddit
)
from src.prediction_storage import (
    save_prediction, 
    load_predictions, 
    update_prediction_with_actual, 
    compare_predictions
)

# --------------------------------------------------------------------------------
# Streamlit Application
# --------------------------------------------------------------------------------
def main():
    st.title("Advanced Stock Price Prediction with LSTM + Reddit Sentiment")
    st.write("""
    This app combines LSTM-based stock prediction with Reddit sentiment analysis to give a more complete picture.
    """)

    # Initialize session state for sentiment data
    if 'sentiment_data' not in st.session_state:
        st.session_state.sentiment_data = None
    if 'sentiment_summary' not in st.session_state:
        st.session_state.sentiment_summary = None
    if 'comments' not in st.session_state:
        st.session_state.comments = None
    if 'current_symbol' not in st.session_state:
        st.session_state.current_symbol = None

    # Define a shared sentiment indicator function
    def sentiment_indicator(val):
        if val > 0.2:
            return "Very Positive"
        elif val > 0.05:
            return "Positive"
        elif val > -0.05:
            return "Neutral"
        elif val > -0.2:
            return "Negative"
        else:
            return "Very Negative"

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Reddit Sentiment", "Stock Prediction", "Prediction History"])
    
    # Sidebar for common configuration
    st.sidebar.header("Stock Configuration")
    symbol = st.sidebar.text_input("Enter a stock symbol (e.g., AAPL, TSLA, INTC):", value="AAPL")
    
    # Clear sentiment data if stock changes
    if st.session_state.current_symbol != symbol:
        st.session_state.sentiment_data = None
        st.session_state.sentiment_summary = None
        st.session_state.comments = None
        st.session_state.current_symbol = symbol

    # ---------------------- TAB 1: REDDIT SENTIMENT ----------------------
    with tab1:
        st.header("Reddit Sentiment Analysis")
        st.write("""
            This analysis fetches Reddit comments mentioning the stock ticker from popular finance
            subreddits and analyzes the sentiment using VADER sentiment analysis.
        """)
        
        col1, col2, col3 = st.columns(3)
        days_back = col1.slider("Days to look back", 1, 30, 30)
        min_upvotes = col2.slider("Minimum upvotes", 0, 20, 2)
        custom_subs = col3.checkbox("Custom subreddits", value=False)
        
        subreddits = None
        if custom_subs:
            subreddits_input = st.text_input("Enter subreddits (comma-separated)", 
                                            value="wallstreetbets,stocks,investing,stockmarket")
            subreddits = [s.strip() for s in subreddits_input.split(",")]
        
        # Check if we already have sentiment data stored
        if st.session_state.comments is not None and st.session_state.sentiment_summary is not None:
            st.info(f"Showing stored sentiment analysis for {symbol}")
            
            # Use the stored data
            comments = st.session_state.comments
            sentiment_summary = st.session_state.sentiment_summary
            
            # Display sentiment data
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Comments", sentiment_summary['total_comments'])
            col2.metric("Positive Comments", sentiment_summary['positive_comments'])
            col3.metric("Negative Comments", sentiment_summary['negative_comments'])
            
            # Format the Pos/Neg ratio
            if np.isinf(sentiment_summary['ratio_pos_neg']):
                ratio_display = "∞"
            else:
                ratio_display = f"{sentiment_summary['ratio_pos_neg']:.2f}"
            
            col4.metric("Pos/Neg Ratio", ratio_display)
            
            # Weighted sentiment score
            st.metric("Weighted Sentiment Score", 
                    f"{sentiment_summary['weighted_sentiment']:.3f}",
                    delta=None,
                    delta_color="normal")
            
            # Create sentiment distribution chart
            st.subheader("Sentiment Distribution")
            fig, ax = plt.subplots()
            sentiment_counts = [
                sentiment_summary['positive_comments'],
                sentiment_summary['neutral_comments'],
                sentiment_summary['negative_comments']
            ]
            ax.pie(sentiment_counts, 
                  labels=['Positive', 'Neutral', 'Negative'],
                  autopct='%1.1f%%',
                  colors=['green', 'gray', 'red'])
            ax.set_title(f"Sentiment Distribution for {symbol}")
            st.pyplot(fig)
            
            # Show sentiment by subreddit
            subreddit_sentiment = get_sentiment_by_subreddit(comments)
            if len(subreddit_sentiment) > 1:
                st.subheader("Sentiment by Subreddit")
                
                # Convert to dataframe for display
                subreddit_df = pd.DataFrame.from_dict(subreddit_sentiment, orient='index')
                subreddit_df['pos_pct'] = (subreddit_df['positive'] / subreddit_df['comments'] * 100).round(1)
                subreddit_df['neg_pct'] = (subreddit_df['negative'] / subreddit_df['comments'] * 100).round(1)
                subreddit_df = subreddit_df.sort_values('comments', ascending=False)
                
                # Display as table
                st.dataframe(subreddit_df[['comments', 'positive', 'negative', 'pos_pct', 'neg_pct', 'average_sentiment']])
                
                # Bar chart for sentiment by subreddit
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                x = np.arange(len(subreddit_df))
                width = 0.35
                
                ax2.bar(x - width/2, subreddit_df['pos_pct'], width, label='Positive %', color='green')
                ax2.bar(x + width/2, subreddit_df['neg_pct'], width, label='Negative %', color='red')
                
                ax2.set_xticks(x)
                ax2.set_xticklabels(subreddit_df.index)
                ax2.set_ylabel('Percentage')
                ax2.set_title(f'Sentiment by Subreddit for {symbol}')
                ax2.legend()
                
                st.pyplot(fig2)
            
            # Sample comments
            st.subheader("Sample Comments")
            
            # Sort by most upvoted
            sorted_comments = sorted(comments, key=lambda x: x['score'], reverse=True)
            
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            sid = SentimentIntensityAnalyzer()
            for i, comment in enumerate(sorted_comments[:10]):  # Show top 10 comments
                scores = sid.polarity_scores(comment['text'])
                sentiment_type = "Positive" if scores['compound'] > 0.05 else "Negative" if scores['compound'] < -0.05 else "Neutral"
                color = "green" if sentiment_type == "Positive" else "red" if sentiment_type == "Negative" else "gray"
                
                # Format the timestamp
                timestamp = datetime.fromtimestamp(comment['created_utc'])
                date_str = timestamp.strftime("%Y-%m-%d %H:%M")
                
                # Create an expandable section for each comment
                with st.expander(f"Comment from r/{comment['subreddit']} • {date_str} • Score: {comment['score']} • {sentiment_type} ({scores['compound']:.2f})"):
                    st.markdown(comment['text'])
            
            # Add a refresh button
            if st.button("Refresh Sentiment Analysis", key="refresh_sentiment"):
                # Fetch new data
                with st.spinner(f"Fetching Reddit comments about {symbol}..."):
                    try:
                        comments = fetch_reddit_comments_about_ticker(
                            symbol, 
                            subreddits=subreddits,
                            min_upvotes=min_upvotes,
                            days_back=days_back
                        )
                        
                        if not comments:
                            st.warning(f"No recent comments found for {symbol} with at least {min_upvotes} upvotes.")
                        else:
                            sentiment_summary = analyze_sentiment_ratio(comments)
                            
                            # Update session state
                            st.session_state.comments = comments
                            st.session_state.sentiment_summary = sentiment_summary
                    
                    except Exception as e:
                        st.error(f"Error analyzing Reddit sentiment: {e}")
                        import traceback
                        st.error(traceback.format_exc())
        
        # Keep the original button for when there's no stored data
        elif st.button("Analyze Reddit Sentiment", key="reddit_button"):
            with st.spinner(f"Fetching Reddit comments about {symbol}..."):
                try:
                    comments = fetch_reddit_comments_about_ticker(
                        symbol, 
                        subreddits=subreddits,
                        min_upvotes=min_upvotes,
                        days_back=days_back
                    )
                    
                    if not comments:
                        st.warning(f"No recent comments found for {symbol} with at least {min_upvotes} upvotes.")
                    else:
                        sentiment_summary = analyze_sentiment_ratio(comments)
                        subreddit_sentiment = get_sentiment_by_subreddit(comments)
                        
                        # Store in session state
                        st.session_state.comments = comments
                        st.session_state.sentiment_summary = sentiment_summary
                        
                        # Display summary metrics
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Total Comments", sentiment_summary['total_comments'])
                        col2.metric("Positive Comments", sentiment_summary['positive_comments'])
                        col3.metric("Negative Comments", sentiment_summary['negative_comments'])
                        
                        # Format the Pos/Neg ratio
                        if np.isinf(sentiment_summary['ratio_pos_neg']):
                            ratio_display = "∞"
                        else:
                            ratio_display = f"{sentiment_summary['ratio_pos_neg']:.2f}"
                        
                        col4.metric("Pos/Neg Ratio", ratio_display)
                        
                        # Weighted sentiment score
                        st.metric("Weighted Sentiment Score", 
                                f"{sentiment_summary['weighted_sentiment']:.3f}",
                                delta=None,
                                delta_color="normal")
                        
                        # Create sentiment distribution chart
                        st.subheader("Sentiment Distribution")
                        fig, ax = plt.subplots()
                        sentiment_counts = [
                            sentiment_summary['positive_comments'],
                            sentiment_summary['neutral_comments'],
                            sentiment_summary['negative_comments']
                        ]
                        ax.pie(sentiment_counts, 
                              labels=['Positive', 'Neutral', 'Negative'],
                              autopct='%1.1f%%',
                              colors=['green', 'gray', 'red'])
                        ax.set_title(f"Sentiment Distribution for {symbol}")
                        st.pyplot(fig)
                        
                        # Show sentiment by subreddit
                        if len(subreddit_sentiment) > 1:
                            st.subheader("Sentiment by Subreddit")
                            
                            # Convert to dataframe for display
                            subreddit_df = pd.DataFrame.from_dict(subreddit_sentiment, orient='index')
                            subreddit_df['pos_pct'] = (subreddit_df['positive'] / subreddit_df['comments'] * 100).round(1)
                            subreddit_df['neg_pct'] = (subreddit_df['negative'] / subreddit_df['comments'] * 100).round(1)
                            subreddit_df = subreddit_df.sort_values('comments', ascending=False)
                            
                            # Display as table
                            st.dataframe(subreddit_df[['comments', 'positive', 'negative', 'pos_pct', 'neg_pct', 'average_sentiment']])
                            
                            # Bar chart for sentiment by subreddit
                            fig2, ax2 = plt.subplots(figsize=(10, 6))
                            x = np.arange(len(subreddit_df))
                            width = 0.35
                            
                            ax2.bar(x - width/2, subreddit_df['pos_pct'], width, label='Positive %', color='green')
                            ax2.bar(x + width/2, subreddit_df['neg_pct'], width, label='Negative %', color='red')
                            
                            ax2.set_xticks(x)
                            ax2.set_xticklabels(subreddit_df.index)
                            ax2.set_ylabel('Percentage')
                            ax2.set_title(f'Sentiment by Subreddit for {symbol}')
                            ax2.legend()
                            
                            st.pyplot(fig2)
                        
                        # Sample comments
                        st.subheader("Sample Comments")
                        
                        # Sort by most upvoted
                        sorted_comments = sorted(comments, key=lambda x: x['score'], reverse=True)
                        
                        from nltk.sentiment.vader import SentimentIntensityAnalyzer
                        sid = SentimentIntensityAnalyzer()
                        for i, comment in enumerate(sorted_comments[:10]):  # Show top 5 comments
                            scores = sid.polarity_scores(comment['text'])
                            sentiment_type = "Positive" if scores['compound'] > 0.05 else "Negative" if scores['compound'] < -0.05 else "Neutral"
                            color = "green" if sentiment_type == "Positive" else "red" if sentiment_type == "Negative" else "gray"
                            
                            # Format the timestamp
                            timestamp = datetime.fromtimestamp(comment['created_utc'])
                            date_str = timestamp.strftime("%Y-%m-%d %H:%M")
                            
                            # Create an expandable section for each comment
                            with st.expander(f"Comment from r/{comment['subreddit']} • {date_str} • Score: {comment['score']} • {sentiment_type} ({scores['compound']:.2f})"):
                                st.markdown(comment['text'])
                
                except Exception as e:
                    st.error(f"Error analyzing Reddit sentiment: {e}")
                    import traceback
                    st.error(traceback.format_exc())
    
    # -------------------- TAB 2: STOCK PREDICTION --------------------
    with tab2:
        st.header("LSTM-based Stock Price Prediction")
        
        # Date Range
        years_data = st.slider("Years of historical data", 1, 70, 5)
        start_date = (datetime.now() - timedelta(days=365*years_data)).strftime('%Y-%m-%d')

        #Custom Backtest
        enable_custom_backtest = st.checkbox("Enable custom backtesting", value=False)
        custom_backtest_date = None
        if enable_custom_backtest:
            custom_backtest_date = st.date_input(
                "Backtest cutoff date (data after this date will be excluded)",
                value=datetime.today(),            # Default today
                min_value=datetime(1970, 1, 1),      # Allow user to select all the way back
                max_value=datetime.now()             # Up to today
            )

            # Validate if it's a weekend
            if custom_backtest_date.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
                st.error("Please select a weekday. Backtest dates cannot be on weekends.")
        
        # Feature Selection - create expandable section
        with st.expander("Feature Selection", expanded=True):
            st.info("Select features to include in the prediction model. 'Close' price is always included.")
            
            # Create tabs for feature categories
            feature_tabs = st.tabs(["Price", "Moving Averages", "Volatility", "Momentum", "Volume", "Technical Indicators"])
            
            with feature_tabs[0]:
                use_open = st.checkbox("Open Price", value=True)
                use_high = st.checkbox("High Price", value=True)
                use_low = st.checkbox("Low Price", value=True)
                
            with feature_tabs[1]:
                use_ma5 = st.checkbox("5-day MA", value=True)
                use_ma10 = st.checkbox("10-day MA", value=False)
                use_ma20 = st.checkbox("20-day MA", value=False)
                use_ma50 = st.checkbox("50-day MA", value=False)
                
            with feature_tabs[2]:
                use_volatility5 = st.checkbox("5-day Volatility", value=True)
                use_volatility20 = st.checkbox("20-day Volatility", value=False)
                
            with feature_tabs[3]:
                use_momentum5 = st.checkbox("5-day Momentum", value=True)
                use_momentum10 = st.checkbox("10-day Momentum", value=False)
                use_price_change1d = st.checkbox("1-day Price Change", value=True)
                use_price_change5d = st.checkbox("5-day Price Change", value=False)
                use_price_change20d = st.checkbox("20-day Price Change", value=False)
                
            with feature_tabs[4]:
                use_volume = st.checkbox("Volume", value=True)
                use_volume_sma5 = st.checkbox("Volume 5-day SMA", value=False)
                use_volume_change = st.checkbox("Volume Change", value=False)
                
            with feature_tabs[5]:
                use_rsi = st.checkbox("RSI (14-day)", value=True)
                use_macd = st.checkbox("MACD", value=True)
                use_macd_signal = st.checkbox("MACD Signal", value=False)
                use_macd_hist = st.checkbox("MACD Histogram", value=False)
                use_bb_upper = st.checkbox("Bollinger Upper Band", value=False)
                use_bb_middle = st.checkbox("Bollinger Middle Band", value=False)
                use_bb_lower = st.checkbox("Bollinger Lower Band", value=False)
        
        # Model Configuration
        with st.expander("Model Configuration", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                sequence_length = st.slider("Sequence Length (Days)", 1, 200, 90,)
                lstm_units = st.slider("LSTM Units", 10, 300, 50,)
                dropout_rate = st.slider("Dropout Rate", 0.0, 0.5, 0.2, 0.05)
                train_size = st.slider("Train/Test Split Ratio", 0.5, 0.95, 0.85, 0.05)
            
            with col2:
                learning_rate = st.select_slider(
                    "Learning Rate",
                    options=[0.0001, 0.0005, 0.001, 0.005, 0.01],
                    value=0.001
                )
                batch_size = st.select_slider(
                    "Batch Size",
                    options=[1, 16, 32, 64, 80, 96, 112, 128],
                    value=32
                )
                epochs = st.slider("Max Epochs", 10, 200, 50)

                calculate_feature_importance = st.checkbox("Calculate Feature Importance", value=True, 
                                        help="Analyzes which features contribute most to prediction accuracy")

        # Validation settings
        with st.expander("Advanced Validation", expanded=False):
            st.info("Walk-forward validation simulates realistic model performance by training on past data and validating on future data.")
            enable_walk_forward = st.checkbox("Enable walk-forward validation", value=False)
            
            if enable_walk_forward:
                col1, col2 = st.columns(2)
                with col1:
                    validation_window = st.slider("Validation Window (Days)", 5, 60, 30)
                    train_min_size = st.slider("Minimum Training Size (Days)", 180, 4000, 756)
                with col2:
                    wf_step_size = st.slider("Step Size (Days)", 10, 60, 30)
                
                st.warning("Note: Walk-forward validation is computationally intensive and may take several hours to complete.")

        # Prediction settings
        prediction_days = st.slider("Days to Predict", 1, 30, 5)
        include_sentiment = st.checkbox("Include Reddit sentiment in prediction display", value=True)
        
        if st.button("Run Stock Prediction", key="predict_button"):
            with st.spinner("Fetching data and training the model..."):
                try:
                    # Create initial feature list with 'Close' (always included)
                    features = ['Close']  

                    # 1. Fetch data first
                    data = fetch_data(symbol, start_date=start_date)
                    st.success(f"Fetched data for {symbol}. Rows: {len(data)}")

                    # Only apply cutoff if custom backtesting is enabled
                    if enable_custom_backtest and custom_backtest_date is not None:
                        cutoff_date = custom_backtest_date.strftime("%Y-%m-%d")
                        original_rows = len(data)
                        data = data.loc[:cutoff_date]
                        st.info(f"Data cut off at {cutoff_date} for backtesting; using {len(data)} of {original_rows} rows")

                    # Get all available columns in the fetched data
                    available_features = data.columns.tolist()

                    # Always include Close price
                    features = ['Close']

                    # Add selected features based on checkboxes
                    if use_open and 'Open' in data.columns:
                        features.append('Open')

                    if use_high and 'High' in data.columns:
                        features.append('High')

                    if use_low and 'Low' in data.columns:
                        features.append('Low')

                    if use_volume and 'Volume' in data.columns:
                        features.append('Volume')

                    # Moving Averages
                    if use_ma5 and 'MA5' in data.columns:
                        features.append('MA5')
                    if use_ma10 and 'MA10' in data.columns:
                        features.append('MA10')
                    if use_ma20 and 'MA20' in data.columns:
                        features.append('MA20')
                    if use_ma50 and 'MA50' in data.columns:
                        features.append('MA50')

                    # Volatility
                    if use_volatility5 and 'Volatility5' in data.columns:
                        features.append('Volatility5')
                    if use_volatility20 and 'Volatility20' in data.columns:
                        features.append('Volatility20')

                    # Momentum
                    if use_momentum5 and 'Momentum5' in data.columns:
                        features.append('Momentum5')
                    if use_momentum10 and 'Momentum10' in data.columns:
                        features.append('Momentum10')

                    # Price Change
                    if use_price_change1d and 'PriceChange1d' in data.columns:
                        features.append('PriceChange1d')
                    if use_price_change5d and 'PriceChange5d' in data.columns:
                        features.append('PriceChange5d')
                    if use_price_change20d and 'PriceChange20d' in data.columns:
                        features.append('PriceChange20d')

                    # Volume Indicators
                    if use_volume_sma5 and 'VolumeSMA5' in data.columns:
                        features.append('VolumeSMA5')
                    if use_volume_change and 'VolumeChange' in data.columns:
                        features.append('VolumeChange')

                    # Technical Indicators
                    if use_rsi and 'RSI14' in data.columns:
                        features.append('RSI14')

                    if use_macd and 'MACD' in data.columns:
                        features.append('MACD')
                    if use_macd_signal and 'MACD_Signal' in data.columns:
                        features.append('MACD_Signal')
                    if use_macd_hist and 'MACD_Hist' in data.columns:
                        features.append('MACD_Hist')

                    # Bollinger Bands
                    if use_bb_upper and 'BB_Upper' in data.columns:
                        features.append('BB_Upper')
                    if use_bb_middle and 'BB_Middle' in data.columns:
                        features.append('BB_Middle')
                    if use_bb_lower and 'BB_Lower' in data.columns:
                        features.append('BB_Lower')

                    
                    # Show selected features
                    st.write(f"**Selected Features:** {', '.join(features)}")
                    
                    # Display a sample of the data
                    st.subheader("Data Sample with Technical Indicators")
                    st.dataframe(data.tail())
                    
                    # Get Reddit sentiment if requested
                    if include_sentiment:
                        with st.spinner("Fetching Reddit sentiment..."):
                            comments = fetch_reddit_comments_about_ticker(
                                symbol, 
                                min_upvotes=0,
                                days_back=30  # Look back a week for sentiment
                            )
                            
                            if comments:
                                sentiment_summary = analyze_sentiment_ratio(comments)
                                weighted_sentiment = sentiment_summary['weighted_sentiment']
                                avg_sentiment = sentiment_summary['average_sentiment']
                                total_comments = sentiment_summary['total_comments']
                                pos_ratio = sentiment_summary['positive_comments'] / sentiment_summary['total_comments']
                                
                                # Store in session state
                                st.session_state.comments = comments
                                st.session_state.sentiment_summary = sentiment_summary
                            else:
                                weighted_sentiment = 0
                                avg_sentiment = 0
                                total_comments = 0
                                pos_ratio = 0
                    
                    # Plot historical data with moving averages
                    st.subheader("Historical Price Chart")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(data.index, data['Close'], label='Close Price')
                    if 'MA5' in data.columns and use_ma5:
                        ax.plot(data.index, data['MA5'], label='5-day MA', alpha=0.7)
                    
                    # Complete the LSTM model training and prediction
                    
                    # 1. Split the raw data into training and testing sets first.
                    split_index = int(len(data) * train_size)
                    train_data = data.iloc[:split_index]
                    test_data = data.iloc[split_index:]

                    # 2. Process the training data:
                    #    - Fit the scaler on training data only.
                    #    - Generate sequences from the training data.
                    X_train, y_train, scaler, columns_to_use = preprocess_daily_data(train_data, use_features=features, sequence_length=sequence_length)

                    # 3. Process the test data using the already-fitted scaler:
                    X_test, y_test = create_sequences(test_data, scaler, columns_to_use, sequence_length)

                    # Log the shapes and split information for confirmation.
                    st.write(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
                    st.write(f"Train-Test Split: {train_size:.0%} / {(1-train_size):.0%}")
                    st.write(f"Model input shape: {X_train.shape}")

                    # 4. Build Model
                    model = build_lstm_model(
                        input_shape=(X_train.shape[1], X_train.shape[2]), 
                        lstm_units=lstm_units,
                        dropout_rate=dropout_rate,
                        learning_rate=learning_rate
                    )
                    
                    # 5. Train Model
                    with st.spinner("Training model... This may take a while"):
                        early_stopping = EarlyStopping(
                            monitor='val_loss', 
                            patience=10, 
                            restore_best_weights=True
                        )
                        reduce_lr = ReduceLROnPlateau(
                            monitor='val_loss',
                            factor=0.2,
                            patience=5,
                            min_lr=0.0001
                        )
                        
                        history = model.fit(
                            X_train, y_train, 
                            epochs=epochs, 
                            batch_size=batch_size, 
                            validation_split=0.1, 
                            callbacks=[early_stopping, reduce_lr],
                            verbose=1
                        )
                        
                        st.success("Model training completed!")

                    # 6. Visualize training history
                    st.subheader("Training History")
                    fig2, ax2 = plt.subplots(figsize=(10, 4))
                    ax2.plot(history.history['loss'], label='Training Loss')
                    ax2.plot(history.history['val_loss'], label='Validation Loss')
                    ax2.set_title('Model Loss During Training')
                    ax2.set_xlabel('Epoch')
                    ax2.set_ylabel('Loss')
                    ax2.legend()
                    st.pyplot(fig2)
                    
                    # 7. Predictions on test set
                    y_pred_scaled = model.predict(X_test).flatten()
                    
                    # 8. Calculate Metrics
                    metrics = calculate_metrics(y_test, y_pred_scaled, scaler, columns_to_use)
                    
                    # Display metrics in a nice format with columns
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Mean Absolute Error (MAE)", f"${metrics['mae']:.2f}")
                    col2.metric("Root Mean Squared Error", f"${metrics['rmse']:.2f}")
                    col3.metric("Direction Accuracy", f"{metrics['direction_accuracy']:.1f}%")

                    col1, col2, col3 = st.columns(3)
                    col1.metric("R² Score", f"{metrics['r2']:.4f}")
                    col2.metric("Mean Squared Error (MSE)", f"${metrics['mse']:.2f}")
                    col3.metric("Mean Absolute % Error", f"{metrics['mape']:.2f}%")

                    # Feature Importance Analysis
                    if calculate_feature_importance:
                        st.subheader("Feature Importance Analysis")
                        with st.spinner("Analyzing feature importance... This may take a minute"):
                            try:
                                # Calculate feature importance
                                feature_importances = calculate_permutation_importance(
                                    model, 
                                    X_test, 
                                    y_test, 
                                    features,
                                    n_repeats=3
                                )
                                
                                # Display numerical results in a table
                                st.write("**Feature Importance Ranking**")
                                st.dataframe(feature_importances.style.format({'Importance': '{:.2f}'}))
                                
                                # Visualize the top features
                                fig_imp = plot_feature_importance(feature_importances, top_n=10, 
                                                            title=f"Feature Importance for {symbol} Prediction")
                                st.pyplot(fig_imp)
                                
                                # Add interpretation of the results
                                st.write(f"""
                                **Interpretation**: 
                                - The most important feature (**{feature_importances.iloc[0]['Feature']}**) has an importance score of {feature_importances.iloc[0]['Importance']:.2f}.
                                - An importance score of 1.0 means the feature has no impact, while higher values indicate greater importance.
                                - Features with scores much higher than 1.0 are crucial for accurate predictions.
                                """)
                                
                            except Exception as e:
                                st.error(f"Error during feature importance calculation: {e}")
                                import traceback
                                st.error(traceback.format_exc())
                    
                    # 9. Plot Actual vs Predicted
                    test_dates = data.index[-len(metrics['y_test_inv']):]

                    fig3, ax3 = plt.subplots(figsize=(10, 6))
                    ax3.plot(test_dates, metrics['y_test_inv'], label="Actual Price")
                    ax3.plot(test_dates, metrics['y_pred_inv'], label="Predicted Price")
                    ax3.set_xlabel("Date")
                    ax3.set_ylabel("Close Price (USD)")
                    ax3.set_title(f"{symbol} Stock Price Prediction (Test Set)")
                    ax3.legend()
                    plt.xticks(rotation=45)
                    st.pyplot(fig3)

                    # Walk-forward validation section
                    if enable_walk_forward:
                        with st.spinner("Performing walk-forward validation..."):
                            try:
                                results_df, predictions_df, summary = walk_forward_validation(
                                    data=data,
                                    features=features,
                                    sequence_length=sequence_length,
                                    validation_window=validation_window,
                                    train_min_size=train_min_size,
                                    step_size=wf_step_size,
                                    lstm_units=lstm_units,
                                    dropout_rate=dropout_rate,
                                    learning_rate=learning_rate,
                                    epochs=epochs,
                                    batch_size=batch_size
                                )
                                
                                # Display summary metrics
                                st.subheader("Walk-Forward Validation Results")
                                
                                col1, col2, col3 = st.columns(3)
                                col1.metric("Average MAE", f"${summary['avg_mae']:.2f}")
                                col2.metric("Average RMSE", f"${summary['avg_rmse']:.2f}")
                                col3.metric("Avg Direction Accuracy", f"{summary['avg_direction_accuracy']:.1f}%")
                                
                                # Plot results
                                fig = plot_walk_forward_results(results_df, predictions_df, summary, symbol)
                                st.pyplot(fig)
                                
                                # Show detailed results
                                with st.expander("Detailed Validation Results", expanded=False):
                                    st.dataframe(results_df)
                                
                            except Exception as e:
                                st.error(f"Error in walk-forward validation: {e}")
                                import traceback
                                st.error(traceback.format_exc())

                    # 10. Predict future prices
                    st.subheader(f"Future Price Predictions ({prediction_days} days)")
                    
                    # Prepare data for prediction
                    full_scaled_data = scaler.transform(data[columns_to_use])
                    
                    # Predict future prices
                    future_prices, last_actual_price = predict_future_prices(
                        model, 
                        full_scaled_data, 
                        scaler, 
                        columns_to_use, 
                        sequence_length=sequence_length,
                        n_days=prediction_days
                    )

                    # Generate future dates (excluding weekends)
                    last_date = data.index[-1]
                    future_dates = []

                    current_date = last_date
                    for _ in range(prediction_days):
                        current_date = current_date + timedelta(days=1)
                        # Skip weekends
                        while current_date.weekday() >= 5:  # Saturday=5, Sunday=6
                            current_date = current_date + timedelta(days=1)
                        future_dates.append(current_date)

                    # Calculate the first day's change
                    first_day_change = ((future_prices[0] / last_actual_price) - 1) * 100

                    # Create a dataframe with predictions and accurate changes
                    prediction_df = pd.DataFrame({
                        'Date': future_dates,
                        'Predicted Price': future_prices,
                        'Change (%)': [first_day_change] + [(future_prices[i] / future_prices[i-1] - 1) * 100 for i in range(1, len(future_prices))]
                    })

                    # Display predictions table
                    st.dataframe(prediction_df)

                    # Save next day prediction
                    today = datetime.now().strftime('%Y-%m-%d')
                    save_prediction(symbol, today, future_prices[0])

                    # Add sentiment information if requested
                    if include_sentiment and total_comments > 0:
                        st.info(f"**Current Reddit Sentiment:** {sentiment_indicator(avg_sentiment)} ({avg_sentiment:.2f}) based on {total_comments} comments")

                    # For the visualization
                    fig4, ax4 = plt.subplots(figsize=(10, 6))

                    # Plot recent actual prices
                    recent_dates = data.index[-30:]
                    recent_prices = data['Close'].iloc[-30:].values
                    ax4.plot(recent_dates, recent_prices, label='Historical Prices', color='blue')

                    # Add a simple connecting line
                    ax4.plot([last_date, future_dates[0]], [last_actual_price, future_prices[0]], 
                            color='gray', linestyle='--', alpha=0.7)

                    # Plot the predictions
                    ax4.plot(future_dates, future_prices, label='Predicted Prices', color='red', marker='o')

                    # Add confidence interval (based on MAE)
                    ax4.fill_between(
                        future_dates,
                        future_prices - metrics['mae'],
                        future_prices + metrics['mae'],
                        color='red',
                        alpha=0.2
                    )

                    # Add a vertical line at the last known price
                    ax4.axvline(x=last_date, color='green', linestyle='--', alpha=0.7)

                    ax4.set_title(f"{symbol} Stock Price Forecast")
                    ax4.set_xlabel("Date")
                    ax4.set_ylabel("Price (USD)")
                    ax4.legend()
                    plt.xticks(rotation=45)
                    st.pyplot(fig4)
                    
                except Exception as e:
                    st.error(f"Error in stock prediction: {e}")
                    import traceback
                    st.error(traceback.format_exc())
    
    # -------------------- TAB 3: PREDICTION HISTORY --------------------
    with tab3:
        st.header("Prediction History & Accuracy")
        
        # Options for comparing past predictions
        days_ago_compare = st.slider("Days ago to compare", 1, 100, 2)
        
        if st.button("View Prediction History", key="history_button"):
            try:
                # Check if we have past predictions
                past_predictions = load_predictions(symbol)
                
                if not past_predictions:
                    st.info("No past predictions available for this symbol.")
                else:
                    # Update past predictions with actual values if possible
                    # First fetch the data again to have access to more recent prices
                    data = fetch_data(symbol)
                    
                    # Update predictions with actual values
                    for date_str, values in past_predictions.items():
                        if values['actual'] is None:
                            prediction_date = datetime.strptime(date_str, '%Y-%m-%d')
                            target_date = prediction_date + timedelta(days=1)
                            
                            # Skip weekends
                            while target_date.weekday() >= 5:
                                target_date = target_date + timedelta(days=1)
                            
                            # Check if we have actual data for this date
                            target_date_str = target_date.strftime('%Y-%m-%d')
                            if target_date_str in data.index:
                                actual_price = data.loc[target_date_str, 'Close']
                                update_prediction_with_actual(symbol, date_str, actual_price)
                    
                    # Get updated predictions
                    past_predictions = load_predictions(symbol)
                    
                    # Convert to dataframe for display
                    pred_list = []
                    for date, values in past_predictions.items():
                        pred_list.append({
                            'Date': date,
                            'Prediction': values['prediction'],
                            'Actual': values['actual'],
                            'Error': abs(values['prediction'] - values['actual']) if values['actual'] is not None else None,
                            'Error %': (abs(values['prediction'] - values['actual']) / values['actual'] * 100) if values['actual'] is not None else None
                        })
                    
                    pred_df = pd.DataFrame(pred_list)
                    pred_df = pred_df.sort_values('Date', ascending=False)
                    
                    # Display the full history
                    st.subheader("All Predictions")
                    st.dataframe(pred_df)
                    
                    # Filter for predictions with actual values
                    predictions_with_actuals = pred_df.dropna(subset=['Actual'])
                    
                    if not predictions_with_actuals.empty:
                        # Calculate average error
                        avg_error = predictions_with_actuals['Error'].mean()
                        avg_error_pct = predictions_with_actuals['Error %'].mean()
                        
                        col1, col2 = st.columns(2)
                        col1.metric("Average Prediction Error", f"${avg_error:.2f}")
                        col2.metric("Average Error Percentage", f"{avg_error_pct:.2f}%")
                        
                        # Plot error over time
                        fig5, ax5 = plt.subplots(figsize=(10, 6))
                        
                        # Sort by date for the plot
                        plot_df = predictions_with_actuals.sort_values('Date')
                        
                        ax5.plot(plot_df['Date'], plot_df['Error %'], marker='o', linestyle='-')
                        ax5.set_title('Prediction Error % Over Time')
                        ax5.set_xlabel('Prediction Date')
                        ax5.set_ylabel('Error %')
                        ax5.grid(True, alpha=0.3)
                        plt.xticks(rotation=45)
                        
                        # Add a horizontal line for average error
                        ax5.axhline(y=avg_error_pct, color='r', linestyle='--', alpha=0.7)
                        ax5.annotate(f'Avg Error: {avg_error_pct:.2f}%', 
                                    xy=(0.02, avg_error_pct),
                                    xycoords=('axes fraction', 'data'),
                                    color='r')
                        
                        st.pyplot(fig5)
                        
                        # Plot predicted vs actual
                        fig6, ax6 = plt.subplots(figsize=(10, 6))
                        
                        ax6.scatter(plot_df['Actual'], plot_df['Prediction'], alpha=0.7)
                        
                        # Add perfect prediction line
                        min_val = min(plot_df['Actual'].min(), plot_df['Prediction'].min())
                        max_val = max(plot_df['Actual'].max(), plot_df['Prediction'].max())
                        ax6.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
                        
                        ax6.set_title('Predicted vs Actual Prices')
                        ax6.set_xlabel('Actual Price')
                        ax6.set_ylabel('Predicted Price')
                        ax6.grid(True, alpha=0.3)
                        
                        st.pyplot(fig6)
                    else:
                        st.info("No predictions have been verified with actual values yet.")
            
            except Exception as e:
                st.error(f"Error analyzing prediction history: {e}")
                import traceback
                st.error(traceback.format_exc())

if __name__ == "__main__":
    main()