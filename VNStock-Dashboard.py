import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import pandas_ta as ta
import numpy as np
from datetime import datetime
import os

# Constants
FOLDER_PATH = "FOLDER_PATH"

class StockData:
    def __init__(self, folder_path):
        """Initialize with folder path containing stock data"""
        self.folder_path = folder_path
        
    def get_available_symbols(self):
        """Get list of available stock symbols without modifying files"""
        try:
            # Only list CSV files
            return [f.replace('.csv', '') for f in os.listdir(self.folder_path) 
                   if f.endswith('.csv')]
        except Exception as e:
            return []
    
    def load_stock_data(self, symbol):
        """Load stock data without modifying original values"""
        try:
            file_path = os.path.join(self.folder_path, f"{symbol}.csv")
            
            # Read CSV with minimal processing
            df = pd.read_csv(file_path, parse_dates=['Date'])
            
            # Create a copy for processing
            processed_df = df.copy()
            
            # Sort by date without modifying original
            processed_df = processed_df.sort_values('Date')
            
            # Remove duplicates if they exist, without modifying source
            if processed_df.duplicated(subset=['Date']).any():
                processed_df = processed_df.drop_duplicates(subset=['Date'], keep='first')
            
            return processed_df
            
        except Exception as e:
            return None

    def process_data(self, data, period='1y', start_date=None, end_date=None):
        """Process data for specified time period"""
        if data is None:
            return None
            
        df = data.copy()
        
        # Filter data between 2000 and 2022
        df = df[(df['Date'].dt.year >= 2000) & (df['Date'].dt.year <= 2022)]
        
        if period == 'Custom' and start_date and end_date:
            mask = (df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date)
            df = df[mask]
        else:
            end_date = df['Date'].max()
            if period == '1d':
                start_date = end_date - pd.Timedelta(days=1)
            elif period == '1wk':
                start_date = end_date - pd.Timedelta(weeks=1)
            elif period == '1mo':
                start_date = end_date - pd.Timedelta(days=30)
            elif period == '1y':
                start_date = end_date - pd.Timedelta(days=365)
            else:  # 'max'
                start_date = df['Date'].min()
            
            df = df[df['Date'] >= start_date]
        
        return df.reset_index(drop=True)

class TechnicalAnalysis:
    @staticmethod
    def calculate_indicators(data, rsi_thresholds=(30, 70)):
        """Calculate technical indicators with customizable parameters"""
        if data is None or data.empty:
            return None
            
        try:
            df = data.copy()
            
            # ... [previous indicator calculations] ...
            # Create a copy to avoid modifying original data
            df = data.copy()
            
            # Handle missing data
            df['Price Close'] = df['Price Close'].fillna(method='ffill')
            df['Price Open'] = df['Price Open'].fillna(df['Price Close'])
            df['Price High'] = df['Price High'].fillna(df['Price Close'])
            df['Price Low'] = df['Price Low'].fillna(df['Price Close'])
            df['Volume'] = df['Volume'].fillna(0)
            
            # SMA and EMA
            df['SMA_20'] = df['Price Close'].rolling(window=20, min_periods=1).mean()
            df['EMA_20'] = df['Price Close'].ewm(span=20, adjust=False, min_periods=1).mean()
            
            # Bollinger Bands
            bb_period = 20
            bb_std = 2
            df['BBM_20_2.0'] = df['Price Close'].rolling(window=bb_period, min_periods=1).mean()
            bb_std_val = df['Price Close'].rolling(window=bb_period, min_periods=1).std()
            df['BBU_20_2.0'] = df['BBM_20_2.0'] + (bb_std * bb_std_val)
            df['BBL_20_2.0'] = df['BBM_20_2.0'] - (bb_std * bb_std_val)
            
            # RSI
            delta = df['Price Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
            rs = gain / loss
            df['RSI_14'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = df['Price Close'].ewm(span=12, adjust=False, min_periods=1).mean()
            exp2 = df['Price Close'].ewm(span=26, adjust=False, min_periods=1).mean()
            df['MACD_12_26_9'] = exp1 - exp2
            df['MACDs_12_26_9'] = df['MACD_12_26_9'].ewm(span=9, adjust=False, min_periods=1).mean()
            df['MACDh_12_26_9'] = df['MACD_12_26_9'] - df['MACDs_12_26_9']
            
            # Calculate support and resistance using the new method
            df['Support'], df['Resistance'] = TechnicalAnalysis.calculate_support_resistance(df)
            
            # Calculate MFI
            df['MFI'] = TechnicalAnalysis.calculate_mfi(df)
            # OBV
            df['Price_Change'] = df['Price Close'].diff()
            df['OBV'] = df['Volume'].copy()
            df.loc[df['Price_Change'] < 0, 'OBV'] = -df['Volume']
            df.loc[df['Price_Change'] == 0, 'OBV'] = 0
            df['OBV'] = df['OBV'].cumsum()
            df['OBV'] = df['OBV'].rolling(window=10, min_periods=1).mean()
            
            # ATR
            high_low = df['Price High'] - df['Price Low']
            high_close = np.abs(df['Price High'] - df['Price Close'].shift())
            low_close = np.abs(df['Price Low'] - df['Price Close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            df['ATR_14'] = true_range.rolling(14, min_periods=1).mean()

            # Support and Resistance levels
            window = 20
            df['Resistance'] = df['Price High'].rolling(window=window, min_periods=1).max()
            df['Support'] = df['Price Low'].rolling(window=window, min_periods=1).min()
            # Trading Signals with customizable thresholds
            df['RSI_Signal'] = 'Hold'
            df.loc[df['RSI_14'] < rsi_thresholds[0], 'RSI_Signal'] = 'Buy'
            df.loc[df['RSI_14'] > rsi_thresholds[1], 'RSI_Signal'] = 'Sell'
            
            df['MACD_Signal'] = 'Hold'
            df.loc[(df['MACD_12_26_9'] > df['MACDs_12_26_9']) & 
                  (df['MACD_12_26_9'].shift(1) <= df['MACDs_12_26_9'].shift(1)), 'MACD_Signal'] = 'Buy'
            df.loc[(df['MACD_12_26_9'] < df['MACDs_12_26_9']) & 
                  (df['MACD_12_26_9'].shift(1) >= df['MACDs_12_26_9'].shift(1)), 'MACD_Signal'] = 'Sell'
            
            return df
            
        except Exception as e:
            print(f"Error calculating indicators: {str(e)}")
            return None
    def calculate_support_resistance(df, window=20):
        """Calculate support and resistance levels using pivot points"""
        df = df.copy()
        
        # Initialize lists to store support and resistance levels
        supports = []
        resistances = []
        
        for i in range(window, len(df)):
            # Get the window of data
            window_data = df.iloc[i-window:i]
            
            # Find pivot highs (resistance)
            pivot_high = window_data['Price High'].max()
            pivot_high_idx = window_data['Price High'].idxmax()
            
            # Find pivot lows (support)
            pivot_low = window_data['Price Low'].min()
            pivot_low_idx = window_data['Price Low'].idxmin()
            
            # Check if pivot points are valid
            is_resistance = all(
                df.iloc[pivot_high_idx]['Price High'] >= df.iloc[j]['Price High']
                for j in range(pivot_high_idx - 2, pivot_high_idx + 3)
                if 0 <= j < len(df) and j != pivot_high_idx
            )
            
            is_support = all(
                df.iloc[pivot_low_idx]['Price Low'] <= df.iloc[j]['Price Low']
                for j in range(pivot_low_idx - 2, pivot_low_idx + 3)
                if 0 <= j < len(df) and j != pivot_low_idx
            )
            
            if is_resistance:
                resistances.append(pivot_high)
            else:
                resistances.append(None)
                
            if is_support:
                supports.append(pivot_low)
            else:
                supports.append(None)
        
        # Pad the beginning of the lists with None values
        supports = [None] * window + supports
        resistances = [None] * window + resistances
        
        # Add to dataframe
        df['Support'] = supports
        df['Resistance'] = resistances
        
        # Forward fill the values
        df['Support'] = df['Support'].fillna(method='ffill')
        df['Resistance'] = df['Resistance'].fillna(method='ffill')
        
        return df['Support'], df['Resistance']
    def calculate_mfi(df, period=14):
        """Calculate Money Flow Index"""
        try:
            # Typical price
            typical_price = (df['Price High'] + df['Price Low'] + df['Price Close']) / 3
            
            # Money flow
            money_flow = typical_price * df['Volume']
            
            # Get positive and negative money flow
            diff = typical_price.diff()
            
            # Separate positive and negative money flow
            positive_flow = money_flow.where(diff > 0, 0.0)
            negative_flow = money_flow.where(diff < 0, 0.0)
            
            # Get money flow ratio
            positive_mf = positive_flow.rolling(window=period, min_periods=1).sum()
            negative_mf = negative_flow.rolling(window=period, min_periods=1).sum()
            
            # Calculate MFI
            money_flow_ratio = positive_mf / negative_mf
            mfi = 100 - (100 / (1 + money_flow_ratio))
            
            return mfi
            
        except Exception as e:
            print(f"Error calculating MFI: {str(e)}")
            return None

class ChartBuilder:
    def __init__(self, data, ticker):
        self.data = data
        self.ticker = ticker
        self.dates = data['Date'].dt.strftime('%Y-%m-%d').tolist()
    
    def create_chart(self, chart_type, show_volume=True, indicators=None, show_resistance_support=False, show_signals=False):
        """Create interactive chart with selected indicators and optional features"""
        if indicators is None:
            indicators = []
            
        num_rows = self._calculate_rows(show_volume, indicators)
        row_heights = self._calculate_row_heights(num_rows, show_volume)
        
        fig = make_subplots(
            rows=num_rows, 
            cols=1, 
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=row_heights
        )
        
        current_row = 1
        
        # Add main price chart
        self._add_price_chart(fig, chart_type, current_row)
        
        # Add resistance/support if enabled
        if show_resistance_support and 'Resistance' in self.data.columns and 'Support' in self.data.columns:
            self._add_resistance_support(fig, current_row)
        
        # Add trading signals if enabled
        if show_signals and 'Combined_Signal' in self.data.columns:
            self._add_trading_signals(fig, current_row)
        
        # Add overlay indicators
        self._add_overlay_indicators(fig, indicators, current_row)
        
        if show_volume:
            current_row += 1
            self._add_volume(fig, current_row)
        
        # Add separate indicator panels
        current_row = self._add_indicator_panels(fig, indicators, current_row)
        
        self._update_layout(fig, num_rows)
        
        return fig

    def _add_resistance_support(self, fig, row):
        """Add resistance and support levels with improved visualization"""
        # Add resistance level
        fig.add_trace(
            go.Scatter(
                x=self.data.index,
                y=self.data['Resistance'],
                name='Resistance',
                line=dict(color='rgba(255, 0, 0, 0.5)', width=1, dash='dash'),
                showlegend=True
            ),
            row=row, col=1
        )
        
        # Add support level
        fig.add_trace(
            go.Scatter(
                x=self.data.index,
                y=self.data['Support'],
                name='Support',
                line=dict(color='rgba(0, 255, 0, 0.5)', width=1, dash='dash'),
                showlegend=True
            ),
            row=row, col=1
        )

    def _add_trading_signals(self, fig, row):
        """Add buy/sell signals with improved visualization"""
        # Add buy signals
        buy_signals = self.data[self.data['Combined_Signal'] == 'Buy']
        if not buy_signals.empty:
            fig.add_trace(
                go.Scatter(
                    x=buy_signals.index,
                    y=buy_signals['Price Close'],
                    name='Buy Signal',
                    mode='markers',
                    marker=dict(
                        symbol='triangle-up',
                        size=10,
                        color='green',
                        line=dict(width=2)
                    ),
                    hovertemplate='Buy Signal<br>Date: %{text}<br>Price: %{y:.2f}<br>RSI: %{customdata[0]:.1f}<br>MACD: %{customdata[1]:.4f}<extra></extra>',
                    text=buy_signals['Date'].dt.strftime('%Y-%m-%d'),
                    customdata=buy_signals[['RSI_14', 'MACD_12_26_9']]
                ),
                row=row, col=1
            )

        # Add sell signals
        sell_signals = self.data[self.data['Combined_Signal'] == 'Sell']
        if not sell_signals.empty:
            fig.add_trace(
                go.Scatter(
                    x=sell_signals.index,
                    y=sell_signals['Price Close'],
                    name='Sell Signal',
                    mode='markers',
                    marker=dict(
                        symbol='triangle-down',
                        size=10,
                        color='red',
                        line=dict(width=2)
                    ),
                    hovertemplate='Sell Signal<br>Date: %{text}<br>Price: %{y:.2f}<br>RSI: %{customdata[0]:.1f}<br>MACD: %{customdata[1]:.4f}<extra></extra>',
                    text=sell_signals['Date'].dt.strftime('%Y-%m-%d'),
                    customdata=sell_signals[['RSI_14', 'MACD_12_26_9']]
                ),
                row=row, col=1
            )      
    def _calculate_rows(self, show_volume, indicators):
        """Calculate number of subplot rows needed"""
        num_rows = 1  # Main chart
        
        # Volume panel
        if show_volume:
            num_rows += 1
            
        # Separate indicator panels
        indicator_panels = {
            'RSI': 1,
            'MACD': 1,
            'MFI': 1,  # Add MFI to indicator panels
            'OBV': 1,
            'ATR': 1
        }
        
        # Count all indicators that need separate panels
        for indicator in indicators:
            if indicator in indicator_panels:
                num_rows += indicator_panels[indicator]
        
        return num_rows
    def _calculate_row_heights(self, num_rows, show_volume):
        """Calculate height ratios for subplots"""
        row_heights = [0.5]  # Main chart is 50% of height
        
        # If showing volume, add it with 20% height
        if show_volume:
            row_heights.append(0.2)
        
        # Remaining indicators get equal portions of remaining height
        remaining_rows = num_rows - len(row_heights)
        if remaining_rows > 0:
            indicator_height = 0.3  # 30% height for each indicator panel
            row_heights.extend([indicator_height] * remaining_rows)
        
        return row_heights
    
    def _add_price_chart(self, fig, chart_type, row):
        """Add main price chart"""
        if chart_type == 'Candlestick':
            fig.add_trace(
                go.Candlestick(
                    x=self.data.index,
                    open=self.data['Price Open'],
                    high=self.data['Price High'],
                    low=self.data['Price Low'],
                    close=self.data['Price Close'],
                    name=self.ticker,
                    increasing_line_color='red',
                    decreasing_line_color='green',
                    whiskerwidth=1,
                    line_width=1,
                    hovertext=self.dates
                ),
                row=row, col=1
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=self.data.index,
                    y=self.data['Price Close'],
                    name=self.ticker,
                    line=dict(width=2),
                    hovertext=self.dates
                ),
                row=row, col=1
            )
    
    def _add_overlay_indicators(self, fig, indicators, row):
        """Add overlay indicators to main chart"""
        if 'SMA 20' in indicators:
            fig.add_trace(
                go.Scatter(
                    x=self.data.index, 
                    y=self.data['SMA_20'],
                    name='SMA 20',
                    line=dict(color='orange', width=1.5)
                ),
                row=row, col=1
            )
        
        if 'EMA 20' in indicators:
            fig.add_trace(
                go.Scatter(
                    x=self.data.index,
                    y=self.data['EMA_20'],
                    name='EMA 20',
                    line=dict(color='blue', width=1.5)
                ),
                row=row, col=1
            )
        
        if 'Bollinger Bands' in indicators:
            for band, color in [('BBU_20_2.0', 'gray'), ('BBM_20_2.0', 'gray'), ('BBL_20_2.0', 'gray')]:
                fig.add_trace(
                    go.Scatter(
                        x=self.data.index,
                        y=self.data[band],
                        name=f'BB {band.split("_")[0][2:]}',
                        line=dict(color=color, width=1, dash='dash' if 'M' not in band else None)
                    ),
                    row=row, col=1
                ) 
    def _add_volume(self, fig, row):
        """Add volume chart"""
        colors = ['red' if row['Price Close'] >= row['Price Open'] else 'green' 
                 for index, row in self.data.iterrows()]
        fig.add_trace(
            go.Bar(
                x=self.data.index,
                y=self.data['Volume'],
                name='Volume',
                marker_color=colors,
                marker_line_width=0,
                opacity=0.8,
                hovertext=self.dates
            ),
            row=row, col=1
        )
    
    def _add_indicator_panels(self, fig, indicators, current_row):
        """Add separate panels for indicators"""
        
        # MFI Panel first
        if 'MFI' in indicators:
            current_row += 1
            fig.add_trace(
                go.Scatter(
                    x=self.data.index,
                    y=self.data['MFI'],
                    name='MFI',
                    line=dict(color='purple')
                ),
                row=current_row, col=1
            )
            # Add MFI overbought/oversold lines
            fig.add_hline(y=80, line_dash="dash", line_color="red", row=current_row)
            fig.add_hline(y=20, line_dash="dash", line_color="green", row=current_row)
            fig.update_yaxes(title_text="MFI", range=[0, 100], row=current_row, col=1)

        # RSI Panel
        if 'RSI' in indicators:
            current_row += 1
            fig.add_trace(
                go.Scatter(
                    x=self.data.index,
                    y=self.data['RSI_14'],
                    name='RSI',
                    line=dict(color='blue')
                ),
                row=current_row, col=1
            )
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=current_row)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=current_row)
            fig.update_yaxes(title_text="RSI", range=[0, 100], row=current_row, col=1)

        # MACD Panel
        if 'MACD' in indicators:
            current_row += 1
            fig.add_trace(
                go.Scatter(
                    x=self.data.index,
                    y=self.data['MACD_12_26_9'],
                    name='MACD',
                    line=dict(color='blue')
                ),
                row=current_row, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=self.data.index,
                    y=self.data['MACDs_12_26_9'],
                    name='Signal',
                    line=dict(color='orange')
                ),
                row=current_row, col=1
            )
            fig.add_trace(
                go.Bar(
                    x=self.data.index,
                    y=self.data['MACDh_12_26_9'],
                    name='MACD Hist',
                    marker_color='gray'
                ),
                row=current_row, col=1
            )
            fig.update_yaxes(title_text="MACD", row=current_row, col=1)

        # OBV Panel
        if 'OBV' in indicators:
            current_row += 1
            fig.add_trace(
                go.Scatter(
                    x=self.data.index,
                    y=self.data['OBV'],
                    name='OBV',
                    line=dict(color='teal')
                ),
                row=current_row, col=1
            )
            fig.update_yaxes(title_text="OBV", row=current_row, col=1)

        # ATR Panel
        if 'ATR' in indicators:
            current_row += 1
            fig.add_trace(
                go.Scatter(
                    x=self.data.index,
                    y=self.data['ATR_14'],
                    name='ATR',
                    line=dict(color='brown')
                ),
                row=current_row, col=1
            )
            fig.update_yaxes(title_text="ATR", row=current_row, col=1)

        return current_row       
    def _update_layout(self, fig, num_rows):
        """Update chart layout and styling"""
        fig.update_layout(
            title=f'{self.ticker} - Price Chart',
            xaxis_title='Time',
            yaxis_title='Price (VND)',
            height=300 + (200 * num_rows),  # Adjust height based on number of panels
            showlegend=True,
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis_rangeslider_visible=False,
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Roboto"
            ),
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

def main():
    st.set_page_config(layout="wide", page_title="Stock Dashboard")
    st.title('Stock Dashboard - Vietnam Market')
    
    # Initialize data and StockData
    data = None
    stock_data = StockData(FOLDER_PATH)
    
    # Check path and available symbols
    if not os.path.exists(FOLDER_PATH):
        st.error("Path does not exist!")
        return
    
    available_symbols = stock_data.get_available_symbols()
    if not available_symbols:
        st.error("No CSV files found!")
        return
    
    # Sidebar - Basic Options
    st.sidebar.header('Chart Options')
    
    # Stock selection
    ticker = st.sidebar.selectbox('Select Stock Symbol', available_symbols)
    
    # Time period selection
    time_period = st.sidebar.selectbox('Time Period', ['Custom', '1d', '1wk', '1mo', '1y', 'max'])
    
    # Custom date range (2000-2022)
    if time_period == 'Custom':
        col1, col2 = st.sidebar.columns(2)
        min_date = datetime(2000, 1, 1).date()
        max_date = datetime(2022, 12, 31).date()
        with col1:
            start_date = st.date_input('Start Date', 
                                     min_value=min_date,
                                     max_value=max_date,
                                     value=min_date)
        with col2:
            end_date = st.date_input('End Date', 
                                   min_value=min_date,
                                   max_value=max_date,
                                   value=max_date)
    
    # Chart type and volume
    chart_type = st.sidebar.selectbox('Chart Type', ['Candlestick', 'Line'])
    show_volume = st.sidebar.checkbox('Show Volume', value=True)
    
    # Technical Analysis Options
    with st.sidebar.expander("Technical Analysis Options", expanded=False):
        # Signal type selection
        signal_type = st.radio(
            "Trading Signal Type",
            ["RSI", "MACD", "Combined"],
            help="Choose which trading signals to display"
        )
        
        # RSI Thresholds
        if signal_type in ["RSI", "Combined"]:
            col1, col2 = st.sidebar.columns(2)
            with col1:
                rsi_lower = st.number_input("RSI Oversold", value=30, min_value=0, max_value=100)
            with col2:
                rsi_upper = st.number_input("RSI Overbought", value=70, min_value=0, max_value=100)
        
        # Display options
        show_signals = st.checkbox('Show Trading Signals', value=False)
        show_support_resistance = st.checkbox('Show Support/Resistance', value=False)
    
    # Technical Indicators Selection
    st.sidebar.subheader('Technical Indicators')
    overlay_indicators = st.sidebar.multiselect(
        'Price Chart Overlay',
        ['SMA 20', 'EMA 20', 'Bollinger Bands']
    )
    separate_indicators = st.sidebar.multiselect(
        'Separate Panels',
        ['RSI', 'MACD', 'MFI', 'OBV', 'ATR']
    )
    
    indicators = overlay_indicators + separate_indicators
    
    # Update Chart Button
    if st.sidebar.button('Update Chart'):
        try:
            # Load and process data
            with st.spinner('Loading data...'):
                # Load data
                data = stock_data.load_stock_data(ticker)
                if data is None:
                    st.error("Failed to load data!")
                    return
                
                # Process time period
                if time_period == 'Custom':
                    data = stock_data.process_data(data, period='Custom', 
                                                 start_date=start_date, 
                                                 end_date=end_date)
                else:
                    data = stock_data.process_data(data, period=time_period)
                
                # Calculate indicators with custom thresholds
                rsi_thresholds = (
                    rsi_lower if signal_type in ["RSI", "Combined"] else 30,
                    rsi_upper if signal_type in ["RSI", "Combined"] else 70
                )
                data = TechnicalAnalysis.calculate_indicators(data, rsi_thresholds=rsi_thresholds)
                
                if data is not None:
                    # Set signal type based on selection
                    if signal_type != "Combined":
                        data['Combined_Signal'] = data[f'{signal_type}_Signal']
                    
                    # Create and display chart
                    chart_builder = ChartBuilder(data, ticker)
                    fig = chart_builder.create_chart(
                        chart_type=chart_type,
                        show_volume=show_volume,
                        indicators=indicators,
                        show_resistance_support=show_support_resistance,
                        show_signals=show_signals
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display Trading Signals Analysis
                    if show_signals:
                        st.subheader(f'{signal_type} Trading Signals Analysis')
                        try:
                            # Prepare signal columns
                            base_columns = ['Date', 'Price Close', 'RSI_14', 'MACD_12_26_9']
                            if signal_type == "Combined":
                                signal_columns = base_columns + ['RSI_Signal', 'MACD_Signal', 'Combined_Signal']
                            else:
                                signal_columns = base_columns + [f'{signal_type}_Signal']
                            
                            # Get signals data
                            signals_df = data[signal_columns]
                            signal_col = 'Combined_Signal' if signal_type == "Combined" else f'{signal_type}_Signal'
                            signals_df = signals_df[signals_df[signal_col] != 'Hold'].copy()
                            
                            if not signals_df.empty:
                                # Format data
                                signals_df['Date'] = signals_df['Date'].dt.strftime('%Y-%m-%d')
                                signals_df['Price Close'] = signals_df['Price Close'].round(2)
                                signals_df['RSI_14'] = signals_df['RSI_14'].round(2)
                                signals_df['MACD_12_26_9'] = signals_df['MACD_12_26_9'].round(4)
                                
                                # Display signal statistics
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    buy_signals = len(signals_df[signals_df[signal_col] == 'Buy'])
                                    st.metric("Buy Signals", buy_signals)
                                with col2:
                                    sell_signals = len(signals_df[signals_df[signal_col] == 'Sell'])
                                    st.metric("Sell Signals", sell_signals)
                                with col3:
                                    total_signals = len(signals_df)
                                    st.metric("Total Signals", total_signals)
                                
                                # Display signals table
                                st.dataframe(
                                    signals_df.style.apply(
                                        lambda x: ['background-color: #c6efcd' if v == 'Buy' 
                                                else 'background-color: #ffc7ce' if v == 'Sell'
                                                else '' for v in x],
                                        subset=[signal_col]
                                    ),
                                    use_container_width=True
                                )
                            else:
                                st.info('No trading signals found in the selected time period.')
                        except Exception as e:
                            st.error(f'Error displaying trading signals: {str(e)}')
                    
                    # Display Detailed Data
                    with st.expander("Detailed Data", expanded=False):
                        st.subheader('Price and Indicator Data')
                        
                        # Select columns to display
                        cols_to_display = ['Date', 'Price Open', 'Price High', 'Price Low', 'Price Close', 'Volume']
                        
                        # Add indicator columns
                        indicator_mapping = {
                            'SMA 20': 'SMA_20',
                            'EMA 20': 'EMA_20',
                            'Bollinger Bands': ['BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0'],
                            'MFI': 'MFI',
                            'RSI': 'RSI_14',
                            'MACD': ['MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9'],
                            'OBV': 'OBV',
                            'ATR': 'ATR_14'
                        }
                        
                        for indicator in indicators:
                            if indicator in indicator_mapping:
                                if isinstance(indicator_mapping[indicator], list):
                                    cols_to_display.extend(indicator_mapping[indicator])
                                else:
                                    cols_to_display.append(indicator_mapping[indicator])
                        
                        # Format and display data
                        display_df = data[cols_to_display].copy()
                        display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
                        st.dataframe(display_df, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
            st.exception(e)

# Create and export PDF files
from fpdf import FPDF
from io import BytesIO
import base64

def get_all_stock_data(folder_path):
    """Load all stock data into a dictionary."""
    stock_data = {}
    try:
        for file in os.listdir(folder_path):
            if file.endswith('.csv'):
                symbol = file.replace('.csv', '')
                df = pd.read_csv(
                    os.path.join(folder_path, file), parse_dates=['Date']
                )
                stock_data[symbol] = df
        return stock_data
    except Exception as e:
        st.error(f"Error loading stock data: {str(e)}")
        return {}

def filter_data_by_date(data, start_date, end_date):
    """Filter stock data based on selected date range."""
    return data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]

def calculate_top_stocks(stock_data, start_date, end_date):
    """Calculate top 10 stocks based on positive, negative, high, and low changes."""
    summary = []
    for symbol, data in stock_data.items():
        filtered_data = filter_data_by_date(data, start_date, end_date)
        if not filtered_data.empty:
            price_change = filtered_data['Price Close'].iloc[-1] - filtered_data['Price Close'].iloc[0]
            high_price = filtered_data['Price High'].max()
            low_price = filtered_data['Price Low'].min()
            summary.append({
                'Symbol': symbol,
                'Price Change': price_change,
                'High Price': high_price,
                'Low Price': low_price
            })
    summary_df = pd.DataFrame(summary)

    # Calculate top stocks
    top_positive = summary_df.nlargest(10, 'Price Change')
    top_negative = summary_df.nsmallest(10, 'Price Change')
    top_high = summary_df.nlargest(10, 'High Price')
    top_low = summary_df.nsmallest(10, 'Low Price')

    return top_positive, top_negative, top_high, top_low

def generate_pdf_report(top_positive, top_negative, top_high, top_low, start_date, end_date):
    """Generate a PDF report for the selected data."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Title Page
    pdf.add_page()
    pdf.set_font("Arial", size=16, style='B')
    pdf.cell(200, 10, txt="Vietnam's Stock Dashboard Report", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Date Range: {start_date} to {end_date}", ln=True, align='C')

    # Add Table for Top Positive
    pdf.add_page()
    pdf.set_font("Arial", size=14, style='B')
    pdf.cell(200, 10, txt="Top 10 Positive Stocks", ln=True, align='C')
    pdf.set_font("Arial", size=10)
    for index, row in top_positive.iterrows():
        pdf.cell(200, 10, txt=f"{row['Symbol']}: {row['Price Change']:.2f}", ln=True)

    # Add Table for Top Negative
    pdf.add_page()
    pdf.set_font("Arial", size=14, style='B')
    pdf.cell(200, 10, txt="Top 10 Negative Stocks", ln=True, align='C')
    pdf.set_font("Arial", size=10)
    for index, row in top_negative.iterrows():
        pdf.cell(200, 10, txt=f"{row['Symbol']}: {row['Price Change']:.2f}", ln=True)

    # Add Table for High Stocks
    pdf.add_page()
    pdf.set_font("Arial", size=14, style='B')
    pdf.cell(200, 10, txt="Top 10 High Stocks", ln=True, align='C')
    pdf.set_font("Arial", size=10)
    for index, row in top_high.iterrows():
        pdf.cell(200, 10, txt=f"{row['Symbol']}: {row['High Price']:.2f}", ln=True)

    # Add Table for Low Stocks
    pdf.add_page()
    pdf.set_font("Arial", size=14, style='B')
    pdf.cell(200, 10, txt="Top 10 Low Stocks", ln=True, align='C')
    pdf.set_font("Arial", size=10)
    for index, row in top_low.iterrows():
        pdf.cell(200, 10, txt=f"{row['Symbol']}: {row['Low Price']:.2f}", ln=True)

    # Save PDF
    pdf_path = "stock_report.pdf"
    pdf.output(pdf_path)
    return pdf_path

def export_pdf_report_ui():
    st.title("Export PDF Report")

    # Load all stock data
    stock_data = get_all_stock_data(FOLDER_PATH)

    # Date Range Selection
    min_date = datetime(2000, 1, 1).date()
    max_date = datetime(2022, 12, 31).date()
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", min_value=min_date, max_value=max_date, value=min_date)
    with col2:
        end_date = st.date_input("End Date", min_value=min_date, max_value=max_date, value=max_date)

    if st.button("Generate Report"):
        if start_date > end_date:
            st.error("Start date must be before end date!")
        else:
            with st.spinner("Processing data..."):
                # Calculate top stocks
                top_positive, top_negative, top_high, top_low = calculate_top_stocks(
                    stock_data, pd.Timestamp(start_date), pd.Timestamp(end_date)
                )

                # Display results
                st.subheader("Top 10 Positive Stocks")
                st.dataframe(top_positive, use_container_width=True)

                st.subheader("Top 10 Negative Stocks")
                st.dataframe(top_negative, use_container_width=True)

                st.subheader("Top 10 High Stocks")
                st.dataframe(top_high, use_container_width=True)

                st.subheader("Top 10 Low Stocks")
                st.dataframe(top_low, use_container_width=True)

                # Generate PDF
                pdf_path = generate_pdf_report(top_positive, top_negative, top_high, top_low, start_date, end_date)
                st.success("PDF Report generated successfully!")

                # Download link
                with open(pdf_path, "rb") as pdf_file:
                    st.download_button(
                        label="Download PDF Report",
                        data=pdf_file,
                        file_name="stock_report.pdf",
                        mime="application/pdf"
                    )

# Run Export PDF Report UI
if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="Vietnam Stock Dashboard")

    page = st.sidebar.selectbox("Select Page", ["Main Dashboard", "Export PDF Report"])

    if page == "Main Dashboard":
        main()
    elif page == "Export PDF Report":
        stock_data = StockData(FOLDER_PATH).get_available_symbols()
        export_pdf_report_ui()