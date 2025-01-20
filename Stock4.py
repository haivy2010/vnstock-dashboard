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
FOLDER_PATH = r'D:\dai hoc\Data\processed\valid_data'

class StockData:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        
    def get_available_symbols(self):
        try:
            return [f.replace('.csv', '') for f in os.listdir(self.folder_path) if f.endswith('.csv')]
        except Exception as e:
            st.error(f"Error reading folder: {str(e)}")
            return []
    
    def load_stock_data(self, symbol):
        try:
            file_path = os.path.join(self.folder_path, f"{symbol}.csv")
            df = pd.read_csv(file_path)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
            df = df.drop_duplicates(subset=['Date'], keep='first')
            return df
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None

    def process_data(self, data, period='1y', start_date=None, end_date=None):
        """Process and filter data based on time period"""
        if data is None:
            return None
            
        df = data.copy()
        
        # Filter data from 2022 onwards
        df = df[df['Date'] >= '2022-01-01']
        
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
    # Your existing TechnicalAnalysis class code remains the same
    pass


class ChartBuilder:
    def __init__(self, data, ticker):
        self.data = data
        self.ticker = ticker
        self.dates = data['Date'].dt.strftime('%Y-%m-%d').tolist()
    
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
            # Upper Band
            fig.add_trace(
                go.Scatter(
                    x=self.data.index,
                    y=self.data['BBU_20_2.0'],
                    name='BB Upper',
                    line=dict(color='rgba(173, 204, 255, 0.7)', width=1),
                    showlegend=True
                ),
                row=row, col=1
            )
            
            # Middle Band
            fig.add_trace(
                go.Scatter(
                    x=self.data.index,
                    y=self.data['BBM_20_2.0'],
                    name='BB Middle',
                    line=dict(color='rgba(102, 153, 255, 0.7)', width=1.5),
                    showlegend=True
                ),
                row=row, col=1
            )
            
            # Lower Band
            fig.add_trace(
                go.Scatter(
                    x=self.data.index,
                    y=self.data['BBL_20_2.0'],
                    name='BB Lower',
                    line=dict(color='rgba(173, 204, 255, 0.7)', width=1),
                    fill='tonexty',  # Fill area between upper and lower bands
                    fillcolor='rgba(173, 204, 255, 0.1)',
                    showlegend=True
                ),
                row=row, col=1
            )
        
        if 'VWAP' in indicators:
            fig.add_trace(
                go.Scatter(
                    x=self.data.index,
                    y=self.data['VWAP'],
                    name='VWAP',
                    line=dict(color='purple', width=1.5, dash='dot'),
                    showlegend=True
                ),
                row=row, col=1
            )

    def _calculate_rows(self, show_volume, indicators):
        """Calculate number of subplot rows needed"""
        num_rows = 1  # Main chart
        if show_volume:
            num_rows += 1
        if 'RSI' in indicators:
            num_rows += 1
        if 'MACD' in indicators:
            num_rows += 1
        if 'Stochastic' in indicators:
            num_rows += 1
        if 'OBV' in indicators:  # Add row for OBV
            num_rows += 1
        return num_rows


class TechnicalAnalysis:
    @staticmethod
    def calculate_indicators(data):
        """Calculate technical indicators"""
        if data is None or data.empty:
            st.error("No data available for technical analysis")
            return None
            
        try:
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
            
            # VWAP calculation
            df['Typical_Price'] = (df['Price High'] + df['Price Low'] + df['Price Close']) / 3
            df['VWAP'] = df['Typical_Price'].copy()

            # Calculate VWAP for each day
            for date in df['Date'].dt.date.unique():
                mask = df['Date'].dt.date == date
                df.loc[mask, 'VWAP'] = (
                    (df.loc[mask, 'Typical_Price'] * df.loc[mask, 'Volume']).cumsum() / 
                    df.loc[mask, 'Volume'].cumsum()
                )
            
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
            
            # Trading Signals
            # RSI Signals
            df['RSI_Signal'] = 'Hold'
            df.loc[df['RSI_14'] < 30, 'RSI_Signal'] = 'Buy'
            df.loc[df['RSI_14'] > 70, 'RSI_Signal'] = 'Sell'
            
            # MACD Signals
            df['MACD_Signal'] = 'Hold'
            df.loc[(df['MACD_12_26_9'] > df['MACDs_12_26_9']) & 
                  (df['MACD_12_26_9'].shift(1) <= df['MACDs_12_26_9'].shift(1)), 'MACD_Signal'] = 'Buy'
            df.loc[(df['MACD_12_26_9'] < df['MACDs_12_26_9']) & 
                  (df['MACD_12_26_9'].shift(1) >= df['MACDs_12_26_9'].shift(1)), 'MACD_Signal'] = 'Sell'
            
            # Combined Signal
            df['Combined_Signal'] = 'Hold'
            df.loc[(df['RSI_Signal'] == 'Buy') & (df['MACD_Signal'] == 'Buy'), 'Combined_Signal'] = 'Buy'
            df.loc[(df['RSI_Signal'] == 'Sell') & (df['MACD_Signal'] == 'Sell'), 'Combined_Signal'] = 'Sell'

            return df
            
        except Exception as e:
            st.error(f"Error calculating technical indicators: {str(e)}")
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
            'Stochastic': 1,
            'OBV': 1,  # OBV in separate panel
            'ATR': 1,  # ATR in separate panel
            'Chaikin Volatility': 1  # Chaikin Vol in separate panel
        }
        
        for indicator in indicators:
            if indicator in indicator_panels:
                num_rows += indicator_panels[indicator]
        
        return num_rows
    
    def _calculate_row_heights(self, num_rows, show_volume):
        """Calculate height ratios for subplots"""
        row_heights = [0.5]  # Main chart
        if show_volume:
            row_heights.append(0.2)
        row_heights.extend([0.3] * (num_rows - len(row_heights)))
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
        # RSI Panel
        if 'RSI' in indicators:
            current_row += 1
            fig.add_trace(
                go.Scatter(
                    x=self.data.index,
                    y=self.data['RSI_14'],
                    name='RSI',
                    line=dict(color='purple')
                ),
                row=current_row, col=1
            )
            # Add RSI overbought/oversold lines
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=current_row)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=current_row)
            # Add RSI title
            fig.update_yaxes(title_text="RSI", row=current_row, col=1)

        # MACD Panel
        if 'MACD' in indicators:
            current_row += 1
            # MACD Line
            fig.add_trace(
                go.Scatter(
                    x=self.data.index,
                    y=self.data['MACD_12_26_9'],
                    name='MACD',
                    line=dict(color='blue')
                ),
                row=current_row, col=1
            )
            # Signal Line
            fig.add_trace(
                go.Scatter(
                    x=self.data.index,
                    y=self.data['MACDs_12_26_9'],
                    name='Signal',
                    line=dict(color='orange')
                ),
                row=current_row, col=1
            )
            # MACD Histogram
            fig.add_trace(
                go.Bar(
                    x=self.data.index,
                    y=self.data['MACDh_12_26_9'],
                    name='MACD Hist',
                    marker_color='gray'
                ),
                row=current_row, col=1
            )
            # Add MACD title
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
    
    # Initialize data
    data = None
    
    # Initialize StockData
    stock_data = StockData(FOLDER_PATH)
    
    if not os.path.exists(FOLDER_PATH):
        st.error("Path does not exist!")
        return
    
    available_symbols = stock_data.get_available_symbols()
    if not available_symbols:
        st.error("No CSV files found!")
        return
    
    # Sidebar options
    st.sidebar.header('Chart Options')
    ticker = st.sidebar.selectbox('Select Stock Symbol', available_symbols)
    time_period = st.sidebar.selectbox('Time Period', ['Custom', '1d', '1wk', '1mo', '1y', 'max'])
    
    if time_period == 'Custom':
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_date = st.date_input('Start Date')
        with col2:
            end_date = st.date_input('End Date')
    
    chart_type = st.sidebar.selectbox('Chart Type', ['Candlestick', 'Line'])
    show_volume = st.sidebar.checkbox('Show Volume', value=True)
    
    # Technical Analysis Options
    with st.sidebar.expander("Technical Analysis Options", expanded=False):
        show_signals = st.checkbox('Show Trading Signals', value=False)
        show_support_resistance = st.checkbox('Show Support/Resistance', value=False)
    
    # Indicator selection
    st.sidebar.subheader('Technical Indicators')
    overlay_indicators = st.sidebar.multiselect(
        'Price Chart Overlay',
        ['SMA 20', 'EMA 20', 'Bollinger Bands', 'VWAP']
    )
    separate_indicators = st.sidebar.multiselect(
        'Separate Panels',
        ['RSI', 'MACD', 'OBV', 'ATR']
    )
    
    indicators = overlay_indicators + separate_indicators
    
    if st.sidebar.button('Update Chart'):
        try:
            # Load and process data
            with st.spinner('Loading data...'):
                data = stock_data.load_stock_data(ticker)
                if data is None:
                    return
                
                # Process data based on time period
                if time_period == 'Custom':
                    data = stock_data.process_data(data, period='Custom', 
                                                 start_date=start_date, 
                                                 end_date=end_date)
                else:
                    data = stock_data.process_data(data, period=time_period)
                
                # Calculate technical indicators
                data = TechnicalAnalysis.calculate_indicators(data)
                
                if data is not None:
                    # Create chart
                    chart_builder = ChartBuilder(data, ticker)
                    fig = chart_builder.create_chart(
                        chart_type=chart_type,
                        show_volume=show_volume,
                        indicators=indicators,
                        show_resistance_support=show_support_resistance,
                        show_signals=show_signals
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display trading signals analysis if enabled
                    if show_signals:
                        st.subheader('Trading Signals Analysis')
                        try:
                            signals_df = data[['Date', 'Price Close', 'RSI_14', 'MACD_12_26_9', 
                                             'RSI_Signal', 'MACD_Signal', 'Combined_Signal']]
                            signals_df = signals_df[signals_df['Combined_Signal'] != 'Hold'].copy()
                            
                            if not signals_df.empty:
                                # Format the data for display
                                signals_df['Date'] = signals_df['Date'].dt.strftime('%Y-%m-%d')
                                signals_df['Price Close'] = signals_df['Price Close'].round(2)
                                signals_df['RSI_14'] = signals_df['RSI_14'].round(2)
                                signals_df['MACD_12_26_9'] = signals_df['MACD_12_26_9'].round(4)
                                
                                # Display signal statistics
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    buy_signals = len(signals_df[signals_df['Combined_Signal'] == 'Buy'])
                                    st.metric("Buy Signals", buy_signals)
                                with col2:
                                    sell_signals = len(signals_df[signals_df['Combined_Signal'] == 'Sell'])
                                    st.metric("Sell Signals", sell_signals)
                                with col3:
                                    total_signals = len(signals_df)
                                    st.metric("Total Signals", total_signals)
                                
                                # Display signals table with styling
                                st.dataframe(
                                    signals_df.style.apply(lambda x: ['background-color: #c6efcd' if v == 'Buy' 
                                                                    else 'background-color: #ffc7ce' if v == 'Sell'
                                                                    else '' for v in x],
                                                         subset=['Combined_Signal']),
                                    use_container_width=True
                                )
                            else:
                                st.info('No trading signals found in the selected time period.')
                        except Exception as e:
                            st.error(f'Error displaying trading signals: {str(e)}')
                    
                    # Display data table
                    with st.expander("Detailed Data", expanded=False):
                        st.subheader('Price and Indicator Data')
                        cols_to_display = ['Date', 'Price Open', 'Price High', 'Price Low', 'Price Close', 'Volume']
                        
                        # Add selected indicators to display columns
                        indicator_mapping = {
                            'SMA 20': 'SMA_20',
                            'EMA 20': 'EMA_20',
                            'Bollinger Bands': ['BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0'],
                            'RSI': 'RSI_14',
                            'MACD': ['MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9'],
                            'OBV': 'OBV',
                            'VWAP': 'VWAP',
                            'ATR': 'ATR_14'
                        }
                        
                        for indicator in indicators:
                            if indicator in indicator_mapping:
                                if isinstance(indicator_mapping[indicator], list):
                                    cols_to_display.extend(indicator_mapping[indicator])
                                else:
                                    cols_to_display.append(indicator_mapping[indicator])
                        
                        # Format and display the dataframe
                        display_df = data[cols_to_display].copy()
                        display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
                        st.dataframe(display_df, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
            st.exception(e)  # This will display the full traceback

if __name__ == "__main__":
    main()