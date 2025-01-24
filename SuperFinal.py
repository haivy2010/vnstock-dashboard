import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import pandas_ta as ta
import numpy as np
from fpdf import FPDF
from datetime import datetime
import os
import io
import seaborn as sns
import matplotlib.pyplot as plt
# Constants
FOLDER_PATH = r'D:\dai hoc\Data\processed\valid_data'

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
            if period == 'D':
                start_date = end_date - pd.Timedelta(days=1)
            elif period == 'W':
                start_date = end_date - pd.Timedelta(weeks=1)
            elif period == 'M':
                start_date = end_date - pd.Timedelta(days=30)
            elif period == 'Y':
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
class ReportGenerator:
    def __init__(self, stock_data, folder_path):
        self.stock_data = stock_data
        self.folder_path = folder_path
        
    def generate_performance_report(self, period='1mo', start_date=None, end_date=None, output_path='stock_report.pdf'):
        """Generate a comprehensive PDF report of stock performance"""
        temp_files = []  # Keep track of temporary files
        
        try:
            pdf = FPDF()
            pdf.add_page()
            
            # Add report header
            self._add_header(pdf, period, start_date, end_date)
            
            # Get performance data for all stocks
            performance_data = self._calculate_all_stocks_performance(period, start_date, end_date)
            
            if not performance_data:
                raise ValueError("No performance data available for the selected period")
            
            # Top gainers and losers
            gainers_losers = self._analyze_gainers_losers(performance_data)
            self._add_gainers_losers_section(pdf, gainers_losers)
            
            
            # Save the PDF
            pdf.output(output_path)
            return output_path
            
        except Exception as e:
            # Clean up any temporary files if an error occurs
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            raise e

    def _add_header(self, pdf, period, start_date=None, end_date=None):
        """
        Add report header with title and date information based on data index dates
        
        Args:
            pdf: FPDF object
            period: str ('W', 'M', 'Y', or 'Custom')
            start_date: datetime.date, optional
            end_date: datetime.date, optional
        """
        from datetime import datetime, timedelta
        
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'Stock Market Performance Report', ln=True, align='C')
        pdf.set_font('Arial', '', 12)
        
        try:
            # Get the available data range
            available_symbols = self.stock_data.get_available_symbols()
            if not available_symbols:
                raise ValueError("No stock data available")
                
            # Get the first stock's data to determine date range
            sample_data = self.stock_data.load_stock_data(available_symbols[0])
            if sample_data is None or sample_data.empty:
                raise ValueError("Could not load sample data")
                
            # Get the maximum date from the data
            max_date = sample_data['Date'].max()
            
            # Calculate start and end dates based on period
            if period == 'Custom':
                if not start_date or not end_date:
                    raise ValueError("Custom period requires both start_date and end_date")
                date_range = f"Period: Custom {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
                
            else:
                end_date = max_date
                
                if period == 'W':
                    start_date = end_date - timedelta(days=7)
                    period_text = "Week"
                elif period == 'M':
                    start_date = end_date - timedelta(days=30)
                    period_text = "Month"
                elif period == 'Y':
                    start_date = end_date - timedelta(days=365)
                    period_text = "Year"
                else:
                    # Default to showing full range
                    start_date = sample_data['Date'].min()
                    period_text = "Full Range"
                    
                date_range = f"Period: {period_text} ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})"
            
            # Add date range to PDF
            pdf.cell(0, 10, date_range, ln=True, align='L')
            pdf.cell(0, 10, f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M")}', ln=True, align='L')
            pdf.ln(10)
            
            # Return the calculated dates for use in other methods
            return start_date, end_date
            
        except Exception as e:
            pdf.cell(0, 10, "Error calculating date range", ln=True, align='L')
            pdf.cell(0, 10, f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M")}', ln=True, align='L')
            pdf.ln(10)
            raise ValueError(f"Error in header generation: {str(e)}")


    def _calculate_all_stocks_performance(self, period, start_date=None, end_date=None):
        """Calculate performance metrics for all stocks"""
        performance_data = []
        errors = []
        
        available_symbols = self.stock_data.get_available_symbols()
        if not available_symbols:
            raise ValueError("No stock symbols found in the data folder")
        
        for symbol in available_symbols:
            try:
                # Load stock data
                data = self.stock_data.load_stock_data(symbol)
                if data is None:
                    errors.append(f"Could not load data for {symbol}")
                    continue
                
                # Process data for the period
                processed_data = self.stock_data.process_data(data, period, start_date, end_date)
                if processed_data is None or processed_data.empty:
                    errors.append(f"No data available for {symbol} in the selected period")
                    continue
                
                # Calculate performance metrics
                try:
                    initial_price = processed_data['Price Close'].iloc[0]
                    final_price = processed_data['Price Close'].iloc[-1]
                    
                    if pd.isna(initial_price) or pd.isna(final_price) or initial_price == 0:
                        errors.append(f"Invalid price data for {symbol}")
                        continue
                    
                    price_change = ((final_price - initial_price) / initial_price) * 100
                    
                    # Calculate average volume safely
                    volumes = processed_data['Volume'].dropna()
                    avg_volume = volumes.mean() if not volumes.empty else 0
                    
                    # Calculate volatility safely
                    returns = processed_data['Price Close'].pct_change().dropna()
                    volatility = returns.std() * 100 if not returns.empty else 0
                    
                    performance_data.append({
                        'symbol': symbol,
                        'initial_price': initial_price,
                        'final_price': final_price,
                        'price_change': price_change,
                        'avg_volume': avg_volume,
                        'volatility': volatility,
                        'data': processed_data
                    })
                    
                except Exception as e:
                    errors.append(f"Error calculating metrics for {symbol}: {str(e)}")
                    
            except Exception as e:
                errors.append(f"Error processing {symbol}: {str(e)}")
        
        # Check if we have any valid data
        if not performance_data:
            error_summary = "\n".join(errors[:5])  # Show first 5 errors
            if len(errors) > 5:
                error_summary += f"\n... and {len(errors) - 5} more errors"
            raise ValueError(f"Could not calculate performance metrics. Errors:\n{error_summary}")
        
        return performance_data

    def _analyze_gainers_losers(self, performance_data):
        """Analyze top gainers and losers"""
        sorted_data = sorted(performance_data, key=lambda x: x['price_change'], reverse=True)
        return {
            'top_gainers': sorted_data[:10],
            'top_losers': sorted_data[-10:]
        }

    def _analyze_volume_trends(self, performance_data):
        """Analyze volume trends"""
        sorted_by_volume = sorted(performance_data, key=lambda x: x['avg_volume'], reverse=True)
        return {
            'highest_volume': sorted_by_volume[:10],
            'lowest_volume': sorted_by_volume[-10:]
        }

    def _calculate_volatility(self, performance_data):
        """Calculate volatility metrics"""
        sorted_by_volatility = sorted(performance_data, key=lambda x: x['volatility'], reverse=True)
        return {
            'most_volatile': sorted_by_volatility[:10],
            'least_volatile': sorted_by_volatility[-10:]
        }

    def _add_gainers_losers_section(self, pdf, gainers_losers):
        """Add top gainers and losers section to the report"""
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'Top Performers', ln=True)
        pdf.ln(5)
        
        # Add top gainers table
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Top Gainers', ln=True)
        self._add_table(pdf, gainers_losers['top_gainers'], 
                       ['Symbol', 'Price Change (%)', 'Final Price'],
                       [30, 40, 40])
        
        pdf.ln(10)
        
        # Add top losers table
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Top Decliners', ln=True)
        self._add_table(pdf, gainers_losers['top_losers'], 
                       ['Symbol', 'Price Change (%)', 'Final Price'],
                       [30, 40, 40])

    def _add_table(self, pdf, data, headers, widths):
        """Helper function to add formatted tables to the report"""
        pdf.set_font('Arial', 'B', 10)
        
        # Add headers
        for header, width in zip(headers, widths):
            pdf.cell(width, 10, header, 1)
        pdf.ln()
        
        # Add data rows
        pdf.set_font('Arial', '', 10)
        for item in data:
            pdf.cell(widths[0], 10, str(item['symbol']), 1)
            pdf.cell(widths[1], 10, f"{item['price_change']:.2f}", 1)
            pdf.cell(widths[2], 10, f"{item['final_price']:.2f}", 1)
            pdf.ln()

def main():
    st.set_page_config(layout="wide", page_title="Stock Dashboard")
    st.title('Stock Dashboard - Vietnam Market')
    
    # Initialize data and StockData
    data = None
    stock_data = StockData(FOLDER_PATH)
    
    def check_data_availability():
        if not os.path.exists(FOLDER_PATH):
            st.error("Path does not exist!")
            return False
        
        available_symbols = stock_data.get_available_symbols()
        if not available_symbols:
            st.error("No CSV files found!")
            return False
        return available_symbols

    def create_sidebar_options(available_symbols):
        # Sidebar - Basic Options
        st.sidebar.header('Chart Options')
        
        # Stock selection
        ticker = st.sidebar.selectbox('Select Stock Symbol', available_symbols)
        
        # Time period selection
        time_period = st.sidebar.selectbox('Time Period', ['Custom', 'D', 'W', 'M', 'Y', 'max'])
        
        # Custom date range (2000-2022)
        start_date = None
        end_date = None
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
        
        return ticker, time_period, start_date, end_date, chart_type, show_volume

    def create_technical_analysis_options():
        with st.sidebar.expander("Technical Analysis Options", expanded=False):
            # Signal type selection
            signal_type = st.radio(
                "Trading Signal Type",
                ["RSI", "MACD", "Combined"],
                help="Choose which trading signals to display"
            )
            
            # RSI Thresholds
            rsi_lower = 30
            rsi_upper = 70
            if signal_type in ["RSI", "Combined"]:
                col1, col2 = st.sidebar.columns(2)
                with col1:
                    rsi_lower = st.number_input("RSI Oversold", value=30, min_value=0, max_value=100)
                with col2:
                    rsi_upper = st.number_input("RSI Overbought", value=70, min_value=0, max_value=100)
            
            # Display options
            show_signals = st.checkbox('Show Trading Signals', value=False)
            show_support_resistance = st.checkbox('Show Support/Resistance', value=False)
        
        return signal_type, rsi_lower, rsi_upper, show_signals, show_support_resistance

    def create_indicator_selection():
        st.sidebar.subheader('Technical Indicators')
        overlay_indicators = st.sidebar.multiselect(
            'Price Chart Overlay',
            ['SMA 20', 'EMA 20', 'Bollinger Bands']
        )
        separate_indicators = st.sidebar.multiselect(
            'Separate Panels',
            ['RSI', 'MACD', 'MFI', 'OBV', 'ATR']
        )
        return overlay_indicators + separate_indicators

    def create_report_options():
        with st.sidebar.expander("Generate Report", expanded=False):
            report_period = st.selectbox(
                'Report Time Period',
                ['W', 'M', 'Y', 'Custom'],
                key='report_period'
            )
            
            report_start_date = None
            report_end_date = None
            if report_period == 'Custom':
                report_start_date = st.date_input(
                    'Report Start Date',
                    min_value=datetime(2000, 1, 1).date(),
                    max_value=datetime(2022, 12, 31).date(),
                    value=datetime(2000, 1, 1).date(),
                    key='report_start_date'
                )
                report_end_date = st.date_input(
                    'Report End Date',
                    min_value=datetime(2000, 1, 1).date(),
                    max_value=datetime(2022, 12, 31).date(),
                    value=datetime(2022, 12, 31).date(),
                    key='report_end_date'
                )
            
            return report_period, report_start_date, report_end_date

    def generate_report(report_period, report_start_date, report_end_date):
        if st.button('Generate PDF Report'):
            try:
                if report_period == 'Custom' and report_start_date > report_end_date:
                    st.error("Start date must be before end date")
                    return
                    
                with st.spinner('Generating report...'):
                    report_generator = ReportGenerator(stock_data, FOLDER_PATH)
                    
                    try:
                        if report_period == 'Custom':
                            report_path = report_generator.generate_performance_report(
                                period='Custom',
                                start_date=report_start_date,
                                end_date=report_end_date,
                                output_path=f"stock_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
                            )
                        else:
                            report_path = report_generator.generate_performance_report(
                                period=report_period,
                                output_path=f"stock_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
                            )
                        
                        if not os.path.exists(report_path):
                            raise FileNotFoundError("Failed to generate report file")
                        
                        with open(report_path, "rb") as pdf_file:
                            PDFbyte = pdf_file.read()
                        
                        st.download_button(
                            label="Download PDF Report",
                            data=PDFbyte,
                            file_name=os.path.basename(report_path),
                            mime="application/pdf"
                        )
                        
                        st.success('Report generated successfully!')
                        
                    finally:
                        if 'report_path' in locals() and os.path.exists(report_path):
                            os.remove(report_path)
                
            except Exception as e:
                error_message = str(e)
                if "no performance data available" in error_message.lower():
                    st.error("No data available for the selected time period. Please try a different time range.")
                elif "date" in error_message.lower():
                    st.error("Invalid date range selected. Please check your dates.")
                else:
                    st.error(f"Error generating report: {error_message}")
                st.exception(e)

    def update_chart(ticker, time_period, start_date, end_date, chart_type, show_volume, 
                    signal_type, rsi_thresholds, indicators, show_support_resistance, show_signals):
        try:
            with st.spinner('Loading data...'):
                data = stock_data.load_stock_data(ticker)
                if data is None:
                    st.error("Failed to load data!")
                    return
                
                if time_period == 'Custom':
                    data = stock_data.process_data(data, period='Custom', 
                                                 start_date=start_date, 
                                                 end_date=end_date)
                else:
                    data = stock_data.process_data(data, period=time_period)
                
                data = TechnicalAnalysis.calculate_indicators(data, rsi_thresholds=rsi_thresholds)
                
                if data is not None:
                    if signal_type != "Combined":
                        data['Combined_Signal'] = data[f'{signal_type}_Signal']
                    
                    chart_builder = ChartBuilder(data, ticker)
                    fig = chart_builder.create_chart(
                        chart_type=chart_type,
                        show_volume=show_volume,
                        indicators=indicators,
                        show_resistance_support=show_support_resistance,
                        show_signals=show_signals
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    display_trading_signals(data, signal_type, show_signals)
                    display_detailed_data(data, indicators)
        
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
            st.exception(e)

    def display_trading_signals(data, signal_type, show_signals):
        if not show_signals:
            return
            
        st.subheader(f'{signal_type} Trading Signals Analysis')
        try:
            base_columns = ['Date', 'Price Close', 'RSI_14', 'MACD_12_26_9']
            if signal_type == "Combined":
                signal_columns = base_columns + ['RSI_Signal', 'MACD_Signal', 'Combined_Signal']
            else:
                signal_columns = base_columns + [f'{signal_type}_Signal']
            
            signal_col = 'Combined_Signal' if signal_type == "Combined" else f'{signal_type}_Signal'
            signals_df = data[signal_columns]
            signals_df = signals_df[signals_df[signal_col] != 'Hold'].copy()
            
            if not signals_df.empty:
                signals_df['Date'] = signals_df['Date'].dt.strftime('%Y-%m-%d')
                signals_df['Price Close'] = signals_df['Price Close'].round(2)
                signals_df['RSI_14'] = signals_df['RSI_14'].round(2)
                signals_df['MACD_12_26_9'] = signals_df['MACD_12_26_9'].round(4)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Buy Signals", len(signals_df[signals_df[signal_col] == 'Buy']))
                with col2:
                    st.metric("Sell Signals", len(signals_df[signals_df[signal_col] == 'Sell']))
                with col3:
                    st.metric("Total Signals", len(signals_df))
                
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

    def display_detailed_data(data, indicators):
        with st.expander("Detailed Data", expanded=False):
            st.subheader('Price and Indicator Data')
            
            cols_to_display = ['Date', 'Price Open', 'Price High', 'Price Low', 'Price Close', 'Volume']
            
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
            
            display_df = data[cols_to_display].copy()
            display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
            st.dataframe(display_df, use_container_width=True)

    # Main execution flow
    available_symbols = check_data_availability()
    if not available_symbols:
        return

    # Create sidebar options
    ticker, time_period, start_date, end_date, chart_type, show_volume = create_sidebar_options(available_symbols)
    signal_type, rsi_lower, rsi_upper, show_signals, show_support_resistance = create_technical_analysis_options()
    indicators = create_indicator_selection()
    report_period, report_start_date, report_end_date = create_report_options()

    # Handle report generation
    generate_report(report_period, report_start_date, report_end_date)

    # Update chart when requested
    if st.sidebar.button('Update Chart'):
        rsi_thresholds = (
            rsi_lower if signal_type in ["RSI", "Combined"] else 30,
            rsi_upper if signal_type in ["RSI", "Combined"] else 70
        )
        update_chart(
            ticker, time_period, start_date, end_date, chart_type, show_volume,
            signal_type, rsi_thresholds, indicators, show_support_resistance, show_signals
        )

if __name__ == "__main__":
    main()