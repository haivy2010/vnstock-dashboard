import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime
import os

# Đặt đường dẫn cố định đến folder chứa dữ liệu
FOLDER_PATH = 'https://github.com/haivy2010/vnstock-dashboard/tree/main/data'

# SECTION 1: Utility Functions
def get_available_symbols(folder_path):
    """Lấy danh sách các mã cổ phiếu từ tên file CSV"""
    try:
        return [f.replace('.csv', '') for f in os.listdir(folder_path) if f.endswith('.csv')]
    except Exception as e:
        print(f"Lỗi khi đọc folder: {str(e)}")
        return []

def load_stock_data(file_path):
    """Đọc và xử lý dữ liệu từ file CSV"""
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Sắp xếp dữ liệu theo thời gian
    df = df.sort_values('Date')
    
    # Đảm bảo không có dữ liệu trùng lặp
    df = df.drop_duplicates(subset=['Date'], keep='first')
    
    return df

def fill_missing_dates(data):
    """Điền các ngày bị thiếu vào dữ liệu."""
    # Tạo dải thời gian đầy đủ từ ngày nhỏ nhất đến ngày lớn nhất
    full_date_range = pd.date_range(start=data['Date'].min(), end=data['Date'].max())
    
    # Chuyển cột Date thành index
    data = data.set_index('Date')
    
    # Reindex để thêm các ngày còn thiếu
    data = data.reindex(full_date_range)
    
    # Chuyển index về cột Date
    data = data.reset_index()
    data.rename(columns={'index': 'Date'}, inplace=True)
    
    # Điền giá trị mặc định cho các cột thiếu dữ liệu
    data['Price Open'] = data['Price Open'].fillna(method='ffill')
    data['Price High'] = data['Price High'].fillna(method='ffill')
    data['Price Low'] = data['Price Low'].fillna(method='ffill')
    data['Price Close'] = data['Price Close'].fillna(method='ffill')
    data['Volume'] = data['Volume'].fillna(0)
    
    return data

def process_data(data, period='1y'):
    """Xử lý và lọc dữ liệu theo khoảng thời gian để đảm bảo tính liên tục."""
    # Tạo bản sao của dữ liệu và chuyển đổi cột Date
    df = data.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Điền các ngày bị thiếu
    df = fill_missing_dates(df)
    
    # Lọc dữ liệu theo khoảng thời gian
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
    
    df = df[df['Date'] >= start_date].copy()
    df.reset_index(drop=True, inplace=True)
    
    return df

# SECTION 2: Technical Indicators
def calculate_indicators(data):
    """Tính toán các chỉ báo kỹ thuật"""
    df = data.copy()
    
    # SMA và EMA
    df['SMA_20'] = df['Price Close'].rolling(window=20).mean()
    df['EMA_20'] = df['Price Close'].ewm(span=20, adjust=False).mean()
    
    # RSI
    delta = df['Price Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Price Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Price Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']
    
    # Volatility (ATR)
    high_low = df['Price High'] - df['Price Low']
    high_close = np.abs(df['Price High'] - df['Price Close'].shift())
    low_close = np.abs(df['Price Low'] - df['Price Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    
    return df

def create_chart(data, ticker, chart_type, show_volume=True, indicators=None):
    """Tạo biểu đồ với các chỉ báo"""
    if indicators is None:
        indicators = []
        
    # Tạo subplot layout
    num_technical_indicators = sum(1 for ind in ['RSI', 'MACD', 'Volatility'] if ind in indicators)
    rows = 2 + num_technical_indicators
    
    row_heights = [0.5, 0.2]
    row_heights.extend([0.3] * num_technical_indicators)
    
    fig = make_subplots(
        rows=rows, 
        cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights
    )

    # Thêm candlestick/line chart
    if chart_type == 'Candlestick':
        fig.add_trace(
            go.Candlestick(
                x=data['Date'],
                open=data['Price Open'],
                high=data['Price High'],
                low=data['Price Low'],
                close=data['Price Close'],
                name=ticker,
                increasing_line_color='red',
                decreasing_line_color='green'
            ),
            row=1, col=1
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=data['Date'],
                y=data['Price Close'],
                name=ticker,
                line=dict(width=2)
            ),
            row=1, col=1
        )

    # Thêm SMA/EMA
    if 'SMA 20' in indicators:
        fig.add_trace(
            go.Scatter(
                x=data['Date'],
                y=data['SMA_20'],
                name='SMA 20',
                line=dict(color='orange', width=1.5)
            ),
            row=1, col=1
        )
    
    if 'EMA 20' in indicators:
        fig.add_trace(
            go.Scatter(
                x=data['Date'],
                y=data['EMA_20'],
                name='EMA 20',
                line=dict(color='blue', width=1.5)
            ),
            row=1, col=1
        )

    # Thêm Volume
    if show_volume:
        colors = ['red' if row['Price Close'] >= row['Price Open'] else 'green' 
                 for index, row in data.iterrows()]
        fig.add_trace(
            go.Bar(
                x=data['Date'],
                y=data['Volume'],
                name='Volume',
                marker_color=colors,
                marker_line_width=0,
                opacity=0.8
            ),
            row=2, col=1
        )

    # Thêm các chỉ báo kỹ thuật
    current_row = 3
    if 'RSI' in indicators:
        fig.add_trace(
            go.Scatter(
                x=data['Date'],
                y=data['RSI'],
                name='RSI',
                line=dict(color='purple')
            ),
            row=current_row, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=current_row, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=current_row, col=1)
        current_row += 1

    if 'MACD' in indicators:
        fig.add_trace(
            go.Scatter(
                x=data['Date'],
                y=data['MACD'],
                name='MACD',
                line=dict(color='blue')
            ),
            row=current_row, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=data['Date'],
                y=data['Signal_Line'],
                name='Signal Line',
                line=dict(color='orange')
            ),
            row=current_row, col=1
        )
        fig.add_trace(
            go.Bar(
                x=data['Date'],
                y=data['MACD_Histogram'],
                name='MACD Histogram',
                marker_color=['red' if x < 0 else 'green' for x in data['MACD_Histogram']]
            ),
            row=current_row, col=1
        )
        current_row += 1

    if 'Volatility' in indicators:
        fig.add_trace(
            go.Scatter(
                x=data['Date'],
                y=data['ATR'],
                name='ATR (14)',
                line=dict(color='brown')
            ),
            row=current_row, col=1
        )

    # Cập nhật layout
    fig.update_layout(
        title=f'{ticker} - Biểu đồ giá',
        xaxis_title='Thời gian',
        yaxis_title='Giá (VND)',
        height=800,
        showlegend=True,
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis_rangeslider_visible=False
    )

    # Cập nhật style cho các trục
    for i in range(1, rows + 1):
        fig.update_yaxes(
            gridcolor='lightgrey',
            gridwidth=0.5,
            zerolinecolor='lightgrey',
            row=i, col=1
        )
        fig.update_xaxes(
            type='date',
            gridcolor='lightgrey',
            gridwidth=0.5,
            row=i, col=1,
            tickformat='%Y-%m-%d',
            nticks=20,
            rangeslider_visible=False
        )

    return fig

def main():
    st.set_page_config(layout="wide")
    st.title('Stock Dashboard - Vietnam Market')

    if not os.path.exists(FOLDER_PATH):
        st.error("Đường dẫn không tồn tại!")
        return

    available_symbols = get_available_symbols(FOLDER_PATH)
    if not available_symbols:
        st.error("Không tìm thấy file CSV nào!")
        return

    # Sidebar
    st.sidebar.header('Tùy chọn biểu đồ')
    ticker = st.sidebar.selectbox('Chọn mã cổ phiếu', available_symbols)
    time_period = st.sidebar.selectbox('Khoảng thời gian', ['1d', '1wk', '1mo', '1y', 'max'])
    chart_type = st.sidebar.selectbox('Loại biểu đồ', ['Candlestick', 'Line'])
    show_volume = st.sidebar.checkbox('Hiển thị Volume', value=True)
    indicators = st.sidebar.multiselect(
        'Chỉ báo kỹ thuật',
        ['SMA 20', 'EMA 20', 'RSI', 'MACD', 'Volatility']
    )

    if st.sidebar.button('Cập nhật'):
        try:
            file_path = os.path.join(FOLDER_PATH, f"{ticker}.csv")
            data = load_stock_data(file_path)
            data = process_data(data, time_period)
            data = calculate_indicators(data)
            
            fig = create_chart(data, ticker, chart_type, show_volume, indicators)
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader('Dữ liệu chi tiết')
            cols_to_display = ['Date', 'Price Open', 'Price High', 'Price Low', 'Price Close', 'Volume']
            if indicators:
                if 'RSI' in indicators:
                    cols_to_display.append('RSI')
                if 'MACD' in indicators:
                    cols_to_display.extend(['MACD', 'Signal_Line'])
                if 'Volatility' in indicators:
                    cols_to_display.append('ATR')
            
            st.dataframe(data[cols_to_display])
            
        except Exception as e:
            st.error(f"Lỗi khi xử lý dữ liệu: {str(e)}")

if __name__ == "__main__":
    main()
