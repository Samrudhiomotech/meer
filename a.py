"""
IoT Environmental Monitoring Dashboard
Predictive Analytics & Spatial Visualization
Research Project: OMO/RE/323
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy.interpolate import griddata
import tensorflow as tf
from tensorflow.keras.models import load_model

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="IoT Environmental Monitoring System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional dashboard
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 500;
        color: #1e3c72;
        text-align: center;
        padding: 1.5rem 0;
        border-bottom: 3px solid #e9ecef;
        margin-bottom: 2rem;
    }
    
    .section-header {
        font-size: 1.4rem;
        font-weight: 500;
        color: #1e3c72;
        padding: 1rem 0;
        margin: 1.5rem 0 1rem 0;
        border-left: 5px solid #667eea;
        padding-left: 1rem;
    }
    
    .critical-warning {
        background: #fee;
        color: #b71c1c;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #c62828;
    }
    
    .high-warning {
        background: #fff3e0;
        color: #bf360c;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #e65100;
    }
    
    .moderate-warning {
        background: #fff9c4;
        color: #b26a00;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #ffa000;
    }
    
    .normal-status {
        background: #e8f5e8;
        color: #1b5e20;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #2e7d32;
    }
    
    .info-box {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e9ecef;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: white;
        padding: 1.2rem;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin: 0.5rem 0;
        border: 1px solid #e9ecef;
    }
    
    .plot-container {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin: 1rem 0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: white;
        padding: 0.5rem;
        border-radius: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #64748b;
        font-weight: 500;
        border-radius: 8px;
        padding: 0.5rem 2rem;
    }
    
    .stTabs [aria-selected="true"] {
        background: #667eea;
        color: white;
    }
    
    .divider {
        height: 2px;
        background: linear-gradient(to right, transparent, #667eea, transparent);
        margin: 2rem 0;
    }
    
    .empty-state {
        text-align: center;
        padding: 3rem;
        background: white;
        border-radius: 10px;
        border: 2px dashed #e9ecef;
        color: #64748b;
    }
    
    .research-info {
        background: #f0f4f8;
        color: #1e3c72;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


class EnvironmentalMonitoringSystem:
    """
    IoT Environmental Monitoring System
    Research Project: OMO/RE/323
    """
    
    def __init__(self):
        self.data_file = 'sensor_data.csv'
        self.lstm_model_file = 'lstm_model.keras'
        self.lstm_scaler_file = 'lstm_scaler.pkl'
        
        # Parameter mapping
        self.parameter_mapping = {
            'DS18B20': 'DS18B20_Temperature_C',
            'DHTTemp': 'Temperature_C',
            'Humidity': 'Humidity_RH',
            'COppm': 'CO_ppm',
            'eCO2': 'eCO2_ppm',
            'TVOC': 'VOC_ppb',
            'PM2.5 concentration (ug/m^3)': 'Dust_PM_ugm3',
            'AQI': 'aqi_value'
        }
        
        # Display names
        self.display_names = {
            'DS18B20_Temperature_C': 'DS18B20 Temperature',
            'Temperature_C': 'DHT22 Temperature',
            'Humidity_RH': 'Humidity',
            'CO_ppm': 'Carbon Monoxide',
            'eCO2_ppm': 'eCO2',
            'VOC_ppb': 'TVOC',
            'Dust_PM_ugm3': 'PM2.5',
            'aqi_value': 'AQI'
        }
        
        # Thresholds for warnings
        self.thresholds = {
            'Dust_PM_ugm3': {
                'good': 25, 'moderate': 50, 'high': 75, 'critical': 100,
                'unit': 'μg/m³', 'name': 'PM2.5'
            },
            'VOC_ppb': {
                'good': 150, 'moderate': 250, 'high': 375, 'critical': 500,
                'unit': 'ppb', 'name': 'TVOC'
            },
            'eCO2_ppm': {
                'good': 600, 'moderate': 800, 'high': 1000, 'critical': 1200,
                'unit': 'ppm', 'name': 'eCO2'
            },
            'CO_ppm': {
                'good': 3, 'moderate': 5, 'high': 7.5, 'critical': 10,
                'unit': 'ppm', 'name': 'Carbon Monoxide'
            },
            'Temperature_C': {
                'good_low': 15, 'good_high': 25,
                'moderate_low': 10, 'moderate_high': 30,
                'high_low': 5, 'high_high': 35,
                'unit': '°C', 'name': 'Temperature'
            },
            'Humidity_RH': {
                'good_low': 40, 'good_high': 60,
                'moderate_low': 30, 'moderate_high': 70,
                'high_low': 20, 'high_high': 80,
                'unit': '%', 'name': 'Humidity'
            }
        }
        
        self.feature_columns = [
            'Dust_PM_ugm3', 'VOC_ppb', 'eCO2_ppm',
            'Temperature_C', 'Humidity_RH', 'CO_ppm',
            'DS18B20_Temperature_C'
        ]
        
        self.expected_columns = [
            'timestamp', 'DS18B20_Temperature_C', 'Temperature_C', 'Humidity_RH',
            'Dust_PM_ugm3', 'CO_ppm', 'eCO2_ppm', 'VOC_ppb',
            'aqi_value', 'aqi_category'
        ]
        
        self.lstm_model = None
        self.lstm_scaler = None
        self.locations = {}  # Store location data for heatmaps
        
        self.initialize_system()
    
    def initialize_system(self):
        """Initialize system components"""
        self.initialize_data_file()
        self.load_models()
    
    def initialize_data_file(self):
        """Initialize CSV data file"""
        if not os.path.exists(self.data_file):
            df = pd.DataFrame(columns=self.expected_columns)
            df.to_csv(self.data_file, index=False)
    
    def load_models(self):
        """Load LSTM model from .keras file"""
        if os.path.exists(self.lstm_model_file):
            try:
                self.lstm_model = load_model(self.lstm_model_file)
                if os.path.exists(self.lstm_scaler_file):
                    with open(self.lstm_scaler_file, 'rb') as f:
                        self.lstm_scaler = pickle.load(f)
                else:
                    self.create_lstm_scaler()
            except Exception as e:
                st.warning(f"Could not load LSTM model: {e}")
                self.lstm_model = None
        else:
            st.warning("LSTM model file not found. Please ensure lstm_model.keras is in the correct directory.")
    
    def create_lstm_scaler(self):
        """Create scaler for LSTM if not exists"""
        self.lstm_scaler = MinMaxScaler()
        # Fit with sample data
        sample_data = np.zeros((100, len(self.feature_columns)))
        self.lstm_scaler.fit(sample_data)
        with open(self.lstm_scaler_file, 'wb') as f:
            pickle.dump(self.lstm_scaler, f)
    
    def set_location(self, location_name, latitude, longitude):
        """Set location for heatmap visualization"""
        self.locations[location_name] = {'lat': latitude, 'lon': longitude}
    
    def get_warning_level(self, parameter, value):
        """Get warning level for a parameter"""
        if parameter not in self.thresholds:
            return 'unknown', 0
        
        thresholds = self.thresholds[parameter]
        
        if parameter in ['Temperature_C', 'Humidity_RH']:
            if value < thresholds.get('high_low', 0) or value > thresholds.get('high_high', 100):
                return 'critical', 4
            elif value < thresholds.get('moderate_low', 0) or value > thresholds.get('moderate_high', 100):
                return 'high', 3
            elif value < thresholds.get('good_low', 0) or value > thresholds.get('good_high', 100):
                return 'moderate', 2
            else:
                return 'good', 1
        else:
            if value >= thresholds['critical']:
                return 'critical', 4
            elif value >= thresholds['high']:
                return 'high', 3
            elif value >= thresholds['moderate']:
                return 'moderate', 2
            else:
                return 'good', 1
    
    def calculate_aqi(self, sensor_data):
        """Calculate Air Quality Index"""
        if 'aqi_value' in sensor_data and sensor_data['aqi_value'] > 0:
            aqi = sensor_data['aqi_value']
        else:
            dust = sensor_data.get('Dust_PM_ugm3', 0)
            
            def calculate_pollutant_aqi(concentration, breakpoints):
                for (c_low, c_high, aqi_low, aqi_high) in breakpoints:
                    if c_low <= concentration <= c_high:
                        return ((aqi_high - aqi_low) / (c_high - c_low)) * (concentration - c_low) + aqi_low
                return 500
            
            pm_breakpoints = [
                (0, 12.0, 0, 50),
                (12.1, 35.4, 51, 100),
                (35.5, 55.4, 101, 150),
                (55.5, 150.4, 151, 200),
                (150.5, 250.4, 201, 300),
                (250.5, 350.4, 301, 400),
                (350.5, 500.4, 401, 500)
            ]
            
            aqi = calculate_pollutant_aqi(dust, pm_breakpoints)
        
        if aqi <= 50:
            category = "Good"
        elif aqi <= 100:
            category = "Moderate"
        elif aqi <= 150:
            category = "Unhealthy for Sensitive Groups"
        elif aqi <= 200:
            category = "Unhealthy"
        elif aqi <= 300:
            category = "Very Unhealthy"
        else:
            category = "Hazardous"
        
        return {'value': aqi, 'category': category}
    
    def predict_72_hours(self, df):
        """Predict next 72 hours using LSTM model"""
        if self.lstm_model is None or len(df) < 10:
            return None, None
        
        try:
            # Get last 10 sequences for prediction
            sequence_length = 10
            if len(df) < sequence_length:
                return None, None
            
            last_sequence = df[self.feature_columns].tail(sequence_length).values
            
            # Scale the sequence
            sequence_scaled = self.lstm_scaler.transform(last_sequence)
            sequence_scaled = sequence_scaled.reshape(1, sequence_length, len(self.feature_columns))
            
            predictions = []
            current_sequence = sequence_scaled.copy()
            
            # Generate 72 hourly predictions
            for _ in range(72):
                # Predict next step
                next_pred = self.lstm_model.predict(current_sequence, verbose=0)
                
                # Inverse scale
                next_pred_original = self.lstm_scaler.inverse_transform(next_pred)
                predictions.append(next_pred_original[0])
                
                # Update sequence for next prediction
                current_sequence = np.roll(current_sequence, -1, axis=1)
                current_sequence[0, -1, :] = next_pred[0]
            
            predictions = np.array(predictions)
            predictions = np.maximum(predictions, 0)  # Ensure non-negative values
            
            return predictions, None
            
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return None, None
    
    def parse_timestamp(self, date_str, time_str, millis_str=None):
        """Parse timestamp from Date, Time, and Millis columns"""
        try:
            datetime_str = f"{date_str} {time_str}"
            timestamp = pd.to_datetime(datetime_str)
            
            if millis_str and pd.notna(millis_str):
                try:
                    millis = float(millis_str)
                    timestamp = timestamp + timedelta(milliseconds=millis)
                except:
                    pass
            
            return timestamp
        except:
            return datetime.now()
    
    def save_sensor_data(self, sensor_data, aqi_info, timestamp=None, location=None):
        """Save sensor data to CSV"""
        try:
            if timestamp is None:
                timestamp = datetime.now()
            
            if isinstance(timestamp, str):
                try:
                    timestamp = pd.to_datetime(timestamp)
                except:
                    timestamp = datetime.now()
            
            row = {
                'timestamp': timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                'DS18B20_Temperature_C': sensor_data.get('DS18B20_Temperature_C', 0),
                'Temperature_C': sensor_data.get('Temperature_C', 0),
                'Humidity_RH': sensor_data.get('Humidity_RH', 0),
                'Dust_PM_ugm3': sensor_data.get('Dust_PM_ugm3', 0),
                'CO_ppm': sensor_data.get('CO_ppm', 0),
                'eCO2_ppm': sensor_data.get('eCO2_ppm', 0),
                'VOC_ppb': sensor_data.get('VOC_ppb', 0),
                'aqi_value': aqi_info.get('value', 0),
                'aqi_category': aqi_info.get('category', 'Good')
            }
            
            # Add location if provided
            if location:
                row['location'] = location
            
            df = pd.DataFrame([row])
            
            if os.path.exists(self.data_file):
                try:
                    existing_df = pd.read_csv(self.data_file, nrows=1)
                    if len(existing_df.columns) == len(row):
                        df.to_csv(self.data_file, mode='a', header=False, index=False)
                    else:
                        df.to_csv(self.data_file, index=False)
                except:
                    df.to_csv(self.data_file, index=False)
            else:
                df.to_csv(self.data_file, index=False)
            
            return True
        except Exception as e:
            print(f"Error saving data: {e}")
            return False
    
    def load_data(self):
        """Load sensor data"""
        try:
            if not os.path.exists(self.data_file):
                return pd.DataFrame()
            
            try:
                df = pd.read_csv(self.data_file)
            except pd.errors.ParserError:
                try:
                    df = pd.read_csv(self.data_file, on_bad_lines='skip')
                except:
                    return pd.DataFrame()
            
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                df = df.dropna(subset=['timestamp'])
                df = df.sort_values('timestamp')
            
            for col in self.feature_columns + ['aqi_value']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df
            
        except:
            return pd.DataFrame()


def display_header():
    """Display system header"""
    st.markdown("""
    <div class="main-header">
        IoT ENVIRONMENTAL MONITORING SYSTEM
    </div>
    <div class="research-info">
        Research Project: OMO/RE/323 | Environmental Engineering & Machine Learning
    </div>
    """, unsafe_allow_html=True)


def display_csv_input(system):
    """CSV input tab content"""
    st.markdown('<div class="info-box">Upload CSV file with environmental sensor data collected from IoT devices</div>', unsafe_allow_html=True)
    
    st.markdown("**CSV Format Requirements:**")
    st.markdown("""
    Required columns: `Date`, `Time`, `Millis`, `DS18B20`, `DHTTemp`, `Humidity`, 
    `DustRaw`, `DustAvgADC`, `MQ7Raw`, `COppm`, `eCO2`, `TVOC`, `PM2.5 concentration (ug/m^3)`, `AQI`
    """)
    
    # Create template
    template_df = pd.DataFrame(columns=[
        'Date', 'Time', 'Millis', 'DS18B20', 'DHTTemp', 'Humidity',
        'DustRaw', 'DustAvgADC', 'MQ7Raw', 'COppm', 'eCO2', 'TVOC',
        'PM2.5 concentration (ug/m^3)', 'AQI'
    ])
    
    sample_data = {
        'Date': ['2024-01-15'],
        'Time': ['14:30:25'],
        'Millis': [123],
        'DS18B20': [23.5],
        'DHTTemp': [24.1],
        'Humidity': [45.2],
        'DustRaw': [350],
        'DustAvgADC': [352],
        'MQ7Raw': [120],
        'COppm': [2.1],
        'eCO2': [850],
        'TVOC': [200],
        'PM2.5 concentration (ug/m^3)': [25.1],
        'AQI': [75]
    }
    template_df = pd.concat([template_df, pd.DataFrame(sample_data)], ignore_index=True)
    
    csv_template = template_df.to_csv(index=False)
    st.download_button(
        label="Download CSV Template",
        data=csv_template,
        file_name="sensor_data_template.csv",
        mime="text/csv"
    )
    
    uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            try:
                df = pd.read_csv(uploaded_file)
            except:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, on_bad_lines='skip')
            
            st.write("File Preview:")
            st.dataframe(df.head(10), use_container_width=True)
            
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                if st.button("Process CSV Data", use_container_width=True):
                    processed = 0
                    errors = 0
                    progress_bar = st.progress(0)
                    
                    for idx, row in df.iterrows():
                        try:
                            timestamp = system.parse_timestamp(
                                row.get('Date', ''),
                                row.get('Time', ''),
                                row.get('Millis', '')
                            )
                            
                            sensor_data = {
                                'DS18B20_Temperature_C': float(row.get('DS18B20', 0)),
                                'Temperature_C': float(row.get('DHTTemp', 0)),
                                'Humidity_RH': float(row.get('Humidity', 0)),
                                'CO_ppm': float(row.get('COppm', 0)),
                                'eCO2_ppm': float(row.get('eCO2', 0)),
                                'VOC_ppb': float(row.get('TVOC', 0)),
                                'Dust_PM_ugm3': float(row.get('PM2.5 concentration (ug/m^3)', 0)),
                                'aqi_value': float(row.get('AQI', 0))
                            }
                            
                            aqi = system.calculate_aqi(sensor_data)
                            
                            if system.save_sensor_data(sensor_data, aqi, timestamp):
                                processed += 1
                            else:
                                errors += 1
                        except Exception as e:
                            errors += 1
                        
                        progress_bar.progress((idx + 1) / len(df))
                    
                    st.success(f"Successfully processed {processed} records ({errors} errors)")
                    st.rerun()
            
            with col2:
                if st.button("Clear Preview", use_container_width=True):
                    st.rerun()
                    
        except Exception as e:
            st.error(f"Error reading CSV file: {str(e)}")


def display_manual_input(system):
    """Manual input tab content"""
    st.markdown('<div class="info-box">Enter environmental parameters from IoT sensors manually</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Temperature Sensors**")
        ds18b20 = st.number_input("DS18B20 Temperature (°C)", value=None, placeholder="23.5", step=0.1)
        dht_temp = st.number_input("DHT22 Temperature (°C)", value=None, placeholder="24.1", step=0.1)
        humidity = st.number_input("Humidity (%)", value=None, placeholder="45.2", step=0.1)
    
    with col2:
        st.markdown("**Air Quality Parameters**")
        pm25 = st.number_input("PM2.5 Concentration (μg/m³)", value=None, placeholder="25.1", step=0.1)
        tvoc = st.number_input("TVOC (ppb)", value=None, placeholder="200", step=0.1)
        eco2 = st.number_input("eCO2 (ppm)", value=None, placeholder="850", step=0.1)
        co = st.number_input("CO (ppm)", value=None, placeholder="2.1", step=0.01)
    
    with col3:
        st.markdown("**Raw Sensor Data (Optional)**")
        dust_raw = st.number_input("Dust Raw", value=None, placeholder="350", step=1)
        dust_avg = st.number_input("Dust Avg ADC", value=None, placeholder="352", step=1)
        mq7_raw = st.number_input("MQ7 Raw", value=None, placeholder="120", step=1)
        aqi_input = st.number_input("AQI (if available)", value=None, placeholder="75", step=1)
        
        st.markdown("**Location (for Heatmap)**")
        location_name = st.text_input("Location Name", value="Main Station")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Submit Data", use_container_width=True):
            if None in [ds18b20, dht_temp, humidity, pm25, tvoc, eco2, co]:
                st.error("Please fill in all required fields")
            else:
                sensor_data = {
                    'DS18B20_Temperature_C': ds18b20,
                    'Temperature_C': dht_temp,
                    'Humidity_RH': humidity,
                    'Dust_PM_ugm3': pm25,
                    'VOC_ppb': tvoc,
                    'eCO2_ppm': eco2,
                    'CO_ppm': co,
                    'aqi_value': aqi_input if aqi_input is not None else 0
                }
                
                aqi = system.calculate_aqi(sensor_data)
                
                if system.save_sensor_data(sensor_data, aqi, location=location_name):
                    st.success("Data saved successfully")
                    st.rerun()
    
    with col2:
        if st.button("Reset Form", use_container_width=True):
            st.rerun()


def display_current_status(system, df):
    """Display current environmental status"""
    if df.empty:
        return False
    
    latest = df.iloc[-1]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        aqi_class = "normal-status"
        if latest['aqi_value'] > 200:
            aqi_class = "critical-warning"
        elif latest['aqi_value'] > 150:
            aqi_class = "high-warning"
        elif latest['aqi_value'] > 100:
            aqi_class = "moderate-warning"
        
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 0.9rem; color: #64748b;">Air Quality Index</div>
            <div style="font-size: 2rem; font-weight: 600; color: #1e3c72;">{latest['aqi_value']:.0f}</div>
            <div class="{aqi_class}" style="padding: 0.3rem;">{latest['aqi_category']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        pm25_status, _ = system.get_warning_level('Dust_PM_ugm3', latest['Dust_PM_ugm3'])
        status_class = f"warning-{pm25_status}" if pm25_status != 'unknown' else "moderate-warning"
        status_class = status_class.replace('warning-', '') + '-warning'
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 0.9rem; color: #64748b;">PM2.5</div>
            <div style="font-size: 2rem; font-weight: 600;">{latest['Dust_PM_ugm3']:.1f} μg/m³</div>
            <div class="{status_class}" style="padding: 0.3rem;">{pm25_status.title()}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        temp_status, _ = system.get_warning_level('Temperature_C', latest['Temperature_C'])
        status_class = f"warning-{temp_status}" if temp_status != 'unknown' else "moderate-warning"
        status_class = status_class.replace('warning-', '') + '-warning'
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 0.9rem; color: #64748b;">Temperature</div>
            <div style="font-size: 2rem; font-weight: 600;">{latest['Temperature_C']:.1f}°C</div>
            <div class="{status_class}" style="padding: 0.3rem;">{temp_status.title()}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        humid_status, _ = system.get_warning_level('Humidity_RH', latest['Humidity_RH'])
        status_class = f"warning-{humid_status}" if humid_status != 'unknown' else "moderate-warning"
        status_class = status_class.replace('warning-', '') + '-warning'
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 0.9rem; color: #64748b;">Humidity</div>
            <div style="font-size: 2rem; font-weight: 600;">{latest['Humidity_RH']:.1f}%</div>
            <div class="{status_class}" style="padding: 0.3rem;">{humid_status.title()}</div>
        </div>
        """, unsafe_allow_html=True)
    
    return True


def display_72hour_forecast(system, df):
    """Display 72-hour forecast using LSTM model"""
    st.markdown("### 72-Hour Predictive Analytics")
    st.markdown("*LSTM-based time series forecasting*")
    
    if df.empty:
        st.markdown("""
        <div class="empty-state">
            No data available for predictive modeling.
        </div>
        """, unsafe_allow_html=True)
        return
    
    if len(df) >= 10:
        predictions, _ = system.predict_72_hours(df)
        
        if predictions is not None:
            last_time = df['timestamp'].iloc[-1]
            future_times = pd.date_range(start=last_time + timedelta(hours=1), periods=72, freq='H')
            
            # Calculate AQI for predictions
            future_aqi = []
            for pred in predictions:
                sensor_data = {
                    'Dust_PM_ugm3': pred[0],
                    'VOC_ppb': pred[1],
                    'CO_ppm': pred[5]
                }
                aqi_info = system.calculate_aqi(sensor_data)
                future_aqi.append(aqi_info['value'])
            
            st.markdown("#### Air Quality Index - 72 Hour Forecast")
            
            fig = go.Figure()
            
            # Historical data (last 48 hours)
            hist_window = min(48, len(df))
            hist_aqi = df['aqi_value'].tail(hist_window).values
            hist_times = df['timestamp'].tail(hist_window)
            
            fig.add_trace(go.Scatter(
                x=hist_times,
                y=hist_aqi,
                mode='lines+markers',
                name='Historical AQI',
                line=dict(color='#1e3c72', width=3),
                marker=dict(size=5)
            ))
            
            fig.add_trace(go.Scatter(
                x=future_times,
                y=future_aqi,
                mode='lines+markers',
                name='Forecasted AQI',
                line=dict(color='#dc2626', width=3, dash='dash'),
                marker=dict(size=3)
            ))
            
            # Add threshold lines
            fig.add_hline(y=50, line_dash="dot", line_color="#10b981", 
                         annotation_text="Good", annotation_position="right")
            fig.add_hline(y=100, line_dash="dot", line_color="#f59e0b", 
                         annotation_text="Moderate", annotation_position="right")
            fig.add_hline(y=150, line_dash="dot", line_color="#f97316", 
                         annotation_text="Unhealthy (Sensitive)", annotation_position="right")
            fig.add_hline(y=200, line_dash="dot", line_color="#ef4444", 
                         annotation_text="Unhealthy", annotation_position="right")
            
            fig.update_layout(
                title="AQI Prediction - Next 72 Hours",
                xaxis_title="Time",
                yaxis_title="AQI Value",
                height=450,
                hovermode='x unified',
                paper_bgcolor='white',
                plot_bgcolor='white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Parameter-specific forecasts
            st.markdown("#### Parameter-Specific Forecasts")
            
            param = st.selectbox("Select Parameter", 
                               system.feature_columns,
                               format_func=lambda x: system.display_names.get(x, x))
            
            col1, col2 = st.columns([1, 3])
            
            with col1:
                st.markdown("**Forecast Summary**")
                pred_idx = system.feature_columns.index(param)
                pred_values = predictions[:, pred_idx]
                
                avg_val = np.mean(pred_values)
                max_val = np.max(pred_values)
                min_val = np.min(pred_values)
                current_val = df[param].iloc[-1] if param in df.columns else 0
                
                st.metric("Current", f"{current_val:.2f}")
                st.metric("Average (72h)", f"{avg_val:.2f}", 
                         f"{((avg_val - current_val) / current_val * 100):.1f}%" if current_val > 0 else "")
                st.metric("Peak", f"{max_val:.2f}")
                st.metric("Low", f"{min_val:.2f}")
            
            with col2:
                fig = go.Figure()
                
                # Historical data
                hist_data = df.tail(48)
                fig.add_trace(go.Scatter(
                    x=hist_data['timestamp'],
                    y=hist_data[param],
                    mode='lines+markers',
                    name='Historical',
                    line=dict(color='#1e3c72', width=2),
                    marker=dict(size=4)
                ))
                
                # Predictions
                fig.add_trace(go.Scatter(
                    x=future_times,
                    y=pred_values,
                    mode='lines+markers',
                    name='Forecast',
                    line=dict(color='#dc2626', width=2, dash='dash'),
                    marker=dict(size=2)
                ))
                
                # Add thresholds if available
                if param in system.thresholds:
                    thresh = system.thresholds[param]
                    if 'critical' in thresh:
                        fig.add_hline(y=thresh['critical'], line_dash="dash", 
                                    line_color="#dc2626", annotation_text="Critical")
                    if 'high' in thresh:
                        fig.add_hline(y=thresh['high'], line_dash="dash", 
                                    line_color="#f97316", annotation_text="High")
                    if 'moderate' in thresh:
                        fig.add_hline(y=thresh['moderate'], line_dash="dash", 
                                    line_color="#f59e0b", annotation_text="Moderate")
                
                unit = system.thresholds[param]['unit'] if param in system.thresholds else ''
                
                fig.update_layout(
                    title=f"{system.display_names.get(param, param)} - 72 Hour Forecast",
                    xaxis_title="Time",
                    yaxis_title=unit,
                    height=400,
                    hovermode='x unified',
                    paper_bgcolor='white',
                    plot_bgcolor='white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Warning timeline
            st.markdown("#### Predicted Warning Timeline")
            
            warning_levels = []
            for hour in range(72):
                max_level = 1  # Normal
                for idx, param_name in enumerate(system.feature_columns):
                    value = predictions[hour, idx]
                    status, level = system.get_warning_level(param_name, value)
                    max_level = max(max_level, level)
                warning_levels.append(max_level)
            
            fig = go.Figure()
            
            colors = ['#10b981' if l == 1 else '#f59e0b' if l == 2 else '#f97316' if l == 3 else '#dc2626' 
                     for l in warning_levels]
            
            fig.add_trace(go.Bar(
                x=list(range(72)),
                y=warning_levels,
                marker_color=colors,
                name='Warning Level',
                hovertemplate='Hour: %{x}<br>Level: %{y}<extra></extra>'
            ))
            
            fig.update_layout(
                title="Warning Intensity Over Next 72 Hours",
                xaxis_title="Hours from Now",
                yaxis_title="Warning Level",
                yaxis=dict(
                    tickmode='array',
                    tickvals=[1, 2, 3, 4],
                    ticktext=['Normal', 'Moderate', 'High', 'Critical']
                ),
                height=250,
                paper_bgcolor='white',
                plot_bgcolor='white',
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary statistics
            critical_hours = sum(1 for l in warning_levels if l == 4)
            high_hours = sum(1 for l in warning_levels if l == 3)
            moderate_hours = sum(1 for l in warning_levels if l == 2)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Critical Hours", critical_hours)
            with col2:
                st.metric("High Alert Hours", high_hours)
            with col3:
                st.metric("Moderate Hours", moderate_hours)
            with col4:
                st.metric("Normal Hours", 72 - critical_hours - high_hours - moderate_hours)
            
        else:
            st.info("Unable to generate predictions. Please check LSTM model.")
    else:
        st.info(f"Need at least 10 data points for time series analysis. Current: {len(df)}")


def display_spatial_heatmap(system, df):
    """Display spatial heatmap for environmental parameters"""
    st.markdown("### Spatial Heatmap Visualization")
    st.markdown("*Geographic distribution of environmental parameters*")
    
    if df.empty:
        st.markdown("""
        <div class="empty-state">
            No location data available for heatmap visualization.
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Check if location data exists
    if 'location' not in df.columns:
        st.warning("Location data not found. Add location information to create heatmaps.")
        return
    
    # Get unique locations and their latest readings
    latest_by_location = df.groupby('location').last().reset_index()
    
    if len(latest_by_location) < 3:
        st.info(f"Need at least 3 locations for heatmap generation. Current: {len(latest_by_location)}")
        return
    
    # Sample location coordinates (in real implementation, these would come from a database)
    # For demonstration, create synthetic coordinates
    np.random.seed(42)
    locations = latest_by_location['location'].unique()
    lats = np.random.uniform(40.5, 40.9, len(locations))
    lons = np.random.uniform(-74.2, -73.8, len(locations))
    
    param_to_viz = st.selectbox(
        "Select Parameter for Heatmap",
        ['aqi_value', 'Dust_PM_ugm3', 'Temperature_C', 'Humidity_RH', 'VOC_ppb'],
        format_func=lambda x: system.display_names.get(x, x)
    )
    
    # Create grid for interpolation
    grid_lat, grid_lon = np.mgrid[min(lats):max(lats):100j, min(lons):max(lons):100j]
    
    # Interpolate values
    values = latest_by_location[param_to_viz].values
    grid_values = griddata(
        (lats, lons), values, 
        (grid_lat, grid_lon), 
        method='cubic'
    )
    
    # Create heatmap
    fig = go.Figure()
    
    # Add heatmap layer
    fig.add_trace(go.Heatmap(
        z=grid_values,
        x=grid_lon[0],
        y=grid_lat[:, 0],
        colorscale='RdYlGn_r',
        zmin=values.min(),
        zmax=values.max(),
        showscale=True,
        colorbar=dict(title=system.display_names.get(param_to_viz, param_to_viz))
    ))
    
    # Add location markers
    fig.add_trace(go.Scatter(
        x=lons,
        y=lats,
        mode='markers+text',
        marker=dict(
            size=10,
            color='black',
            line=dict(color='white', width=2)
        ),
        text=locations,
        textposition="top center",
        name='Monitoring Stations',
        hoverinfo='text',
        hovertext=[f"{loc}<br>{param_to_viz}: {val:.2f}" 
                  for loc, val in zip(locations, values)]
    ))
    
    fig.update_layout(
        title=f"Spatial Distribution of {system.display_names.get(param_to_viz, param_to_viz)}",
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        height=500,
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Station comparison
    st.markdown("#### Station Comparison")
    
    fig = go.Figure()
    for idx, loc in enumerate(locations):
        loc_data = df[df['location'] == loc].tail(24)  # Last 24 hours
        fig.add_trace(go.Scatter(
            x=loc_data['timestamp'],
            y=loc_data[param_to_viz],
            mode='lines',
            name=loc,
            line=dict(width=2)
        ))
    
    fig.update_layout(
        title=f"{system.display_names.get(param_to_viz, param_to_viz)} - Last 24 Hours by Location",
        xaxis_title="Time",
        yaxis_title=system.thresholds.get(param_to_viz, {}).get('unit', ''),
        height=400,
        hovermode='x unified',
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    
    st.plotly_chart(fig, use_container_width=True)


def display_temporal_visualizations(system, df):
    """Display temporal visualizations"""
    st.markdown("### Temporal Visualization")
    st.markdown("*Interactive visualizations for pattern analysis*")
    
    if df.empty or len(df) < 3:
        st.markdown("""
        <div class="empty-state">
            Need at least 3 data points for visualizations.
        </div>
        """, unsafe_allow_html=True)
        return
    
    viz_type = st.radio("Visualization Type", 
                       ["Time Series Analysis", "Correlation Matrix", "Distribution Analysis"],
                       horizontal=True)
    
    if viz_type == "Time Series Analysis":
        available_params = [col for col in system.feature_columns if col in df.columns]
        params = st.multiselect("Select Parameters",
                               available_params,
                               default=available_params[:3] if len(available_params) >= 3 else available_params,
                               format_func=lambda x: system.display_names.get(x, x))
        
        if params:
            fig = go.Figure()
            colors = ['#1e3c72', '#dc2626', '#059669', '#d97706', '#7c3aed', '#db2777']
            
            for idx, param in enumerate(params):
                fig.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=df[param],
                    mode='lines',
                    name=system.display_names.get(param, param),
                    line=dict(color=colors[idx % len(colors)], width=2)
                ))
            
            fig.update_layout(
                title="Multi-Parameter Time Series",
                xaxis_title="Time",
                yaxis_title="Value",
                height=450,
                hovermode='x unified',
                paper_bgcolor='white',
                plot_bgcolor='white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Correlation Matrix":
        available_params = [col for col in system.feature_columns if col in df.columns]
        numeric_df = df[available_params].select_dtypes(include=[np.number])
        
        if not numeric_df.empty and len(numeric_df.columns) > 1:
            corr = numeric_df.corr()
            
            x_labels = [system.display_names.get(col, col) for col in corr.columns]
            y_labels = [system.display_names.get(col, col) for col in corr.columns]
            
            fig = go.Figure(data=go.Heatmap(
                z=corr.values,
                x=x_labels,
                y=y_labels,
                colorscale='RdBu',
                zmid=0,
                text=np.round(corr.values, 2),
                texttemplate='%{text}',
                textfont={"size": 10}
            ))
            
            fig.update_layout(
                title="Parameter Correlation Analysis",
                height=500,
                xaxis_tickangle=-45,
                paper_bgcolor='white',
                plot_bgcolor='white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    else:  # Distribution Analysis
        available_params = [col for col in system.feature_columns if col in df.columns]
        param = st.selectbox("Parameter", 
                           available_params,
                           format_func=lambda x: system.display_names.get(x, x))
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(df, x=param, nbins=30,
                             title=f"Distribution: {system.display_names.get(param, param)}",
                             labels={'x': system.display_names.get(param, param)})
            fig.update_layout(paper_bgcolor='white', plot_bgcolor='white', height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(df, y=param,
                       title=f"Box Plot: {system.display_names.get(param, param)}")
            fig.update_layout(paper_bgcolor='white', plot_bgcolor='white', height=400)
            st.plotly_chart(fig, use_container_width=True)


def display_statistics(system, df):
    """Display statistical analysis"""
    st.markdown("### Statistical Analysis")
    st.markdown("*Data-driven environmental assessment*")
    
    if df.empty:
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Descriptive Statistics**")
        available_params = [col for col in system.feature_columns if col in df.columns]
        stats_df = df[available_params].describe()
        stats_df.columns = [system.display_names.get(col, col) for col in stats_df.columns]
        st.dataframe(stats_df.style.format("{:.2f}"), use_container_width=True)
    
    with col2:
        st.markdown("**AQI Distribution**")
        if 'aqi_category' in df.columns and not df['aqi_category'].isna().all():
            counts = df['aqi_category'].value_counts()
            
            fig = go.Figure(data=[go.Pie(
                labels=counts.index,
                values=counts.values,
                marker=dict(colors=['#10b981', '#f59e0b', '#f97316', '#ef4444', '#7f1d1d']),
                textinfo='label+percent',
                hole=0.4
            )])
            
            fig.update_layout(
                title="Air Quality Categories",
                paper_bgcolor='white',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Trend analysis
    st.markdown("### Temporal Trend Analysis")
    
    trend_df = df.copy()
    for param in ['Dust_PM_ugm3', 'VOC_ppb', 'CO_ppm', 'aqi_value']:
        if param in trend_df.columns:
            window_size = min(7, len(trend_df))
            trend_df[f'{param}_MA'] = trend_df[param].rolling(window=window_size, min_periods=1).mean()
    
    fig = go.Figure()
    
    params_to_plot = ['Dust_PM_ugm3', 'VOC_ppb', 'CO_ppm']
    colors = ['#dc2626', '#f97316', '#7c3aed']
    
    for idx, param in enumerate(params_to_plot):
        if param in trend_df.columns and f'{param}_MA' in trend_df.columns:
            fig.add_trace(go.Scatter(
                x=trend_df['timestamp'],
                y=trend_df[f'{param}_MA'],
                mode='lines',
                name=f"{system.display_names.get(param, param)} (Moving Average)",
                line=dict(color=colors[idx], width=2)
            ))
    
    if len(fig.data) > 0:
        fig.update_layout(
            title="Moving Average Trends - Key Pollutants",
            xaxis_title="Time",
            yaxis_title="Concentration",
            height=400,
            hovermode='x unified',
            paper_bgcolor='white',
            plot_bgcolor='white'
        )
        
        st.plotly_chart(fig, use_container_width=True)


def main():
    """Main application"""
    
    if 'system' not in st.session_state:
        st.session_state.system = EnvironmentalMonitoringSystem()
    
    system = st.session_state.system
    
    display_header()
    
    with st.sidebar:
        st.markdown("### System Status")
        
        df = system.load_data()
        
        st.markdown(f"**Total Records:** {len(df)}")
        
        st.markdown("---")
        st.markdown("**ML Model Status:**")
        
        if system.lstm_model:
            st.markdown("LSTM Network: Active")
        else:
            st.markdown("LSTM Network: Not Loaded")
            st.markdown("Please ensure lstm_model.keras exists")
        
        if not df.empty:
            latest_time = df['timestamp'].iloc[-1]
            if pd.notna(latest_time):
                st.markdown(f"**Latest Update:**")
                st.markdown(f"{latest_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        st.markdown("---")
        st.markdown("**Research Objectives:**")
        st.markdown("""
        1. IoT sensor network design
        2. Data transmission and storage
        3. Heatmap visualization
        4. ML pattern recognition
        """)
        
        st.markdown("---")
        
        if st.button("Clear All Data", use_container_width=True):
            if os.path.exists(system.data_file):
                os.remove(system.data_file)
                system.initialize_data_file()
                st.rerun()
    
    # Data Input Tabs
    tab1, tab2 = st.tabs(["CSV Upload", "Manual Entry"])
    
    with tab1:
        display_csv_input(system)
    
    with tab2:
        display_manual_input(system)
    
    # Load data after input
    df = system.load_data()
    
    # Display analysis sections only if data exists
    if not df.empty:
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        # Current Status
        if display_current_status(system, df):
            
            # 72-Hour Forecast
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            display_72hour_forecast(system, df)
            
            # Spatial Heatmap
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            display_spatial_heatmap(system, df)
            
            # Temporal Visualizations
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            display_temporal_visualizations(system, df)
            
            # Statistics
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            display_statistics(system, df)
            
            # Data Export
            st.markdown("### Data Export")
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download Dataset (CSV)",
                    data=csv,
                    file_name=f"environmental_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            with col2:
                st.markdown(f"**Records:** {len(df)}")
                st.markdown(f"**Time Range:** {(df['timestamp'].max() - df['timestamp'].min()).days} days")
            st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
