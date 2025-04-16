import os
from flask import Flask, render_template, send_from_directory, jsonify, request
import pandas as pd
from datetime import datetime
import json
import logging

# Get the base directory of the application
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, 
            static_folder=os.path.join(BASE_DIR, 'static'),
            template_folder=os.path.join(BASE_DIR, 'templates'))

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/forecast')
def forecast():
    return render_template('forecast.html')

@app.route('/climate_zones')
def climate_zones():
    return render_template('climate_zones.html')

@app.route('/climate_zones_map')
def climate_zones_map():
    return render_template('climate_zones_map.html')

@app.route('/risk_assessment')
def risk_assessment():
    return render_template('risk_assessment.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/api/forecast/<location>')
def get_forecast_data(location):
    try:
        # Convert location name to match file naming
        location = location.lower()  # Keep lowercase for internal use
        location_cap = location.capitalize()  # For file name
        
        # Define path for forecast file using relative path
        forecast_file = os.path.join(BASE_DIR, 'data', 'forecasts', f'{location_cap}_forecast.csv')
        
        logger.debug(f"Looking for forecast file at: {forecast_file}")
        
        if not os.path.exists(forecast_file):
            return jsonify({
                'success': False,
                'error': f'No forecast data available for {location}'
            })
        
        try:
            # Skip the metrics and read only the data portion
            data_df = pd.read_csv(forecast_file, skiprows=4, encoding='latin-1')
            logger.debug("Successfully read data rows")
            logger.debug(f"Columns found: {data_df.columns.tolist()}")
            
            # Replace empty strings and whitespace with NaN
            data_df = data_df.replace(r'^\s*$', pd.NA, regex=True)
            data_df = data_df.replace('', pd.NA)
            
            # Get column names with proper capitalization
            rain_col = f'{location_cap}_rain_forecast'
            soil_col = f'{location_cap}_soil_forecast'
            temp_col = f'{location_cap}_temp_forecast'
            
            # Verify column names exist
            if rain_col not in data_df.columns or soil_col not in data_df.columns or temp_col not in data_df.columns:
                logger.error(f"Expected columns not found. Looking for: {rain_col}, {soil_col}, {temp_col}")
                logger.error(f"Available columns: {data_df.columns.tolist()}")
                return jsonify({
                    'success': False,
                    'error': f'Column names not found in forecast file'
                })
            
            # Convert columns to float, replacing any remaining non-numeric values with NaN
            try:
                data_df[rain_col] = pd.to_numeric(data_df[rain_col], errors='coerce')
                data_df[soil_col] = pd.to_numeric(data_df[soil_col], errors='coerce')
                data_df[temp_col] = pd.to_numeric(data_df[temp_col], errors='coerce')
            except Exception as e:
                logger.error(f"Error converting columns to numeric: {str(e)}")
                return jsonify({
                    'success': False,
                    'error': f'Error converting data to numeric: {str(e)}'
                })
            
            # Remove rows with NaN values
            data_df = data_df.dropna()
            
            if len(data_df) == 0:
                return jsonify({
                    'success': False,
                    'error': 'No valid data rows found after cleaning'
                })
            
            # Extract forecast data (without metrics)
            data = {
                'dates': data_df['Date'].tolist(),
                'rainfall': data_df[rain_col].tolist(),
                'soil_moisture': data_df[soil_col].tolist(),
                'temperature': data_df[temp_col].tolist(),
                'metrics': {
                    'rainfall': {'rmse': 0, 'mae': 0, 'r2': 0},
                    'soil_moisture': {'rmse': 0, 'mae': 0, 'r2': 0},
                    'temperature': {'rmse': 0, 'mae': 0, 'r2': 0}
                }
            }
            
            logger.debug(f"Processed data for {location}:")
            logger.debug(f"Number of dates: {len(data['dates'])}")
            logger.debug(f"Number of values per variable: {len(data['rainfall'])}")
            
            return jsonify({
                'success': True,
                'data': data
            })
            
        except Exception as e:
            logger.error(f"Error processing {forecast_file}: {str(e)}")
            return jsonify({
                'success': False,
                'error': f'Error processing forecast data: {str(e)}'
            })
            
    except Exception as e:
        logger.error(f"Error processing request for {location}: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/risk_assessment/<location>')
def get_risk_assessment(location):
    try:
        # Path to pre-computed risk assessment file using relative path
        risk_file = os.path.join(BASE_DIR, 'data', 'risk_assesment', 'combined_risk_assessment.json')
        
        logger.debug(f"Looking for risk assessment file at: {risk_file}")
        
        try:
            # Try to read the pre-computed risk assessments
            with open(risk_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                logger.debug("Successfully loaded risk assessment data")
                
                if 'county_risks' not in data:
                    return jsonify({
                        'error': 'Invalid data structure: missing county_risks'
                    }), 500
                
                risk_data = data['county_risks']
                logger.debug(f"Available locations: {list(risk_data.keys())}")
                logger.debug(f"Requested location: {location}")
                
                # Try to find a case-insensitive match
                location_found = None
                for key in risk_data.keys():
                    if key.lower() == location.lower():
                        location_found = key
                        break
            
                if location_found:
                    logger.debug(f"Found matching location: {location_found}")
                    return jsonify({
                        'location': location_found,
                        'risk_assessment': risk_data[location_found]
                    })
                else:
                    logger.error(f"Location not found. Available locations: {list(risk_data.keys())}")
                    return jsonify({
                        'error': f'No risk assessment data available for {location}. Available locations: {", ".join(sorted(risk_data.keys()))}'
                    }), 404
                
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Error reading risk assessment file: {str(e)}")
            return jsonify({
                'error': f'Could not read risk assessment data: {str(e)}'
            }), 500
            
    except Exception as e:
        logger.error(f"Error loading risk assessment data: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)

if __name__ == '__main__':
    app.run(debug=True) 