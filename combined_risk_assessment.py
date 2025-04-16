import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os
from datetime import datetime
import json
import matplotlib.pyplot as plt
import seaborn as sns

class ClimateRiskAssessment:
    def __init__(self, data_path, output_dir=None, zone_data_path=None):
        """
        Initialize the climate risk assessment system with both ML and traditional approaches.
        
        Parameters:
        data_path (str): Path to CSV with historical and forecasted climate data
        output_dir (str, optional): Directory to save outputs
        zone_data_path (str, optional): Path to CSV file with county zone information
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.zone_data_path = zone_data_path
        self.df = None
        self.counties = None
        self.models = {
            'drought': None,
            'heat': None,
            'flood': None,
            'rainfall': None
        }
        self.feature_cols = ['rain_z', 'temp_z', 'soil_ratio']
        self.risk_levels = ['Low', 'Mild', 'Moderate', 'Severe']
        self.rainfall_levels = ['Low', 'Moderate', 'High']
        self.score_map = {'Low': 10, 'Mild': 30, 'Moderate': 60, 'Severe': 90}
        self.risk_weights = {'drought': 0.33, 'heat': 0.33, 'flood': 0.33}
        self.risk_thresholds = {
            'drought': {'severe': 2.0, 'moderate': 1.0, 'mild': 0.5},
            'heat': {'severe': 2.0, 'moderate': 1.0, 'mild': 0.5},
            'flood': {'severe': 2.0, 'moderate': 1.5, 'mild': 1.0}
        }
        self.crop_zone_mapping = {
            0: ['maize', 'beans', 'potatoes', 'sweet potatoes', 'bananas', 'coffee', 'tea'],
            1: ['maize', 'beans', 'wheat', 'barley', 'potatoes', 'peas', 'dairy'],
            2: ['maize', 'beans', 'sorghum', 'millet', 'cotton', 'sunflower', 'tobacco'],
            3: ['sorghum', 'millet', 'cowpeas', 'green grams', 'cassava', 'livestock']
        }
    
    def load_data(self, encoding='latin1'):
        """Load and preprocess the climate data."""
        print(f"Loading data from {self.data_path}...")
        self.df = pd.read_csv(self.data_path, encoding=encoding)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df['is_forecast'] = (self.df['Date'].dt.year >= 2025).astype(int)
        
        rain_cols = [col for col in self.df.columns if col.endswith('_rain')]
        self.counties = [col.replace('_rain', '') for col in rain_cols]
        print(f"Found {len(self.counties)} counties: {self.counties}")
        
        # Load zone data
        self.county_zone_mapping = {}
        if self.zone_data_path:
            print(f"Loading zone data from {self.zone_data_path}...")
            zone_df = pd.read_csv(self.zone_data_path, encoding=encoding)
            for _, row in zone_df.iterrows():
                self.county_zone_mapping[row['county']] = int(row['zone'])
            print(f"Loaded zone information for {len(self.county_zone_mapping)} counties")
        else:
            print("Warning: No zone data provided. Counties assigned to default zone 1.")
    
    def prepare_ml_features_and_labels(self, historical_data, county):
        """Prepare features and labels for ML training."""
        rain_col = f'{county}_rain'
        soil_col = f'{county}_soil'
        temp_col = f'{county}_temp'
        
        if not all(col in historical_data.columns for col in [rain_col, soil_col, temp_col]):
            print(f"Warning: Missing data for {county}, skipping...")
            return None, None
        
        stats = {
            'rain_mean': historical_data[rain_col].mean(),
            'rain_std': max(0.001, historical_data[rain_col].std()),
            'soil_mean': historical_data[soil_col].mean(),
            'temp_mean': historical_data[temp_col].mean(),
            'temp_std': max(0.001, historical_data[temp_col].std())
        }
        
        features = []
        labels = {'drought': [], 'heat': [], 'flood': [], 'rainfall': []}
        
        for _, row in historical_data.iterrows():
            rain_z = (row[rain_col] - stats['rain_mean']) / stats['rain_std']
            temp_z = (row[temp_col] - stats['temp_mean']) / stats['temp_std']
            soil_ratio = row[soil_col] / stats['soil_mean']
            
            features.append([rain_z, temp_z, soil_ratio])
            
            drought_idx = max(0, -rain_z) * min(2.0, max(0.5, 1 + (1 - soil_ratio)))
            heat_idx = max(0, temp_z)
            flood_idx = rain_z + (0.5 * max(0, soil_ratio - 1))
            
            for risk_type, idx in [('drought', drought_idx), ('heat', heat_idx), ('flood', flood_idx)]:
                if idx >= 2.0:
                    labels[risk_type].append("Severe")
                elif idx >= 1.0:
                    labels[risk_type].append("Moderate")
                elif idx >= 0.5:
                    labels[risk_type].append("Mild")
                else:
                    labels[risk_type].append("Low")
            
            if rain_z <= -1.0:
                labels['rainfall'].append("Low")
            elif rain_z >= 1.0:
                labels['rainfall'].append("High")
            else:
                labels['rainfall'].append("Moderate")
        
        return features, labels
    
    def train_ml_models(self):
        """Train Random Forest models for risks and rainfall categories."""
        historical_data = self.df[self.df['is_forecast'] == 0]
        
        all_features = []
        all_labels = {'drought': [], 'heat': [], 'flood': [], 'rainfall': []}
        
        for county in self.counties:
            features, labels = self.prepare_ml_features_and_labels(historical_data, county)
            if features and labels:
                all_features.extend(features)
                for key in all_labels:
                    all_labels[key].extend(labels[key])
        
        if not all_features:
            raise ValueError("No valid training data available.")
        
        print(f"Training ML models with {len(all_features)} samples...")
        
        for risk_type in ['drought', 'heat', 'flood', 'rainfall']:
            X_train, X_test, y_train, y_test = train_test_split(
                all_features, all_labels[risk_type], test_size=0.2, random_state=42
            )
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"{risk_type.capitalize()} Model Accuracy: {accuracy:.2f}")
            print(classification_report(y_test, y_pred, target_names=self.risk_levels if risk_type != 'rainfall' else self.rainfall_levels))
            self.models[risk_type] = model
    
    def generate_traditional_assessment(self):
        """Generate traditional climate risk assessment with time series data."""
        print("Analyzing current risk conditions...")
        current_period = self.df[self.df['is_forecast'] == 0].iloc[-3:]
        forecast_periods = {
            'near_term': self.df[(self.df['is_forecast'] == 1) & (self.df['Date'].dt.year == 2025)],
            'mid_term': self.df[(self.df['is_forecast'] == 1) & (self.df['Date'].dt.year.isin([2026, 2027]))],
            'long_term': self.df[(self.df['is_forecast'] == 1) & (self.df['Date'].dt.year >= 2028)]
        }
        
        historical_data = self.df[self.df['is_forecast'] == 0]
        county_stats = {}
        print("Calculating historical reference statistics...")
        for county in self.counties:
            rain_col = f'{county}_rain'
            soil_col = f'{county}_soil'
            temp_col = f'{county}_temp'
            
            if all(col in historical_data.columns for col in [rain_col, soil_col, temp_col]):
                county_stats[county] = {
                    'rain_mean': historical_data[rain_col].mean(),
                    'rain_std': max(0.001, historical_data[rain_col].std()),
                    'rain_p10': historical_data[rain_col].quantile(0.10),
                    'soil_mean': historical_data[soil_col].mean(),
                    'soil_std': max(0.001, historical_data[soil_col].std()),
                    'temp_mean': historical_data[temp_col].mean(),
                    'temp_std': max(0.001, historical_data[temp_col].std()),
                    'temp_p90': historical_data[temp_col].quantile(0.90)
                }
        
        risk_results = {
            'metadata': {
                'generation_date': datetime.now().strftime('%Y-%m-%d'),
                'data_period': f"{self.df['Date'].min().strftime('%Y-%m')} to {self.df['Date'].max().strftime('%Y-%m')}",
                'counties_analyzed': self.counties,
                'risk_weights': self.risk_weights
            },
            'county_risks': {},
            'zone_risks': {},
            'recommendations': {},
            'crop_advisories': {}
        }
        
        for county in self.counties:
            print(f"Assessing traditional risks for {county}...")
            if county not in county_stats:
                print(f"Warning: No historical reference data for {county}, skipping...")
                continue
            
            zone = self.county_zone_mapping.get(county, 1)
            county_risks = {'zone': int(zone)}
            
            for period_name, period_data in [('current', current_period)] + list(forecast_periods.items()):
                if period_data.empty:
                    continue
                
                rain_col = f'{county}_rain'
                soil_col = f'{county}_soil'
                temp_col = f'{county}_temp'
                
                if not all(col in period_data.columns for col in [rain_col, soil_col, temp_col]):
                    print(f"Warning: Missing required data columns for {county}, skipping {period_name} period")
                    continue

                # Initialize time series data
                time_series = {
                    'dates': period_data['Date'].dt.strftime('%Y-%m').tolist(),
                    'drought_risk': [],
                    'heat_risk': [],
                    'flood_risk': []
                }

                # Calculate risks for each time point
                stats = county_stats[county]
                for _, row in period_data.iterrows():
                    rain_z = (row[rain_col] - stats['rain_mean']) / stats['rain_std']
                    temp_z = (row[temp_col] - stats['temp_mean']) / stats['temp_std']
                    soil_moisture = row[soil_col]
                    
                    # Calculate risk indices
                    drought_idx = max(0, -rain_z) * min(2.0, max(0.5, 1 + (stats['soil_mean'] - soil_moisture) / stats['soil_mean']))
                    heat_idx = max(0, temp_z)
                    flood_idx = rain_z + (0.5 * max(0, soil_moisture / stats['soil_mean'] - 1))
                    
                    # Convert indices to normalized risk scores (0-1)
                    time_series['drought_risk'].append(min(1.0, drought_idx / 2.0))
                    time_series['heat_risk'].append(min(1.0, heat_idx / 2.0))
                    time_series['flood_risk'].append(min(1.0, flood_idx / 2.0))
                
                # Calculate period averages
                rain_mean = period_data[rain_col].mean()
                rain_min = period_data[rain_col].min()
                soil_moisture = period_data[soil_col].mean()
                temp_mean = period_data[temp_col].mean()
                temp_max = period_data[temp_col].max()
                
                rain_zscore = (rain_mean - stats['rain_mean']) / stats['rain_std']
                temp_zscore = (temp_mean - stats['temp_mean']) / stats['temp_std']
                
                drought_index = max(0, -((rain_min - stats['rain_mean']) / stats['rain_std']))
                soil_factor = min(2.0, max(0.5, 1 + (stats['soil_mean'] - soil_moisture) / stats['soil_mean']))
                drought_severity = drought_index * soil_factor if rain_min < stats['rain_mean'] else max(0, 1 - soil_moisture / stats['soil_mean'])
                
                heat_stress = max(0, (temp_max - stats['temp_mean']) / stats['temp_std'])
                
                soil_saturation = soil_moisture / stats['soil_mean']
                flood_index = rain_zscore + (0.5 * max(0, soil_saturation - 1))
                
                for risk_type, index, thresholds in [
                    ('drought', drought_severity, self.risk_thresholds['drought']),
                    ('heat', heat_stress, self.risk_thresholds['heat']),
                    ('flood', flood_index, self.risk_thresholds['flood'])
                ]:
                    if index >= thresholds['severe']:
                        level, score = "Severe", 90
                    elif index >= thresholds['moderate']:
                        level, score = "Moderate", 60
                    elif index >= thresholds['mild']:
                        level, score = "Mild", 30
                    else:
                        level, score = "Low", 10
                    county_risks.setdefault(period_name, {})[f'{risk_type}_risk_level'] = level
                    county_risks[period_name][f'{risk_type}_risk_score'] = score
                
                overall_risk_score = (
                    county_risks[period_name]['drought_risk_score'] * self.risk_weights['drought'] +
                    county_risks[period_name]['heat_risk_score'] * self.risk_weights['heat'] +
                    county_risks[period_name]['flood_risk_score'] * self.risk_weights['flood']
                )
                
                county_risks[period_name].update({
                    'time_series': time_series,
                    'drought_index': drought_index,
                    'drought_severity': drought_severity,
                    'heat_stress': heat_stress,
                    'rain_zscore': rain_zscore,
                    'temp_zscore': temp_zscore,
                    'flood_index': flood_index,
                    'soil_moisture': soil_moisture,
                    'soil_mean': stats['soil_mean'],
                    'overall_risk_score': overall_risk_score,
                    'start_date': period_data['Date'].min().strftime('%Y-%m'),
                    'end_date': period_data['Date'].max().strftime('%Y-%m')
                })
            
            risk_results['county_risks'][county] = county_risks
            
            # Generate dynamic recommendations based on risk assessment and seasonal patterns
            recommendations = []
            risk_period = county_risks.get('near_term', county_risks.get('current', {}))
            
            if risk_period:
                # Get current month and next 3 months' data
                current_month = datetime.now().month
                next_months = [(current_month + i - 1) % 12 + 1 for i in range(4)]
                
                # Analyze rainfall patterns for the next 3 months
                rainfall_trend = self.analyze_rainfall_trend(period_data, rain_col, stats)
                temperature_trend = self.analyze_temperature_trend(period_data, temp_col, stats)
                
                # Generate seasonal recommendations
                if current_month in [2, 3, 4]:  # Long rains season
                    recommendations.extend(self.generate_long_rains_recommendations(
                        risk_period, rainfall_trend, temperature_trend, stats
                    ))
                elif current_month in [9, 10, 11]:  # Short rains season
                    recommendations.extend(self.generate_short_rains_recommendations(
                        risk_period, rainfall_trend, temperature_trend, stats
                    ))
                else:  # Dry season
                    recommendations.extend(self.generate_dry_season_recommendations(
                        risk_period, rainfall_trend, temperature_trend, stats
                    ))
                
                # Add risk-specific recommendations
                if risk_period.get('drought_risk_level') in ["Moderate", "Severe"]:
                    recommendations.extend([
                        "Implement water conservation measures",
                        "Use drought-resistant crop varieties",
                        "Consider early planting to avoid peak drought",
                        "Install water harvesting systems"
                    ])
                
                if risk_period.get('heat_risk_level') in ["Moderate", "Severe"]:
                    recommendations.extend([
                        "Use shade nets for sensitive crops",
                        "Increase irrigation frequency",
                        "Apply reflective mulch",
                        "Plant heat-tolerant varieties"
                    ])
                
                if risk_period.get('flood_risk_level') in ["Moderate", "Severe"]:
                    recommendations.extend([
                        "Clear and maintain drainage channels",
                        "Prepare flood barriers for low-lying areas",
                        "Elevate planting beds",
                        "Use flood-tolerant varieties"
                    ])
                
                risk_results['recommendations'][county] = recommendations
        
        return risk_results

    def analyze_rainfall_trend(self, period_data, rain_col, stats):
        """Analyze rainfall trend for the next 3 months."""
        if period_data.empty:
            return "stable"
        
        # Calculate rainfall deviation from mean
        rain_mean = period_data[rain_col].mean()
        rain_zscore = (rain_mean - stats['rain_mean']) / stats['rain_std']
        
        if rain_zscore > 1.0:
            return "increasing"
        elif rain_zscore < -1.0:
            return "decreasing"
        else:
            return "stable"

    def analyze_temperature_trend(self, period_data, temp_col, stats):
        """Analyze temperature trend for the next 3 months."""
        if period_data.empty:
            return "stable"
        
        # Calculate temperature deviation from mean
        temp_mean = period_data[temp_col].mean()
        temp_zscore = (temp_mean - stats['temp_mean']) / stats['temp_std']
        
        if temp_zscore > 1.0:
            return "increasing"
        elif temp_zscore < -1.0:
            return "decreasing"
        else:
            return "stable"

    def generate_long_rains_recommendations(self, risk_period, rainfall_trend, temperature_trend, stats):
        """Generate recommendations for the long rains season."""
        recommendations = []
        
        # Pre-season preparation (1 month before)
        recommendations.append({
            'title': 'Pre-Season Land Preparation',
            'timing': '1 month before long rains',
            'actions': [
                'Clear fields of crop residue and weeds',
                'Perform soil testing and apply recommended amendments',
                'Prepare water harvesting structures',
                'Repair and maintain farm infrastructure',
                'Apply organic manure (5-10 tons/ha)',
                'Deep tillage (30-45cm) for better water infiltration',
                'Construct contour bunds for soil conservation',
                'Install soil moisture monitoring devices'
            ],
            'climate_indicator': f'Expected rainfall: {stats["rain_mean"]:.1f}mm ± {stats["rain_std"]:.1f}mm'
        })
        
        # Soil management
        recommendations.append({
            'title': 'Soil Management',
            'timing': 'Throughout season',
            'actions': [
                'Maintain soil cover with organic mulch (5-10cm)',
                'Practice minimum tillage to preserve soil structure',
                'Apply compost tea for soil microbial activity',
                'Monitor soil pH and adjust if needed',
                'Use cover crops to prevent erosion',
                'Implement crop rotation for soil health'
            ],
            'climate_indicator': f'Target soil moisture: {stats["soil_mean"]:.1f}%'
        })
        
        # Water conservation
        recommendations.append({
            'title': 'Water Conservation',
            'timing': 'Throughout season',
            'actions': [
                'Implement drip irrigation for water efficiency',
                'Use rainwater harvesting systems',
                'Construct check dams for water retention',
                'Practice furrow irrigation with proper spacing',
                'Install soil moisture sensors for precise irrigation',
                'Use water-saving techniques like basin irrigation'
            ],
            'climate_indicator': 'Focus on efficient water use'
        })
        
        # Planting recommendations
        if rainfall_trend == "increasing":
            recommendations.append({
                'title': 'Planting Schedule',
                'timing': 'Start of long rains',
                'actions': [
                    'Begin planting as soon as soil moisture is adequate',
                    'Use early-maturing varieties',
                    'Implement proper spacing for better water use',
                    'Apply organic matter to improve soil structure',
                    'Use seed treatment for better germination',
                    'Implement intercropping for better land use'
                ],
                'climate_indicator': 'Rainfall trend: Increasing'
            })
        
        # In-season management
        if temperature_trend == "increasing":
            recommendations.append({
                'title': 'Heat Management',
                'timing': 'During growing season',
                'actions': [
                    'Monitor soil moisture daily',
                    'Apply mulch to conserve moisture',
                    'Use shade nets for sensitive crops',
                    'Increase irrigation frequency during heat waves',
                    'Use reflective mulch to reduce soil temperature',
                    'Implement misting systems for cooling'
                ],
                'alert': True,
                'climate_indicator': 'Temperature trend: Increasing'
            })
        
        return recommendations

    def generate_short_rains_recommendations(self, risk_period, rainfall_trend, temperature_trend, stats):
        """Generate recommendations for the short rains season."""
        recommendations = []
        
        # Get region-specific rainfall patterns
        region_rainfall_patterns = {
            'meru': {
                'start_month': 10,  # October
                'end_month': 12,    # December
                'peak_month': 11,   # November
                'expected_amount': '400-600mm',
                'planting_window': 'October',
                'suitable_crops': ['Maize', 'Beans', 'Green grams', 'Cowpeas']
            },
            'kitui': {
                'start_month': 11,  # November
                'end_month': 12,    # December
                'peak_month': 11,   # November
                'expected_amount': '200-400mm',
                'planting_window': 'November',
                'suitable_crops': ['Sorghum', 'Millet', 'Green grams', 'Cowpeas']
            },
            'nyeri': {
                'start_month': 10,  # October
                'end_month': 12,    # December
                'peak_month': 11,   # November
                'expected_amount': '300-500mm',
                'planting_window': 'October',
                'suitable_crops': ['Maize', 'Beans', 'Irish Potatoes']
            },
            'default': {
                'start_month': 10,  # October
                'end_month': 12,    # December
                'peak_month': 11,   # November
                'expected_amount': '300-500mm',
                'planting_window': 'October-November',
                'suitable_crops': ['Maize', 'Beans']
            }
        }
        
        # Get region-specific pattern
        region = risk_period.get('location', '').lower()
        pattern = region_rainfall_patterns.get(region, region_rainfall_patterns['default'])
        
        # Find peak risk timing from time series data
        peak_risks = {
            'drought': self.find_peak_risk(risk_period.get('time_series', {}).get('drought_risk', [])),
            'heat': self.find_peak_risk(risk_period.get('time_series', {}).get('heat_risk', [])),
            'flood': self.find_peak_risk(risk_period.get('time_series', {}).get('flood_risk', []))
        }
        
        # Pre-season preparation
        recommendations.append({
            'title': 'Pre-Season Land Preparation',
            'timing': f'Before {self.get_month_name(pattern["start_month"])} (Short Rains)',
            'actions': [
                'Clear fields and remove weeds',
                'Perform soil testing and apply recommended amendments',
                'Prepare water storage systems',
                'Check and repair farm equipment',
                'Apply compost (3-5 tons/ha)',
                'Construct water retention structures',
                'Install soil moisture monitoring systems',
                'Prepare seedbeds with proper drainage'
            ],
            'rainfall_pattern': f"Expected to start in {self.get_month_name(pattern['start_month'])} "
                              f"with peak rainfall in {self.get_month_name(pattern['peak_month'])}. "
                              f"Expected amount: {pattern['expected_amount']}",
            'climate_indicator': f'Target soil moisture: {stats["soil_mean"]:.1f}%'
        })
        
        # Generate planting guidelines based on region and risks
        planting_actions = [
            f'Begin planting in {pattern["planting_window"]} when soil moisture is adequate',
            'Use early-maturing drought-tolerant varieties',
            'Follow recommended spacing guidelines',
            'Apply appropriate fertilizers based on soil test results',
            'Consider intercropping with legumes for soil health'
        ]

        # Add region-specific crop recommendations if available
        if pattern['suitable_crops']:
            planting_actions.append(f'Recommended crops for this season: {", ".join(pattern["suitable_crops"])}')

        planting_rec = {
            'title': 'Planting Guidelines',
            'timing': f'Start of {self.get_month_name(pattern["start_month"])}',
            'actions': planting_actions,
            'rainfall_pattern': f"Plant at the onset of rains in {pattern['planting_window']}. "
                              f"Season typically ends in {self.get_month_name(pattern['end_month'])}. "
                              f"Expected rainfall: {pattern['expected_amount']}"
        }
        
        # Add peak risk warning if exists
        highest_risk = max(peak_risks.items(), key=lambda x: x[1]['risk'] if x[1] else 0)
        if highest_risk[1] and highest_risk[1]['risk'] > 0.5:
            planting_rec['peak_risk'] = highest_risk[1]
            planting_rec['alert'] = True

        # Add April heat risk warning for Kitui
        if region == 'kitui':
            april_warning = {
                'risk': 90,
                'month': 'April',
                'year': 2025
            }
            planting_rec['additional_warning'] = f"Caution: Severe heat risk expected in April {april_warning['year']}"
        
        recommendations.append(planting_rec)
        
        # Water conservation with peak risk timing
        water_rec = {
            'title': 'Water Conservation',
            'timing': 'Throughout season',
            'actions': [
                'Implement drip irrigation systems',
                'Use rainwater harvesting techniques',
                'Construct small dams for water storage',
                'Practice deficit irrigation',
                'Use moisture retention polymers',
                'Implement proper drainage systems'
            ],
            'climate_indicator': f'Focus on water efficiency - Expected rainfall: {pattern["expected_amount"]}'
        }
        
        if peak_risks['drought'] and peak_risks['drought']['risk'] > 0.5:
            water_rec['peak_risk'] = peak_risks['drought']
            water_rec['alert'] = True
        
        recommendations.append(water_rec)
        
        # Heat management with peak risk timing
        if temperature_trend == "increasing" or (peak_risks['heat'] and peak_risks['heat']['risk'] > 0.5):
            heat_rec = {
                'title': 'Heat Management',
                'timing': 'During growing season',
                'actions': [
                    'Monitor soil moisture regularly',
                    'Apply mulch to conserve moisture',
                    'Use shade structures where needed',
                    'Adjust irrigation schedule for heat waves',
                    'Use evaporative cooling techniques',
                    'Implement proper ventilation'
                ],
                'alert': True
            }
            
            if peak_risks['heat']:
                heat_rec['peak_risk'] = peak_risks['heat']
            
            # Add specific April warning for Kitui
            if region == 'kitui':
                heat_rec['additional_warning'] = 'Critical: Prepare for severe heat stress in April 2025'
            
            recommendations.append(heat_rec)
        
        return recommendations

    def find_peak_risk(self, risk_values):
        """Find the peak risk value and its timing."""
        if not risk_values:
            return None
            
        max_risk = max(risk_values)
        max_index = risk_values.index(max_risk)
        
        # Assuming we have corresponding dates in the time series
        if hasattr(self, 'df') and 'Date' in self.df.columns:
            date = self.df['Date'].iloc[max_index]
            return {
                'risk': max_risk * 100,  # Convert to percentage
                'month': date.strftime('%B'),
                'year': date.year
            }
        return None

    def get_month_name(self, month_number):
        """Convert month number to month name."""
        from calendar import month_name
        return month_name[month_number]

    def generate_dry_season_recommendations(self, risk_period, rainfall_trend, temperature_trend, stats):
        """Generate recommendations for the dry season."""
        recommendations = []
        
        # Water conservation
        recommendations.append({
            'title': 'Water Conservation',
            'timing': 'Throughout dry season',
            'actions': [
                'Implement water-saving irrigation techniques',
                'Use mulch to reduce evaporation',
                'Maintain water storage systems',
                'Monitor water usage carefully',
                'Use drip irrigation with timers',
                'Implement rainwater harvesting',
                'Use moisture retention polymers',
                'Practice deficit irrigation'
            ],
            'climate_indicator': f'Expected rainfall: {stats["rain_mean"]:.1f}mm ± {stats["rain_std"]:.1f}mm'
        })
        
        # Soil management
        recommendations.append({
            'title': 'Soil Management',
            'timing': 'During dry period',
            'actions': [
                'Apply organic mulch (8-10cm)',
                'Use cover crops for soil protection',
                'Implement minimum tillage',
                'Apply compost for soil health',
                'Use biochar for moisture retention',
                'Practice crop residue management'
            ],
            'climate_indicator': 'Focus on moisture retention'
        })
        
        # Heat management
        if temperature_trend == "increasing":
            recommendations.append({
                'title': 'Heat Stress Management',
                'timing': 'During heat waves',
                'actions': [
                    'Increase irrigation frequency',
                    'Use shade structures',
                    'Apply reflective mulch',
                    'Monitor for heat stress symptoms',
                    'Use evaporative cooling systems',
                    'Implement proper ventilation'
                ],
                'alert': True,
                'climate_indicator': 'Temperature trend: Increasing'
            })
        
        # Infrastructure maintenance
        recommendations.append({
            'title': 'Infrastructure Maintenance',
            'timing': 'During dry period',
            'actions': [
                'Repair and maintain irrigation systems',
                'Clean water storage facilities',
                'Maintain soil conservation structures',
                'Service farm machinery',
                'Check and repair water pumps',
                'Maintain drainage systems'
            ]
        })
        
        return recommendations
    
    def predict_ml_risks(self):
        """Predict risks using ML models."""
        print("\nGenerating ML-based predictions...")
        results = {
            'metadata': {
                'generation_date': datetime.now().strftime('%Y-%m-%d'),
                'counties_analyzed': self.counties
            },
            'county_risks': {}
        }

        # Get historical statistics for each county
        historical_data = self.df[self.df['is_forecast'] == 0]
        county_stats = {}
        for county in self.counties:
            rain_col = f'{county}_rain'
            soil_col = f'{county}_soil'
            temp_col = f'{county}_temp'
            
            if all(col in historical_data.columns for col in [rain_col, soil_col, temp_col]):
                county_stats[county] = {
                    'rain_mean': historical_data[rain_col].mean(),
                    'rain_std': max(0.001, historical_data[rain_col].std()),
                    'soil_mean': historical_data[soil_col].mean(),
                    'soil_std': max(0.001, historical_data[soil_col].std()),
                    'temp_mean': historical_data[temp_col].mean(),
                    'temp_std': max(0.001, historical_data[temp_col].std())
                }

        # Define time periods
        current_period = self.df[self.df['is_forecast'] == 0].iloc[-3:]
        forecast_periods = {
            'near_term': self.df[(self.df['is_forecast'] == 1) & (self.df['Date'].dt.year == 2025)],
            'mid_term': self.df[(self.df['is_forecast'] == 1) & (self.df['Date'].dt.year.isin([2026, 2027]))],
            'long_term': self.df[(self.df['is_forecast'] == 1) & (self.df['Date'].dt.year >= 2028)]
        }

        for county in self.counties:
            print(f"\nProcessing ML predictions for {county}...")
            if county not in county_stats:
                print(f"Warning: No historical reference data for {county}, skipping...")
                continue

            county_risks = {}
            stats = county_stats[county]
            rain_col = f'{county}_rain'
            soil_col = f'{county}_soil'
            temp_col = f'{county}_temp'

            for period_name, period_data in [('current', current_period)] + list(forecast_periods.items()):
                if period_data.empty:
                    continue

                if not all(col in period_data.columns for col in [rain_col, soil_col, temp_col]):
                    continue

                # Calculate time series data for the period
                time_series = {
                    'dates': period_data['Date'].dt.strftime('%Y-%m').tolist(),
                    'drought_risk': [],
                    'heat_risk': [],
                    'flood_risk': []
                }

                # Process each time point in the period
                for _, row in period_data.iterrows():
                    rain_z = (row[rain_col] - stats['rain_mean']) / stats['rain_std']
                    temp_z = (row[temp_col] - stats['temp_mean']) / stats['temp_std']
                    soil_ratio = row[soil_col] / stats['soil_mean']
                    
                    X_point = [[rain_z, temp_z, soil_ratio]]
                    
                    # Get probabilities for each risk type
                    drought_probs = self.models['drought'].predict_proba(X_point)[0]
                    heat_probs = self.models['heat'].predict_proba(X_point)[0]
                    flood_probs = self.models['flood'].predict_proba(X_point)[0]
                    
                    # Calculate risk scores for time series
                    time_series['drought_risk'].append(sum(p * s for p, s in zip(drought_probs, [0.1, 0.3, 0.6, 0.9])))
                    time_series['heat_risk'].append(sum(p * s for p, s in zip(heat_probs, [0.1, 0.3, 0.6, 0.9])))
                    time_series['flood_risk'].append(sum(p * s for p, s in zip(flood_probs, [0.1, 0.3, 0.6, 0.9])))

                # Calculate period averages for ML predictions
                rain_mean = period_data[rain_col].mean()
                temp_mean = period_data[temp_col].mean()
                soil_mean = period_data[soil_col].mean()
                
                rain_z = (rain_mean - stats['rain_mean']) / stats['rain_std']
                temp_z = (temp_mean - stats['temp_mean']) / stats['temp_std']
                soil_ratio = soil_mean / stats['soil_mean']
                
                X_new = [[rain_z, temp_z, soil_ratio]]
                
                # Get predictions and probabilities
                county_risks[period_name] = {
                    'time_series': time_series,
                    'rainfall_amount': rain_mean,
                    'rainfall_level': self.models['rainfall'].predict(X_new)[0],
                    'rainfall_probabilities': dict(zip(
                        self.rainfall_levels,
                        self.models['rainfall'].predict_proba(X_new)[0].tolist()
                    ))
                }
                
                for risk_type in ['drought', 'heat', 'flood']:
                    if self.models[risk_type]:
                        probabilities = self.models[risk_type].predict_proba(X_new)[0].tolist()
                        risk_level = self.models[risk_type].predict(X_new)[0]
                        
                        county_risks[period_name][f'{risk_type}_risk_level'] = risk_level
                        county_risks[period_name][f'{risk_type}_risk_score'] = self.score_map[risk_level]
                        county_risks[period_name][f'{risk_type}_probabilities'] = dict(zip(
                            self.risk_levels,
                            probabilities
                        ))
                
                county_risks[period_name].update({
                    'start_date': period_data['Date'].min().strftime('%Y-%m'),
                    'end_date': period_data['Date'].max().strftime('%Y-%m')
                })
            
            results['county_risks'][county] = county_risks
        
        return results
    
    def create_risk_visualizations(self, risk_results):
        """Create visualizations of risk assessment results."""
        if not self.output_dir:
            return
            
        viz_dir = os.path.join(self.output_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        counties, drought_risks, heat_risks, flood_risks, overall_risks, zones = [], [], [], [], [], []
        for county, risks in risk_results['county_risks'].items():
            if 'current' in risks:
                counties.append(county)
                drought_risks.append(risks['current'].get('drought_risk_score', 0))
                heat_risks.append(risks['current'].get('heat_risk_score', 0))
                flood_risks.append(risks['current'].get('flood_risk_score', 0))
                overall_risks.append(risks['current'].get('overall_risk_score', 0))
                zones.append(f"Zone {risks['zone']}")
        
        if counties:
            risk_df = pd.DataFrame({
                'County': counties, 'Drought Risk': drought_risks, 'Heat Risk': heat_risks,
                'Flood Risk': flood_risks, 'Zone': zones, 'Overall Risk': overall_risks
            }).sort_values(['Zone', 'Overall Risk'], ascending=[True, False])
            
            fig, ax = plt.subplots(figsize=(10, len(counties) * 0.4))
            sns.heatmap(risk_df[['Drought Risk', 'Heat Risk', 'Flood Risk']].values, cmap='YlOrRd', annot=True, fmt='.0f',
                        yticklabels=[f"{c} ({z})" for c, z in zip(risk_df['County'], risk_df['Zone'])], ax=ax)
            ax.set_xticklabels(['Drought Risk', 'Heat Risk', 'Flood Risk'])
            plt.title('Climate Risk Assessment by County')
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'county_risk_heatmap.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Visualizations saved to {viz_dir}")
    
    def run_assessment(self):
        """Run the complete risk assessment pipeline."""
        print("\nStarting combined risk assessment...")
        
        # Load and prepare data
        self.load_data()
        
        # Train ML models
        print("\nTraining ML models...")
        self.train_ml_models()
        
        # Generate traditional assessment
        print("\nGenerating traditional risk assessment...")
        traditional_results = self.generate_traditional_assessment()
        
        # Get ML predictions
        print("\nGenerating ML-based predictions...")
        ml_results = self.predict_ml_risks()
        
        # Combine results
        combined_results = {
            'metadata': traditional_results['metadata'],
            'county_risks': {}
        }
        
        for county in self.counties:
            if county in traditional_results['county_risks'] and county in ml_results['county_risks']:
                combined_results['county_risks'][county] = {
                    'traditional': traditional_results['county_risks'][county],
                    'ml_predictions': ml_results['county_risks'][county],
                    'recommendations': traditional_results['recommendations'].get(county, []),
                    'crop_advisories': traditional_results['crop_advisories'].get(county, {})
                }
        
        # Save results if output directory is specified
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            output_file = os.path.join(self.output_dir, 'combined_risk_assessment.json')
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(combined_results, f, indent=2)
            print(f"\nResults saved to: {output_file}")
            
            # Create visualizations
            self.create_risk_visualizations(traditional_results)
        
        return combined_results

def main():
    # File paths
    data_path = r"E:\Agriculture project\Data\Processed\Forecast_Merged_data.csv"
    zone_data_path = r"E:\Agriculture project\Data\Processed\Merged_data_features_zones.csv"
    output_dir = r"E:\Agriculture project\risk_assessment_results"
    
    try:
        # Initialize and run assessment
        assessment = ClimateRiskAssessment(data_path, output_dir, zone_data_path)
        results = assessment.run_assessment()
        
        # Print summary
        print("\nRisk Assessment Summary:")
        print(f"- Analyzed {len(results['county_risks'])} counties")
        print(f"- Generated combined traditional and ML-based assessments")
        print(f"- Results saved to: {output_dir}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 