from flask import Flask, jsonify, render_template_string
import folium
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import requests
import os
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)

class WeatherRiskAnalyzer:
    def __init__(self):
        self.weather_risk_weights = {
            'Thunderstorm': 0.9,
            'Tornado': 1.0,
            'Hurricane': 1.0,
            'Tropical Storm': 0.9,
            'Rain': 0.6,
            'Snow': 0.7,
            'Extreme': 0.8,
            'Drizzle': 0.3,
            'Clouds': 0.2,
            'Clear': 0.1
        }
    
    def calculate_risk(self, weather_data):
        """Calculate risk score based on weather conditions"""
        base_risk = 0.1  # base risk score
        
        # risk due to weather
        main_condition = weather_data.get('weather', [{}])[0].get('main', 'Clear')
        weather_risk = self.weather_risk_weights.get(main_condition, 0.1)
        
        # risk due to extreme temperatures
        temp = weather_data.get('main', {}).get('temp', 0)
        temp_risk = 0
        if temp > 35 or temp < -10:  # extreme risk
            temp_risk = 0.3
        elif temp > 30 or temp < 0:  # moderate risk
            temp_risk = 0.2
        
        # risk due to wind speed
        wind_speed = weather_data.get('wind', {}).get('speed', 0)
        wind_risk = min(wind_speed / 100, 0.4)
        
        # risk due to precipitation
        rain = weather_data.get('rain', {}).get('1h', 0)
        rain_risk = min(rain / 50, 0.3)  # volume risk
        
        # final calculation
        total_risk = min(0.95, base_risk + weather_risk + temp_risk + wind_risk + rain_risk)
        
        return total_risk

class DisasterResponseSystem:
    def __init__(self):
        self.weather_data = {}
        self.risk_analyzer = WeatherRiskAnalyzer()
        self.api_key = os.getenv('OPENWEATHER_API_KEY', 'your-api-key-here')
        
        self.monitored_locations = [
            {"name": "New York", "lat": 40.7128, "lon": -74.0060},
            {"name": "Los Angeles", "lat": 34.0522, "lon": -118.2437},
            {"name": "Chicago", "lat": 41.8781, "lon": -87.6298},
            {"name": "Houston", "lat": 29.7604, "lon": -95.3698},
            {"name": "Phoenix", "lat": 33.4484, "lon": -112.0740},
            {"name": "Miami", "lat": 25.7617, "lon": -80.1918},
            {"name": "Seattle", "lat": 47.6062, "lon": -122.3321},
            {"name": "Denver", "lat": 39.7392, "lon": -104.9903},
            {"name": "New Orleans", "lat": 29.9511, "lon": -90.0715},
            {"name": "Kansas City", "lat": 39.0997, "lon": -94.5786}
        ]

    def fetch_weather_data(self, location):
        """Fetch weather data for a specific location"""
        base_url = "http://api.openweathermap.org/data/2.5/weather"
        params = {
            "lat": location["lat"],
            "lon": location["lon"],
            "appid": self.api_key,
            "units": "metric"
        }
        
        try:
            response = requests.get(base_url, params=params)
            if response.status_code == 200:
                weather_data = response.json()
                risk_score = self.risk_analyzer.calculate_risk(weather_data)
                
                return {
                    "location": location["name"],
                    "latitude": location["lat"],
                    "longitude": location["lon"],
                    "weather_data": weather_data,
                    "risk_score": risk_score,
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            print(f"Error fetching weather data for {location['name']}: {e}")
            return None

    def update_all_locations(self):
        """Update weather data for all monitored locations"""
        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(self.fetch_weather_data, self.monitored_locations))
            
        valid_results = [r for r in results if r is not None] # filter out none results
        self.weather_data = {result["location"]: result for result in valid_results}
        return valid_results

    def create_heatmap(self):
        """Create a Folium map with weather-based risk heatmap"""
        m = folium.Map(location=[39.8283, -98.5795], zoom_start=4) # map center
        self.update_all_locations()
        
        # add marker for each location
        for location_data in self.weather_data.values():
            weather_info = location_data["weather_data"]
            risk_score = location_data["risk_score"]
            popup_html = f"""
                <div style='width: 200px'>
                    <h4>{location_data['location']}</h4>
                    <p>Temperature: {weather_info['main']['temp']}°C</p>
                    <p>Conditions: {weather_info['weather'][0]['main']}</p>
                    <p>Wind Speed: {weather_info['wind']['speed']} m/s</p>
                    <p>Humidity: {weather_info['main']['humidity']}%</p>
                    <p>Risk Score: {risk_score:.2f}</p>
                </div>
            """
            folium.CircleMarker(
                location=[location_data['latitude'], location_data['longitude']],
                radius=20,
                popup=folium.Popup(popup_html, max_width=300),
                color=self.get_risk_color(risk_score),
                fill=True,
                fill_color=self.get_risk_color(risk_score)
            ).add_to(m)
            
        return m._repr_html_()
    
    def get_risk_color(self, risk_score):
        """Return color based on risk score"""
        if risk_score < 0.3:
            return 'green'
        elif risk_score < 0.7:
            return 'orange'
        else:
            return 'red'

drs = DisasterResponseSystem() # initialization

@app.route('/')
def dashboard():
    """Render the main dashboard with interactive map"""
    heatmap = drs.create_heatmap()
    
    dashboard_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Disaster Response Dashboard</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/tailwindcss/2.2.19/tailwind.min.css" rel="stylesheet">
        <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.0/jquery.min.js"></script>
    </head>
    <body class="bg-gray-100">
        <div class="container mx-auto px-4 py-8">
            <header class="bg-white shadow rounded-lg p-6 mb-8">
                <h1 class="text-3xl font-bold text-gray-800">AI-Augmented Disaster Response System</h1>
                <p class="text-gray-600">Real-time weather monitoring and risk assessment</p>
            </header>
            
         <div class="grid grid-cols-1 lg:grid-cols-3 gap-8">
            <!-- Map Section -->
            <div class="lg:col-span-2 bg-white shadow rounded-lg p-6">
               <h2 class="text-xl font-semibold mb-4">Weather Risk Heatmap</h2>
               <div id="map" class="h-[600px] w-full">
                     {{ heatmap|safe }}
               </div>
            </div>
            
            <!-- Weather Alerts Section -->
            <div class="bg-white shadow rounded-lg p-6 h-[600px] overflow-hidden">
               <h2 class="text-xl font-semibold mb-4">Weather Alerts</h2>
               <div id="alerts" class="space-y-4 h-[500px] overflow-y-auto">
               </div>
            </div>
         </div>
        </div>
        
        <script>
            function updateAlerts() {
                $.getJSON('/api/alerts', function(data) {
                    const alertsContainer = $('#alerts');
                    alertsContainer.empty();
                    
                    data.forEach(alert => {
                        const alertClass = alert.risk_score >= 0.7 ? 'bg-red-100' :
                                         alert.risk_score >= 0.3 ? 'bg-yellow-100' : 'bg-green-100';
                        
                        const alertHtml = `
                            <div class="${alertClass} p-4 rounded-lg">
                                <h3 class="font-semibold">${alert.location}</h3>
                                <p>Weather: ${alert.weather_data.weather[0].main}</p>
                                <p>Temperature: ${alert.weather_data.main.temp}°C</p>
                                <p>Risk Score: ${(alert.risk_score * 100).toFixed(1)}%</p>
                                <p class="text-sm text-gray-600">${new Date(alert.timestamp).toLocaleString()}</p>
                            </div>
                        `;
                        alertsContainer.append(alertHtml);
                    });
                });
            }
            
            // Update alerts every 30 seconds
            updateAlerts();
            setInterval(updateAlerts, 30000);
        </script>
    </body>
    </html>
    """
    return render_template_string(dashboard_html, heatmap=heatmap)

@app.route('/api/alerts')
def get_alerts():
    """API endpoint to get current weather-based alerts"""
    return jsonify(list(drs.weather_data.values()))

if __name__ == '__main__':
    app.run(debug=True)
