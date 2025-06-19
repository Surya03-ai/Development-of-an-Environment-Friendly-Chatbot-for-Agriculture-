from flask import Flask, request, jsonify, render_template, redirect, url_for
import google.generativeai as genai
from flask_cors import CORS
import os
import traceback
from datetime import datetime
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# Configure Gemini AI with API key
API_KEY = "AIzaSyDqgRQe8nu1lJry7NI0MgF21WSdRSOLEmw"
genai.configure(api_key=API_KEY)

# Load crop recommendation model
try:
    crop_model = joblib.load("models/crop_recommender.pkl")
    print("Crop recommendation model loaded successfully!")
except Exception as e:
    print(f"Error loading crop model: {str(e)}")
    crop_model = None

# Multi-language support dictionary
LANGUAGES = {
    "English": {
        "land_types": ["Clay Soil", "Sandy Soil", "Loamy Soil", "Silt Soil", "Black Soil"],
        "seasons": ["Kharif (Monsoon)", "Rabi (Winter)", "Zaid (Summer)"],
        "crops": ["Rice", "Wheat", "Cotton", "Sugarcane", "Pulses", "Vegetables", "Fruits", "Oil Seeds"]
    },
    "Hindi": {
        "land_types": ["मिट्टी की मिट्टी", "रेतीली मिट्टी", "दोमट मिट्टी", "गाद मिट्टी", "काली मिट्टी"],
        "seasons": ["खरीफ (मानसून)", "रबी (सर्दी)", "जायद (गर्मी)"],
        "crops": ["चावल", "गेहूं", "कपास", "गन्ना", "दालें", "सब्जियां", "फल", "तिलहन"]
    },
    "Telugu": {
        "land_types": ["బంక మట్టి", "ఇసుక నేల", "లోమి నేల", "బురద నేల", "నల్ల నేల"],
        "seasons": ["ఖరీఫ్ (వర్షాకాలం)", "రబీ (శీతాకాలం)", "జైద్ (వేసవి)"],
        "crops": ["వరి", "గోధుమ", "పత్తి", "చెరకు", "పప్పు ధాన్యాలు", "కూరగాయలు", "పండ్లు", "నూనె గింజలు"]
    }
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

@app.route('/crop-recommendation')
def crop_recommendation():
    return render_template('crop_recommend.html')

@app.route('/api/get_options/<language>')
def get_options(language):
    if language in LANGUAGES:
        return jsonify(LANGUAGES[language])
    return jsonify({"error": "Language not supported"}), 400

@app.route('/api/generate_solution', methods=['POST'])
def generate_solution():
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        
        data = request.json
        
        required_fields = ['land_type', 'season', 'crop_type', 'acres', 'problem']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        language = data.get('language', 'English')
        model_name = 'models/gemini-1.5-pro'
        model = genai.GenerativeModel(model_name)
        
        prompt = f"""
        As an agricultural expert, provide a detailed solution in {language} for the following farming situation:
        
        Land Type: {data['land_type']}
        Season: {data['season']}
        Crop Type: {data['crop_type']}
        Land Area: {data['acres']} acres
        Problem Description: {data['problem']}
        
        Please provide:
        1. Problem analysis
        2. Recommended solutions
        3. Preventive measures for the future
        4. Additional tips specific to the land type, crop, and season
        """
        
        response = model.generate_content(prompt)
        
        os.makedirs('solutions', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"solutions/farm_solution_{timestamp}.txt"
        
        with open(filename, "w", encoding="utf-8") as f:
            f.write("FARM PROBLEM DETAILS\n")
            f.write("-------------------\n\n")
            for key, value in data.items():
                f.write(f"{key.title()}: {value}\n")
            f.write("\nRECOMMENDED SOLUTION\n")
            f.write("-------------------\n\n")
            f.write(response.text)
        
        return jsonify({
            "solution": response.text,
            "filename": filename
        })
        
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error in generate_solution: {str(e)}")
        return jsonify({"error": str(e), "traceback": error_details}), 500

@app.route('/api/recommend-crop', methods=['POST'])
def recommend_crop():
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        
        data = request.json
        
        required_features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        for feature in required_features:
            if feature not in data:
                return jsonify({"error": f"Missing required feature: {feature}"}), 400
        
        input_data = np.array([
            data['N'],
            data['P'],
            data['K'],
            data['temperature'],
            data['humidity'],
            data['ph'],
            data['rainfall']
        ]).reshape(1, -1)
        
        prediction = crop_model.predict(input_data)[0]
        probabilities = crop_model.predict_proba(input_data)[0]
        confidence = round(np.max(probabilities) * 100, 2)
        
        # Get top 3 crops
        top3_idx = np.argsort(probabilities)[-3:][::-1]
        top3_crops = crop_model.classes_[top3_idx]
        top3_conf = [round(probabilities[i]*100, 2) for i in top3_idx]
        
        return jsonify({
            "recommended_crop": prediction,
            "confidence": confidence,
            "top_recommendations": [
                {"crop": crop, "confidence": conf} 
                for crop, conf in zip(top3_crops, top3_conf)
            ]
        })
        
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error in recommend_crop: {str(e)}")
        return jsonify({"error": str(e), "traceback": error_details}), 500

if __name__ == '__main__':
    app.run(debug=True)
