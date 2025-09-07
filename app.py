from flask import Flask, request, render_template, session
import numpy as np
import pandas
import sklearn
import pickle
import datetime
from collections import deque

# importing model
model = pickle.load(open('model.pkl','rb'))
sc = pickle.load(open('standscaler.pkl','rb'))
ms = pickle.load(open('minmaxscaler.pkl','rb'))

# creating flask app
app = Flask(__name__, static_folder='templates/static', static_url_path='/static')
app.secret_key = 'crop_recommendation_secret_key'

# Store recent searches (using deque for efficient fixed-size list)
search_history = deque(maxlen=5)

# Define crop seasons mapping
crop_seasons = {
    "Rice": ["summer", "monsoon"],
    "Maize": ["summer", "monsoon"],
    "Jute": ["monsoon"],
    "Cotton": ["monsoon", "autumn"],
    "Coconut": ["all"],
    "Papaya": ["all"],
    "Orange": ["winter", "autumn"],
    "Apple": ["autumn", "winter"],
    "Muskmelon": ["summer"],
    "Watermelon": ["summer"],
    "Grapes": ["winter", "spring"],
    "Mango": ["summer"],
    "Banana": ["all"],
    "Pomegranate": ["monsoon", "autumn"],
    "Lentil": ["winter"],
    "Blackgram": ["monsoon", "autumn"],
    "Mungbean": ["monsoon", "summer"],
    "Mothbeans": ["monsoon"],
    "Pigeonpeas": ["monsoon"],
    "Kidneybeans": ["monsoon"],
    "Chickpea": ["winter"],
    "Coffee": ["monsoon"]
}

@app.route('/')
def index():
    return render_template("index.html", search_history=list(search_history))

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")

@app.route('/submit_contact', methods=['POST'])
def submit_contact():
    # This would typically save the contact form data to a database
    # For now, we'll just redirect back to the contact page with a success message
    return render_template('contact.html')

@app.route("/predict", methods=['POST'])
def predict():
    N = request.form['Nitrogen']
    P = request.form['Phosporus']
    K = request.form['Potassium']
    temp = request.form['Temperature']
    humidity = request.form['Humidity']
    ph = request.form['Ph']
    rainfall = request.form['Rainfall']

    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    scaled_features = ms.transform(single_pred)
    final_features = sc.transform(scaled_features)
    prediction = model.predict(final_features)

    crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                 14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                 19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        crop_name = crop.lower()  # Convert to lowercase for filename matching
        
        # Determine suitable seasons for the recommended crop
        suitable_seasons = crop_seasons.get(crop, [])
        if "all" in suitable_seasons:
            season_text = "all seasons"
        else:
            season_text = ", ".join(suitable_seasons)
            
        result = "{} is the best crop to be cultivated right there. It grows well in {}".format(crop, season_text)
        
        # Find other crops that can grow in the same seasons
        similar_season_crops = []
        for season in suitable_seasons:
            if season == "all":
                # Find all crops that can grow in all seasons
                for c, seasons in crop_seasons.items():
                    if c != crop and "all" in seasons and c not in similar_season_crops:
                        similar_season_crops.append(c)
            else:
                # Find crops that can grow in this specific season
                for c, seasons in crop_seasons.items():
                    if c != crop and (season in seasons or "all" in seasons) and c not in similar_season_crops:
                        similar_season_crops.append(c)
    else:
        result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
        crop_name = None
        suitable_seasons = []
        similar_season_crops = []
    
    # Store search data with timestamp
    search_data = {
        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'inputs': {
            'Nitrogen': N,
            'Phosphorus': P,
            'Potassium': K,
            'Temperature': temp,
            'Humidity': humidity,
            'pH': ph,
            'Rainfall': rainfall
        },
        'result': result,
        'crop_name': crop_name,
        'suitable_seasons': suitable_seasons,
        'similar_season_crops': similar_season_crops
    }
    
    # Add to search history
    search_history.appendleft(search_data)
    
    # Determine current season for display
    current_season = "all"
    if suitable_seasons and suitable_seasons[0] != "all":
        current_season = suitable_seasons[0]
    
    return render_template('index.html', 
                           result=result, 
                           crop_name=crop_name, 
                           search_history=list(search_history), 
                           current_inputs=search_data['inputs'],
                           suitable_seasons=suitable_seasons,
                           similar_season_crops=similar_season_crops,
                           crop_seasons=crop_seasons,
                           current_season=current_season)




@app.route('/reset', methods=['POST'])
def reset():
    # Clear form data
    return render_template('index.html', search_history=list(search_history))

# python main
if __name__ == "__main__":
    app.run(debug=True)