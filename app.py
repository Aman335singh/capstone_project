import os
import pickle
import random
import string
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from functools import wraps

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

# Initialize Flask application
app = Flask(__name__)
app.config['SECRET_KEY'] = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(32))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///agriculture_app.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAIL_SERVER'] = 'smtp.gmail.com'  # Change according to your email provider
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'sahpranav025@gmail.com'  # Replace with your email
app.config['MAIL_PASSWORD'] = 'weev uore txue hkpd'  # Replace with your app password
app.config['OTP_VALIDITY'] = 10  # OTP validity in minutes

# Initialize database
db = SQLAlchemy(app)


# Define User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    is_verified = db.Column(db.Boolean, default=False)
    otp = db.Column(db.String(6), nullable=True)
    otp_expiry = db.Column(db.DateTime, nullable=True)

    def __repr__(self):
        return f'<User {self.username}>'


# Authentication helper functions
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please login to access this page', 'danger')
            return redirect(url_for('login'))
        return f(*args, **kwargs)

    return decorated_function


def generate_otp():
    return ''.join(random.choices(string.digits, k=6))


def send_otp_email(email, otp):
    try:
        msg = MIMEMultipart()
        msg['From'] = app.config['MAIL_USERNAME']
        msg['To'] = email
        msg['Subject'] = 'Email Verification OTP'

        body = f"""
        <html>
        <body>
            <h2>Email Verification</h2>
            <p>Your OTP for email verification is: <strong>{otp}</strong></p>
            <p>This OTP is valid for {app.config['OTP_VALIDITY']} minutes.</p>
            <p>If you did not request this verification, please ignore this email.</p>
        </body>
        </html>
        """

        msg.attach(MIMEText(body, 'html'))

        server = smtplib.SMTP(app.config['MAIL_SERVER'], app.config['MAIL_PORT'])
        server.starttls()
        server.login(app.config['MAIL_USERNAME'], app.config['MAIL_PASSWORD'])
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        print(f"Error sending email: {e}")
        return False


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        # Form validation
        if not username or not email or not password or not confirm_password:
            flash('All fields are required', 'danger')
            return redirect(url_for('register'))

        if password != confirm_password:
            flash('Passwords do not match', 'danger')
            return redirect(url_for('register'))

        # Check if user exists - check username and email separately for better error messages
        existing_username = User.query.filter_by(username=username).first()
        if existing_username:
            flash('Username already exists. Please choose a different username.', 'danger')
            return redirect(url_for('register'))

        existing_email = User.query.filter_by(email=email).first()
        if existing_email:
            flash('Email already exists. Please use a different email or try to log in.', 'danger')
            return redirect(url_for('register'))

        try:
            # Create new user
            hashed_password = generate_password_hash(password)
            otp = generate_otp()
            otp_expiry = datetime.now() + timedelta(minutes=app.config['OTP_VALIDITY'])

            new_user = User(
                username=username,
                email=email,
                password=hashed_password,
                otp=otp,
                otp_expiry=otp_expiry
            )

            # Send OTP email
            if send_otp_email(email, otp):
                db.session.add(new_user)
                db.session.commit()
                flash('Registration successful! Please verify your email with the OTP sent to your email address.',
                      'success')
                return redirect(url_for('verify_email', email=email))
            else:
                flash('Failed to send verification email. Please try again.', 'danger')
                return redirect(url_for('register'))

        except Exception as e:
            db.session.rollback()
            flash(f'Registration failed: {str(e)}', 'danger')
            return redirect(url_for('register'))

    return render_template('register.html')

@app.route('/verify-email/<email>', methods=['GET', 'POST'])
def verify_email(email):
    if request.method == 'POST':
        otp = request.form.get('otp')

        if not otp:
            flash('OTP is required', 'danger')
            return redirect(url_for('verify_email', email=email))

        user = User.query.filter_by(email=email).first()
        if not user:
            flash('User not found', 'danger')
            return redirect(url_for('register'))

        if user.otp != otp:
            flash('Invalid OTP', 'danger')
            return redirect(url_for('verify_email', email=email))

        if datetime.now() > user.otp_expiry:
            flash('OTP has expired', 'danger')
            return redirect(url_for('verify_email', email=email))

        # Verify user
        user.is_verified = True
        user.otp = None
        user.otp_expiry = None
        db.session.commit()

        flash('Email verification successful! You can now login.', 'success')
        return redirect(url_for('login'))

    return render_template('verify_email.html', email=email)


@app.route('/resend-otp/<email>')
def resend_otp(email):
    user = User.query.filter_by(email=email).first()

    if not user:
        flash('User not found', 'danger')
        return redirect(url_for('register'))

    otp = generate_otp()
    otp_expiry = datetime.now() + timedelta(minutes=app.config['OTP_VALIDITY'])

    user.otp = otp
    user.otp_expiry = otp_expiry
    db.session.commit()

    if send_otp_email(email, otp):
        flash('OTP has been resent to your email', 'success')
    else:
        flash('Failed to resend OTP. Please try again.', 'danger')

    return redirect(url_for('verify_email', email=email))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username_or_email = request.form.get('username_or_email')
        password = request.form.get('password')

        if not username_or_email or not password:
            flash('All fields are required', 'danger')
            return redirect(url_for('login'))

        # Find user
        user = User.query.filter((User.username == username_or_email) | (User.email == username_or_email)).first()

        if not user:
            flash('Invalid username/email or password', 'danger')
            return redirect(url_for('login'))

        if not check_password_hash(user.password, password):
            flash('Invalid username/email or password', 'danger')
            return redirect(url_for('login'))

        if not user.is_verified:
            flash('Please verify your email before logging in', 'warning')
            return redirect(url_for('verify_email', email=user.email))

        # Login successful
        session['user_id'] = user.id
        session['username'] = user.username

        flash('Login successful!', 'success')
        return redirect(url_for('dashboard'))

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('username', None)
    flash('You have been logged out', 'success')
    return redirect(url_for('login'))


@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')


@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email')

        if not email:
            flash('Email is required', 'danger')
            return redirect(url_for('forgot_password'))

        user = User.query.filter_by(email=email).first()

        if not user:
            flash('No account found with this email', 'danger')
            return redirect(url_for('forgot_password'))

        otp = generate_otp()
        otp_expiry = datetime.now() + timedelta(minutes=app.config['OTP_VALIDITY'])

        user.otp = otp
        user.otp_expiry = otp_expiry
        db.session.commit()

        if send_otp_email(email, otp):
            flash('OTP has been sent to your email for password reset', 'success')
            return redirect(url_for('reset_password', email=email))
        else:
            flash('Failed to send OTP. Please try again.', 'danger')
            return redirect(url_for('forgot_password'))

    return render_template('forgot_password.html')


@app.route('/reset-password/<email>', methods=['GET', 'POST'])
def reset_password(email):
    if request.method == 'POST':
        otp = request.form.get('otp')
        new_password = request.form.get('new_password')
        confirm_password = request.form.get('confirm_password')

        if not otp or not new_password or not confirm_password:
            flash('All fields are required', 'danger')
            return redirect(url_for('reset_password', email=email))

        if new_password != confirm_password:
            flash('Passwords do not match', 'danger')
            return redirect(url_for('reset_password', email=email))

        user = User.query.filter_by(email=email).first()

        if not user:
            flash('User not found', 'danger')
            return redirect(url_for('forgot_password'))

        if user.otp != otp:
            flash('Invalid OTP', 'danger')
            return redirect(url_for('reset_password', email=email))

        if datetime.now() > user.otp_expiry:
            flash('OTP has expired', 'danger')
            return redirect(url_for('reset_password', email=email))

        # Reset password
        user.password = generate_password_hash(new_password)
        user.otp = None
        user.otp_expiry = None
        db.session.commit()

        flash('Password has been reset successfully! You can now login.', 'success')
        return redirect(url_for('login'))

    return render_template('reset_password.html', email=email)


# ----- CROP RECOMMENDATION FUNCTIONALITY -----

def load_crop_models():
    """Load all necessary models and preprocessors for crop recommendation"""
    models = {}

    # Load the best model (assuming Random Forest based on previous results)
    model_path = 'models/random_forest_model.pkl'
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            models['model'] = pickle.load(f)
    else:
        # Fallback to any available model
        model_files = [f for f in os.listdir('models') if f.endswith('_model.pkl')]
        if model_files:
            with open(f'models/{model_files[0]}', 'rb') as f:
                models['model'] = pickle.load(f)
        else:
            raise FileNotFoundError("No model files found in the models directory")

    # Load the label encoder
    with open('models/label_encoder.pkl', 'rb') as f:
        models['label_encoder'] = pickle.load(f)

    # Load the scaler
    with open('models/scaler.pkl', 'rb') as f:
        models['scaler'] = pickle.load(f)

    # Load the region encoder
    with open('models/region_encoder.pkl', 'rb') as f:
        models['region_encoder'] = pickle.load(f)

    return models


def predict_crop(input_data):
    """Make a prediction using the trained crop recommendation model"""
    models = load_crop_models()
    model = models['model']
    label_encoder = models['label_encoder']
    scaler = models['scaler']
    region_encoder = models['region_encoder']

    # Process input data
    input_df = pd.DataFrame([input_data])

    # Extract region and encode it
    region = input_df[['region']]
    region_encoded = region_encoder.transform(region)

    # Drop region from input_df
    features_df = input_df.drop('region', axis=1)

    # Add feature engineering (if used in training)
    if 'NPK_sum' not in features_df.columns and hasattr(model, 'feature_names_in_'):
        feature_names = model.feature_names_in_

        # Check if engineered features were used in training
        if 'NPK_sum' in feature_names:
            features_df['NPK_sum'] = features_df['N'] + features_df['P'] + features_df['K']
        if 'NPK_ratio' in feature_names:
            features_df['NPK_ratio'] = features_df['N'] / (features_df['P'] + features_df['K'] + 1)
        if 'temp_humidity_interaction' in feature_names:
            features_df['temp_humidity_interaction'] = features_df['temperature'] * features_df['humidity'] / 100
        if 'rainfall_per_temp' in feature_names:
            features_df['rainfall_per_temp'] = features_df['rainfall'] / (features_df['temperature'] + 1)
        if 'humidity_squared' in feature_names:
            features_df['humidity_squared'] = features_df['humidity'] ** 2
        if 'ph_squared' in feature_names:
            features_df['ph_squared'] = features_df['ph'] ** 2

    # Get region column names dynamically
    region_columns = [col for col in feature_names if col.startswith('region_')] if hasattr(model,
                                                                                          'feature_names_in_') else []

    # Create region dataframe with appropriate column names
    if region_columns:
        region_df = pd.DataFrame(
            region_encoded,
            columns=region_columns,
            index=features_df.index
        )
    else:
        # If no region column names found, use default naming
        region_df = pd.DataFrame(
            region_encoded,
            columns=[f'region_{i}' for i in range(region_encoded.shape[1])],
            index=features_df.index
        )

    # Combine features
    X = pd.concat([features_df, region_df], axis=1)

    # Ensure we have all required columns
    if hasattr(model, 'feature_names_in_'):
        required_columns = list(model.feature_names_in_)
        for col in required_columns:
            if col not in X.columns:
                X[col] = 0  # Add missing columns with default value

        # Reorder columns to match training data
        X = X[required_columns]

    # Scale features
    X_scaled = scaler.transform(X)

    # Make prediction
    prediction = model.predict(X_scaled)[0]
    predicted_crop = label_encoder.inverse_transform([prediction])[0]

    # Get probability (confidence)
    probabilities = model.predict_proba(X_scaled)[0]
    confidence = probabilities[prediction]

    # Get top 3 recommendations
    top_indices = np.argsort(probabilities)[::-1][:3]
    top_crops = label_encoder.inverse_transform(top_indices)
    top_probabilities = probabilities[top_indices]

    recommendations = []
    for crop, prob in zip(top_crops, top_probabilities):
        recommendations.append({
            'crop': crop,
            'confidence': float(prob)  # Convert numpy float to Python float for JSON serialization
        })

    result = {
        'predicted_crop': predicted_crop,
        'confidence': float(confidence),
        'top_recommendations': recommendations
    }

    return result


@app.route('/crop-recommendation')
@login_required
def crop_recommendation():
    # List of available regions
    regions = ["Bihar", "Chhattisgarh", "Punjab", "Uttar Pradesh"]
    return render_template('crop_recommendation.html', regions=regions)


@app.route('/predict-crop', methods=['POST'])
@login_required
def predict_crop_route():
    try:
        # Get input data from form
        input_data = {
            'N': float(request.form['nitrogen']),
            'P': float(request.form['phosphorus']),
            'K': float(request.form['potassium']),
            'temperature': float(request.form['temperature']),
            'humidity': float(request.form['humidity']),
            'ph': float(request.form['ph']),
            'rainfall': float(request.form['rainfall']),
            'region': request.form['region']
        }

        # Make prediction
        result = predict_crop(input_data)

        return render_template('crop_result.html', result=result, input_data=input_data)

    except Exception as e:
        flash(f"Error: {str(e)}", 'danger')
        return redirect(url_for('crop_recommendation'))


@app.route('/api/predict-crop', methods=['POST'])
@login_required
def api_predict_crop():
    try:
        # Get input data from JSON
        data = request.get_json()

        input_data = {
            'N': float(data['N']),
            'P': float(data['P']),
            'K': float(data['K']),
            'temperature': float(data['temperature']),
            'humidity': float(data['humidity']),
            'ph': float(data['ph']),
            'rainfall': float(data['rainfall']),
            'region': data['region']
        }

        # Make prediction
        result = predict_crop(input_data)

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 400


# ----- COMMODITY PRICE PREDICTION FUNCTIONALITY -----

@app.route('/commodity-prediction')
@login_required
def commodity_prediction():
    # Define common commodities and states based on your dataset
    commodities = ["Rice", "Wheat", "Maize", "Barley", "Ragi", "Jowar", "Bajra"]
    states = ["Andhra Pradesh", "Assam", "Bihar", "Chhattisgarh", "Gujarat",
              "Haryana", "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra",
              "Odisha", "Punjab", "Rajasthan", "Tamil Nadu", "Telangana", "Uttar Pradesh", "West Bengal"]

    # Try to load actual values from the CSV if available
    try:
        data = pd.read_csv('all_commodity_price_with_season.csv')
        commodities = sorted(data['Commodity'].unique())
        states = sorted(data['State'].unique())
    except Exception as e:
        print(f"Could not load commodity data: {e}")

    return render_template('commodity_prediction.html', commodities=commodities, states=states)


@app.route('/predict-price', methods=['POST'])
@login_required
def predict_price():
    # Get input values from the form
    commodity = request.form.get('commodity')
    state = request.form.get('state')
    average = float(request.form.get('average'))
    season = int(request.form.get('season'))

    # Get date
    date_str = request.form.get('date')
    date = datetime.strptime(date_str, '%Y-%m-%d')
    month = date.month
    day = date.day

    # Load base price data if available
    base_price = average
    try:
        data = pd.read_csv('all_commodity_price_with_season.csv')
        # Filter for this commodity and state
        filtered_data = data[(data['Commodity'] == commodity) & (data['State'] == state)]
        if not filtered_data.empty:
            # Use the historical average price as a base
            base_price = filtered_data['Price'].mean()
    except Exception as e:
        print(f"Could not load historical data: {e}")

    # Simple fallback prediction logic (since we can't load the model)
    # This will adjust the base price based on season and month
    season_factors = {1: 0.95, 2: 1.0, 3: 1.05, 4: 1.02}
    month_factors = {
        1: 1.02, 2: 1.03, 3: 1.01, 4: 0.99, 5: 0.98, 6: 0.97,
        7: 0.98, 8: 0.99, 9: 1.01, 10: 1.02, 11: 1.03, 12: 1.04
    }

    # Calculate prediction using the factors
    prediction = base_price * season_factors.get(season, 1.0) * month_factors.get(month, 1.0)

    # Adjust based on the provided average compared to historical data
    prediction = prediction * (average / base_price) if base_price > 0 else average * 1.05

    result = {
        'prediction': round(prediction, 2),
        'commodity': commodity,
        'state': state,
        'date': date_str
    }

    return render_template('commodity_result.html', result=result)


@app.route('/api/predict-price', methods=['POST'])
@login_required
def api_predict_price():
    try:
        data = request.get_json()

        # Get input values from JSON
        commodity = data.get('commodity')
        state = data.get('state')
        average = float(data.get('average'))
        season = int(data.get('season'))
        date_str = data.get('date')

        date = datetime.strptime(date_str, '%Y-%m-%d')
        month = date.month

        # Load base price data if available
        base_price = average
        try:
            data = pd.read_csv('all_commodity_price_with_season.csv')
            # Filter for this commodity and state
            filtered_data = data[(data['Commodity'] == commodity) & (data['State'] == state)]
            if not filtered_data.empty:
                # Use the historical average price as a base
                base_price = filtered_data['Price'].mean()
        except Exception as e:
            print(f"Could not load historical data: {e}")

        # Simple fallback prediction logic
        season_factors = {1: 0.95, 2: 1.0, 3: 1.05, 4: 1.02}
        month_factors = {
            1: 1.02, 2: 1.03, 3: 1.01, 4: 0.99, 5: 0.98, 6: 0.97,
            7: 0.98, 8: 0.99, 9: 1.01, 10: 1.02, 11: 1.03, 12: 1.04
        }

        # Calculate prediction using the factors
        prediction = base_price * season_factors.get(season, 1.0) * month_factors.get(month, 1.0)

        # Adjust based on the provided average compared to historical data
        prediction = prediction * (average / base_price) if base_price > 0 else average * 1.05

        return jsonify({
            'prediction': round(prediction, 2),
            'commodity': commodity,
            'state': state,
            'date': date_str
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


# Main route
@app.route('/')
def home():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return render_template('home.html')


# Initialize database and create necessary directories
def initialize_app():
    # Create database if it doesn't exist
    db.create_all()

    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')

    # Create templates directory if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')

    # Create static directory if it doesn't exist
    if not os.path.exists('static'):
        os.makedirs('static')


if __name__ == '__main__':
    # Create an application context
    with app.app_context():
        # Initialize the app
        initialize_app()

    # Run the Flask app
    app.run(debug=True)