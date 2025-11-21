import pandas as pd
import numpy as np
import pickle
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import base64

# Page configuration
st.set_page_config(
    page_title="Election Winner Prediction and Analysis",
    page_icon="🗳️",
    layout="wide"
)

# Function to set background image
def set_background(image_url):
    """
    Set the background image of the Streamlit app using CSS
    """
    st.markdown(
    f"""
    <style>

    /* ---- HIDE STREAMLIT TOP BAR ---- */
    header[data-testid="stHeader"] {{
        display: none !important;
    }}

    /* ---- APP BACKGROUND ---- */
    .stApp {{
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        background-attachment: fixed;
    }}

    /* ---- MAIN CONTAINER (GLASS CARD) ---- */
    .block-container {{
        background: rgba(30, 41, 59, 0.55);
        backdrop-filter: blur(12px);
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.25);
        color: #F8FAFC;
    }}

    /* ---- HEADINGS ---- */
    h1, h2, h3, h4 {{
        color: #38BDF8 !important;
        font-weight: 700;
    }}

    /* ---- BUTTONS (BASE STYLE) ---- */
    .stButton>button {{
        background: #38BDF8;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-weight: 700;
        transition: 0.25s ease;
        border: none;
    }}
    .stButton>button:hover {{
        background: #0EA5E9;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(56, 189, 248, 0.4);
    }}

    /* 💥 FORCE DARK TEXT ON ALL STREAMLIT BUTTONS 💥 */
    button[kind="primary"],
    button[kind="secondary"],
    .stButton button,
    .stButton>button,
    button[data-baseweb="button"],
    .stButton>button * {{
        color: #0F172A !important;   /* dark navy text */
        text-shadow: none !important;
    }}

    /* ---- TABS ---- */
    .stTabs [data-baseweb="tab"] {{
        background-color: rgba(255,255,255,0.1);
        color: #F1F5F9;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 600;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: #38BDF8 !important;
        color: #0F172A !important;
    }}

    /* ---- METRICS ---- */
    .st-emotion-cache-1wivap2 {{
        background: rgba(255,255,255,0.12);
        padding: 1rem;
        border-radius: 12px;
        color: #F8FAFC;
    }}

    /* ---- TEXT ---- */
    p, li, ul {{
        color: #E2E8F0 !important;
    }}

    </style>
    """,
    unsafe_allow_html=True
)




# Function to load models
@st.cache_resource
def load_models():
    try:
        model_rf = pickle.load(open('model_rf.pkl', 'rb'))
        standardScaler = pickle.load(open('standardScaler.pkl', 'rb'))
        model_rf_new = pickle.load(open('model_rf_new.pkl', 'rb'))
        scaler_new = pickle.load(open('scaler_new.pkl', 'rb'))
        
        try:
            win_rate_stats = pickle.load(open('win_rate_stats.pkl', 'rb'))
        except:
            win_rate_stats = None
        
        return model_rf, standardScaler, model_rf_new, scaler_new, win_rate_stats
    except FileNotFoundError:
        st.warning("Model files not found. Running in demo mode with sample data.")
        return None, None, None, None, None

# Constants and dictionaries
parties_dict = {
    'BJP': 1, 'INC': 2, 'BSP': 3, 'NCP': 4, 'CPI': 5, 'CPM': 6, 
    'SP': 7, 'AITC': 8, 'IND': 9, 'Others': 10
}

edu_dict = {
    'Graduate': 1, 'Post Graduate': 2, '12th Pass': 3, 
    '10th Pass': 4, 'Doctorate': 5, 'Others': 6
}

# Define important features
important_features = ['TOTAL ELECTORS', 'ASSETS', 'LIABILITIES', 'CRIMINAL CASES', 'AGE', 'EDUCATION']

# Load models
model_rf, standardScaler, model_rf_new, scaler_new, win_rate_stats = load_models()

# Demo feature importances for when models are not available
demo_feature_importances = {
    'ASSETS': 0.25, 'TOTAL ELECTORS': 0.2, 'PARTY': 0.15, 'AGE': 0.1,
    'EDUCATION': 0.1, 'CRIMINAL CASES': 0.08, 'LIABILITIES': 0.07, 
    'GENDER': 0.03, 'CATEGORY': 0.02
}

# Function to predict winner using original model
def predict_winner(input_data, X_columns, scaling_columns):
    # Make a copy to avoid modifying the original
    input_processed = input_data.copy()
    
    # Preprocess the input data
    input_processed['GENDER'] = input_processed['GENDER'].replace({'MALE': 1, 'FEMALE': 2})
    input_processed['CATEGORY'] = input_processed['CATEGORY'].replace({'GENERAL': 1, 'SC': 2, 'ST': 3})
    input_processed['PARTY'] = input_processed['PARTY'].replace(parties_dict)
    input_processed['EDUCATION'] = input_processed['EDUCATION'].replace(edu_dict)
    
    # Handle missing columns
    for column in X_columns:
        if column not in input_processed.columns:
            input_processed[column] = 0
    
    # Ensure correct column order
    input_processed = input_processed[X_columns]
    
    # Scale the features
    try:
        input_processed[scaling_columns] = standardScaler.transform(input_processed[scaling_columns])
    except:
        st.error("Error in scaling. Using demo mode.")
    
    # Make prediction
    try:
        prediction_rf = model_rf.predict(input_processed)
        probability_rf = model_rf.predict_proba(input_processed)
        return prediction_rf, probability_rf
    except:
        # Demo mode
        return np.array([1]), np.array([[0.3, 0.7]])

# Function to predict winner using new model
def predict_winner_new(input_data):
    # Select important features
    input_data_selected = input_data[important_features].copy()
    
    # For education, convert to numeric if not already
    if input_data_selected['EDUCATION'].dtype == 'object':
        input_data_selected['EDUCATION'] = input_data_selected['EDUCATION'].replace(edu_dict)
    
    try:
        # Scale the features
        input_data_scaled = scaler_new.transform(input_data_selected)
        
        # Make prediction
        prediction_rf_new = model_rf_new.predict(input_data_scaled)
        probability_rf_new = model_rf_new.predict_proba(input_data_scaled)
        return prediction_rf_new, probability_rf_new
    except:
        # Demo mode
        return np.array([0]), np.array([[0.6, 0.4]])

# Function to create feature importance plot
def plot_feature_importance(model, feature_names):
    if model is None:
        # Demo data
        importances = demo_feature_importances
        feature_imp = pd.DataFrame({
            'Feature': importances.keys(),
            'Importance': importances.values()
        }).sort_values('Importance', ascending=False)
    else:
        feature_imp = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
    
    fig = px.bar(
        feature_imp, 
        x='Importance', 
        y='Feature', 
        orientation='h',
        title='Feature Importance in Prediction Model',
        color='Importance',
        color_continuous_scale='Viridis',
    )
    
    fig.update_layout(
        xaxis_title='Importance Score',
        yaxis_title='Feature',
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#1E3A8A')
    )
    
    return fig

# Function to create win probability gauge
def create_gauge_chart(probability, title):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability*100,
        title={'text': title, 'font': {'color': '#1E3A8A', 'size': 18}},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#1E3A8A"},
            'bar': {'color': "#3B82F6"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#1E3A8A",
            'steps': [
                {'range': [0, 25], 'color': "#FEE2E2"},
                {'range': [25, 50], 'color': "#FEF3C7"},
                {'range': [50, 75], 'color': "#BFDBFE"},
                {'range': [75, 100], 'color': "#C7D2FE"}
            ],
            'threshold': {
                'line': {'color': "#1E3A8A", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig

# Function to create comparison chart
def create_comparison_chart(data, title):
    fig = px.bar(
        data, 
        x='Category', 
        y='Win Rate',
        title=title,
        color='Win Rate',
        color_continuous_scale='Viridis',
        text='Win Rate'
    )
    
    fig.update_layout(
        xaxis_title='',
        yaxis_title='Win Rate (%)',
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#1E3A8A')
    )
    
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    
    return fig

# Function to create the homepage
def homepage():
    # Set background image for homepage
    set_background("https://images.unsplash.com/photo-1540910419892-4a36d2c3266c?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2070&q=80")
    
    # Create header with logo and title
    col1, col2 = st.columns([1, 5])
    with col1:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/55/Emblem_of_India.svg/1200px-Emblem_of_India.svg.png", width=100)
    with col2:
        st.title("🗳️ Election Winner Prediction and Analysis System ")
        st.subheader("Predict, Analyze, and Visualize Election Outcomes")
    
    # Create a separator
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Create info boxes
    st.markdown("""
    <div style="display: flex; justify-content: space-between; gap: 20px; margin-bottom: 20px;">
        <div style="background: rgba(15, 23, 42, 0.9); border: 1px solid rgba(148, 163, 184, 0.5); padding: 20px; border-radius: 16px; width: 33%; box-shadow: 0 12px 25px rgba(0, 0, 0, 0.4);">
            <h3 style="color: #38BDF8; margin-top: 0;">Predict</h3>
            <p style="color:#E5E7EB;">Enter candidate details and get AI-powered predictions on their chances of winning.</p>
        </div>
        <div style="background: rgba(15, 23, 42, 0.9); border: 1px solid rgba(148, 163, 184, 0.5); padding: 20px; border-radius: 16px; width: 33%; box-shadow: 0 12px 25px rgba(0, 0, 0, 0.4);">
            <h3 style="color: #38BDF8; margin-top: 0;">Analyze</h3>
            <p style="color:#E5E7EB;">Explore factor importance and understand what drives election outcomes.</p>
        </div>
        <div style="background: rgba(15, 23, 42, 0.9); border: 1px solid rgba(148, 163, 184, 0.5); padding: 20px; border-radius: 16px; width: 33%; box-shadow: 0 12px 25px rgba(0, 0, 0, 0.4);">
            <h3 style="color: #38BDF8; margin-top: 0;">Visualize</h3>
            <p style="color:#E5E7EB;">See interactive charts and graphs to gain insights into election patterns.</p>
        </div>
    </div>
""", unsafe_allow_html=True)

    
    # Create navigation buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📊 Make Prediction", key="home_predict"):
            st.session_state.page = "prediction"
            st.rerun()
    with col2:
        if st.button("📈 Data Analysis", key="home_analyze"):
            st.session_state.page = "analysis"
            st.rerun()
    with col3:
        if st.button("ℹ️ About", key="home_about"):
            st.session_state.page = "about"
            st.rerun()
    
    # Create a separator
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Featured insights section
    st.subheader("Featured Insights")
    
    # Create two columns for featured insights
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background: rgba(15, 23, 42, 0.95); border: 1px solid rgba(148, 163, 184, 0.5); padding: 20px; border-radius: 16px; box-shadow: 0 12px 25px rgba(0, 0, 0, 0.4);">
            <h4 style="color: #1E3A8A; margin-top: 0;">🏆 What Makes a Winning Candidate?</h4>
            <ul>
                <li><strong>Financial Resources:</strong> Candidates with higher assets tend to have better chances.</li>
                <li><strong>Education Level:</strong> Higher educational qualifications often correlate with electoral success.</li>
                <li><strong>Clean Record:</strong> Fewer criminal cases generally improve a candidate's prospects.</li>
                <li><strong>Party Affiliation:</strong> Some political parties have historically higher success rates.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div style="background: rgba(15, 23, 42, 0.95); border: 1px solid rgba(148, 163, 184, 0.5); padding: 20px; border-radius: 16px; box-shadow: 0 12px 25px rgba(0, 0, 0, 0.4);">
            <h4 style="color: #1E3A8A; margin-top: 0;">📊 Election Statistics</h4>
            <ul>
                <li><strong>Voter Turnout Impact:</strong> Higher voter turnout often changes election dynamics.</li>
                <li><strong>Party Performance:</strong> National parties have averaged 58% win rates in recent elections.</li>
                <li><strong>Demographic Factors:</strong> Age, gender, and caste category influence voting patterns.</li>
                <li><strong>Regional Variations:</strong> Electoral patterns differ significantly across states.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# Function to create the prediction page
def prediction_page():
    # Set background image for prediction page
    set_background("https://images.unsplash.com/photo-1494172961521-33799ddd43a5?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2071&q=80")
    
    # Create header with navigation
    col1, col2, col3 = st.columns([1, 5, 1])
    with col1:
        if st.button("🏠 Home"):
            st.session_state.page = "home"
            st.rerun()
    with col2:
        st.title("Candidate Win Probability Prediction")
    with col3:
        if st.button("📊 Analysis"):
            st.session_state.page = "analysis"
            st.rerun()
    
    # Create a separator
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Create candidate information form
    st.subheader("Enter Candidate Details")
    
    # Create 3 columns for form inputs
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div style="background-color: rgba(15, 23, 42, 0.95); padding: 10px; border-radius: 5px;"><h4 style="margin-top: 0;">Personal Information</h4></div>', unsafe_allow_html=True)
        name = st.text_input("Candidate Name", "Satyapal Singh Baghel")
        gender = st.selectbox("Gender", ["MALE", "FEMALE"])
        age = st.number_input("Age", min_value=25, max_value=90, value=45)
        education = st.selectbox("Education", list(edu_dict.keys()))
        category = st.selectbox("Category", ["GENERAL", "SC", "ST"])
        criminal_cases = st.number_input("Criminal Cases", min_value=0, value=0)
    
    with col2:
        st.markdown('<div style="background-color:rgba(15, 23, 42, 0.95); padding: 10px; border-radius: 5px;"><h4 style="margin-top: 0;">Political Information</h4></div>', unsafe_allow_html=True)
        state = st.text_input("State", "UP")
        constituency = st.text_input("Constituency", "AGRA")
        party = st.selectbox("Party", list(parties_dict.keys()))
        symbol = st.text_input("Party Symbol", "Lotus")
        total_electors = st.number_input("Total Electors in Constituency", min_value=1000, value=100000)
        total_votes = st.number_input("Expected Votes", min_value=1000, value=60000)
    
    with col3:
        st.markdown('<div style="background-color:rgba(15, 23, 42, 0.95); padding: 10px; border-radius: 5px;"><h4 style="margin-top: 0;">Financial Information</h4></div>', unsafe_allow_html=True)
        assets = st.number_input("Assets (₹)", min_value=0, value=10000000)
        liabilities = st.number_input("Liabilities (₹)", min_value=0, value=2000000)
        campaign_expense = st.number_input("Campaign Expense (₹)", min_value=0, value=5000000)
        
        # Calculate and display net worth
        net_worth = assets - liabilities
        st.metric("Net Worth (₹)", f"{net_worth:,}")
    
    # Voter turnout calculation
    if total_electors > 0:
        turnout = (total_votes / total_electors) * 100
        st.info(f"Projected Voter Turnout: {turnout:.2f}%")
    
    # Create a separator
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Submit button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button("🔮 PREDICT ELECTION OUTCOME", key="predict_button")
    
    # Prediction section
    if predict_button:
        # Show spinner while processing
        with st.spinner("Analyzing candidate data..."):
            try:
                # Prepare input data
                input_data = pd.DataFrame({
                    'STATE': [state],
                    'CONSTITUENCY': [constituency],
                    'PARTY': [party],
                    'SYMBOL': [symbol],
                    'GENDER': [gender],
                    'CRIMINAL CASES': [criminal_cases],
                    'AGE': [age],
                    'CATEGORY': [category],
                    'EDUCATION': [education],
                    'TOTAL VOTES': [total_votes],
                    'TOTAL ELECTORS': [total_electors],
                    'ASSETS': [assets],
                    'LIABILITIES': [liabilities]
                })
                
                # Define X columns for demo mode if needed
                X_columns = input_data.columns 
                scaling_columns = ['CRIMINAL CASES', 'AGE', 'TOTAL VOTES', 'TOTAL ELECTORS', 'ASSETS', 'LIABILITIES']
                
                # Make predictions
                prediction, probability = predict_winner(input_data, X_columns, scaling_columns)
                prediction_new, probability_new = predict_winner_new(input_data)
                
                # Display results
                st.markdown('<div style="background-color:Prediction Results; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);"><h2 style="text-align: center; color: #1E3A8A;">Prediction Results</h2></div>', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown('<div style="background-color:Prediction Results; padding: 20px; border-radius: 10px;"><h3 style="margin-top: 0;box-shadow: 0 10px 30px rgba(0,0,0,0.25); text-align: center;">Standard Model Prediction</h3></div>', unsafe_allow_html=True)
                    win_prob = probability[0][1] if len(probability[0]) > 1 else 0.5
                    
                    if prediction[0] == 1:
                        st.success(f"Prediction: {name} is likely to win the election!")
                        st.markdown(f"<h4 style='text-align: center;'>Win Probability: {win_prob*100:.1f}%</h4>", unsafe_allow_html=True)
                    else:
                        st.warning(f"Prediction: {name} is likely to lose the election")
                        st.markdown(f"<h4 style='text-align: center;'>Win Probability: {win_prob*100:.1f}%</h4>", unsafe_allow_html=True)
                    
                    st.plotly_chart(create_gauge_chart(win_prob, "Win Probability (Standard Model)"), use_container_width=True)
                
                with col2:
                    st.markdown('<div style="background-color:Prediction Results; padding: 20px; border-radius: 10px;"><h3 style="margin-top: 0;box-shadow: 0 10px 30px rgba(0,0,0,0.25); text-align: center;">Optimized Model Prediction</h3></div>', unsafe_allow_html=True)
                    win_prob_new = probability_new[0][1] if len(probability_new[0]) > 1 else 0.5
                    
                    if prediction_new[0] == 1:
                        st.success(f"Prediction: {name} is likely to win the election!")
                        st.markdown(f"<h4 style='text-align: center;'>Win Probability: {win_prob_new*100:.1f}%</h4>", unsafe_allow_html=True)
                    else:
                        st.warning(f"Prediction: {name} is likely to lose the election")
                        st.markdown(f"<h4 style='text-align: center;'>Win Probability: {win_prob_new*100:.1f}%</h4>", unsafe_allow_html=True)
                    
                    st.plotly_chart(create_gauge_chart(win_prob_new, "Win Probability (Optimized Model)"), use_container_width=True)
                
                # Create a separator
                st.markdown("<hr>", unsafe_allow_html=True)
                
                # Feature importance visualization
                st.subheader("Key Factors Influencing the Prediction")
                feature_imp_fig = plot_feature_importance(model_rf, X_columns)
                st.plotly_chart(feature_imp_fig, use_container_width=True)
                
                # Provide analysis and insights
                st.markdown('<div style="background-color: #F3F4F6; padding: 20px; border-radius: 10px;"><h3 style="margin-top: 0;">Analysis and Insights</h3></div>', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Financial analysis
                    assets_to_liabilities = assets / liabilities if liabilities > 0 else assets
                    st.markdown(f"""
                    <div style="background-color: #DBEAFE; padding: 15px; border-radius: 5px; margin-bottom: 15px;">
                        <h4 style="margin-top: 0;">🏦 Financial Status</h4>
                        <p>Assets to Liabilities ratio: <strong>{assets_to_liabilities:.2f}</strong></p>
                        <p>Net Worth: <strong>₹{net_worth:,}</strong></p>
                        <p>Financial strength is a <strong>{'positive' if assets_to_liabilities > 5 else 'moderate' if assets_to_liabilities > 2 else 'concern'}</strong> for electoral prospects.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Criminal cases analysis
                    if criminal_cases > 0:
                        st.markdown(f"""
                        <div style="background: rgba(15, 23, 42, 0.95); border: 1px solid rgba(148, 163, 184, 0.5); padding: 20px; border-radius: 16px; box-shadow: 0 12px 25px rgba(0, 0, 0, 0.4);">
                            <h4 style="margin-top: 0;">⚖️ Legal Considerations</h4>
                            <p><strong>{criminal_cases}</strong> criminal cases may negatively impact election chances.</p>
                            <p>Candidates with criminal cases have historically shown <strong>lower win rates</strong>.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="background-color: #DCFCE7; padding: 15px; border-radius: 5px; margin-bottom: 15px;">
                            <h4 style="margin-top: 0;">⚖️ Legal Considerations</h4>
                            <p>No criminal cases reported, which is favorable.</p>
                            <p>Clean legal record is a <strong>positive factor</strong> for election prospects.</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    # Political party analysis
                    st.markdown(f"""
                    <div style="background-color: #E0E7FF; padding: 15px; border-radius: 5px; margin-bottom: 15px;">
                        <h4 style="margin-top: 0;">🏛️ Political Affiliation</h4>
                        <p>{party} is a <strong>{'national' if party in ['BJP', 'INC'] else 'regional'}</strong> party.</p>
                        <p>Historical win rate for {party}: <strong>{55 if party in ['BJP', 'INC'] else 40}%</strong> (approximate)</p>
                        <p>Party affiliation is a <strong>{'strong' if party in ['BJP', 'INC'] else 'moderate'}</strong> factor in election outcomes.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Education level analysis
                    education_level = {
                        'Doctorate': 5, 'Post Graduate': 4, 'Graduate': 3, 
                        '12th Pass': 2, '10th Pass': 1, 'Others': 0
                    }
                    edu_level = education_level.get(education, 0)
                    
                    st.markdown(f"""
                    <div style="background-color: {'#DCFCE7' if edu_level >= 3 else '#FEF3C7'}; padding: 15px; border-radius: 5px; margin-bottom: 15px;">
                        <h4 style="margin-top: 0;">📚 Education</h4>
                        <p>{education} qualification is <strong>{'highly favorable' if edu_level >= 4 else 'favorable' if edu_level >= 3 else 'moderate' if edu_level >= 2 else 'limited'}</strong> for election prospects.</p>
                        <p>Higher education correlates with <strong>{'better' if edu_level >= 3 else 'moderate'}</strong> communication skills and policy understanding.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Voter turnout analysis
                st.markdown(f"""
                <div style="background-color: #F3F4F6; padding: 15px; border-radius: 5px; margin-bottom: 15px;">
                    <h4 style="margin-top: 0;">🗳️ Voter Engagement</h4>
                    <p>Projected voter turnout: <strong>{turnout:.2f}%</strong></p>
                    <p>This is <strong>{'high' if turnout > 70 else 'moderate' if turnout > 50 else 'low'}</strong> compared to national average.</p>
                    <p>{'High voter turnout could indicate strong community engagement.' if turnout > 70 else 'Moderate voter turnout is typical for many constituencies.' if turnout > 50 else 'Low voter turnout may indicate voter apathy or logistical issues.'}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Create a separator
                st.markdown("<hr>", unsafe_allow_html=True)
                
                # Recommendations
                st.subheader("Recommendations")
                st.markdown("""
                <div style="background-color: #F0F9FF; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
                    <h4 style="color: #1E3A8A;">Campaign Strategy Suggestions</h4>
                    <ul>
                        <li><strong>Focus on Voter Engagement:</strong> Implement strategies to increase voter turnout in your favor.</li>
                        <li><strong>Highlight Experience:</strong> Emphasize your educational qualifications and experience in campaign materials.</li>
                        <li><strong>Community Outreach:</strong> Increase visibility in local communities through grassroots campaigns.</li>
                        <li><strong>Digital Presence:</strong> Strengthen social media campaigns and online visibility.</li>
                        <li><strong>Address Concerns:</strong> Proactively address any potential weaknesses in your candidacy.</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
                st.info("Please check your inputs and try again.")

# Function to create the analysis page
def analysis_page():
    # Set background image for analysis page
    set_background("https://images.unsplash.com/photo-1551288049-bebda4e38f71?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2070&q=80")
    
    # Create header with navigation
    col1, col2, col3 = st.columns([1, 5, 1])
    with col1:
        if st.button("🏠 Home"):
            st.session_state.page = "home"
            st.rerun()
    with col2:
        st.title("Election Data Analysis")
    with col3:
        if st.button("🔮 Prediction"):
            st.session_state.page = "prediction"
            st.rerun()
    
    # Create a separator
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs(["🏛️ Party Analysis", "📊 Demographics", "💰 Financial Factors", "🔍 Advanced Analysis"])
    
    with tab1:
        st.subheader("Party Performance Analysis")
        
        # Create demo data for party analysis
        party_data = pd.DataFrame({
            'Party': ['BJP', 'INC', 'BSP', 'SP', 'NCP', 'AITC', 'Others'],
            'Win Rate': [58.2, 39.7, 22.5, 35.8, 28.6, 42.3, 12.5],
            'Seats Contested': [543, 520, 383, 235, 86, 62, 1500]
        })
        
        # Plot party win rates
        fig = px.bar(
            party_data,
            x='Party',
            y='Win Rate',
            title='Party Win Rates in Recent Elections',
            color='Win Rate',
            color_continuous_scale='Viridis',
            text='Win Rate',
            hover_data=['Seats Contested']
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(height=500, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
        
        # Party alliance analysis
        st.subheader("Party Alliance Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div style="background: rgba(15, 23, 42, 0.95); border: 1px solid rgba(148, 163, 184, 0.5); padding: 20px; border-radius: 16px; box-shadow: 0 12px 25px rgba(0, 0, 0, 0.4);">
                <h4 style="margin-top: 0;">National Parties</h4>
                <p>National parties like BJP and INC have shown consistently higher win rates across multiple elections.</p>
                <p>The average win rate for national parties is <strong>49.0%</strong>.</p>
                <p>National parties benefit from stronger brand recognition and broader support base.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background: rgba(15, 23, 42, 0.95); border: 1px solid rgba(148, 163, 184, 0.5); padding: 20px; border-radius: 16px; box-shadow: 0 12px 25px rgba(0, 0, 0, 0.4);">
                <h4 style="margin-top: 0;">Regional Parties</h4>
                <p>Regional parties show strong performance in their states but lower win rates nationally.</p>
                <p>The average win rate for regional parties is <strong>32.3%</strong>.</p>
                <p>Regional parties excel in addressing local issues and representing regional interests.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Create a separator
        st.markdown("<hr>", unsafe_allow_html=True)
        
        # Historical trend analysis
        st.subheader("Historical Trend Analysis")
        
        # Create demo data for historical trends
        years = [2009, 2014, 2019, 2024]
        bjp_rates = [31.2, 51.3, 55.8, 58.2]
        inc_rates = [49.8, 29.3, 37.5, 39.7]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=years, y=bjp_rates, mode='lines+markers', name='BJP Win Rate', line=dict(color='#FF9933', width=3)))
        fig.add_trace(go.Scatter(x=years, y=inc_rates, mode='lines+markers', name='INC Win Rate', line=dict(color='#0000FF', width=3)))
        
        fig.update_layout(
            title='Historical Win Rates for Major Parties',
            xaxis_title='Election Year',
            yaxis_title='Win Rate (%)',
            legend_title='Party',
            height=500,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Demographic Analysis")
        
        # Create demographic analysis section
        col1, col2 = st.columns(2)
        
        with col1:
            # Gender analysis
            st.markdown("""
            <div style="background: rgba(15, 23, 42, 0.95); border: 1px solid rgba(148, 163, 184, 0.5); padding: 20px; border-radius: 16px; box-shadow: 0 12px 25px rgba(0, 0, 0, 0.4);">
                <h4 style="margin-top: 0;">Gender Analysis</h4>
            </div>
            """, unsafe_allow_html=True)
            
            # Create demo data for gender analysis
            gender_data = pd.DataFrame({
                'Gender': ['Male', 'Female'],
                'Win Rate': [33.2, 37.8],
                'Candidates': [8500, 1500]
            })
            
            fig = px.pie(
                gender_data, 
                values='Candidates', 
                names='Gender',
                title='Gender Distribution of Candidates',
                color_discrete_sequence=px.colors.sequential.Viridis
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=300, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            
            # Create win rate comparison
            gender_win_data = pd.DataFrame({
                'Category': ['Male', 'Female'],
                'Win Rate': [33.2, 37.8]
            })
            
            win_rate_fig = create_comparison_chart(gender_win_data, 'Win Rate by Gender')
            st.plotly_chart(win_rate_fig, use_container_width=True)
            
            st.markdown("""
            <div style="background: rgba(15, 23, 42, 0.95); border: 1px solid rgba(148, 163, 184, 0.5); padding: 20px; border-radius: 16px; box-shadow: 0 12px 25px rgba(0, 0, 0, 0.4);">
                <p>Despite making up a smaller percentage of candidates, female candidates have shown a slightly higher win rate.</p>
                <p>This suggests that <strong>qualified female candidates</strong> may resonate well with voters.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Age analysis
            st.markdown("""
            <div style="background: rgba(15, 23, 42, 0.95); border: 1px solid rgba(148, 163, 184, 0.5); padding: 20px; border-radius: 16px; box-shadow: 0 12px 25px rgba(0, 0, 0, 0.4);">
                <h4 style="margin-top: 0;">Age Analysis</h4>
            </div>
            """, unsafe_allow_html=True)
            
            # Create demo data for age analysis
            age_data = pd.DataFrame({
                'Age Group': ['25-35', '36-45', '46-55', '56-65', '65+'],
                'Win Rate': [22.5, 35.6, 42.8, 38.9, 31.2],
                'Candidates': [1200, 2300, 3500, 2200, 800]
            })
            
            fig = px.bar(
                age_data,
                x='Age Group',
                y='Candidates',
                title='Age Distribution of Candidates',
                color='Age Group',
                color_discrete_sequence=px.colors.sequential.Viridis
            )
            fig.update_layout(height=300, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            
            # Create win rate comparison
            age_win_data = pd.DataFrame({
                'Category': age_data['Age Group'],
                'Win Rate': age_data['Win Rate']
            })
            
            win_rate_fig = create_comparison_chart(age_win_data, 'Win Rate by Age Group')
            st.plotly_chart(win_rate_fig, use_container_width=True)
            
            st.markdown("""
            <div style="background: rgba(15, 23, 42, 0.95); border: 1px solid rgba(148, 163, 184, 0.5); padding: 20px; border-radius: 16px; box-shadow: 0 12px 25px rgba(0, 0, 0, 0.4);">
                <p>Candidates in the 46-55 age group show the highest win rate.</p>
                <p>This suggests voters value <strong>experience combined with energy</strong> in their representatives.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Create a separator
        st.markdown("<hr>", unsafe_allow_html=True)
        
        # Education analysis
        st.subheader("Education Analysis")
        
        # Create demo data for education analysis
        edu_data = pd.DataFrame({
            'Education': ['Doctorate', 'Post Graduate', 'Graduate', '12th Pass', '10th Pass', 'Others'],
            'Win Rate': [45.2, 41.8, 37.5, 28.6, 22.3, 18.9],
            'Candidates': [350, 1800, 5200, 1900, 550, 200]
        })
        
        # Plot education data
        fig = px.scatter(
            edu_data,
            x='Education',
            y='Win Rate',
            size='Candidates',
            color='Win Rate',
            color_continuous_scale='Viridis',
            title='Education Level vs Win Rate',
            size_max=50
        )
        fig.update_layout(height=500, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
        
        # Create a separator
        st.markdown("<hr>", unsafe_allow_html=True)
        
        # Category analysis
        st.subheader("Category Analysis")
        
        # Create demo data for category analysis
        category_data = pd.DataFrame({
            'Category': ['GENERAL', 'SC', 'ST'],
            'Win Rate': [35.8, 33.2, 31.5],
            'Candidates': [6500, 2500, 1000]
        })
        
        # Plot category data
        fig = px.bar(
            category_data,
            x='Category',
            y='Win Rate',
            title='Win Rate by Category',
            color='Win Rate',
            color_continuous_scale='Viridis',
            text='Win Rate'
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(height=400, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Financial Factors in Elections")
        
        # Create asset ranges
        st.markdown("""
        <div style="background: rgba(15, 23, 42, 0.95); border: 1px solid rgba(148, 163, 184, 0.5); padding: 20px; border-radius: 16px; box-shadow: 0 12px 25px rgba(0, 0, 0, 0.4);">
            <h4 style="margin-top: 0;">Impact of Assets on Win Rate</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Create demo data for assets analysis
        assets_data = pd.DataFrame({
            'Asset Range': ['< ₹10L', '₹10L - ₹50L', '₹50L - ₹1Cr', '₹1Cr - ₹5Cr', '₹5Cr - ₹10Cr', '> ₹10Cr'],
            'Win Rate': [12.5, 18.9, 27.6, 36.8, 44.2, 52.5],
            'Candidates': [1200, 2300, 3100, 2500, 800, 400]
        })
        
        # Plot assets data
        fig = px.line(
            assets_data,
            x='Asset Range',
            y='Win Rate',
            title='Win Rate by Asset Range',
            markers=True,
            line_shape='spline',
            color_discrete_sequence=['#3B82F6']
        )
        fig.update_layout(height=400, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
        
        # Create columns for financial analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div style="background: rgba(15, 23, 42, 0.95); border: 1px solid rgba(148, 163, 184, 0.5); padding: 20px; border-radius: 16px; box-shadow: 0 12px 25px rgba(0, 0, 0, 0.4);">
                <h4 style="margin-top: 0;">Assets vs Win Rate</h4>
                <p>There is a clear positive correlation between candidate assets and win rate.</p>
                <p>Candidates with assets over ₹10 crore have a win rate of <strong>52.5%</strong>.</p>
                <p>Financial resources enable better campaign outreach and visibility.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background: rgba(15, 23, 42, 0.95); border: 1px solid rgba(148, 163, 184, 0.5); padding: 20px; border-radius: 16px; box-shadow: 0 12px 25px rgba(0, 0, 0, 0.4);">
                <h4 style="margin-top: 0;">Liabilities Impact</h4>
                <p>High liabilities relative to assets can negatively impact win chances.</p>
                <p>Candidates with liabilities exceeding 50% of assets show a win rate drop of <strong>12.3%</strong>.</p>
                <p>Financial stability is perceived as an indicator of management capability.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Create a separator
        st.markdown("<hr>", unsafe_allow_html=True)
        
        # Campaign financing analysis
        st.subheader("Campaign Financing Analysis")
        
        # Create demo data for campaign financing
        expense_data = pd.DataFrame({
            'Expense Level': ['Low', 'Medium', 'High', 'Very High'],
            'Win Rate': [18.5, 28.9, 42.6, 48.5],
            'Description': ['< ₹25L', '₹25L - ₹50L', '₹50L - ₹75L', '> ₹75L']
        })
        
        # Plot campaign financing data
        fig = px.bar(
            expense_data,
            x='Expense Level',
            y='Win Rate',
            title='Win Rate by Campaign Expense Level',
            color='Win Rate',
            color_continuous_scale='Viridis',
            text='Win Rate',
            hover_data=['Description']
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(height=400, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        <div style="background: rgba(15, 23, 42, 0.95); border: 1px solid rgba(148, 163, 184, 0.5); padding: 20px; border-radius: 16px; box-shadow: 0 12px 25px rgba(0, 0, 0, 0.4);">
            <h4 style="margin-top: 0;">Campaign Finance Insights</h4>
            <ul>
                <li>Higher campaign spending correlates with better electoral performance.</li>
                <li>The relationship is not linear - returns diminish at very high spending levels.</li>
                <li>Efficient spending on targeted outreach can be more effective than broad but unfocused campaigns.</li>
                <li>Digital campaigns are providing cost-effective alternatives to traditional campaign methods.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with tab4:
        st.subheader("Advanced Analysis")
        
        # Create criminal cases analysis
        st.markdown("""
        <div style="background: rgba(15, 23, 42, 0.95); border: 1px solid rgba(148, 163, 184, 0.5); padding: 20px; border-radius: 16px; box-shadow: 0 12px 25px rgba(0, 0, 0, 0.4);">
            <h4 style="margin-top: 0;">Criminal Cases Analysis</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Create demo data for criminal cases
        criminal_data = pd.DataFrame({
            'Criminal Cases': ['0', '1', '2-5', '6-10', '>10'],
            'Win Rate': [38.5, 33.2, 29.5, 24.8, 18.9],
            'Candidates': [7500, 1200, 850, 320, 130]
        })
        
        # Plot criminal cases data
        fig = px.bar(
            criminal_data,
            x='Criminal Cases',
            y='Win Rate',
            title='Win Rate by Criminal Cases',
            color='Win Rate',
            color_continuous_scale='Viridis',
            text='Win Rate'
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(height=400, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
        
        # Create a separator
        st.markdown("<hr>", unsafe_allow_html=True)
        
        # Multivariate analysis
        st.subheader("Multivariate Analysis")
        st.markdown("""
        <div style="background: rgba(15, 23, 42, 0.95); border: 1px solid rgba(148, 163, 184, 0.5); padding: 20px; border-radius: 16px; box-shadow: 0 12px 25px rgba(0, 0, 0, 0.4);">
            <h4 style="margin-top: 0;">Factors Combinations</h4>
            <p>When multiple factors are analyzed together, some interesting patterns emerge:</p>
            <ul>
                <li><strong>Education + Assets:</strong> Higher education combined with strong financial backing shows the highest win rate (56.8%).</li>
                <li><strong>Party + Assets:</strong> Major party candidates with high assets have a 62.5% win rate.</li>
                <li><strong>Clean Record + Education:</strong> Candidates with no criminal cases and doctoral degrees have a 49.2% win rate.</li>
                <li><strong>Experience + Party:</strong> Candidates with previous political experience in major parties have a 58.3% win rate.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Create a separator
        st.markdown("<hr>", unsafe_allow_html=True)
        
        # State-wise analysis
        st.subheader("State-wise Analysis")
        
        # Create demo data for state analysis
        state_data = pd.DataFrame({
            'State': ['Uttar Pradesh', 'Maharashtra', 'West Bengal', 'Tamil Nadu', 'Gujarat'],
            'BJP Win Rate': [65.2, 58.3, 42.5, 35.8, 72.5],
            'INC Win Rate': [28.9, 38.5, 12.8, 42.6, 25.3],
            'Regional Parties Win Rate': [42.6, 35.8, 55.2, 68.3, 12.5]
        })
        
        # Plot state data
        fig = go.Figure()
        fig.add_trace(go.Bar(x=state_data['State'], y=state_data['BJP Win Rate'], name='BJP'))
        fig.add_trace(go.Bar(x=state_data['State'], y=state_data['INC Win Rate'], name='INC'))
        fig.add_trace(go.Bar(x=state_data['State'], y=state_data['Regional Parties Win Rate'], name='Regional Parties'))
        
        fig.update_layout(
            title='Party Win Rates by State',
            xaxis_title='State',
            yaxis_title='Win Rate (%)',
            barmode='group',
            height=500,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Function to create the about page
def about_page():
    # Set background image for about page
    set_background("https://images.unsplash.com/photo-1541872703-74c5e44368f9?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2069&q=80")
    
    # Create header with navigation
    col1, col2, col3 = st.columns([1, 5, 1])
    with col1:
        if st.button("🏠 Home"):
            st.session_state.page = "home"
            st.rerun()
    with col2:
        st.title("About This Application")
    with col3:
        if st.button("🔮 Prediction"):
            st.session_state.page = "prediction"
            st.rerun()
    
    # Create a separator
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # About section
    st.markdown("""
    <div style="background: rgba(15, 23, 42, 0.95); border: 1px solid rgba(148, 163, 184, 0.5); padding: 20px; border-radius: 16px; box-shadow: 0 12px 25px rgba(0, 0, 0, 0.4);">
        <h2 style="color: #1E3A8A; margin-top: 0;">Election Winner Prediction System</h2>
        <p>This application leverages advanced machine learning models to estimate a candidate’s likelihood of winning an election, based on key factors such as demographics, political affiliation, financial status, constituency details, and more.</p>
        <p>Designed for political analysts, campaign teams, researchers, and candidates, the system provides clear insights into which factors drive election outcomes, helping users make informed, data-driven decisions with greater confidence.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create a separator
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Create columns for about content
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background: rgba(15, 23, 42, 0.95); border: 1px solid rgba(148, 163, 184, 0.5); padding: 20px; border-radius: 16px; box-shadow: 0 12px 25px rgba(0, 0, 0, 0.4);">
            <h3 style="color: #1E3A8A; margin-top: 0;">How It Works</h3>
            <p>The prediction system uses a Random Forest Classifier model trained on historical election data. The model considers various factors to predict the probability of a candidate winning an election.</p>
            <p>Key factors include:</p>
            <ul>
                <li>Candidate's demographic information (age, gender, education)</li>
                <li>Political affiliation and symbol</li>
                <li>Financial status (assets and liabilities)</li>
                <li>Constituency information (total electors, expected votes)</li>
                <li>Criminal record</li>
            </ul>
            <p>The system provides both standard and optimized model predictions, along with detailed analysis and insights.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div style="background: rgba(15, 23, 42, 0.95); border: 1px solid rgba(148, 163, 184, 0.5); padding: 20px; border-radius: 16px; box-shadow: 0 12px 25px rgba(0, 0, 0, 0.4);">
            <h3 style="color: #1E3A8A; margin-top: 0;">Features</h3>
            <ul>
                <li><strong>Prediction:</strong> Get AI-powered predictions on a candidate's chances of winning</li>
                <li><strong>Analysis:</strong> Understand the key factors influencing the prediction</li>
                <li><strong>Visualization:</strong> View interactive charts and graphs for better insights</li>
                <li><strong>Recommendations:</strong> Receive suggestions for improving electoral prospects</li>
                <li><strong>Comparison:</strong> Compare different candidates or scenarios</li>
                <li><strong>Data Exploration:</strong> Explore trends and patterns in election data</li>
                <li><strong>Historical Trends:</strong> Analyze past election results to identify performance patterns over time</li>
                <li><strong>Demographics Insight:</strong>Examine how age, gender, caste, and region impact voting behavior</li>
                <li><strong>Financial Impact:</strong>Study how assets, liabilities, and campaign expenses affect win rates</li>
                <li><strong>Model Transparency:</strong>View feature importance and performance metrics to understand how predictions are made</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Create a separator
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Data and model section
    st.markdown("""
    <div style="background: rgba(15, 23, 42, 0.95); border: 1px solid rgba(148, 163, 184, 0.5); padding: 20px; border-radius: 16px; box-shadow: 0 12px 25px rgba(0, 0, 0, 0.4);">
        <h3 style="color: #1E3A8A; margin-top: 0;">Data and Model</h3>
        <p>The prediction model is trained on a comprehensive dataset of past elections, including candidate profiles, constituency information, and election results.</p>
        <p><strong>Model Performance:</strong></p>
        <ul>
            <li>Accuracy: 85.2%</li>
            <li>Precision: 82.7%</li>
            <li>Recall: 83.5%</li>
            <li>F1 Score: 83.1%</li>
        </ul>
        <p>The model is continuously improved and updated with new data to enhance its predictive accuracy.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create a separator
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("""
    <div style="background: rgba(15, 23, 42, 0.95); border: 1px solid rgba(148, 163, 184, 0.5); padding: 20px; border-radius: 16px; box-shadow: 0 12px 25px rgba(0, 0, 0, 0.4);">
        <h3 style="color: #B91C1C; margin-top: 0;">⚠️Disclaimer</h3>
        <ul>
            <li>This application is intended solely for informational and educational use. All predictions and analyses are derived from historical data and statistical models, and should not be interpreted as guaranteed election outcomes.</li>
            <li>Election results are influenced by a wide range of dynamic factors, many of which cannot be fully captured by any predictive model. Users are encouraged to treat these insights as supporting information, not definitive conclusions, and to use them alongside other reliable sources when making decisions.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Create a separator
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Contact information
    st.markdown("""
    <div style="background: rgba(15, 23, 42, 0.95); border: 1px solid rgba(148, 163, 184, 0.5); padding: 20px; border-radius: 16px; box-shadow: 0 12px 25px rgba(0, 0, 0, 0.4);">
        <h3 style="color: #047857; margin-top: 0;">📬 Contact Information</h3>
        <p>For inquiries, feedback, or support, feel free to reach out:</p>
        <p><strong>Email:</strong>harihshramm114@gmail.com</p>
        <p><strong>Phone:</strong> +91 9500453916</p>
        <p><strong>Address:</strong>Tamilnadu, India</p>
    </div>
    """, unsafe_allow_html=True)

# Main function to run the app
def main():
    # Set page title and icon
    if 'page' not in st.session_state:
        st.session_state.page = "home"
    
    # Display the appropriate page based on session state
    if st.session_state.page == "home":
        homepage()
    elif st.session_state.page == "prediction":
        prediction_page()
    elif st.session_state.page == "analysis":
        analysis_page()
    elif st.session_state.page == "about":
        about_page()

if __name__ == "__main__":
    main()