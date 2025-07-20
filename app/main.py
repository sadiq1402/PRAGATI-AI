import sys
from pathlib import Path

import streamlit as st

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from src.utils.api_utils import (
    get_coordinates_from_city,
    get_gemini_recommendation,
    get_location_from_ip,
    get_weather_data,
    map_weather_code,
)
from src.utils.media_processing import process_media


# Inject custom CSS for a botanical theme
def inject_custom_css():
    st.markdown(
        """
    <style>
        body {
            # background-color: #f9f9f9;
            color: #2e2e2e;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .stApp {
            # background-color: #f9f9f9;
        }
        .main-title {
            background: linear-gradient(90deg,rgba(89, 155, 42, 1) 0%, rgba(46, 135, 62, 1) 50%, rgba(36, 110, 35, 1) 100%);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .main-title h1 {
            font-size: 2.5rem;
            margin: 0;
        }
        .main-title p {
            font-size: 1rem;
            margin: 0;
        }
        .card {
            background-color: white;
            padding: 0.1rem;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.05);
            margin-bottom: 1rem;
        }
        .card h3 {
            color: #2e7d32;
            font-size: 1.2rem;
            margin-bottom: 0.5rem;
            border-bottom: 2px solid #81c784;
            padding-bottom: 0.3rem;
        }
        .metric {
            background-color: #e8f5e9;
            padding: 0.8rem;
            border-radius: 8px;
            margin: 0.5rem;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }
        .metric span {
            font-size: 1.5rem;
            font-weight: bold;
            color: #2e7d32;
        }
        .metric p {
            margin: 0;
            font-size: 0.9rem;
            color: #555;
        }
        .stButton > button {
            background-color: #4caf50;
            color: white !important;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            border: none;
            font-weight: 600;
            transition: background-color 0.3s ease;
        }
        .stButton > button:hover,
        .stButton > button:active,
        .stButton > button:focus {
            background-color: #81c784;
            transform: translateY(-2px);
            color: white;
        }
        .stTextInput input {
            border-radius: 8px;
            padding: 0.5rem;
            border: 1px solid #ccc;
        }
        .stSelectbox select {
            border-radius: 8px;
            padding: 0.5rem;
            border: 1px solid #ccc;
        }
        .stFileUploader .stProgress .st-bo {
            background-color: #81c784;
        }
        .stImage {
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .chat-container {
            background-color: #f0fff0;
            padding: 1rem;
            border-radius: 10px;
            margin-top: 1rem;
        }
        .chat-container h4 {
            color: #2e7d32;
        }
        .sidebar .sidebar-content {
            background-color: #f0fff0;
            border-radius: 10px;
            padding: 1rem;
        }
        .footer {
            font-size: 0.8rem;
            color: #aaa;
            text-align: center;
            margin-top: 2rem;
        }
        .weather-card {
            background-color: #1a1a1a;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
            border: 1px solid #2a2a2a;
            margin-bottom: 1.5rem;
        }

        .weather-metric {
            background-color: #0f0f0f;
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
            margin: 0.5rem;
            border-left: 4px solid #4caf50;
            transition: transform 0.2s ease;
        }

        .weather-metric:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
        }

        .metric-value {
            font-size: 1.2rem;
            font-weight: bold;
            color: #4caf50;
            display: block;
            margin-top: 0.3rem;
        }

        .weather-condition {
            background-color: #0f0f0f;
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
            margin: 0.5rem;
            border-left: 4px solid #81c784;
        }

        .condition-badge {
            display: inline-block;
            padding: 0.3rem 0.6rem;
            border-radius: 20px;
            color: white;
            font-weight: bold;
            margin-top: 0.3rem;
            font-size: 0.9rem;
        }
        .spacer {
            margin-top: 1.5rem;
        }

        .weather-metric {
            background-color: #1a1a1a;
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
            border-left: 4px solid #4caf50;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }

        .weather-metric:hover {
            transform: translateY(-3px);
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.4);
        }

        .metric-value {
            font-size: 1.2rem;
            font-weight: bold;
            display: block;
            margin-top: 0.4rem;
        }

        .weather-condition {
            background-color: #1a1a1a;
            padding: 1.2rem;
            border-radius: 12px;
            text-align: center;
            border-left: 4px solid #81c784;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }

        .weather-condition:hover {
            transform: translateY(-3px);
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.4);
        }

        .condition-badge {
            display: inline-block;
            padding: 0.4rem 0.8rem;
            border-radius: 20px;
            color: white;
            font-weight: bold;
            margin-top: 0.4rem;
            font-size: 0.95rem;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(layout="centered")
    inject_custom_css()

    with st.container():

        # Initialize session state variables
        if "page" not in st.session_state:
            st.session_state.page = "main"
        if "recommendation" not in st.session_state:
            st.session_state["recommendation"] = ""
        if "recommendation_hindi" not in st.session_state:
            st.session_state["recommendation_hindi"] = ""
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "insect_count" not in st.session_state:
            st.session_state["insect_count"] = 0
        if "flower_count" not in st.session_state:
            st.session_state["flower_count"] = 0
        if "weather_data" not in st.session_state:
            st.session_state["weather_data"] = {}
        if "language" not in st.session_state:
            st.session_state["language"] = "English"

        # Language selection
        # st.markdown("### 🌐 Select Language", unsafe_allow_html=True)
        # st.session_state["language"] = st.selectbox(
        #     "Choose Language", ["English", "Hindi"], index=0
        # )

        # Main Title
        st.markdown(
            """
        <div class="main-title">
            <h1>🌿 PRAGATI - Pollination Real-time Analysis for Growth in Agriculture using Tracking and Insights</h1>
            <p>Analyze insect and strawberry flower activity in your field and get tailored recommendations.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Step 1: Location handling
        # st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🌍 Step 1: Location", unsafe_allow_html=True)
        if st.button("📍 Fetch My Location"):
            latitude, longitude, city, country = get_location_from_ip()
            if latitude and longitude:
                st.session_state["latitude"] = latitude
                st.session_state["longitude"] = longitude
                st.session_state["city"] = city
                st.session_state["country"] = country
                st.success(
                    f"📍 Location fetched: {city}, {country} (Lat: {latitude}, Lon: {longitude})"
                )
            else:
                st.error(
                    "Could not fetch location. Please enter city and country manually."
                )
        st.markdown("</div>", unsafe_allow_html=True)

        # Fallback: Manual city/country input
        if "latitude" not in st.session_state:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 📝 Enter Location Manually", unsafe_allow_html=True)
            with st.form("location_form"):
                city = st.text_input("Enter City")
                country = st.selectbox(
                    "Select Country",
                    ["India", "United States", "China", "Brazil", "Other"],
                )
                if country == "Other":
                    country = st.text_input("Enter Country Name")
                submitted = st.form_submit_button("Submit Location")
                if submitted and city and country:
                    latitude, longitude = get_coordinates_from_city(city, country)
                    if latitude and longitude:
                        st.session_state["latitude"] = latitude
                        st.session_state["longitude"] = longitude
                        st.session_state["city"] = city
                        st.session_state["country"] = country
                        st.success(
                            f"📍 Location set: {city}, {country} (Lat: {latitude}, Lon: {longitude})"
                        )
                    else:
                        st.error("Invalid city/country. Please try again.")
            st.markdown("</div>", unsafe_allow_html=True)

        # Step 2: Upload media
        if "latitude" in st.session_state:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 📸 Step 2: Upload Video or Photo", unsafe_allow_html=True)
            uploaded_file = st.file_uploader(
                "Upload a video (.mp4, .avi) or photo (.jpg, .jpeg, .png)",
                type=["mp4", "avi", "jpg", "jpeg", "png"],
            )
            model_path = str(
                Path(__file__).parent.parent / "src" / "models" / "best(1).pt"
            )

            if uploaded_file and st.button("🔍 Analyze Media"):
                with st.spinner("Processing media..."):
                    results = process_media(uploaded_file, model_path)
                    st.session_state["insect_count"] = results["total_insects"]
                    st.session_state["flower_count"] = results["total_flowers"]
                    st.session_state["interaction_data"] = results[
                        "interaction_analysis"
                    ]

                    st.markdown("### 🐝 Detection Results")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(
                            f'<div class="metric"><span>{results["total_insects"]}</span><p>Pollinators</p></div>',
                            unsafe_allow_html=True,
                        )
                    with col2:
                        st.markdown(
                            f'<div class="metric"><span>{results["total_flowers"]}</span><p>Flowers</p></div>',
                            unsafe_allow_html=True,
                        )

                    # Add interaction metrics display
                    if results["interaction_analysis"]:
                        st.write("### Pollination Interaction Analysis")
                        st.write(
                            f"Total Interactions: {results["interaction_analysis"]['direct_contacts'] + results['interaction_analysis']['close_proximities']}"
                        )
                        st.write(
                            f"Direct Contacts: {results["interaction_analysis"]['direct_contacts']}"
                        )
                        st.write(
                            f"Close Proximities: {results["interaction_analysis"]["close_proximities"]}"
                        )
                        st.write(
                            f"Sufficiently Pollinated Flowers (>5 visits): {results['interaction_analysis']['sufficiently_pollinated_flowers']}"
                        )
                        # st.write(
                        #     f"Sufficient Pollination Percentage: {results['interaction_analysis']['sufficient_pollination_percentage']:.1f}%"
                        # )

                        # Calculate and display efficiency
                        efficiency = (
                            (
                                results["interaction_analysis"]["total_interactions"]
                                / results["total_flowers"]
                                * 100
                            )
                            if results["total_flowers"] > 0
                            else 0
                        )
                        st.write(f"Pollination Efficiency: {efficiency:.1f}%")

                    if results["total_insects"] < 5:
                        st.warning(
                            "Low pollinator count detected. Ensure video/photo quality or try another sample."
                        )

                    if results["sample_frame"] is not None:
                        with st.expander("🔍 View Sample Detection Frame"):
                            st.image(
                                results["sample_frame"],
                                caption="🌱 Sample Detection Frame",
                                channels="RGB",
                            )

                with st.spinner("Fetching weather data..."):
                    weather_data = get_weather_data(
                        st.session_state["latitude"], st.session_state["longitude"]
                    )
                    if weather_data:
                        st.session_state["weather_data"] = weather_data
                        st.markdown("### 🌤️ Weather Report")

                        # First row: 4 weather metrics
                        cols_top = st.columns(4)
                        weather_metrics = [
                            (
                                "🌡️ Temperature",
                                f"{weather_data['temperature']}°C",
                                "#4caf50",
                            ),
                            (
                                "🌬️ Wind Speed",
                                f"{weather_data['windspeed']} km/h",
                                "#4caf50",
                            ),
                            (
                                "💧 Precipitation",
                                f"{weather_data['precipitation']} mm",
                                "#4caf50",
                            ),
                            ("💧 Humidity", f"{weather_data['humidity']}%", "#4caf50"),
                        ]

                        for i, (label, value, color) in enumerate(weather_metrics):
                            with cols_top[i]:
                                st.markdown(
                                    f"""
                                <div class="weather-metric">
                                    {label}<br>
                                    <span class="metric-value" style="color: {color};">{value}</span>
                                </div>
                                """,
                                    unsafe_allow_html=True,
                                )

                        # Second row: Centered weather condition card
                        st.markdown(
                            '<div class="spacer"></div>', unsafe_allow_html=True
                        )  # Add vertical space

                        cols_bottom = st.columns(
                            [1, 2, 1]
                        )  # Left (1), Center (2), Right (1)
                        with cols_bottom[1]:  # Center column
                            condition = map_weather_code(weather_data["weathercode"])
                            badge_color = (
                                "#4caf50" if "clear" in condition.lower() else "#81c784"
                            )

                            st.markdown(
                                f"""
                            <div class="weather-condition">
                                🌤️ Condition<br>
                                <span class="condition-badge" style="background-color: {badge_color};">
                                    {condition}
                                </span>
                            </div>
                            """,
                                unsafe_allow_html=True,
                            )

                if "weather_data" in st.session_state:
                    with st.spinner("Generating AI-based recommendations..."):
                        recommendation = get_gemini_recommendation(
                            st.session_state["insect_count"],
                            st.session_state["flower_count"],
                            st.session_state["weather_data"],
                        )
                        st.session_state["recommendation"] = recommendation
                        st.markdown(
                            "### 🌱 Smart Recommendation", unsafe_allow_html=True
                        )
                        st.success(recommendation)
            st.markdown("</div>", unsafe_allow_html=True)

        # Chatbot Sidebar
        if st.session_state.get("recommendation", ""):
            with st.sidebar:
                st.markdown('<div class="sidebar">', unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

        # Footer
        st.markdown(
            '<div class="footer">© 2025 Pollination Monitoring System | Powered by Team8A ©',
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
