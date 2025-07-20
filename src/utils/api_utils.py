import json
import os

import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types
from ratelimit import limits, sleep_and_retry

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set")
client = genai.Client(api_key=api_key)


# Function to get location from IP
def get_location_from_ip():
    try:
        response = requests.get("https://ipinfo.io/json")
        data = response.json()
        loc = data["loc"].split(",")
        city = data.get("city", "Unknown")
        country = data.get("country", "Unknown")
        return float(loc[0]), float(loc[1]), city, country

    except Exception as e:
        return Exception(f"Failed to get location from IP: {str(e)}")


ONE_SECOND = 1


# Function to get coordinates from city/country
@sleep_and_retry
@limits(calls=1, period=ONE_SECOND)
def get_coordinates_from_city(city, country):
    try:
        url = f"https://nominatim.openstreetmap.org/search?city={city}&country={country}&format=json"
        headers = {"User-Agent": "PollinationMonitoring/1.0"}
        response = requests.get(url, headers=headers)
        if response.status_code == 200 and response.json():
            data = response.json()[0]
            return float(data["lat"]), float(data["lon"])
        raise Exception("No coordinates found for the given city and country")

    except Exception as e:
        return Exception(f"Failed to get coordinates: {str(e)}")


# Funtion to fetch weather data from API
def get_weather_data(latitude, longitude):
    if not isinstance(latitude, (int, float)) or not isinstance(
        longitude, (int, float)
    ):
        raise ValueError("Latitude and longitude must be numeric")

    if not (-90 <= latitude <= 90) or not (-180 <= longitude <= 180):
        raise ValueError("Invalid latitude or longitude value")
    url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,windspeed_10m,precipitation,weathercode,relativehumidity_2m"
    try:
        response = requests.get(url)
        response.raise_for_status()

        data = response.json()["current"]
        return {
            "temperature": data["temperature_2m"],
            "windspeed": data["windspeed_10m"],
            "precipitation": data["precipitation"],
            "humidity": data["relativehumidity_2m"],
            "weathercode": data["weathercode"],
        }
    except Exception as e:
        return Exception(f"Failed to fetch weather data: {str(e)}")


# Map Open-Meteo weather codes to description
def map_weather_code(code):
    weather_codes = {
        0: "Clear sky",
        1: "Mainly clear",
        2: "Partly cloudy",
        3: "Overcast",
        45: "Fog",
        48: "Depositing rime fog",
        51: "Light drizzle",
        53: "Moderate drizzle",
        55: "Dense drizzle",
        56: "Light freezing drizzle",
        57: "Dense freezing drizzle",
        61: "Light rain",
        63: "Moderate rain",
        65: "Heavy rain",
        66: "Light freezing rain",
        67: "Heavy freezing rain",
        71: "Light snow fall",
        73: "Moderate snow fall",
        75: "Heavy snow fall",
        77: "Snow grains",
        80: "Light rain showers",
        81: "Moderate rain showers",
        82: "Violent rain showers",
        85: "Light snow showers",
        86: "Heavy snow showers",
        95: "Thunderstorm",
        96: "Thunderstorm with slight hail",
        99: "Thunderstorm with heavy hail",
    }
    return weather_codes.get(code, f"Unknown weather code: {code}")


MODEL = "gemini-2.5-flash"


# Integration of Gemini
def get_gemini_recommendation(pollinator_count, flower_count, weather_data):
    prompt = f"""
    You are an agriculture advisor specializing in strawberry cultivation for farmers in India. Your task is to provide brief, actionable recommendations to a strawberry farmer based on the provided field data. Do not greet.

**Field Data:**
- Pollinators Detected: {pollinator_count}
- Flowers Detected: {flower_count}
- Temperature: {weather_data['temperature']}°C
- Wind Speed: {weather_data['windspeed']} km/h
- Precipitation: {weather_data['precipitation']} mm
- Humidity: {weather_data['humidity']}%
- Weather Condition: {map_weather_code(weather_data['weathercode'])}

**Your Response Must Include:**

1.  **Overview:** A concise summary of the current field conditions and their immediate implications for the strawberry crop, specifically highlighting any urgent actions needed.
2.  **Pollination Health Advice:** Specific, practical recommendations regarding the current pollination status. If pollinator count is low relative to flowers, advise on assisting pollination, clearly stating the *purpose* (e.g., to release pollen).
3.  **Flower Condition Observations:** Interpretations of the flower data, including any potential issues or positive indicators. Advise on identifying damaged or unhealthy flowers.
4.  **Weather-Based Actions/Precautions:** Practical steps the farmer should take or precautions to observe based on the temperature, wind, precipitation, humidity, and overall weather condition. Prioritize actions for immediate weather impacts (e.g., post-storm checks).
5.  **Budget-Specific Recommendations:** Categorized advice based on the farmer's available budget, with clear, concise actions:
    * **Low Budget:** Cost-effective and essential actions.
    * **Medium Budget:** Moderately priced, beneficial improvements.
    * **High Budget:** More significant investments for optimal yield and protection.

**Key Requirements:**
* Keep the recommendations brief, practical, and easy for an Indian farmer to understand and implement.
* The tone should be advisory and helpful.
* Prioritize actions that directly impact strawberry yield and plant health.
* Use clear, straightforward language; avoid jargon.
* Bold keywords where appropriate for emphasis.
* Ensure a logical flow and structure.
* Conclude with a prompt for further action or monitoring.
    """

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.7, candidate_count=1),
    )

    return response.text


def get_gemini_chatbot_response_with_interactions(
    user_question,
    recommendation,
    pollinator_count,
    flower_count,
    weather_data,
    chat_history,
    interaction_data,
):
    """
    Enhanced chatbot response that provides detailed interaction-based advice
    """

    if not isinstance(pollinator_count, int) or pollinator_count < 0:
        raise ValueError("pollinator_count must be a non-negative integer")
    if not isinstance(flower_count, int) or flower_count < 0:
        raise ValueError("flower_count must be a non-negative integer")
    required_weather_keys = [
        "temperature",
        "windspeed",
        "precipitation",
        "humidity",
        "weathercode",
    ]
    if not isinstance(weather_data, dict) or not all(
        key in weather_data for key in required_weather_keys
    ):
        raise ValueError("weather_data must be a dictionary with required keys")
    required_interaction_keys = [
        "total_interactions",
        "direct_contacts",
        "close_proximities",
        "avg_distance",
    ]
    if not isinstance(interaction_data, dict) or not all(
        key in interaction_data for key in required_interaction_keys
    ):
        raise ValueError("interaction_data must be a dictionary with required keys")
    # Calculate additional metrics
    pollination_efficiency = (
        (interaction_data.get("total_interactions", 0) / flower_count * 100)
        if flower_count > 0
        else 0
    )
    pollinator_to_flower_ratio = (
        pollinator_count / flower_count if flower_count > 0 else 0
    )
    interaction_density = (
        interaction_data.get("total_interactions", 0) / pollinator_count
        if pollinator_count > 0
        else 0
    )

    # Create comprehensive context
    context = f"""
    **CURRENT FIELD STATUS:**
    - Pollinators Detected: {pollinator_count}
    - Flowers Detected: {flower_count}
    - Pollinator-to-Flower Ratio: {pollinator_to_flower_ratio:.2f}
    
    **POLLINATION INTERACTION METRICS:**
    - Total Interactions: {interaction_data.get('total_interactions', 0)}
    - Direct Contacts: {interaction_data.get('direct_contacts', 0)}
    - Close Proximities: {interaction_data.get('close_proximities', 0)}
    - Pollination Efficiency: {pollination_efficiency:.1f}%
    - Interaction Density: {interaction_density:.2f} interactions per pollinator
    - Average Interaction Distance: {interaction_data.get('avg_distance', 0):.2f} pixels
    
    **WEATHER CONDITIONS:**
    - Temperature: {weather_data.get('temperature', 'N/A')}°C
    - Wind Speed: {weather_data.get('windspeed', 'N/A')} km/h
    - Precipitation: {weather_data.get('precipitation', 'N/A')} mm
    - Humidity: {weather_data.get('humidity', 'N/A')}%
    - Weather Condition: {map_weather_code(weather_data.get('weathercode', 0))}
    
    **CURRENT RECOMMENDATION CONTEXT:**
    {recommendation}
    """

    # Add conversation history
    conversation_history = ""
    if chat_history:
        conversation_history = "\n**RECENT CONVERSATION:**\n"
        for i, (user_msg, bot_response) in enumerate(chat_history[-3:]):
            conversation_history += f"Farmer: {user_msg}\nAssistant: {bot_response}\n\n"

    prompt = f"""
    You are an expert agricultural advisor specializing in strawberry cultivation and pollination monitoring for farmers in India. A farmer is asking a follow-up question about their field conditions, with particular focus on pollinator-flower interactions and pollination effectiveness.

    {context}
    {conversation_history}

    **FARMER'S QUESTION:** {user_question}

    **Your Response Must Address:**

    1. **Direct Answer:** Provide a clear, specific answer to the farmer's question
    
    2. **Interaction Context:** When relevant, explain how the interaction data relates to their question:
       - What the current pollination efficiency means for their specific concern
       - How direct contacts vs proximities affect the issue they're asking about
       - Whether interaction quality is contributing to their problem
    
    3. **Actionable Advice:** Provide specific steps the farmer can take, considering:
       - Current interaction levels and what they indicate
       - Weather conditions affecting pollination activity
       - Budget-friendly solutions available in India
       - Timing recommendations for optimal results
    
    4. **Success Indicators:** Tell the farmer what to monitor to track improvement:
       - Specific interaction metrics to watch
       - Signs of successful pollination
       - Timeline for seeing results
    
    5. **Follow-up Guidance:** Suggest next steps or additional monitoring

    **Response Guidelines:**
    - Keep response focused and practical (200-400 words)
    - Use **bold** for key actions and metrics
    - Explain technical terms in simple language
    - Provide specific numerical targets when applicable
    - Include cost considerations for small-scale farmers
    - Reference interaction data meaningfully, not just superficially
    - Give timing recommendations (best hours, days, seasons)
    - Include safety precautions when relevant

    **Key Topics to Address When Relevant:**
    - How to interpret pollination efficiency percentages
    - What direct contacts vs proximities mean for fruit set
    - How weather affects pollinator activity and flower receptivity
    - Signs of poor vs good pollination (visual cues)
    - Methods to increase pollinator-flower interactions
    - Flower positioning and accessibility
    - Pollinator behavior patterns and timing
    - Natural vs managed pollination approaches
    - Cost-effective solutions for improving pollination

    **Response Format:**
    - Start with direct answer to their question
    - Provide specific steps with clear actions
    - Include relevant interaction metrics and their meaning
    - End with monitoring recommendations and next steps
    - Maintain supportive, encouraging tone throughout

    Please provide a comprehensive response that helps the farmer understand and improve their strawberry pollination based on the interaction data.
    """

    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.7, candidate_count=1),
        )
        return response.text
    except Exception as e:
        return f"I'm experiencing technical difficulties analyzing your pollination data. Please try asking your question again, and I'll provide guidance based on your current field conditions. (Error: {str(e)})"


def get_interaction_summary_report(
    pollinator_count, flower_count, interaction_data, weather_data
):
    """
    Generate a comprehensive interaction summary report
    """
    pollination_efficiency = (
        (interaction_data.get("total_interactions", 0) / flower_count * 100)
        if flower_count > 0
        else 0
    )
    pollinator_to_flower_ratio = (
        pollinator_count / flower_count if flower_count > 0 else 0
    )

    prompt = f"""
    Generate a comprehensive pollination interaction report for a strawberry farmer in India.

**Field Data:**
- Pollinators: {pollinator_count}, Flowers: {flower_count}
- Pollinator-to-Flower Ratio: {pollinator_to_flower_ratio:.2f}
- Total Interactions: {interaction_data.get('total_interactions', 0)}
- Direct Contacts: {interaction_data.get('direct_contacts', 0)}
- Close Proximities: {interaction_data.get('close_proximities', 0)}
- Pollination Efficiency: {pollination_efficiency:.1f}%
- Interaction Quality: {interaction_data.get('interaction_quality', 'Unknown')}
- Weather: {map_weather_code(weather_data.get('weathercode', 0))}, {weather_data.get('temperature', 'N/A')}°C

**Provide a structured report with:**

1. **Pollination Status Overview** (Current effectiveness level)
2. **Interaction Analysis** (What the numbers mean for yield)
3. **Immediate Actions Required** (Priority interventions)
4. **Performance Benchmarks** (Target metrics to achieve)
5. **Monitoring Schedule** (When and what to check)

Keep it concise, practical, and focused on actionable insights for Indian strawberry farmers.
    """

    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.6, candidate_count=1),
        )
        return response.text
    except Exception as e:
        return f"Unable to generate interaction report. Please check your data and try again. (Error: {str(e)})"


def get_gemini_chatbot_response(
    user_question, recommendation, bee_count, flower_count, weather_data, chat_history
):
    """
    Get chatbot response for farmer's follow-up questions using Gemini API
    """
    # Create context from the current data
    context = f"""
    CONTEXT INFORMATION:
    - Current Recommendation: {recommendation}
    - Bees Detected: {bee_count}
    - Flowers Detected: {flower_count}
    - Temperature: {weather_data.get('temperature', 'N/A')}°C
    - Wind Speed: {weather_data.get('windspeed', 'N/A')} km/h
    - Precipitation: {weather_data.get('precipitation', 'N/A')} mm
    - Humidity: {weather_data.get('humidity', 'N/A')}%
    - Weather Condition: {map_weather_code(weather_data.get('weathercode', 0))}
    """

    # Add recent chat history for context
    conversation_history = ""
    if chat_history:
        conversation_history = "\n\nRECENT CONVERSATION:\n"
        for i, (user_msg, bot_response) in enumerate(
            chat_history[-3:]
        ):  # Last 3 exchanges for context
            conversation_history += f"Farmer: {user_msg}\nAssistant: {bot_response}\n\n"

    prompt = f"""
    You are an expert agricultural advisor specializing in strawberry cultivation for farmers in India. A farmer has received recommendations based on their field analysis and now has a follow-up question.

    {context}
    {conversation_history}

    FARMER'S CURRENT QUESTION: {user_question}

    **Instructions:**
    1. Answer the farmer's question directly and clearly
    2. Reference the current field conditions and recommendations when relevant
    3. Provide practical, actionable advice suitable for Indian farming conditions
    4. Use simple, easy-to-understand language appropriate for Indian farmers
    5. If the question is about implementing a recommendation, provide step-by-step guidance
    6. If the question is about potential concerns, address them with practical solutions
    7. Keep responses concise but comprehensive (200-300 words max)
    8. If you need more information to give a complete answer, ask specific clarifying questions

    **Response Guidelines:**
    - Be supportive and encouraging
    - Focus on practical solutions that farmers can implement
    - Consider budget constraints of small-scale farmers
    - Provide seasonal and weather-appropriate advice
    - Include safety precautions when relevant
    - Reference previous conversation context when helpful
    - Use **bold** for important keywords or actions
    - Suggest cost-effective alternatives when possible

    **Response Format:**
    - Start with a direct answer to the question
    - Provide specific steps or recommendations
    - Include any relevant warnings or precautions
    - End with encouragement or next steps

    Please provide a helpful response to the farmer's question.
    """

    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.7, candidate_count=1),
        )
        return response.text
    except Exception as e:
        return f"I'm experiencing technical difficulties. Please try asking your question again. (Error: {str(e)})"
