from langchain_core.tools import tool
import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("AVIATION_STACK_API_KEY")
BASE_URL = "https://api.aviationstack.com/v1/flights"
DEFAULT_ORIGIN_IATA = os.getenv("DEFAULT_ORIGIN_IATA", "DEL")


@tool
def search_flights(
    destination_iata: str,
    origin_iata: str | None = None,
    limit: int = 5
) -> str:
    """
    Search available flight information between two airports.

    Always call this tool when the user asks about flights or travel
    between two locations.

    Do NOT ask the user for travel dates because this tool does not
    require a date.

    destination_iata:
        Destination airport IATA code, e.g. CDG.

    origin_iata:
        Departure airport IATA code, e.g. DEL.
        If omitted, the default origin airport is used.

    limit:
        Maximum number of flight results.
    """ 

    origin_iata = origin_iata or DEFAULT_ORIGIN_IATA


    if not API_KEY:
        return "AVIATIONSTACK_API_KEY is not configured."

    params = {
        "access_key": API_KEY,
        "dep_iata": origin_iata.upper(),
        "arr_iata": destination_iata.upper(),
        "limit": limit,
    }

    try:
        response = requests.get(
            BASE_URL,
            params=params,
            timeout=15
        )

        response.raise_for_status()
        data = response.json()

    except requests.RequestException as e:
        return f"Flight API request failed: {e}"

    if "error" in data:
        return f"Flight API error: {data['error']}"

    flights = data.get("data", [])

    if not flights:
        return "No flights found."

    results = []

    for flight in flights:
        results.append({
            "airline": flight.get("airline", {}).get("name"),
            "flight": flight.get("flight", {}).get("iata"),
            "departure": flight.get("departure", {}).get("iata"),
            "arrival": flight.get("arrival", {}).get("iata"),
            "status": flight.get("flight_status"),
        })

    return str(results)