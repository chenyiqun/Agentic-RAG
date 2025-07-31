import requests

def google_search(api_key, query, num=10, page=1):
    # API endpoint
    url = f"{API_BASE}/plugins/google-search"

    # Headers for the API request
    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    # Parameters for the API
    params = {
        "query": query,
        "num": num,
        "page": page
    }

    try:
        # Sending GET request to the API with headers and parameters
        response = requests.get(url, headers=headers, params=params)
        
        # Check if the request was successful
        response.raise_for_status()

        # Parsing the JSON response
        results = response.json()

        return results

    except requests.exceptions.RequestException as e:
        # Handle any errors that occur
        print(f"An error occurred: {e}")
        return None

# Example usage
API_KEY = ""
API_BASE = "https://api.openai-proxy.org/v1"

query = "OpenAI"
results = google_search(API_KEY, query, num=5, page=1)

if results:
    print("Search Results:")
    for result in results:
        print(result)