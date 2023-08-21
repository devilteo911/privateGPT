import requests

url = "http://localhost:7700/api/simpleChat"
while True:
    query = input("USER: ")
    data = {
        "query": f"USER: {query}. \
            ASSISTANT:",
        "params": {"temperature": 0.8, "max_tokens": 8192},
    }

    response = requests.post(url, json=data)
    print(f"AI: {response.text}")
