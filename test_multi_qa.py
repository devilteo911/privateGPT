import requests

url = "http://localhost:7700/api/multiTest"
data = {"query": "", "params": {}}

response = requests.post(url, json=data)
print(response.json())
