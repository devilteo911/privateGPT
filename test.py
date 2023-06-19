import requests

url = "http://localhost:7700/api/simple_gen"
q = """
Mi creeresti 20 domande plausibili per il seguente testo? Ecco il testo: "Il sistema Operativo necessario per far girare Smathhall è Windows 7/10 Client con un PC con di 4 GB di ram e processore post 2011. Il browser web necessario è Chrome (versione 80.0.3987 o successiva), Edge (versione 79.0.309) e Firefox (versione 72.0)"
"""
data = {
    "query": q,
    "params": {"temperature": 0.1, "top_p": 0.95, "model_n_ctx": 8192},
    "callbacks": [],
}

response = requests.post(url, json=data)

if response.status_code == 200:
    print(response.json()["generations"][0]["text"])
else:
    print(response.status_code)
