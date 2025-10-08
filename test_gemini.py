import requests
import json

url = "http://localhost:11434/api/generate"
payload = {
    "model": "deepseek-r1",
    "prompt": "Write a short note explaining what GSBG company could specialize in."
}

response = requests.post(url, json=payload, stream=True)

full_reply = ""
for line in response.iter_lines():
    if line:
        data = json.loads(line.decode("utf-8"))
        if "response" in data:
            full_reply += data["response"]

print("\n🧠 Model Response:\n")
print(full_reply)
