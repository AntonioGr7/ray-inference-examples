import requests
import json

url = "http://localhost:8000/v1/chat/completions"

payload = json.dumps({
  "model": "llm-1",
  "messages": [
    {
      "role": "user",
      "content": "How are you?"
    }
  ]
})
headers = {
  'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)