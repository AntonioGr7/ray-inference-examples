import requests
import json



url = "http://localhost:8000/v1/chat/completions"

payload = json.dumps({
"request_id": "id-1234",
"messages": [{"role": "user", "content": "Hello, how are you?"}],
"max_new_tokens": 100,
"temperature": 0.7
})
headers = {
'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)


for line in response.iter_lines():
    if line:
        t = json.loads(line.decode('utf-8'))
        c = t['text']
        print(c,end='')
