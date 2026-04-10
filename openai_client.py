import openai

# Configure the OpenAI client to connect to your local server
client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="your-api-key"  # You can use any string here, as the local server doesn't likely validate it
)

# Create a chat completion request
try:
    response = client.chat.completions.create(
        model="HuggingFaceTB/SmolLM2-135M-Instruct",
        messages=[
            {"role": "user", "content": "Hello! Who are you?"}
        ]
    )
    print(response.choices[0].message.content)

except openai.APIConnectionError as e:
    print(f"Failed to connect to the server: {e}")
except openai.APIStatusError as e:
    print(f"Received an error response from the server: {e.status_code}")
    print(e.response)