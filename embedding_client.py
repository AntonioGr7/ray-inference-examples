import asyncio
import httpx # You will need to install this: pip install httpx
import json
import time
import random
from typing import List, Dict, Any

# --- Configuration ---

# The URL of your running Ray Serve application
URL = "http://127.0.0.1:8000/embed"

# Set the headers to indicate JSON content
HEADERS = {
    "Content-Type": "application/json"
}

# A list of different texts to choose from randomly
QUERIES = [
    "What is the capital of France?",
    "How does a transformer model work?",
    "Explain the theory of relativity in simple terms.",
    "What are the main ingredients in a pizza margherita?",
    "Who wrote the novel 'Pride and Prejudice'?",
    "How to train a dog to sit?",
    "What is the tallest mountain in the world?",
    "Describe the process of photosynthesis.",
    "What is Ray Serve used for?",
    "How to make the perfect cup of coffee?",
    "What are the benefits of a Mediterranean diet?",
    "Explain the concept of blockchain technology."
]

# The shared task description for all queries
TASK = "Given a web search query, retrieve relevant passages that answer the query"

async def send_async_request(client: httpx.AsyncClient) -> Dict[str, Any]:
    """
    Selects a random query and sends a single POST request asynchronously.
    """
    # Randomly pick a query from the list
    random_query = random.choice(QUERIES)
    
    query_data = {
        "text": random_query,
        "task": TASK
    }
    
    try:
        # Use await with the async httpx client
        response = await client.post(URL, data=json.dumps(query_data))
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        
        embeddings = response.json().get("embeddings", [])
        return {
            "query": random_query,
            "success": True,
            "error": None,
            "dim": len(embeddings[0]) if embeddings and embeddings[0] else 0
        }
    except httpx.RequestError as e:
        return {
            "query": random_query,
            "success": False,
            "error": f"Request failed: {e}"
        }
    except Exception as e:
         return {
            "query": random_query,
            "success": False,
            "error": f"An unexpected error occurred: {e}"
        }


async def main_async(num_requests: int):
    """
    Sends a specified number of concurrent requests using asyncio.
    """
    print(f"Sending {num_requests} concurrent requests using asyncio...")
    start_time = time.time()
    
    # httpx.AsyncClient is used to manage connection pooling and headers
    async with httpx.AsyncClient(headers=HEADERS, timeout=30.0) as client:
        # Create a list of all concurrent tasks
        tasks = [send_async_request(client) for _ in range(num_requests)]
        
        # Run all tasks concurrently and wait for them to complete
        results: List[Dict[str, Any]] = await asyncio.gather(*tasks)

    end_time = time.time()
    duration = end_time - start_time
    
    # --- Process Results ---
    successful_requests = 0
    failed_requests = 0
    
    for result in results:
        if result["success"]:
            print(f"Success for query: '{result['query']}' -> Received embedding with dimension {result['dim']}")
            successful_requests += 1
        else:
            print(f"Request for '{result['query']}' failed: {result['error']}")
            failed_requests += 1

    print("\n" + "="*40)
    print("           Request Summary")
    print("="*40)
    print(f"Total requests sent: {num_requests}")
    print(f"Successful requests: {successful_requests}")
    print(f"Failed requests:     {failed_requests}")
    print(f"Total time taken:    {duration:.2f} seconds")
    if duration > 0:
        print(f"Requests per second: {num_requests / duration:.2f}")
    print("="*40)
    print("\nCheck your Ray Serve logs to see the actual batch size handled by the model!")


if __name__ == "__main__":
    N = 512
    # Run the asynchronous main function
    asyncio.run(main_async(N))