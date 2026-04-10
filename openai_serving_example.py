import asyncio
from typing import List

from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_models import OpenAIServingModels, BaseModelPath
from vllm.config import ModelConfig
from vllm.entrypoints.openai.protocol import ChatCompletionRequest, ErrorResponse
from typing import AsyncGenerator
import uuid

async def main():
    
    # Define the model to be served.
    # or a local path to your model.
    model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"

    # Define the engine arguments.
    engine_args = AsyncEngineArgs(
        model=model_name,
        tensor_parallel_size=1,
        enforce_eager=True,
        gpu_memory_utilization=0.5,
    )

    # Create the vLLM engine.
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    model_config: ModelConfig = await engine.get_model_config()
    # Define the served model.
    served_model = BaseModelPath(
        name=model_name.split("/")[-1],  # Extract model name from the path
        model_path=model_name
    )
    
    # Create an instance of OpenAIServingModels.
    # This class manages the models that are being served.
    serving_models = OpenAIServingModels(
        engine_client=engine,
        base_model_paths = [served_model],
        model_config=model_config
    )

    # Create an instance of OpenAIServingChat.
    # This class handles the chat completion logic.
    serving_chat = OpenAIServingChat(
        engine_client=engine,
        models=serving_models, # Pass the served_model list
        response_role="assistant", # This can be different. You need to manage it dynamically based on models if you have multiple models with different roles.
        model_config=model_config,
        chat_template=engine.tokenizer.chat_template,  # Use default chat template
        chat_template_content_format="string",
        request_logger=None
    )

    print("Successfully initialized OpenAIServingModels and OpenAIServingChat.")
    print("To run a full OpenAI-compatible server, you would typically use vllm.entrypoints.openai.api_server.")
    
    # In a real application, you would now use these objects within a web server
    # framework (like FastAPI) to handle incoming API requests.
    # For this example, we will just print the initialized objects.

    print(f"\nOpenAIServingModels instance: {serving_models}")
    print(f"\nOpenAIServingChat instance: {serving_chat}")
    stream = True
    
    request = ChatCompletionRequest(
        model="SmolLM2-135M-Instruct",
        messages=[
            {"role": "system", "content": "You are a helpful and concise assistant."},
            {"role": "user", "content": "What is the capital of France?"}
        ],
        temperature=0.7,
        max_tokens=256,
        # A unique ID for the request.
        request_id=f"cmpl-{uuid.uuid4().hex}", 
        # Set stream=True to get a streaming response.
        stream=stream
    )

    print(f"Sending inference request for model: '{request.model}'")
    print(f"User Message: '{request.messages[-1]['content']}'")
    print("-" * 40)
    print("Assistant's Response:")
    import json
    import re
    response_generator: AsyncGenerator = await serving_chat.create_chat_completion(request)
    if stream==False:
        print(response_generator)
    else:
        async for chunk in response_generator:
            s = chunk
            data_reg = re.search(r'data:\s*(\{.*\})', s)
            if data_reg is not None:
                json_str = data_reg.group(1)
            obj = json.loads(json_str)
            print(obj['choices'][0]['delta'].get('content'),end='')
            #print(obj)
    
    #start_time = time.perf_counter()

if __name__ == "__main__":
    asyncio.run(main())