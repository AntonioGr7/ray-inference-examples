from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import asyncio

from ray import serve
from ray.serve.handle import DeploymentHandle
from ray.serve.llm import LLMConfig, LLMServer
from ray.serve.llm.openai_api_models import ChatCompletionRequest
from fastapi.responses import JSONResponse, StreamingResponse

import json


app = FastAPI()

class MockChunk:
    def __init__(self, content):
        class Message:
            def __init__(self, content):
                self.content = content
        
        class Choice:
            def __init__(self, content):
                self.message = Message(content)
        
        # A mock list of choices containing the content chunk
        self.choices = [Choice(content)]

@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 0.2, "num_gpus": 0.0})
class SimpleCiaoLLM:
    """
    A simple deployment that simulates an LLM by always returning the stream "ciao".
    It returns an async generator of MockChunk objects to match the expected format.
    """
    async def chat(self, request: ChatCompletionRequest):
        # This function returns an async generator that yields chunks
        async def generator():
            for char in "ciao":
                # Yield a new chunk for each character
                yield MockChunk(char)
                # Introduce a small delay to simulate streaming
                await asyncio.sleep(0.01)
            # Yield the final chunk with None content to signal the end of the stream
            yield MockChunk(None) 
        
        return generator()

@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 0.2, "num_gpus": 0.0})
@serve.ingress(app)
class APIIngress:
    def __init__(self, llm_1_handle: DeploymentHandle, llm_2_handle: DeploymentHandle):
        self.llm_1_handle = llm_1_handle
        self.llm_2_handle = llm_2_handle

    @app.post("/v1/chat/completions")
    async def chat(self, request: Request):
        """
        This endpoint mimics the OpenAI Chat Completions API.
        It routes requests to the appropriate model based on the 'model' field in the request body.
        """
        body = await request.json()
        model_id = body.get("model")

        if model_id == "llm-1":
            handle = self.llm_1_handle
        elif model_id == "llm-2":
            handle = self.llm_2_handle
        else:
            return JSONResponse({"error": f"Invalid model_id: {model_id}"}, status_code=400)

        chat_request = ChatCompletionRequest(**body)
        response = handle.options(stream=True).chat.remote(chat_request)
        async def response_generator(response): ### Questo è da rivedere 
            async for chunk in response:
                content = chunk.choices[0].message.content
                if content is None:
                    # After the stream is complete, yield the final [DONE] message 
                    # as required by the Server-Sent Events (SSE) protocol for OpenAI API.
                    yield "data: [DONE]\n\n"
                    break 
                chunk_data = json.dumps({"text": content}) 
                yield f"data: {chunk_data}\n\n"
     
        return StreamingResponse(response_generator(response), media_type="text/event-stream")
    

'''llm_config_0 = LLMConfig(
    model_loading_config={
        "model_id":"llm-1",
        "model_source":"HuggingFaceTB/SmolLM2-135M-Instruct"
    },
    engine_kwargs=dict(
       
        gpu_memory_utilization=0.5,
        max_model_len=512,
        tensor_parallel_size=1,
    ),
    placement_group_config=dict(
        bundles=[{"CPU": 0.5, "GPU": 0.5}],
        strategy="PACK",
    ),
    deployment_config=dict(
        #num_replicas=1,
        #ray_actor_options={"num_gpus": 0.0,"num_cpus":0.2},
        autoscaling_config=dict(
            min_replicas=1,
            max_replicas=1,
        ),
        
    ),
    runtime_env=dict(
        env_vars={
            # Tell vLLM to use half a GPU
            "VLLM_USE_V1": "1",
        }
    )
)'''

#server_options_0 = LLMServer.get_deployment_options(llm_config_0)
#llm_server_0 = serve.deployment(LLMServer).options(**server_options_0).bind(llm_config_0)

llm_server_0 = SimpleCiaoLLM.bind()
llm_server_1 = SimpleCiaoLLM.bind()


llm_app = APIIngress.bind(llm_server_0, llm_server_1)
# to test it locally uncomment this
#serve.run(app, blocking=True)