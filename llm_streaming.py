from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from ray import serve
from ray.serve.handle import DeploymentHandle
from fastapi.responses import JSONResponse, StreamingResponse
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm import SamplingParams
from vllm.engine.async_llm_engine import AsyncLLMEngine
import json

app = FastAPI()


@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 0.2, "num_gpus": 0.0})
@serve.ingress(app)
class APIIngress:
    def __init__(self, llm: DeploymentHandle):
        self.handle = llm
       

    @app.post("/v1/chat/completions")
    async def chat(self, request: Request):
        body = await request.json()
        print("Received request body:", body)
        request_id = body.get("request_id", "default_request_id")
        #chat_request = ChatCompletionRequest(**body)
        messages = body.get("messages")
        if not messages: # This must be checked before using Pydantic
            return JSONResponse({"error": "Messages are required"}, status_code=400)
        max_tokens = body.get("max_new_tokens", 100)
        temperature = body.get("temperature", 0.8)

        #response = self.handle.options(stream=True).chat.remote(chat_request)
        results_generator = self.handle.options(stream=True).inference.remote(request_id, messages, max_tokens, temperature)
       
        async def response_generator():
            async for chunk in results_generator:
                # The chunk here is the raw text yielded by the VLLMDeployment
                if chunk:
                    chunk_data = json.dumps({"text": chunk})
                    yield f"{chunk_data}\n\n"
            
            
     
        return StreamingResponse(response_generator(), media_type="text/event-stream")
    

@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 0.2, "num_gpus": 0.5})
class VLLMDeployment:
    def __init__(self):
        engine_args = AsyncEngineArgs(
            model="HuggingFaceTB/SmolLM2-135M-Instruct",
            enforce_eager=True, # Set to false for better performance in production
            gpu_memory_utilization=0.5) 
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.tokenizer = self.engine.tokenizer
    

    async def inference(self, request_id, prompt: str, max_tokens: int = 100, temperature: float = 0.8, top_p: float = 0.95, seed: int = 42):
            sampling_params = SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                seed=seed,)
            
            prompt = self.tokenizer.apply_chat_template(prompt, tokenize=False)

            results_generator = self.engine.generate(
            prompt, sampling_params, request_id)

            try:
                last_text = ""
                async for request_output in results_generator:
                    current_text = request_output.outputs[0].text
                    # Yield only the newly generated text delta
                    yield current_text[len(last_text):]
                    last_text = current_text

            except Exception as e:
                print(f"Error during vLLM generation: {e}")
                # You might want to yield an error message or handle it differently
                yield " [ERROR] "
    
llm_serving = VLLMDeployment.bind()
llm_app = APIIngress.bind(llm_serving)


if __name__ == "__main__":
    serve.run(llm_app, blocking=True)