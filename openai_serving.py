from fastapi import HTTPException, Request
from typing import Dict
from vllm import AsyncEngineArgs, AsyncLLMEngine
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from ray import serve
from ray.serve.handle import DeploymentHandle
import uuid
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ErrorResponse,
    ChatCompletionResponse,
)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_models import OpenAIServingModels
from vllm.entrypoints.openai.serving_models import BaseModelPath
from vllm.config import ModelConfig

app = FastAPI()
            
@serve.deployment(autoscaling_config={"min_replicas": 1, "max_replicas": 1}, ray_actor_options={"num_cpus": 0.2, "num_gpus": 0.0})
@serve.ingress(app)
class APIIngress:

    def __init__(self, handles: Dict[str, DeploymentHandle]) -> None:
        self.models = handles 
        print("Models:", self.models)
    
    @app.post("/v1/chat/completions")
    async def chat(self, request: ChatCompletionRequest, raw_request: Request):
        if request.model not in self.models:
            raise HTTPException(status_code=404, detail="Model not found")
        
        request_id = None
        if request_id is None:
            request_id = f"chat-{str(uuid.uuid4())}"
        
        if request.model not in self.models:
            raise HTTPException(status_code=404, detail="Model not found")

        handle = self.models.get(request.model)
        if handle is None:
            raise HTTPException(status_code=404, detail="Model not found")
        
        request_dict = request.model_dump()

        if request.stream:
            generator = handle.options(stream=True).serve_request.remote(request_dict)
            return StreamingResponse(
                content=generator, 
                media_type="text/event-stream"
            ) 
        else: 
            print("Request:", request)
            generator = await handle.serve_request.remote(request_dict)

         
        if isinstance(generator, ErrorResponse):
            return JSONResponse(
                content=generator.model_dump(), status_code=generator.code
            ) 
        else:
            assert isinstance(generator, ChatCompletionResponse)
            return JSONResponse(content=generator.model_dump())
        


@serve.deployment(health_check_period_s=10, ray_actor_options={"num_cpus": 0.2, "num_gpus": 0.5})
class VLLMDeployment:
    def __init__(self,engine_args: Dict[str, str]): 
        self.engine_args = engine_args
        self.model_path = self.engine_args['model']
        self.model_name = self.model_path.split("/")[-1] #Extract model name from path
        self.model_config = None
            
    async def init_service_engine(self):
        print("Initializing service engine...")
        engine_args = AsyncEngineArgs(**self.engine_args)

        # Create the vLLM engine.
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.model_config: ModelConfig = await self.engine.get_model_config()
        # Define the served model.
        served_model = BaseModelPath(
            name=self.model_name,
            model_path=self.model_path
        )
        
        # Create an instance of OpenAIServingModels.
        # This class manages the models that are being served.
        self.serving_models = OpenAIServingModels(
            engine_client=self.engine,
            base_model_paths = [served_model],
            model_config=self.model_config
        )

        # Create an instance of OpenAIServingChat.
        # This class handles the chat completion logic.
        self.openai_serving_chat = OpenAIServingChat(
            engine_client=self.engine,
            models=self.serving_models, # Pass the served_model list
            response_role="assistant", # This can be different. You need to manage it dynamically based on models if you have multiple models with different roles.
            model_config=self.model_config,
            chat_template=self.engine.tokenizer.chat_template,  # Use default chat template
            chat_template_content_format="string",
            request_logger=None
        )

        

    async def serve_request(self, request : ChatCompletionRequest):
        print("Serving request...")
        if self.model_config is None: # The model is loaded at first request. You can change this behavior as you wish.
            await self.init_service_engine()
        request = ChatCompletionRequest(**request)
        request.model = self.model_name
        try:
            generator = await self.openai_serving_chat.create_chat_completion(request, raw_request=None)
        except Exception as e:
            print("Error in serving request:", e)
            raise e
        return generator 
        

if __name__ == "__main__":
        model= "HuggingFaceTB/SmolLM2-135M-Instruct"
        engine_args = {
            "model":model,
            "enforce_eager": True, # Set to false for better performance in production
            "tensor_parallel_size":1,
            "gpu_memory_utilization": 0.5
        }
        llm_serving = VLLMDeployment.bind(engine_args)
        llm_app = APIIngress.bind({model:llm_serving})
        serve.run(llm_app, blocking=True)