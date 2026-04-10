import ray
from ray import serve
from ray.serve.handle import DeploymentHandle


from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F
from torch import Tensor
from fastapi import FastAPI, Request

from pydantic import BaseModel, Field
from typing import List, Optional


class EmbeddingRequest(BaseModel):
    text: str
    task: Optional[str] = Field(None, description="The one-sentence instruction that describes the task for the queries.")

class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]


@serve.deployment(num_replicas=1, max_ongoing_requests=512, ray_actor_options={"num_cpus": 0.5, "num_gpus": 0.5})
class EmbeddingModel:
    def __init__(self, device: str = "cuda"):
        self.device = device
        print(f"Using device: {self.device}")
        model_name = "Qwen/Qwen3-Embedding-0.6B"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        self.model = AutoModel.from_pretrained(model_name, device_map=self.device)
       
        self.model.eval()
    
    def get_detailed_instruct(self,task_description: str, query: str) -> str:
        return f'Instruct: {task_description}\nQuery:{query}'

    def last_token_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor: 
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    async def _embed(self, requests: List[EmbeddingRequest]) -> List[List[float]]:
        # Prepend instruction to texts if a task is provided
        input_data = []
        for r in requests:
            if r.task:
                input_data.append(self.get_detailed_instruct(r.task, r.text))
            else:
                input_data.append(r.text)

        max_length = 8192
        # Tokenize the input texts
        batch_dict = self.tokenizer(
            input_data,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        # Move tensors to the correct device
        batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}

        with torch.no_grad():
            outputs = self.model(**batch_dict)
    
        embeddings = self.last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        # Normalize embeddings
        normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
        return normalized_embeddings.cpu().tolist()

    @serve.batch(max_batch_size=64, batch_wait_timeout_s=1)
    async def __call__(self, requests: List[EmbeddingRequest]) -> List[List[float]]:
        outputs = await self._embed(requests)
        print(f"Batch size: {len(requests)}, Embeddings generated: {len(outputs)}")
        return outputs  # Dummy output for testing

app = FastAPI()

@serve.deployment(num_replicas=1, max_ongoing_requests=1024, ray_actor_options={"num_cpus": 0.5, "num_gpus": 0.0})
@serve.ingress(app)
class APIIngress:
    def __init__(self, embedding_handle: DeploymentHandle):
        self.embedding_handle = embedding_handle

    @app.post("/embed", response_model=EmbeddingResponse)
    async def embed(self, request: EmbeddingRequest):
        """
        Receives a list of texts and an optional task description,
        and returns the corresponding embeddings.
        """
        # Call the embedding model. Ray Serve's dynamic batching will automatically
        # batch these requests if multiple calls arrive in a short window.
        embeddings = await self.embedding_handle.remote(request)        
        return EmbeddingResponse(embeddings=[embeddings])

# Deploy the applications
embedding_app = EmbeddingModel.bind()
api_app = APIIngress.bind(embedding_handle=embedding_app)

# You can adjust batching parameters if needed.
# For example, to enable dynamic batching with a max batch size and timeout:
# You can add this to the @serve.deployment decorator for EmbeddingModel
# or configure it when binding:
# serve.run(embedding_app.options(
#     batching_params=serve.BatchingParams(
#         max_batch_size=32,
#         batch_wait_timeout_s=0.05
#     )
# ))

# To deploy with a specific route:
serve.run(api_app, blocking=True)