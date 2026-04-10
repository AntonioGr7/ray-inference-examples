from vllm.engine.arg_utils import AsyncEngineArgs
from ray.serve.llm import LLMConfig, LLMServer
from vllm import SamplingParams

from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.async_llm import AsyncLLM


engine_args = AsyncEngineArgs(
            model="HuggingFaceTB/SmolLM2-135M-Instruct",
            enforce_eager=True, # Set to false for better performance in production
            gpu_memory_utilization=0.5) 
engine = AsyncLLM.from_engine_args(engine_args)


tokenizer = engine.tokenizer
chat_template = tokenizer.chat_template
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello, how are you?"}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False)

print(f"Prompt:\n{prompt}")
output = engine.generate(request_id="test_request", prompt=prompt, sampling_params=SamplingParams())

for completion in output.outputs:
    print(f"Completion:\n{completion.text}")