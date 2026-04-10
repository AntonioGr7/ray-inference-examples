from ray import serve
from ray.serve.llm import LLMConfig, LLMServingArgs, LLMServer,  build_openai_app
from ray.serve.llm.ingress import OpenAiIngress, make_fastapi_ingress

# --- Configuration for first model ---
llm_config_0 = LLMConfig(
    model_loading_config={
        "model_id":"llm-1",
        "model_source":"HuggingFaceTB/SmolLM2-135M-Instruct"
    },
    engine_kwargs=dict(
       
        gpu_memory_utilization=0.2,
        max_model_len=512,
        tensor_parallel_size=1,
    ),
    placement_group_config=dict(
        bundles=[{"CPU": 0.2, "GPU": 0.3}],
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
)

# --- Configuration for the second model ---
llm_config_1 = LLMConfig(
    model_loading_config={
        "model_id":"llm-2",
        "model_source":"HuggingFaceTB/SmolLM2-135M-Instruct"
    },
    engine_kwargs=dict(
        gpu_memory_utilization=0.0,
        max_model_len=256,
        tensor_parallel_size=1,
    ),
    placement_group_config=dict(
        bundles=[{"CPU": 0.1, "GPU": 0.3}],
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
            "VLLM_USE_V1": "1",
        }
    )
)




# Deploy the application, combining both configs
llm_app = build_openai_app(
    LLMServingArgs(
        llm_configs=[
            llm_config_0,
            llm_config_1,
        ],
        
    )
)


serve.run(llm_app,blocking=True)