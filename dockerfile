FROM rayproject/ray-llm:2.46.0-py311-cu124

WORKDIR /home/ray/app

COPY llm_app_ingress.py ./
# COPY requirements.txt ./   # Uncomment if you have dependencies
# RUN pip install -r requirements.txt