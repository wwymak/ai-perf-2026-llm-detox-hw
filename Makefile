
include .env
export


connect_vm_with_forwarding:
	ssh -A -L 3000:localhost:3000 \
		-L 9090:localhost:9090 \
		-L 3001:localhost:3001 \
		-L 8000:localhost:8000 \
		-L 8001:localhost:8001 \
		-i ~/.ssh/$(SSH_KEY_NAME) -o IdentitiesOnly=yes \
		$(VM_USER)@$(IP_ADDRESS)

connect_vm:
	ssh -A -i ~/.ssh/$(SSH_KEY_NAME) -o IdentitiesOnly=yes \
		$(VM_USER)@$(IP_ADDRESS)
#
#ssh -A -i ~/.ssh/id_ed25519 -o IdentitiesOnly=yes \
#		wwymak@195.242.13.226
forward_ssh_keys:
	# edit ~/.ssh/config with the correct host and ForwardAgent: yes
	ssh-agent && ssh-add ~/.ssh/id_ed25519

# lambda labs setup
setup:
	curl -LsSf https://astral.sh/uv/install.sh | sh \
	cd ~/nebius-hw-us-east1 \
	git clone git@github.com:wwymak/ai-perf-course-2026-mlops-assignment-3.git && \
	cd ai-perf-course-2026-mlops-assignment-3 \
	sudo apt-get update -y && \
	sudo apt-get install -y nvidia-container-toolkit && \
	sudo nvidia-ctk runtime configure --runtime=docker && \
	sudo apt install -y docker-compose && \
	sudo adduser "$(id -un)" docker && \
	sudo systemctl restart docker \
	uv sync --system-certs && \
	uv run python scripts/load_data.py \
	source $UV_PROJECT_ENVIRONMENT/bin/activate \
	uv pip install vllm --torch-backend=auto

prepare_data:
	uv run --env-file .env python -m data_prep.build_pairs --out-dir data --max-rows 80000

run_vllm:
	docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:v0.20.0-cu130 \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
   --max-model-len 19904

run_vllm_v2:
	docker run --gpus all -v ./infra:/infra -v ~/.cache/huggingface:/root/.cache/huggingface \
	-p 8000:8000 \
		--ipc=host \
	 vllm/vllm-openai:v0.22.1 --config /infra/vllm_config.yaml

run_guide_llm:
	guidellm benchmark \
	  --target http://localhost:8000 \
	  --profile concurrent \
	  --rate 16 \
	  --warmup 0.1 \
	  --cooldown 0.1 \
	  --max-errors 5 \
	  --max-seconds 120 \
	  --data "kind=synthetic_text,prompt_tokens=2094,output_tokens=2094" \
	  --detect-saturation




#docker run --gpus all -v ./infra/vllm_config.yaml:/config.yaml -v ~/.cache/huggingface:/root/.cache/huggingface \
 #-p 8000:8000 \
 #    --ipc=host \
 # vllm/vllm-openai:latest --config /config.yaml
