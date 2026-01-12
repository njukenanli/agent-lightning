# Eval with Claude Code

## Compatibility on linux and windows

SWE-agent and OpenHands cannot run on raw windows, but Claude Code can.

To run Claude Code with any LLMs we need a middleware router to convert Claude Code format request <=> OpenAI format request for most LLMs. Here we use the router of Agent-Lightning.

## Setup

```bash
python3.12 -m venv .venv

source .venv/bin/activate

bash setup.sh
```

The setup on Windows should be the same.

## Run Inference

```bash
cd examples/cc
```

Run Internal AOAI models

```bash
nohup python cc_agent.py \
        --model_name azure/gpt-5.2-20251211\
        --agent_config agent_config.yaml > log.out 2>&1 &
```

Run Openrouter models

```bash
nohup python cc_agent.py \
        --model_name openrouter/deepseek/deepseek-v3.1-terminus \
        --api_base https://openrouter.ai/api/v1 \
        --api_key sk-or-v1-...... \
        --agent_config agent_config.yaml > log.out 2>&1 &
```

Note the end of api base should not have `\`.

Results would be saved to patch/model_name/instance_id.diff

### Modify agent_config.yaml for your own setting

The dataset_path should point to a jsonl file.

The jsonl file should contain these fields:
- instance_id
- problem_statement
- docker_image

### Parallelism

Not fully implemented. 

Please split the jsonl file into 4-8 files and start cc_agent.py for each file.

### Windows Compatibility

Currently the code is only ok on Linux.

To support running on Windows you need to add some if-else into `examples/cc/utils/claude_code_controller.py`.
