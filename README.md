We will use QLora to finetune a Qwen3-8B model.

Setup: 2xCPU cores with 32GB of memory each, 1xA40 GPU on the Misha server. 

1. Recognizing the instability and poor quality inherent in QLoRA, I chose to rely on unsloth's much higher quality quantization of a pretrained model, in my case Qwen3-8B to finetune using LoRA. This also has the added benefit of being able to use unsloth's FastLanguageModel package allowing for faster finetuning and inference. Run the command:

    huggingface-cli download --resume-download --local-dir-use-symlinks False /unsloth/Qwen3-8B-unsloth-bnb-4bit --local-dir unsloth

2. Next, to use unsloth, it is highly recommended to use a virtual environment with a newly installed base python 3.10 to run unsloth due to the rapid updating of the library. Me and Ms. Ping Luo worked on creating compatibility between unsloth, CUDA, and pytorch over the course of a 4 hour discussion and arrived at the below solution. We tried using conda environment and this did not work:

    module load Python/3.10.8-GCCcore-12.2.0-bare
    python -m venv unsloth_venv
    source unsloth_venv/bin/activate
    pip install "unsloth[cu124-torch250] @ git+https://github.com/unslothai/unsloth.git"

3. Next, set up your local directory so that it contains this folder which will contain your finetuned model:

    ./lora_4bit_qwen3_unsloth

4. Run:

    python finetune.py
    python summary.py

