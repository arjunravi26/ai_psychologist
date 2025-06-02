import os
import torch
import logging
import threading
from typing import Optional, Dict, Any

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedTokenizerBase,
    GenerationConfig,
    TextIteratorStreamer,
)
from peft import PeftModel, PeftConfig

# ─── Logging Configuration ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def _build_bnb_config(
    load_in_4bit: bool = True,
    bnb_4bit_compute_dtype: str = "float16",
    bnb_4bit_quant_type: str = "nf4",
    bnb_4bit_use_double_quant: bool = False,
) -> BitsAndBytesConfig:
    """
    Construct a BitsAndBytesConfig for QLoRA (4-bit quantization).
    Adjust compute dtype based on GPU capability to minimize memory footprint.
    """
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
    if load_in_4bit and torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        if major < 8:
            logger.warning(
                f"GPU compute capability {major} < 8.0: "
                "4-bit bfloat16 acceleration may be suboptimal."
            )

    return BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
    )


class PsychologistAssistant:
    """
    Singleton‐style class that:
      1) Loads a base causal LLM + LoRA‐finetuned weights (QLoRA).
      2) Exposes `respond(...)` for non-streaming inference.
      3) Exposes `respond_stream(...)` for streaming token-by-token output.
    """

    _instance: Optional["PsychologistAssistant"] = None

    def __new__(cls, model_dir: str):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize(model_dir)
        return cls._instance

    def _initialize(self, model_dir: str) -> None:
        """
        Load model + tokenizer into GPU/CPU with 4-bit quantization + device_map="auto".
        """
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        # 1) Load PEFT config to find the base checkpoint path
        logger.info(f"Locating PEFT config at: {model_dir}")
        peft_cfg = PeftConfig.from_pretrained(model_dir)
        base_checkpoint = peft_cfg.base_model_name_or_path

        # # 2) Prepare 4-bit quantization config (minimize VRAM)
        bnb_config = _build_bnb_config(
            load_in_4bit=True,
            bnb_4bit_compute_dtype="float16",
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=False,
        )

        # 3) Load the base model in 4-bit + device_map="auto"
        logger.info(f"Loading base model: {base_checkpoint} (4-bit QLoRA)...")
        self.model: torch.nn.Module = AutoModelForCausalLM.from_pretrained(
            base_checkpoint,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )

        # 4) Merge LoRA weights
        logger.info("Applying LoRA (PEFT) weights...")
        self.model = PeftModel.from_pretrained(self.model, model_dir)

        # 5) Load tokenizer (fast if available)
        logger.info(f"Loading tokenizer for: {base_checkpoint}")
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            base_checkpoint,
            use_fast=True,
        )

        # 6) Configure generation defaults
        self.gen_config = GenerationConfig(
            do_sample=True,
            temperature=0.7,
            max_new_tokens=100,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # 7) Switch to eval mode and disable grad to reduce memory
        self.model.eval()
        torch.set_grad_enabled(False)
        logger.info("Model ready for inference.")

    def respond(self, prompt: str, **override_gen_kwargs) -> str:
        """
        Generate a psychologist‐style response for a given prompt (non-streaming).

        Args:
            prompt (str): The user’s input text.
            override_gen_kwargs: Optional generation parameters to override defaults.

        Returns:
            str: The generated assistant response (trimmed of whitespace).
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt must be a non-empty string.")

        formatted = f"<s>[INST] {prompt.strip()} [/INST]"
        logger.debug(f"Formatted prompt: {formatted}")

        inputs = self.tokenizer(
            formatted, return_tensors="pt").to(self.model.device)

        gen_kwargs = {**self.gen_config.to_dict(), **override_gen_kwargs}

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        text = self.tokenizer.decode(
            outputs[0], skip_special_tokens=True).strip()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Generation complete.")
        return text

    def respond_stream(self, prompt: str, **override_gen_kwargs) -> None:
        """
        Stream a psychologist‐style response token by token to stdout.

        Args:
            prompt (str): The user’s input text.
            override_gen_kwargs: Optional generation parameters to override defaults.

        Yields:
            None: Prints tokens to stdout as they arrive.
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt must be a non-empty string.")

        formatted = f"<s>[INST] {prompt.strip()} [/INST]"
        logger.debug(f"Formatted prompt for streaming: {formatted}")

        inputs = self.tokenizer(
            formatted, return_tensors="pt").to(self.model.device)

        gen_kwargs: Dict[str, Any] = {
            **self.gen_config.to_dict(), **override_gen_kwargs}

        # TextIteratorStreamer will yield tokens as they are generated
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        gen_kwargs["streamer"] = streamer
        gen_kwargs.update(inputs)

        # Launch generation in a separate thread
        thread = threading.Thread(
            target=self.model.generate, kwargs=gen_kwargs)
        thread.start()

        # Print tokens as they come in
        logger.info("Streaming generation started...")
        print("AI: ", end="", flush=True)
        for token in streamer:
            print(token, end="", flush=True)

        # Ensure thread finishes before clearing cache
        thread.join()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Streaming generation complete.")


if __name__ == "__main__":
    """
    Example usage:
      $ export LORA_MODEL_PATH=/path/to/finetuned/weights
      $ python psych_stream_engine.py
    """
    try:
        model_path = os.getenv(
            "LORA_MODEL_PATH",
            r"fine_tuned_weights\Llama-2-7b-chat-hf-finetune",
        )
        prompt_text = "I often feel overwhelmed at work. What should I do?"

        logger.info("Instantiating PsychologistAssistant...")
        assistant = PsychologistAssistant(model_path)

        logger.info("Generating non-streamed response...")
        reply = assistant.respond(
            prompt_text, temperature=0.6, max_new_tokens=80)
        print("\n\n=== Non-Streamed Reply ===\n")
        print(reply)
        print("\n==========================\n")

        logger.info("Generating streamed response...")
        assistant.respond_stream(
            prompt_text, temperature=0.6, max_new_tokens=80)
        print("\n")  # finalize newline after streaming

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
