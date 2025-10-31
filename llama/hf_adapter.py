from typing import Optional
from peft import PeftModel
import torch

try:
    from transformers import LlamaForCausalLM, LlamaTokenizerFast
except Exception as e:
    raise ImportError(
        "transformers is required for HF adapter. Install with: pip install transformers"
    ) from e

class HFTransformerAdapter(torch.nn.Module):
    def __init__(self, hf_model_dir: str, lora_dir: str = None, device=None):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LlamaForCausalLM.from_pretrained(
            hf_model_dir,
            torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
            device_map="auto"
        )
        # Lora权重
        if lora_dir is not None:
            print(f"加载Lora权重，源：{lora_dir}")
            self.model = PeftModel.from_pretrained(self.model, lora_dir)
            self.model.merge_and_unload()

        self.model.eval()
        self.vocab_size = self.model.config.vocab_size

    @torch.no_grad()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        model_device = next(self.model.parameters()).device
        input_ids = tokens.to(model_device)

        # use_cache=False 防止只输出最后一步
        outputs = self.model(input_ids=input_ids, use_cache=False, return_dict=True)
        logits = outputs.logits # (B, T, V)

        #返回最后 token 的 logits
        last = logits[:, -1, :] # (B, V)
        return last.float()