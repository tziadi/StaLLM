# StaLLM_llm.py
import os
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
from openai import AzureOpenAI, OpenAI

__all__ = [
    "LLMConfig",
    "ChatModel",
    "available_models",
    "load_llm_registry",
    "build_llm_from_slot",
    "default_slot_key",
]

@dataclass
class LLMConfig:
    provider: str = "azure-openai"  # "azure-openai" | "openai" | "ollama"
    model: str = ""                 # Azure: deployment; OpenAI: model id; Ollama: model name
    api_base: Optional[str] = None  # Azure endpoint; OpenAI base_url; Ollama host
    api_key: Optional[str] = None
    api_version: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 1400

def _guess_slots():
    declared = [s.strip() for s in os.getenv("STALLM_SLOTS", "").split(",") if s.strip()]
    if declared:
        return declared
    slots = []
    for key in os.environ:
        if key.endswith("_PROVIDER"):
            slots.append(key[:-9])
    return sorted(set(slots))

def _guess_azure_slot_env():
    cands = [k[:-9] for k, v in os.environ.items() if k.endswith("_PROVIDER") and v.lower().strip() == "azure-openai"]
    if not cands:
        for k, v in os.environ.items():
            if k.endswith("_API_BASE") and v:
                val = v.lower()
                if "openai.azure.com" in val or ".ais.azure.com" in val:
                    cands.append(k[:-8])
    for s in cands:
        key = os.getenv(f"{s}_API_KEY")
        base = os.getenv(f"{s}_API_BASE") or os.getenv(f"{s}_AZURE_ENDPOINT")
        ver  = os.getenv(f"{s}_API_VERSION")
        dep  = os.getenv(f"{s}_DEPLOYMENT")
        if key and base:
            return key, base, ver, dep
    return None, None, None, None

class ChatModel:
    """
    Wrapper LLM.
    - chat(..., return_meta=True) -> (text, meta) avec tokens.
    - slot_key: rempli si construit via build_llm_from_slot (utilisé pour tarifs).
    """
    slot_key: Optional[str] = None

    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        prov = (cfg.provider or "azure-openai").lower()

        if prov == "azure-openai":
            key = cfg.api_key or os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
            endpoint = cfg.api_base or os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("OPENAI_API_BASE")
            version = cfg.api_version or os.getenv("AZURE_OPENAI_API_VERSION") or os.getenv("OPENAI_API_VERSION", "2024-05-01-preview")
            if not key or not endpoint:
                k2, b2, v2, d2 = _guess_azure_slot_env()
                key = key or k2
                endpoint = endpoint or b2
                version = version or v2
                if not cfg.model:
                    cfg.model = d2 or cfg.model
            if not key:
                raise ValueError("Azure key missing. Set AZURE_OPENAI_API_KEY or provide <SLOT>_API_KEY.")
            if not endpoint:
                raise ValueError("Azure endpoint missing. Set AZURE_OPENAI_ENDPOINT or provide <SLOT>_API_BASE.")
            self.client = AzureOpenAI(api_key=key, api_version=version or "2024-05-01-preview", azure_endpoint=endpoint)

        elif prov == "openai":
            self.client = OpenAI(
                api_key=cfg.api_key or os.getenv("OPENAI_API_KEY"),
                base_url=(cfg.api_base or os.getenv("OPENAI_BASE_URL") or None),
            )

        elif prov == "ollama":
            if cfg.api_base:
                os.environ["OLLAMA_HOST"] = cfg.api_base
            import ollama  # type: ignore
            self.client = ollama
        else:
            raise ValueError(f"Unknown provider: {prov}")

    @staticmethod
    def _usage_dict(obj: Any) -> Dict[str, int]:
        """
        Normalise 'usage' en dict: prompt/completion/total tokens.
        """
        pt = ct = tt = 0
        if obj is None:
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        if hasattr(obj, "prompt_tokens") or hasattr(obj, "completion_tokens") or hasattr(obj, "total_tokens"):
            pt = getattr(obj, "prompt_tokens", 0)
            ct = getattr(obj, "completion_tokens", 0)
            tt = getattr(obj, "total_tokens", (pt or 0) + (ct or 0))
        if hasattr(obj, "input_tokens") or hasattr(obj, "output_tokens"):
            pt = pt or getattr(obj, "input_tokens", 0)
            ct = ct or getattr(obj, "output_tokens", 0)
            tt = tt or (pt + ct)
        if isinstance(obj, dict):
            pt = obj.get("prompt_eval_count", obj.get("prompt_tokens", pt))
            ct = obj.get("eval_count", obj.get("completion_tokens", ct))
            tt = obj.get("total_tokens", pt + ct)
        return {
            "prompt_tokens": int(pt or 0),
            "completion_tokens": int(ct or 0),
            "total_tokens": int(tt or (pt or 0) + (ct or 0))
        }

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        return_meta: bool = False,
    ):
        prov = (self.cfg.provider or "azure-openai").lower()
        temperature = self.cfg.temperature if temperature is None else temperature
        max_tokens = self.cfg.max_tokens if max_tokens is None else max_tokens

        if prov == "azure-openai":
            resp = self.client.chat.completions.create(
                model=self.cfg.model or os.getenv("OPENAI_DEPLOYMENT_NAME", ""),
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            text = resp.choices[0].message.content or ""
            meta = self._usage_dict(getattr(resp, "usage", None))
            meta["model"] = getattr(resp, "model", self.model_label())
            return (text, meta) if return_meta else text

        elif prov == "openai":
            resp = self.client.chat.completions.create(
                model=self.cfg.model or os.getenv("OPENAI_MODEL", "gpt-4o"),
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            text = resp.choices[0].message.content or ""
            meta = self._usage_dict(getattr(resp, "usage", None))
            meta["model"] = getattr(resp, "model", self.model_label())
            return (text, meta) if return_meta else text

        elif prov == "ollama":
            out = self.client.chat(model=self.cfg.model or "llama3", messages=messages)
            text = (out.get("message", {}) or {}).get("content", "")
            meta = self._usage_dict(out)
            meta["model"] = self.model_label()
            return (text, meta) if return_meta else text

        raise ValueError("Unsupported provider")

    def model_label(self) -> str:
        prov = (self.cfg.provider or "azure-openai").lower()
        if prov == "azure-openai":
            return f"azure:{self.cfg.model or 'deployment'}"
        return f"{prov}:{self.cfg.model or 'model'}"

def load_llm_registry():
    reg = {}
    for slot in _guess_slots():
        prov = os.getenv(f"{slot}_PROVIDER", "").lower()
        if not prov:
            continue
        label = os.getenv(f"{slot}_LABEL", slot)
        if prov == "azure-openai":
            cfg = LLMConfig(
                provider="azure-openai",
                model=os.getenv(f"{slot}_DEPLOYMENT", ""),
                api_base=os.getenv(f"{slot}_API_BASE") or os.getenv(f"{slot}_AZURE_ENDPOINT") or os.getenv("OPENAI_API_BASE"),
                api_key=os.getenv(f"{slot}_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY"),
                api_version=os.getenv(f"{slot}_API_VERSION") or os.getenv("AZURE_OPENAI_API_VERSION") or os.getenv("OPENAI_API_VERSION", "2024-05-01-preview"),
            )
        elif prov == "openai":
            cfg = LLMConfig(
                provider="openai",
                model=os.getenv(f"{slot}_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o")),
                api_base=os.getenv(f"{slot}_API_BASE") or os.getenv("OPENAI_BASE_URL"),
                api_key=os.getenv(f"{slot}_API_KEY") or os.getenv("OPENAI_API_KEY"),
            )
        elif prov == "ollama":
            cfg = LLMConfig(
                provider="ollama",
                model=os.getenv(f"{slot}_MODEL", "llama3"),
                api_base=os.getenv(f"{slot}_HOST") or os.getenv(f"{slot}_API_BASE"),
            )
        else:
            continue
        reg[slot] = {"label": label, "config": cfg}
    return reg

def build_llm_from_slot(slot_key: str) -> ChatModel:
    reg = load_llm_registry()
    if slot_key not in reg:
        raise ValueError(f"Unknown LLM slot '{slot_key}'. Check STALLM_SLOTS or *_PROVIDER in your .env.")
    cm = ChatModel(reg[slot_key]["config"])
    cm.slot_key = slot_key  # pour tarifs .env
    return cm

def default_slot_key() -> Optional[str]:
    reg = load_llm_registry()
    if not reg:
        return None
    explicit = os.getenv("STALLM_DEFAULT_SLOT")
    if explicit and explicit in reg:
        return explicit
    for key in _guess_slots():
        if key in reg:
            return key
    return None

def available_models(provider: str):
    provider = (provider or "azure-openai").lower()
    if provider == "azure-openai":
        env_list = os.getenv("STALLM_MODELS_AZURE_OPENAI", "")
        return [x.strip() for x in env_list.split(",") if x.strip()]
    elif provider == "openai":
        env_list = os.getenv("STALLM_MODELS_OPENAI", "gpt-4o,gpt-4o-mini")
        return [x.strip() for x in env_list.split(",") if x.strip()]
    elif provider == "ollama":
        env_list = os.getenv("STALLM_MODELS_OLLAMA", "llama3,phi3")
        return [x.strip() for x in env_list.split(",") if x.strip()]
    return []
