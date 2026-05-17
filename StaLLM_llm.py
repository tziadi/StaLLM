# StaLLM_llm.py
import os
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any

try:
    from openai import AzureOpenAI, OpenAI
except ModuleNotFoundError as exc:
    if exc.name != "openai":
        raise
    raise ModuleNotFoundError(
        "The 'openai' package is missing from the Python environment used to run StarLLM. "
        "From the StaLLM folder, run the app with './.venv/bin/streamlit run StaLLM_app.py' "
        "or install dependencies with 'python -m pip install -r requirements.txt'."
    ) from exc

__all__ = [
    "LLMConfig",
    "ChatModel",
    "available_models",
    "load_llm_registry",
    "build_llm_from_slot",
    "default_slot_key",
    "test_ollama_connectivity",
    "validate_ollama_config",
    "debug_ollama_response",
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
    - chat(..., return_meta=True) -> (text, meta) with tokens.
    - slot_key: set when built via build_llm_from_slot (used for pricing).
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
            # Do NOT set global env permanently; keep host per-instance
            self.ollama_host = cfg.api_base or os.getenv("OLLAMA_HOST", "http://localhost:11434")
            import ollama  # type: ignore
            self.client = ollama
        else:
            raise ValueError(f"Unknown provider: {prov}")

    @staticmethod
    def _usage_dict(obj: Any) -> Dict[str, int]:
        """
        Normalize token usage into a dict: prompt/completion/total tokens.
        Also handles Ollama fields and provides rough estimates when missing.
        """
        pt = ct = tt = 0
        if obj is None:
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        # OpenAI/Azure styles
        if hasattr(obj, "prompt_tokens") or hasattr(obj, "completion_tokens") or hasattr(obj, "total_tokens"):
            pt = getattr(obj, "prompt_tokens", 0)
            ct = getattr(obj, "completion_tokens", 0)
            tt = getattr(obj, "total_tokens", (pt or 0) + (ct or 0))

        # Alternative names
        if hasattr(obj, "input_tokens") or hasattr(obj, "output_tokens"):
            pt = pt or getattr(obj, "input_tokens", 0)
            ct = ct or getattr(obj, "output_tokens", 0)
            tt = tt or (pt + ct)

        # Dict (Ollama and others)
        if isinstance(obj, dict):
            # Common Ollama fields first
            pt = obj.get("prompt_eval_count", obj.get("prompt_tokens", obj.get("input_tokens", pt)))
            ct = obj.get("eval_count", obj.get("completion_tokens", obj.get("output_tokens", ct)))

            # Nested usage
            if (pt == 0 and ct == 0) and "usage" in obj and isinstance(obj["usage"], dict):
                usage = obj["usage"]
                pt = usage.get("prompt_tokens", usage.get("prompt_eval_count", pt))
                ct = usage.get("completion_tokens", usage.get("eval_count", ct))

            # If still nothing, estimate from message content
            if pt == 0 and ct == 0:
                message = obj.get("message", {})
                if isinstance(message, dict):
                    content = message.get("content", "")
                    if content:
                        est_ct = max(1, len(content) // 4)  # ~1 token ≈ 4 chars
                        est_pt = max(1, est_ct // 2)
                        ct = est_ct
                        pt = est_pt

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
            # Temporarily set env for this call only
            import os as _os
            original_host = _os.environ.get("OLLAMA_HOST")
            try:
                _os.environ["OLLAMA_HOST"] = getattr(self, "ollama_host", "http://localhost:11434")

                model_name = self.cfg.model or "llama3"
                try:
                    out = self.client.chat(
                        model=model_name,
                        messages=messages,
                        options={
                            "temperature": temperature,
                            "num_predict": max_tokens,
                        }
                    )
                except Exception as e:
                    msg = str(e)
                    if "not found" in msg.lower() and "model" in msg.lower():
                        raise ValueError(f"Model '{model_name}' not found on Ollama host {self.ollama_host}.")
                    if "connection" in msg.lower() or "refused" in msg.lower():
                        raise ConnectionError(f"Cannot connect to Ollama host {self.ollama_host}. Is it running?")
                    raise RuntimeError(f"Ollama error: {msg}")

                text = (out.get("message", {}) or {}).get("content", "")
                meta = self._usage_dict(out)
                # Fallback estimate if still 0/0
                if meta["prompt_tokens"] == 0 and meta["completion_tokens"] == 0 and text:
                    est_ct = max(1, len(text) // 4)
                    meta["completion_tokens"] = est_ct
                    total_input_chars = sum(len(m.get("content", "")) for m in messages)
                    est_pt = max(1, total_input_chars // 4)
                    meta["prompt_tokens"] = est_pt
                    meta["total_tokens"] = est_pt + est_ct

                meta["model"] = self.model_label()
                return (text, meta) if return_meta else text
            finally:
                if original_host is not None:
                    _os.environ["OLLAMA_HOST"] = original_host
                elif "OLLAMA_HOST" in _os.environ:
                    del _os.environ["OLLAMA_HOST"]

        raise ValueError("Unsupported provider")

    def model_label(self) -> str:
        prov = (self.cfg.provider or "azure-openai").lower()
        if prov == "azure-openai":
            return f"azure:{self.cfg.model or 'deployment'}"
        elif prov == "ollama":
            host_info = getattr(self, 'ollama_host', 'localhost:11434')
            if host_info.startswith('http://'):
                host_info = host_info[7:]
            elif host_info.startswith('https://'):
                host_info = host_info[8:]
            return f"ollama:{self.cfg.model or 'model'}@{host_info}"
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
                api_base=os.getenv(f"{slot}_HOST") or os.getenv(f"{slot}_API_BASE") or os.getenv("OLLAMA_HOST", "http://localhost:11434"),
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
    cm.slot_key = slot_key  # used for pricing
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

def available_models(provider: str, host: Optional[str] = None):
    provider = (provider or "azure-openai").lower()
    if provider == "azure-openai":
        env_list = os.getenv("STALLM_MODELS_AZURE_OPENAI", "")
        return [x.strip() for x in env_list.split(",") if x.strip()]
    elif provider == "openai":
        env_list = os.getenv("STALLM_MODELS_OPENAI", "gpt-4o,gpt-4o-mini")
        return [x.strip() for x in env_list.split(",") if x.strip()]
    elif provider == "ollama":
        # Try hit the host if provided
        if host:
            try:
                import ollama
                import os as _os
                original_host = _os.environ.get("OLLAMA_HOST")
                try:
                    _os.environ["OLLAMA_HOST"] = host
                    models = ollama.list()
                    return [m["name"] for m in models.get("models", [])]
                finally:
                    if original_host is not None:
                        _os.environ["OLLAMA_HOST"] = original_host
                    elif "OLLAMA_HOST" in _os.environ:
                        del _os.environ["OLLAMA_HOST"]
            except Exception:
                pass
        # Fallback to env list
        env_list = os.getenv("STALLM_MODELS_OLLAMA", "llama3,phi3")
        return [x.strip() for x in env_list.split(",") if x.strip()]
    return []

def test_ollama_connectivity(host: str) -> Tuple[bool, str]:
    """
    Test connectivity to an Ollama host and return (success, message).
    """
    try:
        import requests
        if not host.startswith(('http://', 'https://')):
            host = f"http://{host}"
        test_url = f"{host.rstrip('/')}/api/tags"
        timeout = int(os.getenv("STALLM_OLLAMA_TIMEOUT", "10"))
        response = requests.get(test_url, timeout=timeout)
        if response.status_code == 200:
            try:
                data = response.json()
                models = data.get("models", [])
                return True, f"Connected successfully ({len(models)} models available)"
            except Exception:
                return True, "Connected successfully"
        else:
            return False, f"HTTP {response.status_code}: {response.text}"
    except Exception as e:
        msg = str(e)
        if "Connection refused" in msg or "Failed to establish a new connection" in msg:
            return False, "Connection refused - is Ollama running on this host?"
        if "Invalid URL" in msg:
            return False, "Invalid URL format - use http://host:port or https://host:port"
        return False, f"Error: {msg}"

def validate_ollama_config(host: str, model: str) -> Tuple[bool, str]:
    """
    Validate that a specific model is available on the given Ollama host.
    """
    try:
        import ollama
        import requests  # noqa: F401  (ensures dependency is present for connectivity)
        if not host.startswith(('http://', 'https://')):
            host = f"http://{host}"
        ok, msg = test_ollama_connectivity(host)
        if not ok:
            return False, f"Host connectivity failed: {msg}"
        import os as _os
        original_host = _os.environ.get("OLLAMA_HOST")
        try:
            _os.environ["OLLAMA_HOST"] = host
            models = ollama.list()
            avail = [m["name"] for m in models.get("models", [])]
            if model in avail:
                return True, f"Model '{model}' is available"
            return False, f"Model '{model}' not found. Available: {', '.join(avail[:5])}{'...' if len(avail) > 5 else ''}"
        finally:
            if original_host is not None:
                _os.environ["OLLAMA_HOST"] = original_host
            elif "OLLAMA_HOST" in _os.environ:
                del _os.environ["OLLAMA_HOST"]
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def debug_ollama_response(response: Any) -> Dict[str, Any]:
    """
    Debug helper to inspect Ollama response structure for token information.
    """
    info: Dict[str, Any] = {
        "response_type": type(response).__name__,
        "response_keys": [],
        "has_usage": False,
        "has_eval_count": False,
        "has_prompt_eval_count": False,
        "raw_response": str(response)[:500] + "..." if len(str(response)) > 500 else str(response)
    }
    if isinstance(response, dict):
        info["response_keys"] = list(response.keys())
        info["has_usage"] = "usage" in response
        info["has_eval_count"] = "eval_count" in response
        info["has_prompt_eval_count"] = "prompt_eval_count" in response
        if "usage" in response and isinstance(response["usage"], dict):
            info["usage_keys"] = list(response["usage"].keys())
        if "message" in response and isinstance(response["message"], dict):
            info["message_keys"] = list(response["message"].keys())
    return info
