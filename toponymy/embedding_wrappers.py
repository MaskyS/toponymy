from __future__ import annotations

import os
import warnings
from typing import List, Optional, Protocol, Sequence

import numpy as np
from tqdm.auto import tqdm

from toponymy._ai_sdk_bridge import call_ai_sdk_bridge
from toponymy._utils import handle_verbose_params


class TextEmbedderProtocol(Protocol):
    """Protocol defining the minimal interface required by a text embedder."""

    def encode(
        self,
        texts: Sequence[str],
        show_progress_bar: Optional[bool],
        *args,
        **kwargs,
    ) -> np.typing.NDArray[np.floating]: ...


class _AISDKEmbedder:
    def __init__(self, provider: str, model: str, **provider_config):
        self.provider = provider
        self.model = model
        self.provider_config = provider_config

    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        payload = {
            "action": "embed_many",
            "provider": self.provider,
            "model": self.model,
            "texts": texts,
            **self.provider_config,
        }
        result = call_ai_sdk_bridge(payload)
        embeddings = result.get("embeddings", [])
        return np.asarray(embeddings, dtype=np.float32)

    def encode(
        self, texts: List[str], verbose: bool = None, show_progress_bar: bool = None
    ) -> np.ndarray:
        show_progress_bar_val, _ = handle_verbose_params(
            verbose=verbose,
            show_progress_bar=show_progress_bar,
            default_verbose=False,
        )

        if len(texts) == 0:
            return np.empty((0, 0), dtype=np.float32)

        result = []
        for i in tqdm(
            range(0, len(texts), 96),
            desc="embedding texts",
            disable=(not show_progress_bar_val),
        ):
            result.append(self._encode_batch(texts[i : i + 96]))

        return np.vstack(result)


class CohereEmbedder(_AISDKEmbedder):
    def __init__(
        self,
        api_key,
        model: str = "embed-multilingual-v3.0",
        base_url: str = None,
        httpx_client=None,
    ):
        api_key = api_key or os.environ.get("CO_API_KEY") or os.environ.get("COHERE_API_KEY")
        if not api_key:
            raise ValueError(
                "Cohere API key is required. Set `CO_API_KEY` or `COHERE_API_KEY`, or pass `api_key`."
            )
        if base_url is None:
            base_url = os.environ.get("CO_API_URL")
        if httpx_client is not None:
            warnings.warn(
                "`httpx_client` is ignored by CohereEmbedder in AI SDK mode.",
                stacklevel=2,
            )
        self.input_type = "search_query"
        self.embedding_types = ["float"]
        super().__init__(
            provider="cohere",
            model=model,
            apiKey=api_key,
            baseURL=base_url,
            providerOptions={
                "cohere": {
                    "inputType": self.input_type,
                }
            },
        )


class OpenAIEmbedder(_AISDKEmbedder):
    def __init__(
        self,
        api_key,
        model: str = "text-embedding-3-small",
        base_url: str = None,
        http_client=None,
    ):
        if http_client is not None:
            warnings.warn(
                "`http_client` is ignored by OpenAIEmbedder in AI SDK mode.",
                stacklevel=2,
            )
        super().__init__(
            provider="openai",
            model=model,
            apiKey=api_key,
            baseURL=base_url,
        )


class AzureAIEmbedder(_AISDKEmbedder):
    def __init__(self, api_key: str, endpoint: str, model: str):
        super().__init__(
            provider="azure_inference",
            model=model,
            apiKey=api_key,
            endpoint=endpoint,
        )


class MistralEmbedder(_AISDKEmbedder):
    def __init__(self, api_key: str, model: str = "mistral-embed"):
        super().__init__(
            provider="mistral",
            model=model,
            apiKey=api_key,
        )


class VoyageAIEmbedder(_AISDKEmbedder):
    def __init__(self, api_key: str, model: str = "voyage-2"):
        super().__init__(
            provider="voyage",
            model=model,
            apiKey=api_key,
            baseURL="https://api.voyageai.com/v1",
        )


class VLLMEmbedder(_AISDKEmbedder):
    def __init__(self, model: str = "all-MiniLM-L6-v2", kwargs: dict = {}):
        base_url = kwargs.get("base_url", "http://localhost:8000/v1")
        api_key = kwargs.get("api_key", "EMPTY")
        super().__init__(
            provider="openai_compatible",
            model=model,
            name="vllm",
            baseURL=base_url,
            apiKey=api_key,
        )
