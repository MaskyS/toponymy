import numpy as np
import os
from unittest.mock import patch

import pytest
import toponymy.embedding_wrappers as embedders


def _validate_matrix(result: np.ndarray):
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 2)
    np.testing.assert_almost_equal(result, np.array([[0.1, 0.2], [0.3, 0.4]]))


def _run_embedder_case(embedder):
    with patch(
        "toponymy.embedding_wrappers.call_ai_sdk_bridge",
        return_value={"embeddings": [[0.1, 0.2], [0.3, 0.4]]},
    ) as mock_bridge:
        result = embedder.encode(["sample text 1", "sample text 2"])

    _validate_matrix(result)
    assert mock_bridge.call_count == 1


def test_cohere_embedder_encode():
    _run_embedder_case(embedders.CohereEmbedder(api_key="fake_key"))


def test_openai_embedder_encode():
    _run_embedder_case(embedders.OpenAIEmbedder(api_key="fake_key"))


def test_azure_embedder_encode():
    _run_embedder_case(
        embedders.AzureAIEmbedder(
            api_key="fake_key",
            endpoint="https://fake-endpoint.azure.com/models",
            model="text-embedding",
        )
    )


def test_mistral_embedder_encode():
    _run_embedder_case(embedders.MistralEmbedder(api_key="fake_key"))


def test_voyage_embedder_encode():
    _run_embedder_case(embedders.VoyageAIEmbedder(api_key="fake_key"))


def test_vllm_embedder_encode():
    _run_embedder_case(embedders.VLLMEmbedder(model="all-MiniLM-L6-v2"))


def test_cohere_embedder_legacy_env_and_provider_options():
    captured = []

    def fake_bridge(payload):
        captured.append(payload)
        return {"embeddings": [[0.1, 0.2], [0.3, 0.4]]}

    with patch.dict(os.environ, {"CO_API_KEY": "legacy-co-key"}, clear=False):
        with patch(
            "toponymy.embedding_wrappers.call_ai_sdk_bridge",
            side_effect=fake_bridge,
        ):
            result = embedders.CohereEmbedder(api_key=None).encode(
                ["sample text 1", "sample text 2"]
            )

    _validate_matrix(result)
    assert captured[0]["apiKey"] == "legacy-co-key"
    assert captured[0]["providerOptions"]["cohere"]["inputType"] == "search_query"


def test_embedder_warns_for_ignored_http_clients():
    with pytest.warns(UserWarning, match="http_client"):
        embedders.OpenAIEmbedder(api_key="fake_key", http_client=object())
    with pytest.warns(UserWarning, match="httpx_client"):
        embedders.CohereEmbedder(api_key="fake_key", httpx_client=object())
