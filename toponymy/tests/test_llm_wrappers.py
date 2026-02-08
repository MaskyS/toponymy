import json
import os
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from toponymy.llm_wrappers import (
    AnthropicNamer,
    AzureAINamer,
    CohereNamer,
    GoogleGeminiNamer,
    HuggingFaceNamer,
    LlamaCppNamer,
    OllamaNamer,
    OpenAINamer,
    ReplicateNamer,
    TogetherNamer,
    VLLMNamer,
    repair_json_string_backslashes,
)

VALID_TOPIC_NAME_RESPONSE = {
    "topic_name": "Machine Learning",
    "topic_specificity": 0.6,
}

VALID_CLUSTER_NAMES_RESPONSE = {
    "new_topic_name_mapping": {
        "1. data": "Data Science",
        "2. ml": "Machine Learning\\ML",
        "3. ai": "Artificial Intelligence",
    },
    "topic_specificities": [0.6, 0.8, 0.7],
}

MALFORMED_MAPPING_RESPONSE = (
    '{"new_topic_name_mapping": {"data science": "Data Science", '
    '"data science": "Machine Learning\\ML", "data science": '
    '"Artificial Intelligence"}, "topic_specificities": [0.6, 0.8, 0.7]}'
)


@pytest.fixture
def mock_data():
    return {
        "valid_topic_name": json.dumps(VALID_TOPIC_NAME_RESPONSE),
        "valid_cluster_names": json.dumps(VALID_CLUSTER_NAMES_RESPONSE),
        "old_names": ["data", "ml", "ai"],
        "malformed_mapping": MALFORMED_MAPPING_RESPONSE,
    }


def validate_topic_name(result: str):
    assert result == "Machine Learning"


def validate_cluster_names(result):
    expected = ["Data Science", "Machine Learning\\ML", "Artificial Intelligence"]
    assert result == expected


def test_repair_json_string_backslashes():
    repaired = repair_json_string_backslashes(
        '{"new_topic_name_mapping": {"2. ml": "Machine Learning\\ML"}}'
    )
    assert repaired == '{"new_topic_name_mapping": {"2. ml": "Machine Learning\\\\ML"}}'


def test_openai_generate_topic_name_success(mock_data):
    with patch("toponymy.llm_wrappers.call_ai_sdk_bridge", return_value={"text": mock_data["valid_topic_name"]}):
        wrapper = OpenAINamer(api_key="dummy")
        result = wrapper.generate_topic_name("test prompt")
        validate_topic_name(result)


def test_openai_generate_topic_name_system_prompt_success(mock_data):
    with patch("toponymy.llm_wrappers.call_ai_sdk_bridge", return_value={"text": mock_data["valid_topic_name"]}):
        wrapper = OpenAINamer(api_key="dummy")
        result = wrapper.generate_topic_name({"system": "system", "user": "user"})
        validate_topic_name(result)


def test_generate_cluster_names_success(mock_data):
    with patch("toponymy.llm_wrappers.call_ai_sdk_bridge", return_value={"text": mock_data["valid_cluster_names"]}):
        wrapper = CohereNamer(api_key="dummy")
        result = wrapper.generate_topic_cluster_names("test prompt", mock_data["old_names"])
        validate_cluster_names(result)


def test_generate_cluster_names_fallback_parse_success(mock_data):
    with patch("toponymy.llm_wrappers.call_ai_sdk_bridge", return_value={"text": mock_data["malformed_mapping"]}):
        wrapper = CohereNamer(api_key="dummy")
        result = wrapper.generate_topic_cluster_names("test prompt", mock_data["old_names"])
        validate_cluster_names(result)


def test_generate_topic_name_failure_returns_empty():
    with patch("toponymy.llm_wrappers.call_ai_sdk_bridge", side_effect=RuntimeError("boom")):
        wrapper = AnthropicNamer(api_key="dummy")
        assert wrapper.generate_topic_name("test prompt") == ""


def test_generate_cluster_names_failure_returns_old_names(mock_data):
    with patch("toponymy.llm_wrappers.call_ai_sdk_bridge", side_effect=RuntimeError("boom")):
        wrapper = AnthropicNamer(api_key="dummy")
        assert (
            wrapper.generate_topic_cluster_names("test prompt", mock_data["old_names"])
            == mock_data["old_names"]
        )


def test_llamacpp_rejects_system_prompts():
    fake_llm = Mock(return_value={"choices": [{"text": json.dumps(VALID_TOPIC_NAME_RESPONSE)}]})
    fake_llama_cpp = SimpleNamespace(Llama=Mock(return_value=fake_llm))

    with patch.dict("sys.modules", {"llama_cpp": fake_llama_cpp}):
        wrapper = LlamaCppNamer(model_path="dummy")
        assert wrapper.generate_topic_name({"system": "s", "user": "u"}) == ""


def test_llamacpp_generate_topic_name_uses_local_model(mock_data):
    fake_llm = Mock(return_value={"choices": [{"text": mock_data["valid_topic_name"]}]})
    fake_llama_cpp = SimpleNamespace(Llama=Mock(return_value=fake_llm))

    with patch.dict("sys.modules", {"llama_cpp": fake_llama_cpp}):
        with patch("toponymy.llm_wrappers.call_ai_sdk_bridge") as mock_bridge:
            wrapper = LlamaCppNamer(model_path="dummy")
            result = wrapper.generate_topic_name("prompt")

    validate_topic_name(result)
    mock_bridge.assert_not_called()


def test_payload_mapping_for_representative_wrappers():
    captured = []

    def fake_bridge(payload):
        captured.append(payload)
        return {"text": json.dumps(VALID_TOPIC_NAME_RESPONSE)}

    with patch("toponymy.llm_wrappers.call_ai_sdk_bridge", side_effect=fake_bridge):
        wrappers = [
            OpenAINamer(api_key="openai-key", model="gpt-4o-mini"),
            AzureAINamer(api_key="az-key", endpoint="https://example.services.ai.azure.com/models", model="phi-4"),
            OllamaNamer(model="llama3.2", host="http://localhost:11434"),
            TogetherNamer(api_key="t-key", model="meta-llama/Llama-3-8b-chat-hf"),
            GoogleGeminiNamer(api_key="g-key", model="gemini-1.5-flash"),
            VLLMNamer(model="Qwen/Qwen2.5-7B-Instruct", base_url="http://localhost:8000/v1"),
            ReplicateNamer(api_token="rep-key", model="meta/llama-2-70b-chat"),
        ]

        for wrapper in wrappers:
            validate_topic_name(wrapper.generate_topic_name("test"))

    providers = [payload["provider"] for payload in captured]
    assert providers == [
        "openai",
        "azure_inference",
        "ollama",
        "together",
        "google",
        "openai_compatible",
        "replicate",
    ]

    assert captured[0]["apiKey"] == "openai-key"
    assert captured[0]["maxOutputTokens"] == 128
    assert captured[1]["endpoint"] == "https://example.services.ai.azure.com/models"
    assert captured[2]["host"] == "http://localhost:11434"


def test_huggingface_namer_uses_local_pipeline(mock_data):
    fake_pipeline = Mock(return_value=[{"generated_text": mock_data["valid_topic_name"]}])
    fake_transformers = SimpleNamespace(pipeline=Mock(return_value=fake_pipeline))

    with patch.dict("sys.modules", {"transformers": fake_transformers}):
        with patch("toponymy.llm_wrappers.call_ai_sdk_bridge") as mock_bridge:
            wrapper = HuggingFaceNamer(model="meta-llama/Llama-3.1-8B-Instruct")
            result = wrapper.generate_topic_name("prompt")

    validate_topic_name(result)
    mock_bridge.assert_not_called()
    fake_transformers.pipeline.assert_called_once()


def test_legacy_env_aliases_for_providers(mock_data):
    captured = []

    def fake_bridge(payload):
        captured.append(payload)
        return {"text": mock_data["valid_topic_name"]}

    env = {
        "CO_API_KEY": "co-legacy",
        "CO_API_URL": "https://api.cohere.com",
        "TOGETHER_API_KEY": "tog-legacy",
        "GOOGLE_API_KEY": "goo-legacy",
        "REPLICATE_API_TOKEN": "rep-legacy",
    }
    with patch.dict(os.environ, env, clear=False):
        with patch("toponymy.llm_wrappers.call_ai_sdk_bridge", side_effect=fake_bridge):
            validate_topic_name(CohereNamer(api_key=None).generate_topic_name("test"))
            validate_topic_name(TogetherNamer(api_key=None).generate_topic_name("test"))
            validate_topic_name(GoogleGeminiNamer(api_key=None).generate_topic_name("test"))
            validate_topic_name(ReplicateNamer(api_token=None).generate_topic_name("test"))

    by_provider = {payload["provider"]: payload for payload in captured}
    assert by_provider["cohere"]["apiKey"] == "co-legacy"
    assert by_provider["cohere"]["baseURL"] == "https://api.cohere.com"
    assert by_provider["together"]["apiKey"] == "tog-legacy"
    assert by_provider["google"]["apiKey"] == "goo-legacy"
    assert by_provider["replicate"]["apiKey"] == "rep-legacy"


def test_warns_for_ignored_transport_clients():
    with pytest.warns(UserWarning, match="http_client"):
        OpenAINamer(api_key="openai", http_client=object())
    with pytest.warns(UserWarning, match="httpx_client"):
        CohereNamer(api_key="cohere", httpx_client=object())
