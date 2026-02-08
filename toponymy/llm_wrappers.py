from __future__ import annotations

import asyncio
import io
import json
import os
import re
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union
from uuid import uuid4
from warnings import warn

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from toponymy._ai_sdk_bridge import AISDKBridgeError, call_ai_sdk_bridge, call_ai_sdk_bridge_async
from toponymy.templates import GET_TOPIC_CLUSTER_NAMES_REGEX, GET_TOPIC_NAME_REGEX


class InvalidLLMInputError(ValueError):
    """Raised when an LLM input is invalid and retrying will not help."""


class LLMWrapperImportError(ImportError):
    """Compatibility error used by wrapper initialization failures."""


def _should_retry(e: Exception) -> bool:
    return not isinstance(e, InvalidLLMInputError)


def repair_json_string_backslashes(s: str) -> str:
    """
    Attempts to repair a string that should be JSON by escaping unescaped backslashes.
    This focuses on the common issue of literal backslashes not being escaped.
    """
    placeholders = {
        "\\\\": "__DOUBLE_BACKSLASH_PLACEHOLDER__",
        '\\"': "__ESCAPED_QUOTE_PLACEHOLDER__",
        "\\n": "__NEWLINE_PLACEHOLDER__",
        "\\r": "__CARRIAGE_RETURN_PLACEHOLDER__",
        "\\t": "__TAB_PLACEHOLDER__",
        "\\b": "__BACKSPACE_PLACEHOLDER__",
        "\\f": "__FORMFEED_PLACEHOLDER__",
        "\\/": "__SOLIDUS_PLACEHOLDER__",
    }

    temp_s = s
    for original, placeholder in placeholders.items():
        temp_s = temp_s.replace(original, placeholder)

    temp_s = temp_s.replace("\\", "\\\\")

    for original, placeholder in placeholders.items():
        temp_s = temp_s.replace(placeholder, original)

    return temp_s


def llm_output_to_result(llm_output: str, regex: str) -> dict:
    json_portion = re.findall(regex, llm_output, re.DOTALL)[0]
    try:
        result = json.loads(json_portion)
    except json.JSONDecodeError:
        repaired_json = repair_json_string_backslashes(json_portion)
        result = json.loads(repaired_json)

    return result


def _safe_int(value: str) -> int:
    try:
        return int(value)
    except Exception:
        return 10**9


def _extract_text_from_content(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = []
        for part in content:
            if isinstance(part, str):
                chunks.append(part)
            elif isinstance(part, dict):
                if part.get("type") == "text":
                    chunks.append(str(part.get("text", "")))
                elif "text" in part:
                    chunks.append(str(part["text"]))
            else:
                text = getattr(part, "text", None)
                if text:
                    chunks.append(str(text))
        return "".join(chunks)
    if isinstance(content, dict):
        if "text" in content:
            return str(content["text"])
        if "content" in content:
            return _extract_text_from_content(content["content"])
    text = getattr(content, "text", None)
    if text is not None:
        return str(text)
    return str(content)


def _extract_text_from_chat_completion_body(body) -> str:
    if not body:
        return ""
    if isinstance(body, dict):
        choices = body.get("choices")
    else:
        choices = getattr(body, "choices", None)
    if not choices:
        return ""

    first_choice = choices[0]
    if isinstance(first_choice, dict):
        message = first_choice.get("message", {})
    else:
        message = getattr(first_choice, "message", {})

    if isinstance(message, dict):
        return _extract_text_from_content(message.get("content"))
    return _extract_text_from_content(getattr(message, "content", None))


class LLMWrapper(ABC):

    @abstractmethod
    def _call_llm(self, prompt: str, temperature: float, max_tokens: int) -> str:
        pass

    @abstractmethod
    def _call_llm_with_system_prompt(
        self, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int
    ) -> str:
        pass

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry_error_callback=lambda _: "",
        retry=retry_if_exception(_should_retry),
    )
    def generate_topic_name(
        self, prompt: Union[str, Dict[str, str]], temperature: float = 0.4
    ) -> str:
        try:
            if isinstance(prompt, str):
                topic_name_info_raw = self._call_llm(
                    prompt,
                    temperature,
                    max_tokens=128,
                )
            elif isinstance(prompt, dict) and self.supports_system_prompts:
                topic_name_info_raw = self._call_llm_with_system_prompt(
                    system_prompt=prompt["system"],
                    user_prompt=prompt["user"],
                    temperature=temperature,
                    max_tokens=128,
                )
            else:
                raise InvalidLLMInputError(
                    f"Prompt must be a string or a dictionary, got {type(prompt)}"
                )

            topic_name_info = llm_output_to_result(
                topic_name_info_raw, GET_TOPIC_NAME_REGEX
            )
            topic_name = str(topic_name_info["topic_name"])
        except Exception as exc:
            raise ValueError(
                f"Failed to generate topic name with {self.__class__.__name__}"
            ) from exc

        return topic_name

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry_error_callback=lambda retry_state: (
            retry_state.args[1]
            if len(retry_state.args) > 1 and isinstance(retry_state.args[1], list)
            else []
        ),
        retry=retry_if_exception(_should_retry),
    )
    def generate_topic_cluster_names(
        self,
        prompt: Union[str, Dict[str, str]],
        old_names: List[str],
        temperature: float = 0.4,
    ) -> List[str]:
        try:
            if isinstance(prompt, str):
                topic_name_info_raw = self._call_llm(
                    prompt,
                    temperature,
                    max_tokens=1024,
                )
            elif isinstance(prompt, dict) and self.supports_system_prompts:
                topic_name_info_raw = self._call_llm_with_system_prompt(
                    system_prompt=prompt["system"],
                    user_prompt=prompt["user"],
                    temperature=temperature,
                    max_tokens=1024,
                )
            else:
                raise InvalidLLMInputError(
                    f"Prompt must be a string or a dictionary, got {type(prompt)}"
                )

            topic_name_info = llm_output_to_result(
                topic_name_info_raw, GET_TOPIC_CLUSTER_NAMES_REGEX
            )
        except Exception as exc:
            warn(
                f"Failed to generate topic cluster names with {self.__class__.__name__}: {exc}"
            )
            return old_names

        mapping = topic_name_info["new_topic_name_mapping"]
        if len(mapping) == len(old_names):
            result = []
            for i, old_name_val in enumerate(old_names, start=1):
                key_with_val = f"{i}. {old_name_val}"
                key_just_index = f"{i}."
                if key_with_val in mapping:
                    result.append(mapping[key_with_val])
                elif key_just_index in mapping:
                    result.append(mapping[key_just_index])
                else:
                    result.append(old_name_val)
            return result

        mapping_text = re.findall(
            r'"new_topic_name_mapping":\s*\{(.*?)\}',
            topic_name_info_raw,
            re.DOTALL,
        )[0]
        new_names = re.findall(r'".*?":\s*"(.*?)",?', mapping_text, re.DOTALL)
        if len(new_names) == len(old_names):
            return new_names

        raise ValueError(
            f"Failed to generate enough names when fixing {old_names}; got {mapping_text}"
        )

    @property
    def supports_system_prompts(self) -> bool:
        return True

    def test_llm_connectivity(
        self,
        prompt: str = "Identify yourself and explain that you will be providing topic names for clusters",
    ) -> str:
        try:
            return self._call_llm(prompt, temperature=0.4, max_tokens=128)
        except Exception as exc:
            warn(f"Failed to test LLM connectivity with {self.__class__.__name__}: {exc}")
            return "<error>"


class AsyncLLMWrapper(ABC):

    @abstractmethod
    async def _call_llm_batch(
        self, prompts: List[str], temperature: float, max_tokens: int
    ) -> List[str]:
        pass

    @abstractmethod
    async def _call_llm_with_system_prompt_batch(
        self,
        system_prompts: List[str],
        user_prompts: List[str],
        temperature: float,
        max_tokens: int,
    ) -> List[str]:
        pass

    async def generate_topic_names(
        self, prompts: List[Union[str, Dict[str, str]]], temperature: float = 0.4
    ) -> List[str]:
        if not prompts:
            return []

        if isinstance(prompts[0], str):
            responses = await self._call_llm_batch(prompts, temperature, max_tokens=128)
        elif isinstance(prompts[0], dict) and self.supports_system_prompts:
            system_prompts = [p["system"] for p in prompts]
            user_prompts = [p["user"] for p in prompts]
            responses = await self._call_llm_with_system_prompt_batch(
                system_prompts,
                user_prompts,
                temperature,
                max_tokens=128,
            )
        else:
            raise InvalidLLMInputError(
                f"Prompts must be strings or dictionaries, got {type(prompts[0])}"
            )

        results = []
        for response in responses:
            if not response:
                results.append("")
                continue
            try:
                topic_name_info = llm_output_to_result(response, GET_TOPIC_NAME_REGEX)
                results.append(str(topic_name_info["topic_name"]))
            except Exception as exc:
                warn(
                    f"Failed to generate topic name with {self.__class__.__name__}: {exc}"
                )
                results.append("")

        return results

    async def generate_topic_cluster_names(
        self,
        prompts: List[Union[str, Dict[str, str]]],
        old_names_list: List[List[str]],
        temperature: float = 0.4,
    ) -> List[List[str]]:
        if len(prompts) != len(old_names_list):
            raise ValueError("Number of prompts must match number of old_names lists")
        if not prompts:
            return []

        if isinstance(prompts[0], str):
            responses = await self._call_llm_batch(
                prompts,
                temperature,
                max_tokens=1024,
            )
        elif isinstance(prompts[0], dict) and self.supports_system_prompts:
            system_prompts = [prompt["system"] for prompt in prompts]
            user_prompts = [prompt["user"] for prompt in prompts]
            responses = await self._call_llm_with_system_prompt_batch(
                system_prompts,
                user_prompts,
                temperature,
                max_tokens=1024,
            )
        else:
            raise InvalidLLMInputError(
                f"Prompts must be strings or dictionaries, got {type(prompts[0])}"
            )

        return [
            self._parse_cluster_response(response, old_names)
            for response, old_names in zip(responses, old_names_list)
        ]

    def _parse_cluster_response(self, response: str, old_names: List[str]) -> List[str]:
        try:
            topic_name_info = llm_output_to_result(response, GET_TOPIC_CLUSTER_NAMES_REGEX)
            mapping = topic_name_info["new_topic_name_mapping"]

            if len(mapping) == len(old_names):
                result = []
                for i, old_name_val in enumerate(old_names, start=1):
                    key_with_val = f"{i}. {old_name_val}"
                    key_just_index = f"{i}."
                    if key_with_val in mapping:
                        result.append(mapping[key_with_val])
                    elif key_just_index in mapping:
                        result.append(mapping[key_just_index])
                    else:
                        result.append(old_name_val)
                return result

            mapping_str = re.findall(
                r'"new_topic_name_mapping":\s*\{(.*?)\}',
                response,
                re.DOTALL,
            )[0]
            new_names = re.findall(r'".*?":\s*"(.*?)",?', mapping_str, re.DOTALL)
            if len(new_names) == len(old_names):
                return new_names

            raise ValueError(f"Failed to generate enough names; got {mapping}")
        except Exception as exc:
            warn(f"Failed to parse cluster names: {exc}")
            return old_names

    @property
    def supports_system_prompts(self) -> bool:
        return True

    def test_llm_connectivity(
        self,
        prompt: str = "Identify yourself and explain that you will be providing topic names for clusters",
    ) -> str:
        try:
            response = asyncio.run(
                self._call_llm_batch([prompt], temperature=0.4, max_tokens=128)
            )
            return response[0]
        except Exception as exc:
            warn(f"Failed to test LLM connectivity with {self.__class__.__name__}: {exc}")
            return "<error>"


class FailedImportLLMWrapper(LLMWrapper):
    """Retained for compatibility with previous import-error based wrappers."""

    @classmethod
    def _import_error_message(cls):
        return (
            f"Failed to initialize {cls.__name__}. "
            "Ensure Node.js dependencies for the AI SDK bridge are installed."
        )

    def __init__(self, *args, **kwargs):
        raise LLMWrapperImportError(self._import_error_message())

    def _call_llm(self, prompt: str, temperature: float, max_tokens: int) -> str:
        raise LLMWrapperImportError(self._import_error_message())

    def _call_llm_with_system_prompt(
        self, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int
    ) -> str:
        raise LLMWrapperImportError(self._import_error_message())


class FailedImportAsyncLLMWrapper(AsyncLLMWrapper):
    """Retained for compatibility with previous import-error based wrappers."""

    @classmethod
    def _import_error_message(cls):
        return (
            f"Failed to initialize {cls.__name__}. "
            "Ensure Node.js dependencies for the AI SDK bridge are installed."
        )

    def __init__(self, *args, **kwargs):
        raise LLMWrapperImportError(self._import_error_message())

    async def _call_llm_batch(
        self, prompts: List[str], temperature: float, max_tokens: int
    ) -> List[str]:
        raise LLMWrapperImportError(self._import_error_message())

    async def _call_llm_with_system_prompt_batch(
        self,
        system_prompts: List[str],
        user_prompts: List[str],
        temperature: float,
        max_tokens: int,
    ) -> List[str]:
        raise LLMWrapperImportError(self._import_error_message())


class _AISDKSyncWrapper(LLMWrapper):
    def __init__(
        self,
        provider: str,
        model: str,
        llm_specific_instructions: Optional[str] = None,
        supports_system_prompts: bool = True,
        **provider_config,
    ):
        self.provider = provider
        self.model = model
        self.provider_config = provider_config
        self.extra_prompting = (
            "\n\n" + llm_specific_instructions if llm_specific_instructions else ""
        )
        self._supports_system_prompts = supports_system_prompts

    def _bridge_payload(self, temperature: float, max_tokens: int) -> Dict[str, object]:
        return {
            "action": "generate_text",
            "provider": self.provider,
            "model": self.model,
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
            "maxTokens": max_tokens,
            **self.provider_config,
        }

    def _call_llm(self, prompt: str, temperature: float, max_tokens: int) -> str:
        payload = self._bridge_payload(temperature=temperature, max_tokens=max_tokens)
        payload["prompt"] = {
            "type": "text",
            "prompt": prompt + self.extra_prompting,
        }
        result = call_ai_sdk_bridge(payload)
        return str(result.get("text", ""))

    def _call_llm_with_system_prompt(
        self, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int
    ) -> str:
        if not self.supports_system_prompts:
            raise InvalidLLMInputError(
                f"System prompts are not supported for {self.__class__.__name__}"
            )

        payload = self._bridge_payload(temperature=temperature, max_tokens=max_tokens)
        payload["prompt"] = {
            "type": "chat",
            "system": system_prompt,
            "user": user_prompt + self.extra_prompting,
        }
        result = call_ai_sdk_bridge(payload)
        return str(result.get("text", ""))

    @property
    def supports_system_prompts(self) -> bool:
        return self._supports_system_prompts


class _AISDKAsyncWrapper(AsyncLLMWrapper):
    def __init__(
        self,
        provider: str,
        model: str,
        llm_specific_instructions: Optional[str] = None,
        max_concurrent_requests: int = 10,
        supports_system_prompts: bool = True,
        **provider_config,
    ):
        self.provider = provider
        self.model = model
        self.provider_config = provider_config
        self.max_concurrent_requests = max_concurrent_requests
        self.extra_prompting = (
            "\n\n" + llm_specific_instructions if llm_specific_instructions else ""
        )
        self._supports_system_prompts = supports_system_prompts

    def _bridge_payload(self, temperature: float, max_tokens: int) -> Dict[str, object]:
        return {
            "action": "generate_text",
            "provider": self.provider,
            "model": self.model,
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
            "maxTokens": max_tokens,
            **self.provider_config,
        }

    async def _generate_one(
        self,
        prompt_payload: Dict[str, str],
        temperature: float,
        max_tokens: int,
    ) -> str:
        payload = self._bridge_payload(temperature=temperature, max_tokens=max_tokens)
        payload["prompt"] = prompt_payload
        try:
            result = await call_ai_sdk_bridge_async(payload)
            return str(result.get("text", ""))
        except AISDKBridgeError as exc:
            warn(f"AI SDK bridge call failed for {self.__class__.__name__}: {exc}")
            return ""
        except Exception as exc:
            warn(f"Unexpected async LLM error for {self.__class__.__name__}: {exc}")
            return ""

    async def _call_llm_batch(
        self, prompts: List[str], temperature: float, max_tokens: int
    ) -> List[str]:
        semaphore = asyncio.Semaphore(max(self.max_concurrent_requests, 1))

        async def worker(prompt: str) -> str:
            async with semaphore:
                return await self._generate_one(
                    {
                        "type": "text",
                        "prompt": prompt + self.extra_prompting,
                    },
                    temperature,
                    max_tokens,
                )

        return await asyncio.gather(*[worker(prompt) for prompt in prompts])

    async def _call_llm_with_system_prompt_batch(
        self,
        system_prompts: List[str],
        user_prompts: List[str],
        temperature: float,
        max_tokens: int,
    ) -> List[str]:
        if not self.supports_system_prompts:
            raise InvalidLLMInputError(
                f"System prompts are not supported for {self.__class__.__name__}"
            )

        semaphore = asyncio.Semaphore(max(self.max_concurrent_requests, 1))

        async def worker(system_prompt: str, user_prompt: str) -> str:
            async with semaphore:
                return await self._generate_one(
                    {
                        "type": "chat",
                        "system": system_prompt,
                        "user": user_prompt + self.extra_prompting,
                    },
                    temperature,
                    max_tokens,
                )

        return await asyncio.gather(
            *[
                worker(system_prompt, user_prompt)
                for system_prompt, user_prompt in zip(system_prompts, user_prompts)
            ]
        )

    @property
    def supports_system_prompts(self) -> bool:
        return self._supports_system_prompts

    async def close(self):
        return None


class LlamaCppNamer(LLMWrapper):
    def __init__(self, model_path: str, llm_specific_instructions=None, **kwargs):
        self.model_path = model_path
        try:
            import llama_cpp
        except ImportError as exc:
            raise LLMWrapperImportError(
                "LlamaCppNamer requires `llama-cpp-python`. Install it to use local llama.cpp models."
            ) from exc

        for arg, val in kwargs.items():
            if arg == "n_ctx":
                continue
            setattr(self, arg, val)

        self.llm = llama_cpp.Llama(model_path=model_path, **kwargs)
        self.extra_prompting = (
            "\n\n" + llm_specific_instructions if llm_specific_instructions else ""
        )

    def _call_llm(self, prompt: str, temperature: float, max_tokens: int) -> str:
        response = self.llm(
            prompt + self.extra_prompting,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response["choices"][0]["text"]

    def _call_llm_with_system_prompt(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        raise InvalidLLMInputError("System prompts are not supported for LlamaCpp wrapper")

    @property
    def supports_system_prompts(self) -> bool:
        return False


class HuggingFaceNamer(LLMWrapper):
    def __init__(self, model: str, llm_specific_instructions=None, **kwargs):
        try:
            import transformers
        except ImportError as exc:
            raise LLMWrapperImportError(
                "HuggingFaceNamer requires the `transformers` package."
            ) from exc

        api_key = kwargs.pop("api_key", kwargs.pop("hf_token", os.environ.get("HF_TOKEN")))
        base_url = kwargs.pop("base_url", None)
        if base_url is not None:
            warn("`base_url` is ignored by HuggingFaceNamer local pipeline mode.")
        if api_key is not None and "token" not in kwargs:
            kwargs["token"] = api_key

        self.model = model
        self.llm = transformers.pipeline("text-generation", model=model, **kwargs)
        self.extra_prompting = (
            "\n\n" + llm_specific_instructions if llm_specific_instructions else ""
        )

    def _call_llm(self, prompt: str, temperature: float, max_tokens: int) -> str:
        response = self.llm(
            [{"role": "user", "content": prompt + self.extra_prompting}],
            return_full_text=False,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=self.llm.tokenizer.eos_token_id,
        )
        return response[0]["generated_text"]

    def _call_llm_with_system_prompt(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        response = self.llm(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt + self.extra_prompting},
            ],
            return_full_text=False,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=self.llm.tokenizer.eos_token_id,
        )
        return response[0]["generated_text"]


class AsyncHuggingFaceNamer(AsyncLLMWrapper):
    def __init__(
        self,
        model: str,
        llm_specific_instructions=None,
        max_concurrent_requests=10,
        **kwargs,
    ):
        try:
            import transformers
        except ImportError as exc:
            raise LLMWrapperImportError(
                "AsyncHuggingFaceNamer requires the `transformers` package."
            ) from exc

        api_key = kwargs.pop("api_key", kwargs.pop("hf_token", os.environ.get("HF_TOKEN")))
        base_url = kwargs.pop("base_url", None)
        if base_url is not None:
            warn("`base_url` is ignored by AsyncHuggingFaceNamer local pipeline mode.")
        if api_key is not None and "token" not in kwargs:
            kwargs["token"] = api_key

        self.model = model
        self.llm = transformers.pipeline("text-generation", model=model, **kwargs)
        self.max_concurrent_requests = max_concurrent_requests
        self.extra_prompting = (
            "\n\n" + llm_specific_instructions if llm_specific_instructions else ""
        )

    async def _call_llm_batch(
        self, prompts: List[str], temperature: float, max_tokens: int
    ) -> List[str]:
        responses = []
        for prompt in prompts:
            response = self.llm(
                [{"role": "user", "content": prompt + self.extra_prompting}],
                return_full_text=False,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.llm.tokenizer.eos_token_id,
            )
            responses.append(response[0]["generated_text"])
        return responses

    async def _call_llm_with_system_prompt_batch(
        self,
        system_prompts: List[str],
        user_prompts: List[str],
        temperature: float,
        max_tokens: int,
    ) -> List[str]:
        responses = []
        for system_prompt, user_prompt in zip(system_prompts, user_prompts):
            response = self.llm(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt + self.extra_prompting},
                ],
                return_full_text=False,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.llm.tokenizer.eos_token_id,
            )
            responses.append(response[0]["generated_text"])
        return responses


class VLLMNamer(_AISDKSyncWrapper):
    def __init__(self, model: str, llm_specific_instructions=None, **kwargs):
        base_url = kwargs.pop("base_url", "http://localhost:8000/v1")
        api_key = kwargs.pop("api_key", "EMPTY")
        if kwargs:
            warn(
                "Ignoring unsupported kwargs for VLLMNamer in AI SDK mode: "
                + ", ".join(sorted(kwargs.keys()))
            )
        super().__init__(
            provider="openai_compatible",
            model=model,
            llm_specific_instructions=llm_specific_instructions,
            name="vllm",
            baseURL=base_url,
            apiKey=api_key,
        )


class AsyncVLLMNamer(_AISDKAsyncWrapper):
    def __init__(
        self,
        model: str,
        llm_specific_instructions=None,
        max_concurrent_requests=10,
        **kwargs,
    ):
        base_url = kwargs.pop("base_url", "http://localhost:8000/v1")
        api_key = kwargs.pop("api_key", "EMPTY")
        if kwargs:
            warn(
                "Ignoring unsupported kwargs for AsyncVLLMNamer in AI SDK mode: "
                + ", ".join(sorted(kwargs.keys()))
            )
        super().__init__(
            provider="openai_compatible",
            model=model,
            llm_specific_instructions=llm_specific_instructions,
            max_concurrent_requests=max_concurrent_requests,
            name="vllm",
            baseURL=base_url,
            apiKey=api_key,
        )


class CohereNamer(_AISDKSyncWrapper):
    def __init__(
        self,
        api_key,
        model: str = "command-r-08-2024",
        base_url: str = None,
        httpx_client=None,
        llm_specific_instructions=None,
    ):
        api_key = api_key or os.getenv("CO_API_KEY") or os.getenv("COHERE_API_KEY")
        if not api_key:
            raise ValueError(
                "Cohere API key is required. Set `CO_API_KEY` or `COHERE_API_KEY`, "
                "or pass `api_key` directly."
            )
        if base_url is None:
            base_url = os.getenv("CO_API_URL")
        if httpx_client is not None:
            warn("`httpx_client` is ignored by CohereNamer in AI SDK mode.")
        super().__init__(
            provider="cohere",
            model=model,
            llm_specific_instructions=llm_specific_instructions,
            apiKey=api_key,
            baseURL=base_url,
        )


class AsyncCohereNamer(_AISDKAsyncWrapper):
    def __init__(
        self,
        api_key,
        model: str = "command-r-08-2024",
        llm_specific_instructions=None,
        max_concurrent_requests=10,
        base_url: str = None,
        httpx_client=None,
    ):
        api_key = api_key or os.getenv("CO_API_KEY") or os.getenv("COHERE_API_KEY")
        if not api_key:
            raise ValueError(
                "Cohere API key is required. Set `CO_API_KEY` or `COHERE_API_KEY`, "
                "or pass `api_key` directly."
            )
        if base_url is None:
            base_url = os.getenv("CO_API_URL")
        if httpx_client is not None:
            warn("`httpx_client` is ignored by AsyncCohereNamer in AI SDK mode.")
        super().__init__(
            provider="cohere",
            model=model,
            llm_specific_instructions=llm_specific_instructions,
            max_concurrent_requests=max_concurrent_requests,
            apiKey=api_key,
            baseURL=base_url,
        )


class CohereBatchNamer(AsyncLLMWrapper):
    def __init__(
        self,
        api_key,
        model: str = "command-r-08-2024",
        llm_specific_instructions=None,
        polling_interval=60,
        timeout=7200,
    ):
        self.api_key = api_key
        self.model = model
        self.extra_prompting = (
            "\n\n" + llm_specific_instructions if llm_specific_instructions else ""
        )
        self.polling_interval = polling_interval
        self.timeout = timeout
        self._batch_metadata: Dict[str, Dict[str, object]] = {}
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            import cohere
        except ImportError as exc:
            raise LLMWrapperImportError(
                "CohereBatchNamer requires the `cohere` package."
            ) from exc
        self._client = cohere.ClientV2(api_key=self.api_key)
        return self._client

    def _build_batch_requests(
        self,
        prompts: List[Union[str, Dict[str, str]]],
        temperature: float,
        max_tokens: int,
    ) -> List[dict]:
        requests = []
        for i, prompt in enumerate(prompts):
            if isinstance(prompt, str):
                messages = [{"role": "user", "content": prompt + self.extra_prompting}]
            elif isinstance(prompt, dict):
                messages = [
                    {"role": "system", "content": prompt["system"]},
                    {"role": "user", "content": prompt["user"] + self.extra_prompting},
                ]
            else:
                raise InvalidLLMInputError("Prompt must be a string or a dictionary")

            requests.append(
                {
                    "custom_id": str(i),
                    "params": {
                        "model": self.model,
                        "messages": messages,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                    },
                }
            )
        return requests

    def _submit_legacy_batch(self, requests: List[dict]) -> str:
        client = self._get_client()
        batch = client.beta.messages.batches.create(requests=requests)
        batch_id = batch.id
        self._batch_metadata[batch_id] = {
            "backend": "legacy",
            "num_requests": len(requests),
        }
        return batch_id

    def _wait_for_dataset_validation(self, dataset_id: str) -> None:
        client = self._get_client()
        start_time = time.time()
        while time.time() - start_time < self.timeout:
            dataset = client.datasets.get(dataset_id).dataset
            status = str(dataset.validation_status)
            if status in {"validated", "skipped"}:
                return
            if status == "failed":
                raise RuntimeError(
                    f"Cohere dataset validation failed for {dataset_id}: {dataset.validation_error}"
                )
            time.sleep(min(self.polling_interval, 5))
        raise RuntimeError(
            f"Timed out waiting for Cohere dataset validation for {dataset_id}"
        )

    def _submit_v2_batch(self, requests: List[dict]) -> str:
        client = self._get_client()
        try:
            from cohere.batches import Batch
        except ImportError as exc:
            raise LLMWrapperImportError(
                "CohereBatchNamer requires cohere>=5.x with batch support."
            ) from exc

        payload = "\n".join(json.dumps(item) for item in requests) + "\n"
        dataset = client.datasets.create(
            name=f"toponymy-batch-input-{uuid4().hex}",
            type="batch-chat-v2-input",
            data=("batch_input.jsonl", payload.encode("utf-8"), "application/jsonl"),
            skip_malformed_input=False,
        )
        dataset_id = dataset.id
        if not dataset_id:
            raise RuntimeError("Cohere dataset creation did not return an ID")

        self._wait_for_dataset_validation(dataset_id)

        batch_response = client.batches.create(
            request=Batch(
                name=f"toponymy-batch-{uuid4().hex}",
                input_dataset_id=dataset_id,
                model=self.model,
            )
        )
        batch = batch_response.batch
        batch_id = batch.id
        self._batch_metadata[batch_id] = {
            "backend": "v2",
            "num_requests": len(requests),
            "input_dataset_id": dataset_id,
        }
        return batch_id

    def submit_batch(
        self,
        prompts: List[Union[str, Dict[str, str]]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        client = self._get_client()
        requests = self._build_batch_requests(prompts, temperature, max_tokens)

        if (
            hasattr(client, "beta")
            and hasattr(client.beta, "messages")
            and hasattr(client.beta.messages, "batches")
        ):
            return self._submit_legacy_batch(requests)
        return self._submit_v2_batch(requests)

    def get_batch_status(self, batch_id: str) -> str:
        client = self._get_client()
        meta = self._batch_metadata.get(batch_id, {})
        backend = meta.get("backend")

        if backend == "legacy":
            return str(client.beta.messages.batches.retrieve(batch_id).processing_status)
        if backend == "v2":
            return str(client.batches.retrieve(batch_id).batch.status)

        try:
            return str(client.batches.retrieve(batch_id).batch.status)
        except Exception:
            return str(client.beta.messages.batches.retrieve(batch_id).processing_status)

    async def _wait_for_completion_async(self, batch_id: str) -> bool:
        start_time = time.time()
        while time.time() - start_time < self.timeout:
            status = self.get_batch_status(batch_id)

            if status in {"ended", "BATCH_STATUS_COMPLETED"}:
                return True
            if status in {
                "canceling",
                "canceled",
                "expired",
                "BATCH_STATUS_FAILED",
                "BATCH_STATUS_CANCELED",
            }:
                warn(f"Cohere batch job {batch_id} ended with status: {status}")
                return False

            await asyncio.sleep(self.polling_interval)

        warn(f"Cohere batch job {batch_id} timed out after {self.timeout} seconds")
        return False

    def _parse_legacy_batch_results(self, batch_id: str) -> List[str]:
        client = self._get_client()
        results_obj = client.beta.messages.batches.results(batch_id)
        items = (
            results_obj.results
            if hasattr(results_obj, "results")
            else list(results_obj)
        )
        items = sorted(items, key=lambda x: _safe_int(str(x.custom_id)))

        expected = int(self._batch_metadata.get(batch_id, {}).get("num_requests", len(items)))
        responses = [""] * expected
        for item in items:
            idx = _safe_int(str(item.custom_id))
            if idx >= expected:
                continue
            result = getattr(item, "result", None)
            if result and getattr(result, "type", None) == "succeeded":
                responses[idx] = _extract_text_from_content(result.message.content)
            else:
                warn(f"Cohere batch request {item.custom_id} failed")
        return responses

    def _parse_v2_batch_results(self, batch_id: str) -> List[str]:
        client = self._get_client()
        batch = client.batches.retrieve(batch_id).batch
        output_dataset_id = batch.output_dataset_id
        if not output_dataset_id:
            return []

        dataset = client.datasets.get(output_dataset_id).dataset
        dataset_parts = dataset.dataset_parts or []
        if not dataset_parts:
            return []

        rows = []
        for part in dataset_parts:
            url = part.url or part.original_url
            if not url:
                continue
            data = httpx.get(url, timeout=60).text
            for line in data.splitlines():
                if line.strip():
                    rows.append(json.loads(line))

        expected = int(self._batch_metadata.get(batch_id, {}).get("num_requests", len(rows)))
        responses = [""] * expected

        for row_idx, row in enumerate(rows):
            custom_id = str(row.get("custom_id", row_idx))
            idx = _safe_int(custom_id)
            if idx >= expected:
                continue

            text = ""
            response = row.get("response")
            if isinstance(response, dict):
                body = response.get("body", response)
                text = _extract_text_from_chat_completion_body(body)
                if not text:
                    text = _extract_text_from_content(body.get("message"))
            if not text:
                text = _extract_text_from_content(
                    row.get("text") or row.get("output_text") or row.get("result")
                )
            responses[idx] = text

        return responses

    async def retrieve_batch_text_results(self, batch_id: str) -> List[str]:
        meta = self._batch_metadata.get(batch_id, {})
        backend = meta.get("backend")
        if backend == "legacy":
            return self._parse_legacy_batch_results(batch_id)
        if backend == "v2":
            return self._parse_v2_batch_results(batch_id)

        try:
            return self._parse_v2_batch_results(batch_id)
        except Exception:
            return self._parse_legacy_batch_results(batch_id)

    def cancel_batch(self, batch_id: str):
        client = self._get_client()
        meta = self._batch_metadata.get(batch_id, {})
        backend = meta.get("backend")
        if backend == "legacy":
            client.beta.messages.batches.cancel(batch_id)
        else:
            client.batches.cancel(batch_id)

    async def _call_llm_batch(
        self, prompts: List[str], temperature: float, max_tokens: int
    ) -> List[str]:
        batch_id = self.submit_batch(prompts, temperature, max_tokens)
        if not await self._wait_for_completion_async(batch_id):
            raise RuntimeError(f"Cohere batch job {batch_id} failed or timed out")
        return await self.retrieve_batch_text_results(batch_id)

    async def _call_llm_with_system_prompt_batch(
        self,
        system_prompts: List[str],
        user_prompts: List[str],
        temperature: float,
        max_tokens: int,
    ) -> List[str]:
        if len(system_prompts) != len(user_prompts):
            raise ValueError(
                "Number of system prompts must match number of user prompts"
            )
        prompts = [
            {"system": system_prompt, "user": user_prompt}
            for system_prompt, user_prompt in zip(system_prompts, user_prompts)
        ]
        batch_id = self.submit_batch(prompts, temperature, max_tokens)
        if not await self._wait_for_completion_async(batch_id):
            raise RuntimeError(f"Cohere batch job {batch_id} failed or timed out")
        return await self.retrieve_batch_text_results(batch_id)


class AnthropicNamer(_AISDKSyncWrapper):
    def __init__(
        self,
        api_key,
        model: str = "claude-3-haiku-20240307",
        llm_specific_instructions=None,
    ):
        super().__init__(
            provider="anthropic",
            model=model,
            llm_specific_instructions=llm_specific_instructions,
            apiKey=api_key,
        )


class AsyncAnthropicNamer(_AISDKAsyncWrapper):
    def __init__(
        self,
        api_key,
        model: str = "claude-3-haiku-20240307",
        llm_specific_instructions=None,
        max_concurrent_requests=10,
    ):
        super().__init__(
            provider="anthropic",
            model=model,
            llm_specific_instructions=llm_specific_instructions,
            max_concurrent_requests=max_concurrent_requests,
            apiKey=api_key,
        )


class BatchAnthropicNamer(AsyncLLMWrapper):
    def __init__(
        self,
        api_key,
        model: str = "claude-3-haiku-20240307",
        llm_specific_instructions=None,
        polling_interval=60,
        timeout=7200,
    ):
        try:
            import anthropic
        except ImportError as exc:
            raise LLMWrapperImportError(
                "BatchAnthropicNamer requires the `anthropic` package. Install it to use Anthropic batch jobs."
            ) from exc

        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "Anthropic API key is required. Set ANTHROPIC_API_KEY or pass api_key."
            )

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.extra_prompting = (
            "\n\n" + llm_specific_instructions if llm_specific_instructions else ""
        )
        self.polling_interval = polling_interval
        self.timeout = timeout

    def _build_requests(
        self,
        prompts: List[Union[str, Dict[str, str]]],
        temperature: float,
        max_tokens: int,
    ) -> List[Dict[str, object]]:
        requests = []
        for i, prompt in enumerate(prompts):
            if isinstance(prompt, str):
                params = {
                    "model": self.model,
                    "max_tokens": max_tokens,
                    "messages": [
                        {"role": "user", "content": prompt + self.extra_prompting},
                    ],
                    "temperature": temperature,
                }
            elif isinstance(prompt, dict):
                params = {
                    "model": self.model,
                    "max_tokens": max_tokens,
                    "system": prompt["system"],
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt["user"] + self.extra_prompting,
                        },
                    ],
                    "temperature": temperature,
                }
            else:
                raise InvalidLLMInputError("Prompt must be string or dict")

            requests.append(
                {
                    "custom_id": str(i),
                    "params": params,
                }
            )

        return requests

    async def _run_batch(self, requests: List[Dict[str, object]]) -> List[str]:
        batch = await asyncio.to_thread(
            self.client.messages.batches.create,
            requests=requests,
        )
        batch_id = batch.id
        if not await self._wait_for_completion_async(batch_id):
            raise RuntimeError(f"Batch job {batch_id} failed or timed out")

        return await self._retrieve_batch_results(batch_id, expected_count=len(requests))

    async def _call_llm_batch(
        self, prompts: List[str], temperature: float, max_tokens: int
    ) -> List[str]:
        requests = self._build_requests(prompts, temperature, max_tokens)
        return await self._run_batch(requests)

    async def _call_llm_with_system_prompt_batch(
        self,
        system_prompts: List[str],
        user_prompts: List[str],
        temperature: float,
        max_tokens: int,
    ) -> List[str]:
        if len(system_prompts) != len(user_prompts):
            raise ValueError("Number of system prompts must match number of user prompts")

        prompts = [
            {"system": system_prompt, "user": user_prompt}
            for system_prompt, user_prompt in zip(system_prompts, user_prompts)
        ]
        requests = self._build_requests(prompts, temperature, max_tokens)
        return await self._run_batch(requests)

    async def _wait_for_completion_async(self, batch_id: str) -> bool:
        start_time = time.monotonic()

        while time.monotonic() - start_time < self.timeout:
            batch = await asyncio.to_thread(
                self.client.messages.batches.retrieve,
                batch_id,
            )
            status = batch.processing_status
            if status == "ended":
                return True

            if status in {"canceling", "canceled", "expired"}:
                warn(f"Batch job {batch_id} ended with status: {status}")
                return False

            await asyncio.sleep(self.polling_interval)

        warn(f"Batch job {batch_id} timed out after {self.timeout} seconds")
        return False

    async def _retrieve_batch_results(
        self,
        batch_id: str,
        expected_count: Optional[int] = None,
    ) -> List[str]:
        raw_results = await asyncio.to_thread(
            lambda: list(self.client.messages.batches.results(batch_id))
        )

        parsed_results: Dict[int, str] = {}
        for item in raw_results:
            try:
                index = int(item.custom_id)
            except (TypeError, ValueError):
                continue

            if item.result.type == "succeeded":
                parsed_results[index] = _extract_text_from_content(
                    item.result.message.content
                )
            else:
                error = getattr(item.result, "error", None)
                warn(
                    f"Batch request {item.custom_id} failed: "
                    f"{error if error is not None else item.result.type}"
                )
                parsed_results[index] = ""

        if expected_count is None:
            expected_count = max(parsed_results.keys(), default=-1) + 1

        return [parsed_results.get(i, "") for i in range(expected_count)]

    def submit_batch(
        self,
        prompts: List[Union[str, Dict[str, str]]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        requests = self._build_requests(prompts, temperature, max_tokens)
        batch = self.client.messages.batches.create(requests=requests)
        return batch.id

    def get_batch_status(self, batch_id: str) -> str:
        batch = self.client.messages.batches.retrieve(batch_id)
        return batch.processing_status

    async def retrieve_batch_text_results(self, batch_id: str) -> List[str]:
        return await self._retrieve_batch_results(batch_id)

    def cancel_batch(self, batch_id: str):
        self.client.messages.batches.cancel(batch_id)


class TogetherNamer(_AISDKSyncWrapper):
    def __init__(
        self,
        api_key,
        model: str = "meta-llama/Llama-3-8b-chat-hf",
        llm_specific_instructions=None,
    ):
        api_key = api_key or os.getenv("TOGETHER_API_KEY") or os.getenv(
            "TOGETHER_AI_API_KEY"
        )
        if not api_key:
            raise ValueError(
                "Together API key is required. Set `TOGETHER_API_KEY` or "
                "`TOGETHER_AI_API_KEY`, or pass `api_key` directly."
            )
        super().__init__(
            provider="together",
            model=model,
            llm_specific_instructions=llm_specific_instructions,
            apiKey=api_key,
        )


class AsyncTogether(_AISDKAsyncWrapper):
    def __init__(
        self,
        api_key,
        model: str = "meta-llama/Llama-3-8b-chat-hf",
        llm_specific_instructions=None,
        max_concurrent_requests=10,
    ):
        api_key = api_key or os.getenv("TOGETHER_API_KEY") or os.getenv(
            "TOGETHER_AI_API_KEY"
        )
        if not api_key:
            raise ValueError(
                "Together API key is required. Set `TOGETHER_API_KEY` or "
                "`TOGETHER_AI_API_KEY`, or pass `api_key` directly."
            )
        super().__init__(
            provider="together",
            model=model,
            llm_specific_instructions=llm_specific_instructions,
            max_concurrent_requests=max_concurrent_requests,
            apiKey=api_key,
        )


class OpenAINamer(_AISDKSyncWrapper):
    def __init__(
        self,
        api_key,
        model: str = "gpt-4o-mini",
        base_url: str = None,
        http_client=None,
        llm_specific_instructions=None,
    ):
        if http_client is not None:
            warn("`http_client` is ignored by OpenAINamer in AI SDK mode.")
        super().__init__(
            provider="openai",
            model=model,
            llm_specific_instructions=llm_specific_instructions,
            apiKey=api_key,
            baseURL=base_url,
        )


class AsyncOpenAINamer(_AISDKAsyncWrapper):
    def __init__(
        self,
        api_key,
        model: str = "gpt-4o-mini",
        llm_specific_instructions=None,
        max_concurrent_requests=10,
        organization=None,
        base_url=None,
    ):
        super().__init__(
            provider="openai",
            model=model,
            llm_specific_instructions=llm_specific_instructions,
            max_concurrent_requests=max_concurrent_requests,
            apiKey=api_key,
            organization=organization,
            baseURL=base_url,
        )


class ReplicateNamer(_AISDKSyncWrapper):
    def __init__(
        self,
        api_token=None,
        model: str = "meta/llama-2-70b-chat",
        llm_specific_instructions=None,
    ):
        api_token = api_token or os.getenv("REPLICATE_API_TOKEN")
        if not api_token:
            raise ValueError(
                "Replicate API token is required. Set `REPLICATE_API_TOKEN` "
                "or pass `api_token` directly."
            )
        super().__init__(
            provider="replicate",
            model=model,
            llm_specific_instructions=llm_specific_instructions,
            apiKey=api_token,
            baseURL="https://api.replicate.com/v1",
        )


class AzureAINamer(_AISDKSyncWrapper):
    def __init__(self, api_key: str, endpoint: str, model: str, llm_specific_instructions=None):
        super().__init__(
            provider="azure_inference",
            model=model,
            llm_specific_instructions=llm_specific_instructions,
            apiKey=api_key,
            endpoint=endpoint,
        )


class AsyncAzureAINamer(_AISDKAsyncWrapper):
    def __init__(
        self,
        api_key: str,
        endpoint: str,
        model: str,
        llm_specific_instructions=None,
        max_concurrent_requests=10,
    ):
        super().__init__(
            provider="azure_inference",
            model=model,
            llm_specific_instructions=llm_specific_instructions,
            max_concurrent_requests=max_concurrent_requests,
            apiKey=api_key,
            endpoint=endpoint,
        )


class BatchAzureAINamer(AsyncLLMWrapper):
    def __init__(
        self,
        api_key: str,
        endpoint: str,
        model: str,
        llm_specific_instructions=None,
        polling_interval=60,
        timeout=7200,
        api_version: str = "2024-10-21",
    ):
        self.api_key = api_key
        self.endpoint = endpoint
        self.model = model
        self.api_version = api_version
        self.extra_prompting = (
            "\n\n" + llm_specific_instructions if llm_specific_instructions else ""
        )
        self.polling_interval = polling_interval
        self.timeout = timeout
        self._batch_metadata: Dict[str, Dict[str, object]] = {}
        self._client = None

    def _normalize_endpoint(self) -> str:
        endpoint = self.endpoint.rstrip("/")
        if ".services.ai.azure.com" in endpoint:
            warn(
                "BatchAzureAINamer uses Azure OpenAI Batch API. "
                "Use an Azure OpenAI resource endpoint like "
                "`https://<resource>.openai.azure.com`, not the AI Inference endpoint."
            )
        return endpoint

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            import openai
        except ImportError as exc:
            raise LLMWrapperImportError(
                "BatchAzureAINamer requires the `openai` package."
            ) from exc

        self._client = openai.AzureOpenAI(
            api_key=self.api_key,
            azure_endpoint=self._normalize_endpoint(),
            api_version=self.api_version,
        )
        return self._client

    def _build_jsonl_requests(
        self,
        prompts: List[Union[str, Dict[str, str]]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        records = []
        for i, prompt in enumerate(prompts):
            if isinstance(prompt, str):
                messages = [{"role": "user", "content": prompt + self.extra_prompting}]
            elif isinstance(prompt, dict):
                messages = [
                    {"role": "system", "content": prompt["system"]},
                    {"role": "user", "content": prompt["user"] + self.extra_prompting},
                ]
            else:
                raise InvalidLLMInputError("Prompt must be a string or dictionary")

            body = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            records.append(
                {
                    "custom_id": str(i),
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": body,
                }
            )
        return "\n".join(json.dumps(row) for row in records) + "\n"

    def submit_batch(
        self,
        prompts: List[Union[str, Dict[str, str]]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        client = self._get_client()
        payload = self._build_jsonl_requests(prompts, temperature, max_tokens)
        uploaded_file = client.files.create(
            file=("batch_input.jsonl", io.BytesIO(payload.encode("utf-8"))),
            purpose="batch",
        )
        batch = client.batches.create(
            input_file_id=uploaded_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        self._batch_metadata[batch.id] = {
            "num_requests": len(prompts),
            "input_file_id": uploaded_file.id,
        }
        return batch.id

    def get_batch_status(self, batch_id: str) -> str:
        client = self._get_client()
        batch = client.batches.retrieve(batch_id)
        return str(batch.status)

    async def _wait_for_completion_async(self, batch_id: str) -> bool:
        start_time = time.monotonic()
        while time.monotonic() - start_time < self.timeout:
            status = self.get_batch_status(batch_id)
            if status == "completed":
                return True
            if status in {"failed", "expired", "cancelled"}:
                warn(f"Azure batch job {batch_id} ended with status: {status}")
                return False
            await asyncio.sleep(self.polling_interval)

        warn(f"Azure batch job {batch_id} timed out after {self.timeout} seconds")
        return False

    async def retrieve_batch_text_results(self, batch_id: str) -> List[str]:
        client = self._get_client()
        batch = client.batches.retrieve(batch_id)
        output_file_id = getattr(batch, "output_file_id", None)
        if not output_file_id:
            return []

        content = client.files.content(output_file_id).text
        rows = [json.loads(line) for line in content.splitlines() if line.strip()]
        expected = int(self._batch_metadata.get(batch_id, {}).get("num_requests", len(rows)))
        responses = [""] * expected

        for row in rows:
            idx = _safe_int(str(row.get("custom_id", "")))
            if idx >= expected:
                continue

            text = ""
            response = row.get("response")
            if isinstance(response, dict):
                text = _extract_text_from_chat_completion_body(response.get("body"))
            if not text:
                text = _extract_text_from_content(row.get("error"))
            responses[idx] = text

        return responses

    def cancel_batch(self, batch_id: str):
        self._get_client().batches.cancel(batch_id)

    async def _call_llm_batch(
        self, prompts: List[str], temperature: float, max_tokens: int
    ) -> List[str]:
        batch_id = self.submit_batch(prompts, temperature, max_tokens)
        if not await self._wait_for_completion_async(batch_id):
            raise RuntimeError(f"Batch job {batch_id} failed or timed out")
        return await self.retrieve_batch_text_results(batch_id)

    async def _call_llm_with_system_prompt_batch(
        self,
        system_prompts: List[str],
        user_prompts: List[str],
        temperature: float,
        max_tokens: int,
    ) -> List[str]:
        if len(system_prompts) != len(user_prompts):
            raise ValueError("Number of system prompts must match number of user prompts")

        prompts = [
            {"system": system_prompt, "user": user_prompt}
            for system_prompt, user_prompt in zip(system_prompts, user_prompts)
        ]
        batch_id = self.submit_batch(prompts, temperature, max_tokens)
        if not await self._wait_for_completion_async(batch_id):
            raise RuntimeError(f"Batch job {batch_id} failed or timed out")
        return await self.retrieve_batch_text_results(batch_id)


class OllamaNamer(_AISDKSyncWrapper):
    def __init__(
        self,
        model: str = "llama3.2",
        host: str = "http://localhost:11434",
        llm_specific_instructions=None,
    ):
        self.host = host
        super().__init__(
            provider="ollama",
            model=model,
            llm_specific_instructions=llm_specific_instructions,
            host=host,
        )


class AsyncOllamaNamer(_AISDKAsyncWrapper):
    def __init__(
        self,
        model: str = "llama3.2",
        host: str = "http://localhost:11434",
        llm_specific_instructions=None,
        max_concurrent_requests=5,
    ):
        self.host = host
        super().__init__(
            provider="ollama",
            model=model,
            llm_specific_instructions=llm_specific_instructions,
            max_concurrent_requests=max_concurrent_requests,
            host=host,
        )


class GoogleGeminiNamer(_AISDKSyncWrapper):
    def __init__(
        self,
        api_key,
        model: str = "gemini-1.5-flash",
        llm_specific_instructions=None,
    ):
        api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv(
            "GOOGLE_GENERATIVE_AI_API_KEY"
        )
        if not api_key:
            raise ValueError(
                "Google API key is required. Set `GOOGLE_API_KEY` or "
                "`GOOGLE_GENERATIVE_AI_API_KEY`, or pass `api_key` directly."
            )
        super().__init__(
            provider="google",
            model=model,
            llm_specific_instructions=llm_specific_instructions,
            apiKey=api_key,
        )


class AsyncGoogleGeminiNamer(_AISDKAsyncWrapper):
    def __init__(
        self,
        api_key,
        model: str = "gemini-1.5-flash",
        llm_specific_instructions=None,
        max_concurrent_requests=10,
    ):
        api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv(
            "GOOGLE_GENERATIVE_AI_API_KEY"
        )
        if not api_key:
            raise ValueError(
                "Google API key is required. Set `GOOGLE_API_KEY` or "
                "`GOOGLE_GENERATIVE_AI_API_KEY`, or pass `api_key` directly."
            )
        super().__init__(
            provider="google",
            model=model,
            llm_specific_instructions=llm_specific_instructions,
            max_concurrent_requests=max_concurrent_requests,
            apiKey=api_key,
        )
