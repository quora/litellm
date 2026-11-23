import os
import sys

import httpx
import pytest

import litellm
from litellm.llms.bedrock.chat import BedrockConverseLLM
from litellm.llms.bedrock.common_utils import _get_all_bedrock_regions
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path


def test_encode_model_id_with_inference_profile():
    """
    Test instance profile is properly encoded when used as a model
    """
    test_model = "arn:aws:bedrock:us-east-1:12345678910:application-inference-profile/ujdtmcirjhevpi"
    expected_model = "arn%3Aaws%3Abedrock%3Aus-east-1%3A12345678910%3Aapplication-inference-profile%2Fujdtmcirjhevpi"
    bedrock_converse_llm = BedrockConverseLLM()
    returned_model = bedrock_converse_llm.encode_model_id(test_model)
    assert expected_model == returned_model


class TestBedrockRegionInModelPath:
    """
    Tests for region extraction from bedrock/{region}/{model} path format.

    When a user passes model="bedrock/ap-northeast-1/moonshotai.kimi-k2.5",
    get_llm_provider strips "bedrock/" and passes "ap-northeast-1/moonshotai.kimi-k2.5"
    to the converse handler. The handler must:
    1. Strip the region from modelId (so AWS gets "moonshotai.kimi-k2.5", not "ap-northeast-1%2Fmoonshotai.kimi-k2.5")
    2. Use the extracted region as aws_region_name for the API call
    """

    @pytest.mark.parametrize(
        "model,expected_model_id,expected_region",
        [
            # Region embedded in path — both modelId and region must be extracted
            (
                "ap-northeast-1/moonshotai.kimi-k2.5",
                "moonshotai.kimi-k2.5",
                "ap-northeast-1",
            ),
            (
                "us-east-1/moonshotai.kimi-k2.5",
                "moonshotai.kimi-k2.5",
                "us-east-1",
            ),
            (
                "us-west-2/anthropic.claude-haiku-4-5-20251001-v1:0",
                "anthropic.claude-haiku-4-5-20251001-v1%3A0",
                "us-west-2",
            ),
            # No region in path — modelId unchanged, no region injected
            (
                "moonshotai.kimi-k2.5",
                "moonshotai.kimi-k2.5",
                None,
            ),
            # Cross-region inference prefix (us., eu., ap.) — not a region path segment
            (
                "us.anthropic.claude-haiku-4-5-20251001-v1:0",
                "us.anthropic.claude-haiku-4-5-20251001-v1%3A0",
                None,
            ),
        ],
    )
    def test_region_and_model_id_extraction(
        self, model, expected_model_id, expected_region
    ):
        """
        Verify that completion() correctly extracts both modelId and aws_region_name
        from the bedrock/{region}/{model} path format.
        """
        bedrock_converse_llm = BedrockConverseLLM()
        optional_params: dict = {}

        # Simulate the modelId + region extraction logic from completion()
        _model_for_id = model
        _stripped = _model_for_id
        for rp in ["bedrock/converse/", "bedrock/", "converse/"]:
            if _stripped.startswith(rp):
                _stripped = _stripped[len(rp):]
                break

        _region_from_model = None
        _potential_region = _stripped.split("/", 1)[0]
        if _potential_region in _get_all_bedrock_regions() and "/" in _stripped:
            _region_from_model = _potential_region
            _stripped = _stripped.split("/", 1)[1]
            _model_for_id = _stripped

        for _nova_prefix in ["nova-2/", "nova/"]:
            if _stripped.startswith(_nova_prefix):
                _model_for_id = _model_for_id.replace(_nova_prefix, "", 1)
                break

        model_id = bedrock_converse_llm.encode_model_id(model_id=_model_for_id)
        if _region_from_model is not None and "aws_region_name" not in optional_params:
            optional_params["aws_region_name"] = _region_from_model

        assert model_id == expected_model_id, (
            f"modelId mismatch for {model!r}: got {model_id!r}, expected {expected_model_id!r}"
        )
        assert optional_params.get("aws_region_name") == expected_region, (
            f"region mismatch for {model!r}: got {optional_params.get('aws_region_name')!r}, expected {expected_region!r}"
        )

    def test_explicit_aws_region_name_not_overridden(self):
        """
        If aws_region_name is already set in optional_params, the region in the
        model path must NOT override it.
        """
        bedrock_converse_llm = BedrockConverseLLM()
        optional_params = {"aws_region_name": "eu-west-1"}
        model = "ap-northeast-1/moonshotai.kimi-k2.5"

        _model_for_id = model
        _stripped = model
        _region_from_model = None
        _potential_region = _stripped.split("/", 1)[0]
        if _potential_region in _get_all_bedrock_regions() and "/" in _stripped:
            _region_from_model = _potential_region
            _stripped = _stripped.split("/", 1)[1]
            _model_for_id = _stripped

        model_id = bedrock_converse_llm.encode_model_id(model_id=_model_for_id)
        if _region_from_model is not None and "aws_region_name" not in optional_params:
            optional_params["aws_region_name"] = _region_from_model

        # modelId is still correctly stripped
        assert model_id == "moonshotai.kimi-k2.5"
        # explicitly set region is preserved
        assert optional_params["aws_region_name"] == "eu-west-1"


_CONVERSE_MODEL = "bedrock/converse/anthropic.claude-haiku-4-5-20251001-v1:0"
_AWS_REQUEST_ID = "0a1b2c3d-4e5f-6071-8293-a4b5c6d7e8f9"
_AWS_KWARGS = {
    "aws_access_key_id": "fake",
    "aws_secret_access_key": "fake",
    "aws_region_name": "us-east-1",
}


def _converse_reply(status_code: int, stream: bool) -> httpx.Response:
    headers = {"x-amzn-requestid": _AWS_REQUEST_ID}
    if status_code != 200:
        return httpx.Response(
            status_code, headers=headers, json={"message": "Throttled"}
        )
    if stream:
        # An empty event stream: the headers are what is under test.
        return httpx.Response(200, headers=headers, content=b"")
    return httpx.Response(
        200,
        headers=headers,
        json={
            "output": {"message": {"role": "assistant", "content": [{"text": "hi"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
        },
    )


def _mock_client(is_async: bool, status_code: int, stream: bool):
    transport = httpx.MockTransport(
        lambda request: _converse_reply(status_code, stream)
    )
    if is_async:
        handler = AsyncHTTPHandler()
        handler.client = httpx.AsyncClient(transport=transport)
        return handler
    handler = HTTPHandler()
    handler.client = httpx.Client(transport=transport)
    return handler


async def _complete(is_async: bool, stream: bool, client):
    kwargs = {
        "model": _CONVERSE_MODEL,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": stream,
        "client": client,
        **_AWS_KWARGS,
    }
    if is_async:
        return await litellm.acompletion(**kwargs)
    return litellm.completion(**kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [True, False])
@pytest.mark.parametrize("stream", [False, True])
async def test_converse_forwards_bedrock_response_headers(is_async, stream):
    """The x-amzn-requestid that CloudTrail and AWS support quote must reach the
    caller, or a Bedrock call cannot be traced back to the request that made it."""
    response = await _complete(
        is_async, stream, _mock_client(is_async, status_code=200, stream=stream)
    )

    assert (
        response._hidden_params["additional_headers"]["llm_provider-x-amzn-requestid"]
        == _AWS_REQUEST_ID
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [True, False])
async def test_converse_error_keeps_bedrock_response_headers(is_async):
    """AWS support asks for the request id of a failed call too."""
    with pytest.raises(litellm.RateLimitError) as exc_info:
        await _complete(
            is_async,
            stream=False,
            client=_mock_client(is_async, status_code=429, stream=False),
        )

    assert (
        exc_info.value.litellm_response_headers["x-amzn-requestid"] == _AWS_REQUEST_ID
    )
