"""
Fal.ai video generation configuration for LiteLLM.

Fal.ai uses a queue-based async API:
1. POST https://queue.fal.run/{model_path} submits a generation request
2. Returns request_id + status_url + response_url
3. GET .../requests/{request_id}/status polls for completion
4. GET .../requests/{request_id}/response retrieves the result

Auth: Authorization: Key {FAL_AI_API_KEY}

Docs: https://docs.fal.ai/model-apis/model-endpoints/queue
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import httpx
from httpx._types import RequestFiles

import litellm
from litellm.llms.base_llm.videos.transformation import BaseVideoConfig
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    _get_httpx_client,
    get_async_httpx_client,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.main import VideoCreateOptionalRequestParams, VideoObject
from litellm.types.videos.utils import (
    encode_video_id_with_provider,
    extract_original_video_id,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


FAL_QUEUE_BASE_URL = "https://queue.fal.run"

# Fal queue status -> OpenAI-compatible status
_FAL_STATUS_MAP = {
    "IN_QUEUE": "queued",
    "IN_PROGRESS": "in_progress",
    "COMPLETED": "completed",
    "FAILED": "failed",
}


class FalAIVideoConfig(BaseVideoConfig):
    """
    Configuration for Fal.ai video generation (Kling, etc.).

    Fal.ai queue API:
    - POST /{model_path}                           -> {request_id, status_url, response_url}
    - GET  /{model_path}/requests/{id}/status       -> {status}
    - GET  /{model_path}/requests/{id}/response     -> {video: {url, ...}}

    The litellm model name IS the Fal model path
    (e.g., "fal-ai/kling-video/v2.1/master/text-to-video").
    """

    def get_supported_openai_params(self, model: str) -> list:
        return [
            "model",
            "prompt",
            "input_reference",
            "seconds",
            "size",
            "user",
            "extra_headers",
        ]

    def map_openai_params(
        self,
        video_create_optional_params: VideoCreateOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> Dict:
        mapped_params: Dict[str, Any] = {}

        # input_reference -> image_url
        if "input_reference" in video_create_optional_params:
            input_reference = video_create_optional_params["input_reference"]
            if input_reference is not None:
                mapped_params["image_url"] = str(input_reference)

        # seconds -> duration (Fal uses string "5" or "10")
        if "seconds" in video_create_optional_params:
            seconds = video_create_optional_params["seconds"]
            if seconds is not None:
                mapped_params["duration"] = str(seconds)

        # size -> aspect_ratio (convert "1280x720" to "16:9")
        if "size" in video_create_optional_params:
            size = video_create_optional_params["size"]
            if isinstance(size, str) and "x" in size:
                aspect_ratio = _size_to_aspect_ratio(size)
                if aspect_ratio:
                    mapped_params["aspect_ratio"] = aspect_ratio

        # Pass through provider-specific parameters
        supported_openai_params = self.get_supported_openai_params(model)
        for key, value in video_create_optional_params.items():
            if key not in supported_openai_params:
                mapped_params[key] = value

        return mapped_params

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: Optional[str] = None,
        litellm_params: Optional[GenericLiteLLMParams] = None,
    ) -> dict:
        if litellm_params and litellm_params.api_key:
            api_key = api_key or litellm_params.api_key

        api_key = api_key or litellm.api_key or get_secret_str("FAL_AI_API_KEY")

        if api_key is None:
            raise ValueError(
                "Fal.ai API key is required. Set FAL_AI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        headers.update(
            {
                "Authorization": f"Key {api_key}",
                "Content-Type": "application/json",
            }
        )
        return headers

    def get_complete_url(
        self,
        model: str,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        if api_base is None:
            api_base = get_secret_str("FAL_AI_API_BASE") or FAL_QUEUE_BASE_URL
        return api_base.rstrip("/")

    # -- Create ---------------------------------------------------------------

    def transform_video_create_request(
        self,
        model: str,
        prompt: str,
        api_base: str,
        video_create_optional_request_params: Dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[Dict, RequestFiles, str]:
        request_data: Dict[str, Any] = {"prompt": prompt}
        request_data.update(video_create_optional_request_params)

        files_list: List[Tuple[str, Any]] = []

        # Model name is the Fal model path
        url = f"{api_base}/{model}"
        return request_data, files_list, url

    def transform_video_create_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: Optional[str] = None,
        request_data: Optional[Dict] = None,
    ) -> VideoObject:
        response_data = raw_response.json()

        video_data: Dict[str, Any] = {
            "id": response_data.get("request_id", ""),
            "object": "video",
            "status": "queued",
        }

        if model:
            video_data["model"] = model

        video_obj = VideoObject(**video_data)  # type: ignore[arg-type]

        if custom_llm_provider and video_obj.id:
            video_obj.id = encode_video_id_with_provider(
                video_obj.id, custom_llm_provider, model
            )

        return video_obj

    # -- Status ---------------------------------------------------------------

    def transform_video_status_retrieve_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        original_video_id, provider, model = _decode_video_id(video_id)
        # Fal status endpoint: /{model_path}/requests/{request_id}/status
        url = f"{api_base}/{model}/requests/{original_video_id}/status"
        return url, {}

    def transform_video_status_retrieve_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: Optional[str] = None,
    ) -> VideoObject:
        response_data = raw_response.json()

        fal_status = response_data.get("status", "IN_QUEUE")
        status = _FAL_STATUS_MAP.get(fal_status, "in_progress")

        video_data: Dict[str, Any] = {
            "id": "",
            "object": "video",
            "status": status,
        }

        # Fal includes logs in status response
        logs = response_data.get("logs")
        if logs:
            video_data["progress"] = _estimate_progress(fal_status)

        video_obj = VideoObject(**video_data)  # type: ignore[arg-type]
        return video_obj

    # -- Content (result) -----------------------------------------------------

    def transform_video_content_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
        variant: Optional[str] = None,
    ) -> Tuple[str, Dict]:
        original_video_id, provider, model = _decode_video_id(video_id)
        # Fal result endpoint: /{model_path}/requests/{request_id}/response
        url = f"{api_base}/{model}/requests/{original_video_id}/response"
        return url, {}

    def _extract_video_url(self, response_data: Dict[str, Any]) -> str:
        video_info = response_data.get("video") or {}
        video_url = video_info.get("url")

        if not video_url:
            raise ValueError(
                "Video URL not found in Fal.ai response. Video may not be ready yet."
            )
        return video_url

    def transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> bytes:
        response_data = raw_response.json()
        video_url = self._extract_video_url(response_data)

        httpx_client: HTTPHandler = _get_httpx_client()
        video_response = httpx_client.get(video_url)
        video_response.raise_for_status()
        return video_response.content

    async def async_transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> bytes:
        response_data = raw_response.json()
        video_url = self._extract_video_url(response_data)

        async_httpx_client: AsyncHTTPHandler = get_async_httpx_client(
            llm_provider=litellm.LlmProviders.FAL_AI,
        )
        video_response = await async_httpx_client.get(video_url)
        video_response.raise_for_status()
        return video_response.content

    # -- Unsupported operations -----------------------------------------------

    def transform_video_remix_request(self, *args: Any, **kwargs: Any) -> Tuple[str, Dict]:
        raise NotImplementedError("Video remix is not supported by Fal.ai API")

    def transform_video_remix_response(self, *args: Any, **kwargs: Any) -> VideoObject:
        raise NotImplementedError("Video remix is not supported by Fal.ai API")

    def transform_video_edit_request(self, *args: Any, **kwargs: Any) -> Tuple[str, Dict]:
        raise NotImplementedError("Video editing is not supported by Fal.ai API")

    def transform_video_edit_response(self, *args: Any, **kwargs: Any) -> VideoObject:
        raise NotImplementedError("Video editing is not supported by Fal.ai API")

    def transform_video_extension_request(self, *args: Any, **kwargs: Any) -> Tuple[str, Dict]:
        raise NotImplementedError("Video extension is not supported by Fal.ai API")

    def transform_video_extension_response(self, *args: Any, **kwargs: Any) -> VideoObject:
        raise NotImplementedError("Video extension is not supported by Fal.ai API")

    def transform_video_list_request(self, *args: Any, **kwargs: Any) -> Tuple[str, Dict]:
        raise NotImplementedError("Video listing is not supported by Fal.ai API")

    def transform_video_list_response(self, *args: Any, **kwargs: Any) -> Dict[str, str]:
        raise NotImplementedError("Video listing is not supported by Fal.ai API")

    def transform_video_delete_request(self, *args: Any, **kwargs: Any) -> Tuple[str, Dict]:
        raise NotImplementedError("Video deletion is not supported by Fal.ai API")

    def transform_video_delete_response(self, *args: Any, **kwargs: Any) -> VideoObject:
        raise NotImplementedError("Video deletion is not supported by Fal.ai API")

    def transform_video_create_character_request(self, *args: Any, **kwargs: Any) -> Tuple[str, Dict]:
        raise NotImplementedError("Video character creation is not supported by Fal.ai API")

    def transform_video_create_character_response(self, *args: Any, **kwargs: Any) -> VideoObject:
        raise NotImplementedError("Video character creation is not supported by Fal.ai API")

    def transform_video_get_character_request(self, *args: Any, **kwargs: Any) -> Tuple[str, Dict]:
        raise NotImplementedError("Video character retrieval is not supported by Fal.ai API")

    def transform_video_get_character_response(self, *args: Any, **kwargs: Any) -> VideoObject:
        raise NotImplementedError("Video character retrieval is not supported by Fal.ai API")


def _decode_video_id(video_id: str) -> Tuple[str, str, str]:
    """Extract original_id, provider, and model from encoded video_id."""
    original_id = extract_original_video_id(video_id)
    # The encoded video_id format includes provider and model info
    # We need the model path for Fal URL construction
    parts = video_id.split("_", 2)
    provider = ""
    model = ""
    if len(parts) >= 3:
        # video_{encoded_data} format -- decode to get provider and model
        import base64
        try:
            decoded = base64.b64decode(parts[1] + "==").decode("utf-8")
            # Format: "original_id::provider::model"
            decoded_parts = decoded.split("::")
            if len(decoded_parts) >= 3:
                provider = decoded_parts[1]
                model = decoded_parts[2]
        except Exception:
            pass
    return original_id, provider, model


def _size_to_aspect_ratio(size: str) -> Optional[str]:
    """Convert 'WIDTHxHEIGHT' to Fal aspect ratio."""
    ratios = {
        (16, 9): "16:9",
        (9, 16): "9:16",
        (1, 1): "1:1",
    }
    try:
        w, h = size.lower().split("x")
        width, height = int(w), int(h)
    except (ValueError, AttributeError):
        return None

    if width == 0 or height == 0:
        return None

    actual_ratio = width / height
    best_match = None
    best_diff = float("inf")
    for (rw, rh), label in ratios.items():
        diff = abs(actual_ratio - rw / rh)
        if diff < best_diff:
            best_diff = diff
            best_match = label

    return best_match


def _estimate_progress(fal_status: str) -> int:
    """Estimate progress percentage from Fal status."""
    progress_map = {
        "IN_QUEUE": 0,
        "IN_PROGRESS": 50,
        "COMPLETED": 100,
        "FAILED": 0,
    }
    return progress_map.get(fal_status, 0)
