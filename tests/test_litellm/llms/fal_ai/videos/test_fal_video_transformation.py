"""
Tests for Fal.ai video generation transformation.
"""

from unittest.mock import Mock

import httpx
import pytest

from litellm.llms.fal_ai.videos.transformation import FalAIVideoConfig
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.main import VideoObject
from litellm.types.videos.utils import encode_video_id_with_provider


class TestFalAIVideoTransformation:
    """Test FalAIVideoConfig transformation class."""

    def setup_method(self):
        self.config = FalAIVideoConfig()
        self.mock_logging_obj = Mock()
        self.model = "fal-ai/kling-video/v2.1/master/text-to-video"

    def test_get_supported_openai_params(self):
        params = self.config.get_supported_openai_params(self.model)
        assert "prompt" in params
        assert "input_reference" in params
        assert "seconds" in params
        assert "size" in params

    def test_get_complete_url_default(self):
        url = self.config.get_complete_url(self.model, None, {})
        assert url == "https://queue.fal.run"

    def test_get_complete_url_custom(self):
        url = self.config.get_complete_url(self.model, "https://custom.fal.run/", {})
        assert url == "https://custom.fal.run"

    def test_validate_environment(self):
        headers = self.config.validate_environment(
            {}, self.model, api_key="test-fal-key"
        )
        assert headers["Authorization"] == "Key test-fal-key"
        assert headers["Content-Type"] == "application/json"

    def test_validate_environment_missing_key(self):
        with pytest.raises(ValueError, match="Fal.ai API key is required"):
            self.config.validate_environment({}, self.model, api_key=None)

    def test_map_openai_params_seconds(self):
        mapped = self.config.map_openai_params(
            {"seconds": "10"}, self.model, False
        )
        assert mapped["duration"] == "10"

    def test_map_openai_params_size_to_aspect_ratio(self):
        mapped = self.config.map_openai_params(
            {"size": "1280x720"}, self.model, False
        )
        assert mapped["aspect_ratio"] == "16:9"

    def test_map_openai_params_portrait(self):
        mapped = self.config.map_openai_params(
            {"size": "720x1280"}, self.model, False
        )
        assert mapped["aspect_ratio"] == "9:16"

    def test_map_openai_params_input_reference(self):
        mapped = self.config.map_openai_params(
            {"input_reference": "https://example.com/image.png"},
            self.model,
            False,
        )
        assert mapped["image_url"] == "https://example.com/image.png"

    def test_map_openai_params_passthrough(self):
        mapped = self.config.map_openai_params(
            {"negative_prompt": "blur, distort"},
            self.model,
            False,
        )
        assert mapped["negative_prompt"] == "blur, distort"

    def test_transform_video_create_request(self):
        data, files, url = self.config.transform_video_create_request(
            model=self.model,
            prompt="A cat walking on the beach",
            api_base="https://queue.fal.run",
            video_create_optional_request_params={
                "duration": "5",
                "aspect_ratio": "16:9",
            },
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert data["prompt"] == "A cat walking on the beach"
        assert data["duration"] == "5"
        assert data["aspect_ratio"] == "16:9"
        assert files == []
        assert url == f"https://queue.fal.run/{self.model}"

    def test_transform_video_create_response(self):
        mock_response = Mock(spec=httpx.Response)
        mock_response.json.return_value = {
            "request_id": "fal-req-abc123",
            "status_url": "https://queue.fal.run/.../status",
            "response_url": "https://queue.fal.run/.../response",
        }

        result = self.config.transform_video_create_response(
            model=self.model,
            raw_response=mock_response,
            logging_obj=self.mock_logging_obj,
            custom_llm_provider="fal_ai",
        )

        assert isinstance(result, VideoObject)
        assert result.status == "queued"
        assert result.id.startswith("video_")

    def test_transform_video_status_in_queue(self):
        mock_response = Mock(spec=httpx.Response)
        mock_response.json.return_value = {
            "status": "IN_QUEUE",
        }

        result = self.config.transform_video_status_retrieve_response(
            raw_response=mock_response,
            logging_obj=self.mock_logging_obj,
        )

        assert result.status == "queued"

    def test_transform_video_status_in_progress(self):
        mock_response = Mock(spec=httpx.Response)
        mock_response.json.return_value = {
            "status": "IN_PROGRESS",
            "logs": [{"message": "Generating..."}],
        }

        result = self.config.transform_video_status_retrieve_response(
            raw_response=mock_response,
            logging_obj=self.mock_logging_obj,
        )

        assert result.status == "in_progress"
        assert result.progress == 50

    def test_transform_video_status_completed(self):
        mock_response = Mock(spec=httpx.Response)
        mock_response.json.return_value = {
            "status": "COMPLETED",
        }

        result = self.config.transform_video_status_retrieve_response(
            raw_response=mock_response,
            logging_obj=self.mock_logging_obj,
        )

        assert result.status == "completed"

    def test_transform_video_status_failed(self):
        mock_response = Mock(spec=httpx.Response)
        mock_response.json.return_value = {
            "status": "FAILED",
        }

        result = self.config.transform_video_status_retrieve_response(
            raw_response=mock_response,
            logging_obj=self.mock_logging_obj,
        )

        assert result.status == "failed"

    def test_transform_video_content_request(self):
        video_id = encode_video_id_with_provider(
            "fal-req-abc123", "fal_ai", self.model
        )
        url, params = self.config.transform_video_content_request(
            video_id=video_id,
            api_base="https://queue.fal.run",
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert "requests/fal-req-abc123/response" in url

    def test_extract_video_url(self):
        url = self.config._extract_video_url(
            {"video": {"url": "https://v3.fal.media/files/video.mp4"}}
        )
        assert url == "https://v3.fal.media/files/video.mp4"

    def test_extract_video_url_missing(self):
        with pytest.raises(ValueError, match="Video URL not found"):
            self.config._extract_video_url({"video": {}})

    def test_unsupported_operations(self):
        with pytest.raises(NotImplementedError):
            self.config.transform_video_remix_request()
        with pytest.raises(NotImplementedError):
            self.config.transform_video_edit_request()
        with pytest.raises(NotImplementedError):
            self.config.transform_video_extension_request()
        with pytest.raises(NotImplementedError):
            self.config.transform_video_list_request()
        with pytest.raises(NotImplementedError):
            self.config.transform_video_delete_request()

    def test_full_workflow(self):
        """Test: create -> status (queued) -> status (completed) -> content."""
        # Step 1: Create
        data, files, url = self.config.transform_video_create_request(
            model=self.model,
            prompt="A sunset over the ocean",
            api_base="https://queue.fal.run",
            video_create_optional_request_params={"duration": "5"},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert url == f"https://queue.fal.run/{self.model}"

        # Step 2: Parse create response
        mock_create = Mock(spec=httpx.Response)
        mock_create.json.return_value = {"request_id": "fal-workflow-test"}

        video_obj = self.config.transform_video_create_response(
            model=self.model,
            raw_response=mock_create,
            logging_obj=self.mock_logging_obj,
            custom_llm_provider="fal_ai",
        )
        assert video_obj.status == "queued"

        # Step 3: Status - in queue
        mock_queued = Mock(spec=httpx.Response)
        mock_queued.json.return_value = {"status": "IN_QUEUE"}

        queued_obj = self.config.transform_video_status_retrieve_response(
            raw_response=mock_queued,
            logging_obj=self.mock_logging_obj,
        )
        assert queued_obj.status == "queued"

        # Step 4: Status - completed
        mock_done = Mock(spec=httpx.Response)
        mock_done.json.return_value = {"status": "COMPLETED"}

        done_obj = self.config.transform_video_status_retrieve_response(
            raw_response=mock_done,
            logging_obj=self.mock_logging_obj,
        )
        assert done_obj.status == "completed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
