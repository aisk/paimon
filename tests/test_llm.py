import unittest
from unittest.mock import patch

from paimon.llm import build_model, is_provider_available, split_model_string


class SplitModelStringTest(unittest.TestCase):
    def test_accepts_colon_and_legacy_slash(self) -> None:
        self.assertEqual(split_model_string("zai:glm-5.2"), ("zai", "glm-5.2"))
        self.assertEqual(split_model_string("zai/glm-5.2"), ("zai", "glm-5.2"))

    def test_colon_wins_when_both_separators_appear(self) -> None:
        self.assertEqual(split_model_string("gateway/openai:gpt-4o"), ("gateway/openai", "gpt-4o"))

    def test_unqualified_name_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            split_model_string("glm-5.2")


class BuildModelTest(unittest.TestCase):
    def test_zai_with_custom_endpoint(self) -> None:
        model = build_model("zai/glm-5.2", api_base="https://api.z.ai/api/coding/paas/v4", api_key="k")
        self.assertEqual(type(model).__name__, "ZaiModel")
        self.assertEqual(model.model_name, "glm-5.2")
        self.assertEqual(model.base_url, "https://api.z.ai/api/coding/paas/v4/")
        # The Z.ai profile drives reasoning_content passthrough; the migration
        # depends on these being set for GLM models.
        self.assertEqual(model.profile.get("openai_chat_thinking_field"), "reasoning_content")
        self.assertEqual(model.profile.get("openai_chat_send_back_thinking_parts"), "field")

    def test_openai_provider_takes_base_url_directly(self) -> None:
        model = build_model("openai:gpt-4o", api_base="http://localhost:8080/v1", api_key="k")
        self.assertEqual(model.base_url, "http://localhost:8080/v1/")

    def test_provider_prefix_picks_the_wire_dialect(self) -> None:
        # Same endpoint host, same model name: only the prefix decides whether
        # the request speaks OpenAI chat completions or Anthropic messages.
        openai_style = build_model("zai:glm-5.2", api_base="https://api.z.ai/api/coding/paas/v4", api_key="k")
        anthropic_style = build_model("anthropic:glm-5.2", api_base="https://api.z.ai/api/anthropic", api_key="k")
        self.assertEqual(type(openai_style).__name__, "ZaiModel")
        self.assertEqual(type(anthropic_style).__name__, "AnthropicModel")
        self.assertEqual(anthropic_style.model_name, "glm-5.2")
        self.assertEqual(anthropic_style.base_url, "https://api.z.ai/api/anthropic/")

    def test_custom_base_falls_back_to_the_providers_env_key(self) -> None:
        """AUTH-1: a custom endpoint must not mask ZAI_API_KEY and friends."""
        with patch.dict("os.environ", {"ZAI_API_KEY": "sk-from-env"}):
            model = build_model("zai:glm-5.2", api_base="http://localhost:8080/v4")
        self.assertEqual(model.client.api_key, "sk-from-env")

    def test_custom_base_prefers_the_configured_key_over_env(self) -> None:
        with patch.dict("os.environ", {"ZAI_API_KEY": "sk-from-env"}):
            model = build_model("zai:glm-5.2", api_base="http://localhost:8080/v4",
                                api_key="sk-configured")
        self.assertEqual(model.client.api_key, "sk-configured")

    def test_custom_base_without_any_key_still_builds(self) -> None:
        """Local endpoints that need no auth keep working on a placeholder."""
        with patch.dict("os.environ", {}, clear=True):
            model = build_model("zai:glm-5.2", api_base="http://localhost:8080/v4")
        self.assertEqual(model.client.api_key, "unset")

    def test_missing_sdk_reports_the_provider_by_name(self) -> None:
        with self.assertRaises(ValueError) as caught:
            build_model("bedrock:anthropic.claude-opus-4", api_key="k")
        self.assertIn("bedrock", str(caught.exception))


class ProviderAvailabilityTest(unittest.TestCase):
    def test_shipped_dialects_are_available(self) -> None:
        self.assertTrue(is_provider_available("openai"))
        self.assertTrue(is_provider_available("anthropic"))

    def test_unshipped_and_unknown_providers_are_not(self) -> None:
        self.assertFalse(is_provider_available("bedrock"))
        self.assertFalse(is_provider_available("no-such-provider"))

    def test_login_only_offers_available_providers(self) -> None:
        from paimon.login import _providers

        providers = _providers()
        self.assertIn("anthropic", providers)
        self.assertNotIn("bedrock", providers)
        self.assertTrue(all(is_provider_available(name) for name in providers))


if __name__ == "__main__":
    unittest.main()
