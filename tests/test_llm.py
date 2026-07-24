import unittest

from paimon.llm import build_model, split_model_string


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


if __name__ == "__main__":
    unittest.main()
