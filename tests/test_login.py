import unittest
from unittest.mock import patch

from paimon import llm, login


def _catalog(*names: str):
    return patch("paimon.llm.known_models", return_value=sorted(names))


class ModelsTest(unittest.TestCase):
    def test_only_the_newest_model_of_each_line_is_offered(self) -> None:
        with _catalog("openai:gpt-4o", "openai:gpt-5.5", "openai:gpt-5.6-sol",
                      "openai:gpt-5.6-sol-2026-07-09", "openai:gpt-6-sol", "openai:gpt-6.1-sol"):
            self.assertEqual(llm.models_for("openai"), ["gpt-6.1-sol"])

    def test_a_model_without_a_successor_stays(self) -> None:
        with _catalog("openai:gpt-5.6-luna", "openai:gpt-5.6-sol", "openai:gpt-6.1-sol"):
            self.assertEqual(llm.models_for("openai"), ["gpt-6.1-sol", "gpt-5.6-luna"])

    def test_the_chatgpt_plan_offers_openais_models(self) -> None:
        with _catalog("openai:gpt-5.6-sol", "openai:o3"):
            self.assertEqual(llm.models_for("chatgpt"), ["gpt-5.6-sol"])

    def test_an_unknown_provider_is_shown_whole(self) -> None:
        with _catalog("acme:rocket-1", "acme:rocket-2"):
            self.assertEqual(llm.models_for("acme"), ["rocket-1", "rocket-2"])

    def test_every_provider_on_offer_lists_some_model(self) -> None:
        for provider in login._providers():
            self.assertTrue(llm.models_for(provider), provider)
