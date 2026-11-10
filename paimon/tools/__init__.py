"""The tools the agent can call, split by concern.

Import order is one-way: result, base, files, grep, shell, background,
registry, permissions; each module only imports the ones before it.

This file deliberately imports nothing. The grep worker process imports
paimon.tools.grep, which is standard library only, and anything imported here
would be paid for by every grep call: registry pulls in pydantic and
pydantic-ai, about a second of startup.
"""
