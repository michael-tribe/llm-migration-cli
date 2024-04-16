import logging

from ..client import InstrumentedLlmClient


TOPICS = ["nature", "love", "life", "death", "time", "beauty", "art", "friendship", "happiness", "sadness"]


def write_a_poem(client: InstrumentedLlmClient) -> None:
    template = "Write a haiku about {topic}"
    temperature = 1.0
    max_tokens = 50

    for topic in TOPICS:
        variables = {"topic": topic}
        output_text = client.generate(
            template=template,
            variables=variables,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        logging.info(f"Generated poem: {output_text}")
