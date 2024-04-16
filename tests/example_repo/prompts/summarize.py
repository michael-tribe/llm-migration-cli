import logging

from ..client import InstrumentedLlmClient


TEXTS = [
    "Mary had a little lamb, its fleece was white as snow. And everywhere that Mary went, the lamb was sure to go.",
    "The quick brown fox jumps over the lazy dog. The five boxing wizards jump quickly. Pack my box with five dozen liquor jugs.",
    "The sky above the port was the color of television, tuned to a dead channel.",
    "It was a bright cold day in April, and the clocks were striking thirteen.",
    "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness.",
]


def summarize_text(client: InstrumentedLlmClient) -> None:
    template = "Summarize the following text in as few words as possible: '{text}'"
    temperature = 1.0
    max_tokens = 50

    for text in TEXTS:
        output_text = client.generate(
            template=template,
            variables={"text": text},
            temperature=temperature,
            max_tokens=max_tokens,
        )
        logging.info(f"Summarized text: {output_text}")
