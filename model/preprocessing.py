"""
Text Preprocessing Module for Fake News Detection.

Provides reusable, robust text cleaning functions to prepare raw news articles
and headlines for NLP vectorization and model inference.
Designed for seamless integration into training pipelines and FastAPI endpoints.
"""

import re
from typing import Union, List
import pandas as pd

# Precompile regular expressions for optimal performance
URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
HTML_PATTERN = re.compile(r'<.*?>')
SPECIAL_CHAR_PATTERN = re.compile(r'[^a-zA-Z0-9\s]')
WHITESPACE_PATTERN = re.compile(r'\s+')


def clean_text(text: Union[str, float, None]) -> str:
    """
    Cleans an input text string according to standard NLP guidelines:
      1. Handles missing or non-string input safely.
      2. Converts text to lowercase.
      3. Strips URL hyperlinks.
      4. Removes HTML tags.
      5. Eliminates punctuation and special characters (keeping alphanumerics and spaces).
      6. Normalizes consecutive whitespace into a single space and strips boundaries.

    Args:
        text: Input string, None, or NaN value.

    Returns:
        Cleaned, normalized string.
    """
    # 1. Handle missing, None, or non-string inputs safely
    if text is None or (isinstance(text, float) and pd.isna(text)) or not isinstance(text, str):
        return ""

    # 2. Convert to lowercase
    cleaned = text.lower()

    # 3. Remove URLs
    cleaned = URL_PATTERN.sub(' ', cleaned)

    # 4. Remove HTML tags
    cleaned = HTML_PATTERN.sub(' ', cleaned)

    # 5. Remove punctuation and special characters
    cleaned = SPECIAL_CHAR_PATTERN.sub(' ', cleaned)

    # 6. Normalize whitespace
    cleaned = WHITESPACE_PATTERN.sub(' ', cleaned).strip()

    return cleaned


def combine_and_clean(title: Union[str, float, None], text: Union[str, float, None]) -> str:
    """
    Combines news title and article body, then applies clean_text.
    Useful for single-item inference in API endpoints.

    Args:
        title: News headline.
        text: Main article body text.

    Returns:
        Cleaned combined string.
    """
    safe_title = str(title) if title is not None and not pd.isna(title) else ""
    safe_text = str(text) if text is not None and not pd.isna(text) else ""
    combined = f"{safe_title} {safe_text}".strip()
    return clean_text(combined)


if __name__ == "__main__":
    sample = "Trump WON 2024?! Visit https://fakenews.com/article <p>Details here!!</p>"
    print(f"Original: {sample}")
    print(f"Cleaned : {clean_text(sample)}")
