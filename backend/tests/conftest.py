"""Shared pytest fixtures. Run with `pytest` from the backend/ directory."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.embeddings import get_embedder  # noqa: E402


@pytest.fixture(scope="session")
def embedder():
    return get_embedder()


# A structured, "hard" document with headings + a table, used across tests.
HARD_DOC = """ANNUAL REPORT 2024

ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS
Our total revenue grew 14% year over year, driven by strong demand in the cloud segment. Operating margin expanded by 220 basis points as we controlled headcount growth. We expect continued momentum into the next fiscal year despite macro headwinds.

Consolidated Balance Sheet
Cash and equivalents        1,204     980
Accounts receivable           512     440
Total current assets        2,310   1,905

Note 3. Revenue Recognition
Revenue is recognized when control of goods or services transfers to the customer. Subscription revenue is recognized ratably over the contract term. The company applies the five-step model under ASC 606.

RISK FACTORS
Our business faces competition from larger incumbents with greater resources. Currency fluctuations may materially affect reported results. A prolonged economic downturn could reduce customer spending on our products.
"""
