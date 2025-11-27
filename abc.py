import requests

BASE_URL = "http://localhost:8000"
TOKEN = "CAESIONnP8EQtbRF62ELy3_J5GGvp-9wYvUgFiFGFGiD0W5HGh4KHGh2cy5hSGNIY2ZhcUZ0TkREUTczMEVJVzZrY00"

# Test health
response = requests.get(f"{BASE_URL}/health")
print("Health:", response.json())

# Test ingestion
data = {
    "path": "docs/example.md",
    "repo": "my-repo",
    "commit": "abc123",
    "deleted": False,
    "content": """---
title: Example Document
description: This is a test
tags: test, example
---

# Example Content

This is some example markdown content."""
}

response = requests.post(
    f"{BASE_URL}/ingest",
    json=data,
    headers={"x-ingest-token": TOKEN}
)
print("Ingest:", response.json())

# Test stats
response = requests.get(
    f"{BASE_URL}/stats",
    headers={"x-ingest-token": TOKEN}
)
print("Stats:", response.json())