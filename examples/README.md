# Examples

Standalone examples demonstrating llm-expect usage patterns.

## Setup

```bash
pip install llm-expect[all]
export ANTHROPIC_API_KEY="your-key"  # or OPENAI_API_KEY
```

## Examples

| File | Description |
|------|-------------|
| `basic_yesno.py` | Simple yes/no decision |
| `extract_json.py` | Extract structured JSON from text |
| `classification.py` | Classify items with `expect_choice` |
| `data_extraction.py` | Extract data using regex patterns |
| `code_review.py` | Multi-turn code review conversation |
| `retry_pattern.py` | Robust extraction with retries |

## Running

```bash
python examples/basic_yesno.py
```
