# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2025-02-19

### Added
- Initial release
- `Conversation` class with `send()`, `expect()`, `send_expect()` methods
- OpenAI and Anthropic provider support
- Auto-detect provider from environment variables
- Pattern-to-Prompt: automatic format instructions from regex patterns
- Expect templates: `expect_json()`, `expect_number()`, `expect_choice()`, `expect_yesno()`, `expect_code()`
- `ExpectError` with helpful error messages
- Full type annotations (PEP 561 compatible)
- Conversation history management with `max_history` and `clear_history()`

### Security
- Input validation: timeout must be positive integer
- ReDoS protection: response length limit in `expect_json()`