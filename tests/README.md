# CCGL Analytics Tests

This directory contains test files for the CCGL Analytics System.

## Test Structure

- `unit/` - Unit tests for individual modules
- `integration/` - Integration tests for system components
- `mcp/` - Tests for MCP server functionality
- `data/` - Test data files

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test category
pytest tests/unit/ -v
pytest tests/integration/ -v
pytest tests/mcp/ -v

# Run with coverage
pytest tests/ --cov=ccgl_analytics --cov-report=html
```

## Test Requirements

- pytest>=7.4.0
- pytest-cov>=4.1.0
- pytest-asyncio>=0.21.0
- pytest-mock>=3.11.0