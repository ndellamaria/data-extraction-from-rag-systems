# Wikipedia Article Fetcher

A Python script to fetch thousands of Wikipedia articles created after a specific date in any language and save them to a text file.

## Installation

Install the required dependency:

```bash
pip install requests
```

## Usage

### Quick Start

Edit the configuration values in the `main()` function of `fetch_wikipedia_articles.py`:

```python
LANGUAGE = "de"              # Wikipedia language code
SINCE_DATE = "2025-01-01"    # Fetch articles created after this date
MAX_ARTICLES = 1000          # Maximum number of articles to fetch
OUTPUT_FILE = "wikipedia_articles.txt"  # Output file path
```

Then run:

```bash
python fetch_wikipedia_articles.py
```

### Programmatic Usage

```python
from fetch_wikipedia_articles import WikipediaArticleFetcher

# Create fetcher for German Wikipedia
fetcher = WikipediaArticleFetcher(language="de", max_articles=2000)

# Fetch articles created since January 1, 2025
fetcher.fetch_and_save(since_date="2025-01-01", output_file="german_articles.txt")
```

## Configuration Options

### Language Codes

Common Wikipedia language codes:
- `de` - German
- `en` - English
- `fr` - French
- `es` - Spanish
- `it` - Italian
- `ja` - Japanese
- `zh` - Chinese
- `ru` - Russian
- `pt` - Portuguese
- `ar` - Arabic

See full list at: https://meta.wikimedia.org/wiki/List_of_Wikipedias

### Date Format

Supported date formats:
- `YYYY-MM-DD` (e.g., "2025-01-01")
- `YYYY-MM-DDTHH:MM:SSZ` (e.g., "2025-01-01T00:00:00Z")

### Rate Limiting

The script includes built-in rate limiting (0.2 seconds between article fetches) to be respectful to Wikipedia servers. For very large datasets, consider:
- Running during off-peak hours
- Increasing the delay between requests
- Fetching in smaller batches

## Output Format

The output file contains articles separated by lines of equals signs:

```
================================================================================
TITLE: Article Title Here
================================================================================

Article content in plain text...

================================================================================
```

## Features

- Fetches articles from any Wikipedia language edition
- Filters by creation date
- Handles pagination automatically
- Includes error handling and retry logic
- Rate limiting to respect Wikipedia servers
- Progress tracking during fetch
- Plain text extraction (no HTML/markup)
- Only fetches main namespace articles (no talk pages, user pages, etc.)

## Limitations

- The Wikipedia API may have rate limits for anonymous requests
- Very large requests (10,000+ articles) may take considerable time
- Some articles may be unavailable or deleted between title fetch and content fetch

## Tips for Large Datasets

For fetching many thousands of articles:

1. Run in batches to avoid timeouts
2. Consider using the `rcstart` parameter to resume from a specific timestamp
3. Monitor disk space - text files can grow large
4. Consider compressing the output: `gzip wikipedia_articles.txt`
