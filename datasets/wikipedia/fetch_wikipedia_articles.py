#!/usr/bin/env python3
"""
Wikipedia Article Fetcher

Fetches Wikipedia articles created after a specific date in a specified language
and saves them to a text file.

Dependencies: pip install requests
"""

import requests
import time
import json
from datetime import datetime
from typing import List, Dict, Optional
import argparse

class WikipediaArticleFetcher:
    """Fetches Wikipedia articles based on language and creation date."""

    def __init__(self, language: str = "de", max_articles: int = 1000):
        self.language = language
        self.max_articles = max_articles
        self.base_url = f"https://{language}.wikipedia.org/w/api.php"

        # ✅ Add a proper User-Agent to all session requests
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "cs2881AISafetyP1/1.0 (https://github.com/valerie-chen)"
        })

    def get_recent_articles(self, since_date: str) -> List[str]:
        """Get list of article titles created after a specific date."""
        try:
            if 'T' in since_date:
                dt = datetime.fromisoformat(since_date.replace('Z', '+00:00'))
            else:
                dt = datetime.strptime(since_date, '%Y-%m-%d')
            timestamp = dt.strftime('%Y%m%d%H%M%S')
        except ValueError as e:
            raise ValueError(f"Invalid date format: {since_date}.") from e

        articles = []
        continue_param = {}

        print(f"Fetching articles created after {since_date} from {self.language}.wikipedia.org...")

        while len(articles) < self.max_articles:
            params = {
                'action': 'query',
                'list': 'recentchanges',
                'rctype': 'new',
                'rcnamespace': '0',
                'rcstart': timestamp,
                'rcdir': 'newer',
                'rclimit': '500',
                'rcprop': 'title|timestamp',
                'format': 'json'
            }
            params.update(continue_param)

            try:
                response = self.session.get(self.base_url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()

                if 'query' not in data:
                    break

                rc_entries = data['query'].get('recentchanges', [])
                if not rc_entries:
                    break

                for entry in rc_entries:
                    if len(articles) >= self.max_articles:
                        break
                    articles.append(entry['title'])

                print(f"Fetched {len(articles)} article titles so far...")

                if 'continue' in data and len(articles) < self.max_articles:
                    continue_param = data['continue']
                else:
                    break

                time.sleep(0.2)

            except requests.RequestException as e:
                print(f"Error fetching article list: {e}")
                break

        print(f"Total articles found: {len(articles)}")
        return articles[:self.max_articles]

    def get_article_content(self, title: str) -> Optional[str]:
        """Fetch the text content of a Wikipedia article."""
        params = {
            "action": "query",
            "prop": "extracts",
            "explaintext": True,
            "titles": title,
            "format": "json"
        }

        try:
            response = self.session.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            pages = data['query']['pages']
            page = next(iter(pages.values()))

            return page.get('extract')
        except requests.RequestException as e:
            print(f"Error fetching article '{title}': {e}")
            return None

    def fetch_and_save(self, since_date: str, output_file: str = "wikipedia_articles.txt"):
        """
        Fetch articles and save to a text file.

        Args:
            since_date: Date in format 'YYYY-MM-DD' or 'YYYY-MM-DDTHH:MM:SSZ'
            output_file: Path to output text file
        """
        # Get article titles
        article_titles = self.get_recent_articles(since_date)

        if not article_titles:
            print("No articles found!")
            return

        # Fetch and save article content
        print(f"\nFetching content for {len(article_titles)} articles...")

        with open(output_file, 'w', encoding='utf-8') as f:
            for i, title in enumerate(article_titles, 1):
                print(f"[{i}/{len(article_titles)}] Fetching: {title}")

                content = self.get_article_content(title)

                if content:
                    # Write article with separator
                    f.write(content.strip() + "\n")
                    f.flush()  # Flush to disk periodically

                # Rate limiting - be respectful to Wikipedia servers
                time.sleep(0.2)

        print(f"\nDone! Articles saved to: {output_file}")


def main():
    """
    Main function - configure parameters via command line or defaults.
    """
    parser = argparse.ArgumentParser(description="Fetch recent Wikipedia articles.")
    parser.add_argument("--language", default="en", help="Wikipedia language code (e.g., en, de, fr, zh)")
    parser.add_argument("--since_date", default="2025-01-01", help="Fetch articles created after this date (YYYY-MM-DD)")
    parser.add_argument("--max_articles", type=int, default=10000, help="Maximum number of articles to fetch")
    parser.add_argument("--output_file", default=None, help="Output file path (optional)")

    args = parser.parse_args()

    # Default output filename if not provided
    output_file = (
        args.output_file
        or f"wikipedia_articles_{args.language}_{args.max_articles}_since_{args.since_date}.txt"
    )

    # Create fetcher and run
    fetcher = WikipediaArticleFetcher(language=args.language, max_articles=args.max_articles)
    fetcher.fetch_and_save(since_date=args.since_date, output_file=output_file)


if __name__ == "__main__":
    main()
