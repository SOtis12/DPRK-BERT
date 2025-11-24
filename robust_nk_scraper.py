#!/usr/bin/env python3
"""
Robust NK Website Scraper
Enhanced with user-agent rotation, retry logic, proxy support, and error handling
Optimized for challenging North Korean websites
"""

import os
import json
import time
import random
import logging
import requests
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from urllib.parse import urljoin
import re
import sys

from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('robust_scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Silence SSL warnings since we use verify=False for some sites
try:
    from urllib3.exceptions import InsecureRequestWarning
    import urllib3
    urllib3.disable_warnings(InsecureRequestWarning)
except ImportError:
    try:
        from requests.packages.urllib3.exceptions import InsecureRequestWarning
        requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)
    except Exception:
        pass


class RobustNKScraper:
    """Scraper for (often flaky) NK websites, with Wayback fallback."""

    def __init__(
        self,
        output_dir: str = "scraper_output",
        use_wayback: bool = True,
        max_wayback_snapshots: int = 5,
        proxies: Optional[Dict[str, str]] = None,
    ) -> None:
        self.session = requests.Session()
        self.session.proxies = proxies or {}
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.use_wayback = use_wayback
        self.max_wayback_snapshots = max_wayback_snapshots

        # Simple user-agent rotation
        self.user_agents: List[str] = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
            "(KHTML, like Gecko) Version/17.0 Safari/605.1.15",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/123.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
        ]

        # Basic per-run stats
        self.stats: Dict[str, Any] = {
            "start_time": time.time(),
            "attempts": 0,
            "successes": 0,
            "failures": 0,
            "articles_found": 0,
        }

        # Site configurations (tweak/extend as needed)
        self.nk_sites: List[Dict[str, Any]] = [
            {
                "name": "naenara",
                "description": "Naenara – DPRK portal",
                "domain": "naenara.com.kp",
                "urls": [
                    "https://naenara.com.kp/en/",
                    "http://naenara.com.kp/en/",
                ],
                "content_selectors": [
                    "div#content",
                    "div.content",
                    "div#main",
                    "article",
                    "div.article",
                ],
            },
            {
                "name": "pyongyangtimes",
                "description": "The Pyongyang Times",
                "domain": "pyongyangtimes.com.kp",
                "urls": [
                    "http://www.pyongyangtimes.com.kp/",
                    "https://www.pyongyangtimes.com.kp/",
                ],
                "content_selectors": [
                    "div#content",
                    "div.content",
                    "div#main",
                    "article",
                    "div.article",
                ],
            },
            {
                "name": "koreanbooks",
                "description": "Korean Books",
                "domain": "korean-books.com.kp",
                "urls": [
                    "http://www.korean-books.com.kp/en/",
                    "https://www.korean-books.com.kp/en/",
                ],
                "content_selectors": [
                    "div#content",
                    "div.content",
                    "div#main",
                    "article",
                    "div.article",
                ],
            },
        ]

    # ------------------------------------------------------------------ #
    # HTTP helpers
    # ------------------------------------------------------------------ #
    def get_headers(self) -> Dict[str, str]:
        """Return randomized HTTP headers."""
        ua = random.choice(self.user_agents)
        return {
            "User-Agent": ua,
            "Accept": (
                "text/html,application/xhtml+xml,application/xml;"
                "q=0.9,image/avif,image/webp,*/*;q=0.8"
            ),
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Cache-Control": "max-age=0",
        }

    def make_request(self, url: str, timeout: int = 30) -> Optional[requests.Response]:
        """Make a robust HTTP request with retry logic"""
        self.stats["attempts"] += 1

        for attempt in range(3):  # 3 retry attempts
            try:
                headers = self.get_headers()

                logger.info(f"Attempting to fetch: {url} (attempt {attempt + 1})")

                response = self.session.get(
                    url,
                    headers=headers,
                    timeout=timeout,
                    allow_redirects=True,
                    verify=False,  # Some NK sites have certificate issues
                )

                if response.status_code == 200:
                    self.stats["successes"] += 1
                    logger.info(f"✅ Successfully fetched: {url}")
                    return response

                elif response.status_code in [403, 429]:
                    # Rate limiting or blocking - wait longer
                    wait_time = random.uniform(10, 30)
                    logger.warning(
                        f"⚠️ Rate limited ({response.status_code}), waiting {wait_time:.1f}s"
                    )
                    time.sleep(wait_time)

                else:
                    logger.warning(f"⚠️ HTTP {response.status_code} for {url}")

            except requests.exceptions.Timeout:
                logger.warning(f"⏰ Timeout for {url} (attempt {attempt + 1})")

            except requests.exceptions.ConnectionError:
                logger.warning(f"🔌 Connection error for {url} (attempt {attempt + 1})")

            except Exception as e:
                logger.error(f"❌ Unexpected error for {url}: {e}")

            # Wait between retries with exponential backoff
            if attempt < 2:  # Don't wait after the last attempt
                wait_time = random.uniform(5, 15) * (2 ** attempt)
                logger.info(f"⏳ Waiting {wait_time:.1f}s before retry...")
                time.sleep(wait_time)

        self.stats["failures"] += 1
        logger.error(f"❌ Failed to fetch {url} after 3 attempts")
        return None

    # ------------------------------------------------------------------ #
    # Content extraction helpers
    # ------------------------------------------------------------------ #
    def extract_korean_content(
        self, soup: BeautifulSoup, selectors: List[str]
    ) -> List[str]:
        """Extract Korean text content from HTML, with Wayback fallback."""
        korean_texts: List[str] = []

        # Try each selector
        for selector in selectors:
            elements = soup.select(selector)
            for elem in elements:
                text = elem.get_text(strip=True)
                if self.contains_korean(text) and len(text) > 50:
                    cleaned_text = self.clean_text(text)
                    if cleaned_text:
                        korean_texts.append(cleaned_text)

        # Fallback: try common patterns
        if not korean_texts:
            korean_texts = self.find_korean_text_patterns(soup)

        # Extra fallback: try <body> and <main> tags
        if not korean_texts:
            for tag in ["body", "main"]:
                elem = soup.find(tag)
                if elem:
                    text = elem.get_text(strip=True)
                    if self.contains_korean(text) and len(text) > 50:
                        cleaned_text = self.clean_text(text)
                        if cleaned_text:
                            korean_texts.append(cleaned_text)

        # Debug: log raw HTML if still nothing found
        if not korean_texts:
            logger.warning("Wayback / extraction failed; attempting English fallback and logging raw HTML for debugging.")
            # English fallback: if page contains no Korean, accept large English paragraphs
            english_texts: List[str] = []
            for p in soup.find_all("p"):
                t = p.get_text(separator=" ", strip=True)
                # Accept paragraphs that are long and contain alphabetic characters
                if len(t) > 200 and re.search(r"[A-Za-z]", t):
                    cleaned = self.clean_text(t)
                    if cleaned:
                        english_texts.append(cleaned)

            if english_texts:
                logger.info(f"English fallback found {len(english_texts)} paragraphs; using them as content.")
                return english_texts[:10]

            # Largest-block fallback: choose the biggest text-containing element on the page
            candidates: List[tuple[int, str]] = []
            for tag in ["div", "td", "section", "table", "article", "main", "span"]:
                for el in soup.find_all(tag):
                    t = el.get_text(separator=" ", strip=True)
                    if len(t) > 200:
                        candidates.append((len(t), t))

            if candidates:
                candidates.sort(reverse=True)
                best_text = candidates[0][1]
                cleaned_best = self.clean_text(best_text)
                if cleaned_best:
                    logger.info(f"Large-block fallback selected text of length {len(cleaned_best)}")
                    return [cleaned_best]

            debug_file = self.output_dir / "wayback_failed_debug.html"
            with open(debug_file, "w", encoding="utf-8") as f:
                f.write(str(soup))

        return korean_texts

    def contains_korean(self, text: str) -> bool:
        """Check if text contains Korean characters"""
        korean_pattern = re.compile(r"[가-힣]")
        return bool(korean_pattern.search(text))

    def clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Remove very short texts
        if len(text) < 30:
            return ""

        # Remove texts that are mostly numbers or symbols
        korean_chars = len(re.findall(r"[가-힣]", text))
        if korean_chars < len(text) * 0.3:  # At least 30% Korean characters
            return ""

        return text

    def find_korean_text_patterns(self, soup: BeautifulSoup) -> List[str]:
        """Find Korean text using common patterns"""
        texts: List[str] = []

        # Look for paragraphs with Korean text
        for p in soup.find_all("p"):
            text = p.get_text(strip=True)
            if self.contains_korean(text) and len(text) > 50:
                cleaned = self.clean_text(text)
                if cleaned:
                    texts.append(cleaned)

        # Look for divs with Korean content
        for div in soup.find_all("div"):
            # Skip divs that are likely navigation/menu
            classes = " ".join(div.get("class", [])).lower()
            if any(
                cls in classes
                for cls in ["nav", "menu", "footer", "header", "sidebar"]
            ):
                continue

            text = div.get_text(strip=True)
            if self.contains_korean(text) and len(text) > 50:
                cleaned = self.clean_text(text)
                if cleaned:
                    texts.append(cleaned)

        # Limit to avoid huge duplicates
        return texts[:10]

    # ------------------------------------------------------------------ #
    # Live site scraping
    # ------------------------------------------------------------------ #
    def scrape_site(self, site_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Scrape a single NK website"""
        articles: List[Dict[str, Any]] = []
        site_name = site_config["name"]

        logger.info(f"🌐 Starting to scrape {site_config['description']}")

        # Try each URL for the site
        for url in site_config["urls"]:
            try:
                # Add random delay to mimic human behavior
                time.sleep(random.uniform(3, 8))

                response = self.make_request(url)
                if not response:
                    continue

                soup = BeautifulSoup(response.content, "html.parser")

                # Extract Korean content
                korean_texts = self.extract_korean_content(
                    soup, site_config["content_selectors"]
                )

                if korean_texts:
                    logger.info(
                        f"📄 Found {len(korean_texts)} Korean text sections in {url}"
                    )

                    for i, text in enumerate(korean_texts):
                        article = {
                            "url": url,
                            "source": site_name,
                            "title": f"{site_name}_article_{i+1}",
                            "content": text,
                            "scraped_at": datetime.now().isoformat(),
                            "content_length": len(text),
                            "domain": site_config["domain"],
                        }
                        articles.append(article)
                        self.stats["articles_found"] += 1

                # Try to find additional pages/links
                additional_articles = self.scrape_linked_pages(
                    soup, url, site_config, max_pages=5
                )
                articles.extend(additional_articles)

                if articles:
                    break  # Successfully scraped this site, move to next

            except Exception as e:
                logger.error(f"❌ Error scraping {url}: {e}")
                continue

        logger.info(f"✅ Completed {site_name}: {len(articles)} articles")
        return articles

    def scrape_linked_pages(
        self,
        soup: BeautifulSoup,
        base_url: str,
        site_config: Dict[str, Any],
        max_pages: int = 5,
    ) -> List[Dict[str, Any]]:
        """Scrape additional linked pages from the main page"""
        articles: List[Dict[str, Any]] = []

        try:
            # Find links that might contain articles
            links = soup.find_all("a", href=True)
            article_links: List[str] = []

            domain = site_config["domain"]

            for link in links:
                href = link.get("href")
                if not href:
                    continue

                # Convert relative URLs to absolute
                full_url = urljoin(base_url, href)

                # Filter for relevant links (articles, news, etc.)
                if (
                    domain in full_url
                    and any(
                        keyword in href.lower()
                        for keyword in ["article", "news", "post", "story"]
                    )
                    and full_url not in [base_url]
                    and not any(
                        skip in href.lower()
                        for skip in ["login", "register", "search", "tag", "#"]
                    )
                ):
                    article_links.append(full_url)

            # Scrape a few additional pages
            for i, link in enumerate(article_links[:max_pages]):
                logger.info(f"🔗 Scraping linked page: {link}")

                # Polite delay
                time.sleep(random.uniform(5, 12))

                response = self.make_request(link)
                if response:
                    page_soup = BeautifulSoup(response.content, "html.parser")
                    korean_texts = self.extract_korean_content(
                        page_soup, site_config["content_selectors"]
                    )

                    for j, text in enumerate(korean_texts):
                        article = {
                            "url": link,
                            "source": f"{site_config['name']}_linked",
                            "title": f"{site_config['name']}_linked_{i}_{j}",
                            "content": text,
                            "scraped_at": datetime.now().isoformat(),
                            "content_length": len(text),
                            "domain": site_config["domain"],
                        }
                        articles.append(article)
                        self.stats["articles_found"] += 1

        except Exception as e:
            logger.warning(f"⚠️ Error scraping linked pages: {e}")

        return articles

    # ------------------------------------------------------------------ #
    # Wayback scraping
    # ------------------------------------------------------------------ #
    def scrape_wayback_machine(
        self, site_config: Dict[str, Any], max_snapshots: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Scrape historical content from Wayback Machine"""
        if not self.use_wayback:
            return []

        articles: List[Dict[str, Any]] = []
        max_snapshots = max_snapshots or self.max_wayback_snapshots

        try:
            # Prefer using the Wayback CDX API to find available snapshots, then fetch a handful.
            cdx_base = "http://web.archive.org/cdx/search/cdx"
            # Use the first canonical URL from the site config as the target
            target_url = site_config.get("urls", [None])[0]

            if target_url:
                params = {
                    "url": target_url,
                    "output": "json",
                    "filter": "statuscode:200",
                    "limit": 50,
                    "from": 1990,
                    "to": datetime.now().year,
                    "collapse": "digest",
                }

                logger.info(f"🕰️ Querying Wayback CDX for {target_url}")
                try:
                    resp = self.session.get(
                        cdx_base, params=params, timeout=30, verify=False
                    )
                    if resp.status_code == 200:
                        try:
                            cdx_json = resp.json()
                        except Exception:
                            # CDX sometimes returns NDJSON or non-json; fall back
                            cdx_json = None

                        if cdx_json and len(cdx_json) > 1:
                            # First row is the header
                            entries = cdx_json[1:]
                            # Each entry in CDX JSON:
                            # [urlkey, timestamp, original, mime, statuscode, digest, length]
                            fetched = 0
                            for entry in entries[:max_snapshots]:
                                try:
                                    timestamp = entry[1]
                                    original = entry[2]
                                    archived_url = (
                                        f"https://web.archive.org/web/"
                                        f"{timestamp}/{original}"
                                    )

                                    logger.info(
                                        f"🕸️ Fetching archived snapshot: {archived_url}"
                                    )
                                    time.sleep(random.uniform(2, 6))
                                    r = self.make_request(archived_url, timeout=45)
                                    if not r:
                                        continue
                                    soup = BeautifulSoup(r.content, "html.parser")
                                    korean_texts = self.extract_korean_content(
                                        soup, site_config.get("content_selectors", [])
                                    )
                                    if korean_texts:
                                        logger.info(
                                            f"📚 Found {len(korean_texts)} historical texts from {timestamp}"
                                        )
                                        for i, text in enumerate(korean_texts):
                                            article = {
                                                "url": archived_url,
                                                "source": f"{site_config['name']}_wayback_{timestamp}",
                                                "title": f"{site_config['name']}_historical_{timestamp}_{i}",
                                                "content": text,
                                                "scraped_at": datetime.now().isoformat(),
                                                "content_length": len(text),
                                                "domain": site_config.get("domain"),
                                                "historical_timestamp": timestamp,
                                            }
                                            articles.append(article)
                                            self.stats["articles_found"] += 1
                                        fetched += 1
                                    if fetched >= max_snapshots:
                                        break
                                except Exception as e:
                                    logger.warning(
                                        f"⚠️ Error fetching snapshot entry: {e}"
                                    )

                        else:
                            logger.info(
                                "🔎 No JSON snapshots returned by CDX; falling back to heuristic timestamps"
                            )
                    else:
                        logger.warning(
                            f"⚠️ CDX query returned status {resp.status_code}; falling back to timestamp guesses"
                        )
                except Exception as e:
                    logger.warning(
                        f"⚠️ Error querying CDX API: {e}; falling back to simple approach"
                    )

            # Fallback: try a few heuristic yearly timestamps if CDX didn't produce results
            if not articles:
                years = [
                    str(y) for y in range(datetime.now().year, datetime.now().year - 6, -1)
                ]
                for year in years:
                    for original_url in site_config.get("urls", []):
                        wayback_url = (
                            f"https://web.archive.org/web/{year}0701000000/{original_url}"
                        )
                        logger.info(
                            f"🕰️ Trying Wayback Machine: {year} for {site_config['name']}"
                        )
                        time.sleep(random.uniform(2, 6))
                        response = self.make_request(wayback_url, timeout=45)
                        if response:
                            soup = BeautifulSoup(response.content, "html.parser")
                            korean_texts = self.extract_korean_content(
                                soup, site_config.get("content_selectors", [])
                            )
                            if korean_texts:
                                logger.info(
                                    f"📚 Found {len(korean_texts)} historical texts from {year}"
                                )
                                for i, text in enumerate(korean_texts):
                                    article = {
                                        "url": wayback_url,
                                        "source": f"{site_config['name']}_wayback_{year}",
                                        "title": f"{site_config['name']}_historical_{year}_{i}",
                                        "content": text,
                                        "scraped_at": datetime.now().isoformat(),
                                        "content_length": len(text),
                                        "domain": site_config.get("domain"),
                                        "historical_year": year,
                                    }
                                    articles.append(article)
                                    self.stats["articles_found"] += 1
                                break
                    if articles:
                        break

        except Exception as e:
            logger.warning(f"⚠️ Error with Wayback Machine scraping: {e}")

        return articles

    # ------------------------------------------------------------------ #
    # Orchestration
    # ------------------------------------------------------------------ #
    def run_comprehensive_scraping(self) -> Dict[str, Any]:
        """Execute comprehensive North Korean website scraping"""
        logger.info("🚀 Starting comprehensive robust NK website scraping...")

        all_articles: List[Dict[str, Any]] = []
        site_results: Dict[str, Dict[str, int]] = {}

        # Scrape current websites
        logger.info("📡 Phase 1: Current NK websites")
        for site_config in self.nk_sites:
            try:
                site_name = site_config["name"]
                site_results[site_name] = {
                    "live_articles": 0,
                    "wayback_articles": 0,
                }

                articles = self.scrape_site(site_config)
                all_articles.extend(articles)
                site_results[site_name]["live_articles"] = len(articles)

                # Save intermediate results
                if articles:
                    site_file = self.output_dir / f"{site_config['name']}_current.json"
                    with open(site_file, "w", encoding="utf-8") as f:
                        json.dump(articles, f, ensure_ascii=False, indent=2)
                else:
                    logger.warning(
                        f"⚠️ No live articles from {site_name}; enabling Wayback fallback"
                    )
                    # Immediate Wayback fallback when live scraping fails
                    historical_articles = self.scrape_wayback_machine(site_config)
                    if historical_articles:
                        site_results[site_name]["wayback_articles"] = len(
                            historical_articles
                        )
                        all_articles.extend(historical_articles)
                        wayback_file = self.output_dir / f"{site_name}_wayback.json"
                        with open(wayback_file, "w", encoding="utf-8") as f:
                            json.dump(
                                historical_articles, f, ensure_ascii=False, indent=2
                            )
                    else:
                        logger.warning(
                            f"⚠️ Wayback fallback also returned no articles for {site_name}"
                        )

                # Longer delay between sites
                time.sleep(random.uniform(15, 30))

            except Exception as e:
                logger.error(f"❌ Failed to scrape {site_config['name']}: {e}")
                site_results.setdefault(
                    site_config["name"],
                    {
                        "live_articles": 0,
                        "wayback_articles": 0,
                    },
                )

        # Save consolidated results
        consolidated_file = self.output_dir / "consolidated_nk_articles.json"
        with open(consolidated_file, "w", encoding="utf-8") as f:
            json.dump(all_articles, f, ensure_ascii=False, indent=2)

        # Generate final report
        duration = time.time() - self.stats["start_time"]
        duration_minutes = round(duration / 60.0, 2)
        attempts = self.stats["attempts"] or 1  # avoid div/0
        success_rate = round(100.0 * self.stats["successes"] / attempts, 2)

        articles_per_site = {
            source: len(
                [a for a in all_articles if a.get("source", "").startswith(source)]
            )
            for source in {
                a.get("source", "").split("_")[0] for a in all_articles if "source" in a
            }
        }

        report: Dict[str, Any] = {
            "status": "completed",
            "duration_seconds": duration,
            "duration_minutes": duration_minutes,
            "results": {
                "total_articles": len(all_articles),
                "total_content_length": sum(
                    article.get("content_length", 0) for article in all_articles
                ),
                "by_source": site_results,
                "articles_per_site": articles_per_site,
            },
            "statistics": {
                "attempts": self.stats["attempts"],
                "successes": self.stats["successes"],
                "failures": self.stats["failures"],
                "articles_found": self.stats["articles_found"],
                "success_rate": success_rate,
            },
            "output_files": {
                "consolidated": str(consolidated_file),
                "individual_sites": [
                    str(f) for f in self.output_dir.glob("*.json")
                ],
            },
        }

        report_file = self.output_dir / "comprehensive_scraping_report.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(
            f"🎉 Scraping completed! Found {len(all_articles)} articles in {duration_minutes:.1f} minutes"
        )
        return report


def main(args: Optional["argparse.Namespace"] = None) -> None:
    """Main execution function"""
    print("🚀 Robust NK Website Scraper Starting...")
    print("📋 Features: User-agent rotation, retry logic, stealth mode, Wayback Machine")

    output_dir = getattr(args, "output", "scraper_output")
    use_wayback = getattr(args, "wayback", True)

    scraper = RobustNKScraper(output_dir=output_dir, use_wayback=use_wayback)
    report = scraper.run_comprehensive_scraping()

    print("\n" + "=" * 70)
    print("📊 FINAL SCRAPING REPORT")
    print("=" * 70)
    print(f"✅ Status: {report['status']}")
    print(f"⏱️ Duration: {report['duration_minutes']} minutes")
    print(f"📰 Total articles found: {report['results']['total_articles']}")
    print(f"📊 Success rate: {report['statistics']['success_rate']}%")
    print(
        f"💾 Content length: {report['results']['total_content_length']:,} characters"
    )
    print("\n📋 Articles by source:")
    for source, count in report["results"]["articles_per_site"].items():
        print(f"  • {source}: {count} articles")
    print(f"\n💾 Output directory: {scraper.output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="North Korean website scraper with Wayback support"
    )
    parser.add_argument(
        "--wayback",
        action="store_true",
        help="Enable Wayback scraping (default off in CLI, on in library)",
    )
    parser.add_argument(
        "--only-wayback",
        action="store_true",
        help="Skip live scraping and only attempt Wayback snapshots",
    )
    parser.add_argument(
        "--skip-live",
        action="store_true",
        help="Do not attempt live scraping; still run Wayback (alias for --only-wayback)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="scraper_output",
        help="Output directory (default: scraper_output)",
    )
    parser.add_argument(
        "--jsonl-output",
        type=str,
        default="enhanced_training_data/wayback_scraped.jsonl",
        help="(Optional) Append Wayback results as JSONL to this file",
    )
    parser.add_argument(
        "--cdx-dry-run",
        action="store_true",
        help="List available Wayback snapshots and exit",
    )
    args = parser.parse_args()

    # List of target URLs for the CDX dry-run helper (update as needed)
    target_urls = [
        "https://naenara.com.kp/",
        "http://www.pyongyangtimes.com.kp/",
        "https://www.pyongyangtimes.com.kp/",
        "http://www.korean-books.com.kp/",
        "https://www.korean-books.com.kp/",
    ]

    def cdx_dry_run(urls: List[str]) -> None:
        """List available Wayback snapshots for each URL."""
        cdx_api = "http://web.archive.org/cdx/search/cdx"
        for url in urls:
            print(f"=== {url} ===")
            params = {
                "url": url,
                "output": "json",
                "fl": "timestamp,original",
                "filter": "statuscode:200",
                "limit": "20",
            }
            try:
                resp = requests.get(cdx_api, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                for entry in data[1:]:
                    print(f"{entry[0]} {entry[1]}")
            except Exception as e:
                print(f"Error for {url}: {e}")

    if args.cdx_dry_run:
        cdx_dry_run(target_urls)
        sys.exit(0)

    # Handle wayback-only / skip-live modes without changing the existing
    # comprehensive orchestration function.
    if args.only_wayback or args.skip_live:
        print("🕰️ Running Wayback-only scraping...")
        scraper = RobustNKScraper(output_dir=args.output, use_wayback=True)
        jsonl_path = Path(args.jsonl_output)
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        total = 0
        for site in scraper.nk_sites:
            try:
                items = scraper.scrape_wayback_machine(site)
                if items:
                    # append to JSONL
                    with open(jsonl_path, "a", encoding="utf-8") as jf:
                        for it in items:
                            jf.write(json.dumps(it, ensure_ascii=False) + "\n")
                    total += len(items)
                    print(f"Wrote {len(items)} items from {site['name']} to {jsonl_path}")
                else:
                    print(f"No Wayback items for {site['name']}")
            except Exception as e:
                print(f"Error scraping Wayback for {site['name']}: {e}")

        print(f"Done. Wrote {total} items to {jsonl_path}")
        sys.exit(0)

    main(args)
