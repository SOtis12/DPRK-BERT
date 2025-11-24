#!/usr/bin/env python3
"""
Cloud-based NK Website and Wayback Machine Scraper
Optimized for Google Cloud TPU VM execution
"""

import os
import json
import time
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import subprocess
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CloudNKScraper:
    """Cloud-optimized scraper for NK websites and Wayback Machine"""
    
    def __init__(self, output_dir: str = "scraped_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Create subdirectories
        self.current_dir = self.output_dir / "current_websites"
        self.wayback_dir = self.output_dir / "wayback_archive"
        self.current_dir.mkdir(exist_ok=True)
        self.wayback_dir.mkdir(exist_ok=True)
        
        # NK websites to scrape
        self.nk_sites = [
            {
                "name": "rodong",
                "url": "https://www.rodong.rep.kp/en/",
                "domain": "rodong.rep.kp",
                "description": "Rodong Sinmun - Workers' Party daily"
            },
            {
                "name": "kcna",
                "url": "http://www.kcna.kp/",
                "domain": "kcna.kp", 
                "description": "KCNA - Korean Central News Agency"
            },
            {
                "name": "naenara",
                "url": "https://naenara.com.kp/main/index/en/",
                "domain": "naenara.com.kp",
                "description": "Naenara - Official DPRK portal"
            },
            {
                "name": "pyongyang_times",
                "url": "https://www.pyongyangtimes.com.kp/",
                "domain": "pyongyangtimes.com.kp",
                "description": "Pyongyang Times - English weekly"
            },
            {
                "name": "korean_books",
                "url": "https://www.korean-books.com.kp/en/",
                "domain": "korean-books.com.kp",
                "description": "Korean Books - Publications store"
            },
            {
                "name": "mfa",
                "url": "https://www.mfa.gov.kp/en/",
                "domain": "mfa.gov.kp",
                "description": "Ministry of Foreign Affairs"
            }
        ]
        
    def run_scrapy_command(self, site_data: Dict[str, Any], scraper_type: str = "current") -> bool:
        """Run scrapy command for a specific site"""
        try:
            site_name = site_data["name"]
            output_file = (self.current_dir if scraper_type == "current" else self.wayback_dir) / f"{site_name}.json"
            
            # Scrapy command with politeness settings
            cmd = [
                "scrapy", "crawl", "nk_multisite",
                "-s", "ROBOTSTXT_OBEY=True",
                "-s", "DOWNLOAD_DELAY=3",
                "-s", "RANDOMIZE_DOWNLOAD_DELAY=True", 
                "-s", "CONCURRENT_REQUESTS=1",
                "-s", "AUTOTHROTTLE_ENABLED=True",
                "-s", "AUTOTHROTTLE_START_DELAY=5",
                "-s", "AUTOTHROTTLE_MAX_DELAY=60",
                "-s", "DEPTH_LIMIT=3",
                "-o", str(output_file),
                "-a", f"allowed_domains={site_data['domain']}",
                "-a", f"start_urls={site_data['url']}",
                "-a", f"max_pages=100"
            ]
            
            if scraper_type == "wayback":
                # Add wayback-specific parameters
                wayback_url = f"https://web.archive.org/web/2020*/{site_data['url']}"
                cmd.extend(["-a", f"wayback_urls={wayback_url}"])
            
            logger.info(f"Starting {scraper_type} scraping for {site_name}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
            
            if result.returncode == 0:
                logger.info(f"✅ Successfully scraped {site_name} ({scraper_type})")
                return True
            else:
                logger.error(f"❌ Failed to scrape {site_name}: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"❌ Timeout scraping {site_name}")
            return False
        except Exception as e:
            logger.error(f"❌ Error scraping {site_name}: {e}")
            return False
    
    def scrape_with_requests(self, site_data: Dict[str, Any]) -> bool:
        """Fallback scraping with requests for simple content extraction"""
        import requests
        from bs4 import BeautifulSoup
        
        try:
            site_name = site_data["name"]
            output_file = self.current_dir / f"{site_name}_simple.json"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            logger.info(f"Simple scraping {site_name} with requests...")
            response = requests.get(site_data["url"], headers=headers, timeout=30)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract text content
                articles = []
                for elem in soup.find_all(['article', 'div'], class_=['article', 'content', 'news', 'post']):
                    text = elem.get_text(strip=True)
                    if len(text) > 100:  # Only meaningful content
                        articles.append({
                            'url': site_data["url"],
                            'title': elem.find(['h1', 'h2', 'h3'])['text'] if elem.find(['h1', 'h2', 'h3']) else '',
                            'content': text[:2000],  # Limit length
                            'source': site_name,
                            'scraped_at': datetime.now().isoformat()
                        })
                
                # Save to JSON
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(articles, f, ensure_ascii=False, indent=2)
                
                logger.info(f"✅ Simple scraping successful for {site_name}: {len(articles)} articles")
                return True
            else:
                logger.error(f"❌ HTTP {response.status_code} for {site_name}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Simple scraping failed for {site_name}: {e}")
            return False
    
    def scrape_current_websites(self) -> Dict[str, bool]:
        """Scrape current NK websites"""
        logger.info("🌐 Starting current website scraping...")
        results = {}
        
        for site in self.nk_sites:
            logger.info(f"📡 Scraping {site['description']}")
            
            # Try simple requests first for immediate results
            success = self.scrape_with_requests(site)
            results[f"{site['name']}_simple"] = success
            
            # Also try scrapy if available (more comprehensive but slower)
            try:
                scrapy_success = self.run_scrapy_command(site, "current")
                results[f"{site['name']}_scrapy"] = scrapy_success
            except Exception as e:
                logger.warning(f"Scrapy failed for {site['name']}: {e}")
                results[f"{site['name']}_scrapy"] = False
            
            # Be polite - delay between sites
            time.sleep(5)
        
        return results
    
    def scrape_wayback_machine(self) -> Dict[str, bool]:
        """Scrape Wayback Machine archives"""
        logger.info("📚 Starting Wayback Machine scraping...")
        results = {}
        
        wayback_base = "https://web.archive.org/web"
        
        for site in self.nk_sites[:3]:  # Limit to top 3 sites for Wayback to avoid timeout
            try:
                site_name = site["name"]
                logger.info(f"🕰️ Wayback scraping {site['description']}")
                
                # Try multiple year ranges
                year_ranges = ["2020*/", "2021*/", "2022*/", "2023*/"]
                
                for year in year_ranges:
                    wayback_url = f"{wayback_base}/{year}{site['url']}"
                    
                    try:
                        wayback_site = {
                            **site,
                            "url": wayback_url,
                            "name": f"{site_name}_{year.replace('*/', '')}"
                        }
                        
                        success = self.scrape_with_requests(wayback_site)
                        results[f"{site_name}_{year}"] = success
                        
                        if success:
                            break  # Got content for this site, move to next
                            
                    except Exception as e:
                        logger.warning(f"Failed Wayback {year} for {site_name}: {e}")
                        continue
                        
                time.sleep(10)  # Longer delay for Wayback Machine
                
            except Exception as e:
                logger.error(f"Wayback scraping error for {site['name']}: {e}")
                results[site['name']] = False
        
        return results
    
    def process_scraped_data(self) -> Dict[str, Any]:
        """Process and consolidate scraped data for training"""
        logger.info("🔄 Processing scraped data...")
        
        all_articles = []
        stats = {"total_files": 0, "total_articles": 0, "sources": []}
        
        # Process current website data
        for json_file in self.current_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                if isinstance(data, list):
                    all_articles.extend(data)
                    stats["total_articles"] += len(data)
                    stats["sources"].append(json_file.stem)
                    
                stats["total_files"] += 1
                logger.info(f"Processed {json_file.name}: {len(data) if isinstance(data, list) else 1} articles")
                
            except Exception as e:
                logger.error(f"Error processing {json_file}: {e}")
        
        # Process Wayback data
        for json_file in self.wayback_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                if isinstance(data, list):
                    # Mark as historical content
                    for article in data:
                        article['content_type'] = 'historical'
                    all_articles.extend(data)
                    stats["total_articles"] += len(data)
                    stats["sources"].append(f"wayback_{json_file.stem}")
                    
                stats["total_files"] += 1
                
            except Exception as e:
                logger.error(f"Error processing wayback {json_file}: {e}")
        
        # Save consolidated data
        output_file = self.output_dir / "consolidated_nk_content.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_articles, f, ensure_ascii=False, indent=2)
        
        # Save metadata
        metadata = {
            **stats,
            "processing_date": datetime.now().isoformat(),
            "output_file": str(output_file)
        }
        
        metadata_file = self.output_dir / "scraping_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✅ Consolidated {stats['total_articles']} articles from {stats['total_files']} files")
        return metadata
    
    def run_full_scraping(self) -> Dict[str, Any]:
        """Execute complete scraping pipeline"""
        logger.info("🚀 Starting comprehensive NK web scraping...")
        start_time = time.time()
        
        # Step 1: Current websites
        current_results = self.scrape_current_websites()
        
        # Step 2: Wayback Machine (if current scraping was successful)
        if any(current_results.values()):
            wayback_results = self.scrape_wayback_machine()
        else:
            logger.warning("Skipping Wayback scraping due to current website failures")
            wayback_results = {}
        
        # Step 3: Process and consolidate
        metadata = self.process_scraped_data()
        
        # Final report
        duration = time.time() - start_time
        
        final_report = {
            "status": "completed",
            "duration_minutes": round(duration / 60, 2),
            "current_websites": current_results,
            "wayback_results": wayback_results,
            "consolidation": metadata,
            "output_directory": str(self.output_dir)
        }
        
        report_file = self.output_dir / "scraping_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(final_report, f, indent=2)
        
        logger.info(f"🎉 Scraping completed in {duration/60:.1f} minutes")
        logger.info(f"📊 Report saved to: {report_file}")
        
        return final_report

def main():
    """Main execution function"""
    print("🚀 Cloud NK Website Scraper Starting...")
    
    scraper = CloudNKScraper()
    report = scraper.run_full_scraping()
    
    print("\n" + "="*60)
    print("📋 SCRAPING SUMMARY")
    print("="*60)
    print(f"Status: {report['status']}")
    print(f"Duration: {report['duration_minutes']} minutes") 
    print(f"Output: {report['output_directory']}")
    
    if 'consolidation' in report:
        print(f"Total articles: {report['consolidation']['total_articles']}")
        print(f"Sources: {', '.join(report['consolidation']['sources'])}")
    
    print("="*60)

if __name__ == "__main__":
    main()