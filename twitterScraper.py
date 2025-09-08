import csv
import os
import time
import re
from datetime import datetime, timezone, timedelta
from playwright.sync_api import sync_playwright
import threading
import signal
import sys

# Configuration
BASE_CSV_NAME = "social_media_posts"
SESSION_FILE = "twitter_auth.json"
KEYWORDS = ["Dakota Access Pipeline", "ConcoPhillips Willow Project", "Texas Border Wall"]
CHECK_INTERVAL_SECONDS = 15
INITIAL_SCRAPE_DURATION = 300  # 5 minutes for first cycle
INITIAL_SCROLL_COUNT = 50  # More scrolling for first cycle
REGULAR_SCROLL_COUNT = 10  # Less scrolling for subsequent cycles
MAX_TWEETS_PER_KEYWORD = 200  # Limit per keyword per cycle

# Global flag for graceful shutdown
running = True

def signal_handler(sig, frame):
    global running
    print('\nShutting down gracefully...')
    running = False

signal.signal(signal.SIGINT, signal_handler)

def get_csv_filename(keyword):
    """Generate CSV filename for each keyword"""
    safe_keyword = re.sub(r'[^\w\s-]', '', keyword).replace(' ', '_').lower()
    return f"{BASE_CSV_NAME}_{safe_keyword}.csv"

def initialize_csv(csv_file):
    """Initialize CSV with headers if it doesn't exist"""
    if not os.path.exists(csv_file):
        with open(csv_file, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=[
                "platform", "captured_at", "posted_at", "username", "post_id",
                "content", "source_url", "keyword", "detection_cycle"
            ])
            writer.writeheader()
        print(f"Created new CSV: {csv_file}")
        return True
    else:
        print(f"Using existing CSV: {csv_file}")
        return False

def load_existing_posts_data(csv_file):
    """Load existing post data with timestamps for comparison"""
    if not os.path.exists(csv_file):
        return set(), None
    
    post_ids = set()
    latest_timestamp = None
    
    try:
        with open(csv_file, mode="r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            posts = list(reader)
            
            for row in posts:
                post_ids.add(row["post_id"])
                
                # Track the most recent post timestamp
                if row["posted_at"] != "UNKNOWN":
                    try:
                        post_time = datetime.fromisoformat(row["posted_at"].replace('Z', '+00:00'))
                        if latest_timestamp is None or post_time > latest_timestamp:
                            latest_timestamp = post_time
                    except:
                        pass
                        
    except Exception as e:
        print(f"Error loading existing data: {e}")
        return set(), None
    
    print(f"Loaded {len(post_ids)} existing post IDs")
    if latest_timestamp:
        print(f"Latest post timestamp: {latest_timestamp}")
    
    return post_ids, latest_timestamp

def parse_engagement_number(text):
    """Parse engagement numbers like 1.2K, 5M, etc."""
    if not text:
        return ""
    
    clean_text = re.sub(r'[^\d\.KMB]', '', text.upper())
    if not clean_text:
        return ""
    
    if clean_text.endswith('K'):
        try:
            return str(int(float(clean_text[:-1]) * 1000))
        except:
            return clean_text
    elif clean_text.endswith('M'):
        try:
            return str(int(float(clean_text[:-1]) * 1000000))
        except:
            return clean_text
    elif clean_text.endswith('B'):
        try:
            return str(int(float(clean_text[:-1]) * 1000000000))
        except:
            return clean_text
    
    return clean_text

def extract_engagement(card):
    """Extract likes and views - but we'll skip saving them"""
    return "", ""

def is_post_too_old(posted_at, latest_timestamp, is_initial_scrape):
    """Check if post is too old to be considered new"""
    if is_initial_scrape or posted_at == "UNKNOWN" or not latest_timestamp:
        return False
    
    try:
        post_time = datetime.fromisoformat(posted_at.replace('Z', '+00:00'))
        # Only consider posts newer than our latest recorded post
        return post_time <= latest_timestamp
    except:
        return False

def scrape_section(page, keyword, section_type, existing_post_ids, latest_timestamp, 
                  is_initial_scrape, cycle_number, scroll_count):
    """Scrape a specific section (top or latest) for posts"""
    query = keyword.replace(" ", "%20")
    
    if section_type == "top":
        url = f"https://twitter.com/search?q={query}&src=typed_query"
        print(f"Scraping TOP section for '{keyword}'...")
    else:  # latest
        url = f"https://twitter.com/search?q={query}&src=typed_query&f=live"
        print(f"Scraping LATEST section for '{keyword}'...")
    
    try:
        page.goto(url, wait_until="networkidle", timeout=20000)
        page.wait_for_timeout(3000)
    except:
        print(f"Page load timeout for {keyword} ({section_type}), retrying...")
        page.goto(url)
        page.wait_for_timeout(5000)

    new_posts = []
    posts_found = 0
    consecutive_old_posts = 0
    
    # For latest section, scroll more aggressively to find new posts
    if section_type == "latest":
        scroll_count = scroll_count * 2  # Double the scrolling for latest section
    
    # Scrolling with post collection
    for scroll in range(scroll_count):
        if not running or posts_found >= MAX_TWEETS_PER_KEYWORD:
            break
        
        # Handle content warnings
        try:
            show_buttons = page.query_selector_all("div[role='button']:has-text('Show')")
            for button in show_buttons:
                if button.is_visible():
                    button.click()
                    page.wait_for_timeout(500)
        except:
            pass

        # Get tweet cards
        tweet_cards = page.query_selector_all("article")
        
        for card in tweet_cards:
            if posts_found >= MAX_TWEETS_PER_KEYWORD:
                break
                
            try:
                # Skip ads, promoted content, retweets
                card_text = card.inner_text().lower()
                if any(skip_word in card_text for skip_word in ['promoted', 'ad', 'sponsored']):
                    continue

                # Extract post URL and ID
                link_elem = card.query_selector("a[href*='/status/']")
                if not link_elem:
                    continue
                    
                post_url = f"https://twitter.com{link_elem.get_attribute('href')}"
                post_id = post_url.split("/")[-1] if post_url else "N/A"

                # Skip if already captured
                if post_id in existing_post_ids:
                    continue

                # Extract post timestamp
                time_element = card.query_selector("time")
                posted_at = time_element.get_attribute("datetime") if time_element else "UNKNOWN"

                # Check if post is too old (for subsequent cycles)
                if is_post_too_old(posted_at, latest_timestamp, is_initial_scrape):
                    consecutive_old_posts += 1
                    # If we hit many old posts in a row, continue scrolling for latest section to find newer posts
                    if consecutive_old_posts >= 15 and not is_initial_scrape and section_type == "top":
                        print(f"   Reached older posts, stopping {section_type} section")
                        return new_posts
                    continue
                else:
                    consecutive_old_posts = 0

                # Extract username from URL
                username = "unknown"
                if post_url:
                    url_parts = post_url.split('/')
                    for part in url_parts:
                        if part and not part.isdigit() and part not in ['https:', '', 'twitter.com', 'x.com', 'status']:
                            username = part
                            break

                # Extract post content
                content_elem = card.query_selector("div[lang], div[data-testid='tweetText']")
                content = content_elem.inner_text().strip() if content_elem else ""

                # Skip empty content or retweets
                if not content or content.startswith("RT @") or len(content) < 3:
                    continue

                # Skip engagement extraction since we don't need it
                likes, views = extract_engagement(card)

                # Create post record
                post_data = {
                    "platform": "Twitter",
                    "captured_at": datetime.utcnow().isoformat(),
                    "posted_at": posted_at,
                    "username": username,
                    "post_id": post_id,
                    "content": content,
                    "source_url": post_url,
                    "keyword": keyword,
                    "detection_cycle": cycle_number
                }

                new_posts.append(post_data)
                existing_post_ids.add(post_id)
                posts_found += 1

                # Real-time notification
                timestamp = datetime.now().strftime('%H:%M:%S')
                print(f"   [{timestamp}] @{username} | {content[:50]}...")

            except Exception as e:
                print(f"Error processing post: {e}")
                continue

        # Scroll down more aggressively for latest section
        if section_type == "latest":
            page.mouse.wheel(0, 2000)  # Scroll more for latest
            page.wait_for_timeout(1500)
        else:
            page.mouse.wheel(0, 1500)
            page.wait_for_timeout(1000)
        
        # For initial scrape, add extra wait time
        if is_initial_scrape:
            page.wait_for_timeout(500)

    print(f"   Found {len(new_posts)} new posts in {section_type.upper()} section")
    return new_posts

def save_posts_to_csv(csv_file, new_posts):
    """Save posts to CSV maintaining chronological order (newest first)"""
    if not new_posts:
        return 0

    # Read existing data
    existing_data = []
    if os.path.exists(csv_file):
        try:
            with open(csv_file, mode="r", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                existing_data = list(reader)
        except Exception as e:
            print(f"Error reading existing data: {e}")

    # Combine and sort by posted_at (newest first)
    all_posts = existing_data + new_posts
    
    def sort_key(post):
        if post['posted_at'] == 'UNKNOWN':
            return datetime.min.replace(tzinfo=timezone.utc)
        try:
            return datetime.fromisoformat(post['posted_at'].replace('Z', '+00:00'))
        except:
            return datetime.min.replace(tzinfo=timezone.utc)
    
    all_posts.sort(key=sort_key, reverse=True)

    # Write back to CSV
    try:
        with open(csv_file, mode="w", newline="", encoding="utf-8") as file:
            fieldnames = [
                "platform", "captured_at", "posted_at", "username", "post_id",
                "content", "source_url", "keyword", "detection_cycle"
            ]
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_posts)
        
        return len(new_posts)
    
    except Exception as e:
        print(f"Error saving to CSV: {e}")
        return 0

def scrape_keyword(page, keyword, cycle_number):
    """Scrape all data for a specific keyword"""
    csv_file = get_csv_filename(keyword)
    is_initial_scrape = initialize_csv(csv_file)
    
    existing_post_ids, latest_timestamp = load_existing_posts_data(csv_file)
    
    print(f"\nPROCESSING: {keyword}")
    print(f"   CSV File: {csv_file}")
    print(f"   Cycle Type: {'INITIAL COMPREHENSIVE SCRAPE' if is_initial_scrape else 'INCREMENTAL UPDATE'}")
    
    all_new_posts = []
    
    if is_initial_scrape:
        # First cycle: comprehensive scrape of both sections
        print(f"   Duration: {INITIAL_SCRAPE_DURATION//60} minutes")
        
        # Scrape TOP section
        top_posts = scrape_section(page, keyword, "top", existing_post_ids, latest_timestamp,
                                 is_initial_scrape, cycle_number, INITIAL_SCROLL_COUNT)
        all_new_posts.extend(top_posts)
        
        # Brief pause between sections
        time.sleep(2)
        
        # Scrape LATEST section
        latest_posts = scrape_section(page, keyword, "latest", existing_post_ids, latest_timestamp,
                                    is_initial_scrape, cycle_number, INITIAL_SCROLL_COUNT)
        all_new_posts.extend(latest_posts)
        
    else:
        # Subsequent cycles: focus on latest posts only
        latest_posts = scrape_section(page, keyword, "latest", existing_post_ids, latest_timestamp,
                                    is_initial_scrape, cycle_number, REGULAR_SCROLL_COUNT)
        all_new_posts.extend(latest_posts)
    
    # Save to CSV
    saved_count = save_posts_to_csv(csv_file, all_new_posts)
    
    print(f"   Saved {saved_count} new posts to {csv_file}")
    return saved_count

def real_time_social_media_monitor():
    """Main monitoring function"""
    global running
    
    print("ENHANCED SOCIAL MEDIA MONITORING SYSTEM")
    print("=" * 70)
    print("Press Ctrl+C to stop monitoring\n")

    cycle_count = 0
    total_captured = 0

    with sync_playwright() as p:
        # Launch browser with stealth settings
        browser = p.chromium.launch(
            headless=True,
            args=[
                '--no-sandbox', 
                '--disable-blink-features=AutomationControlled',
                '--disable-dev-shm-usage',
                '--disable-web-security',
                '--disable-features=VizDisplayCompositor'
            ]
        )
        
        context = browser.new_context(
            storage_state=SESSION_FILE,
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        
        page = context.new_page()

        # Anti-detection measures
        page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined,
            });
        """)

        try:
            while running:
                cycle_count += 1
                cycle_start = datetime.now()
                
                print(f"\nMONITORING CYCLE #{cycle_count} | {cycle_start.strftime('%Y-%m-%d %H:%M:%S')}")
                print("=" * 70)
                
                cycle_new_posts = 0
                
                for keyword in KEYWORDS:
                    if not running:
                        break
                    
                    saved_count = scrape_keyword(page, keyword, cycle_count)
                    cycle_new_posts += saved_count
                    total_captured += saved_count
                    
                    # Pause between keywords
                    if running and keyword != KEYWORDS[-1]:
                        time.sleep(2)

                # Cycle summary
                cycle_duration = (datetime.now() - cycle_start).total_seconds()
                print(f"\nCYCLE #{cycle_count} SUMMARY:")
                print(f"   New posts this cycle: {cycle_new_posts}")
                print(f"   Total posts captured: {total_captured}")
                print(f"   Duration: {cycle_duration//60:.0f}m {cycle_duration%60:.0f}s")
                
                # Wait before next cycle
                if running and cycle_count > 0:
                    print(f"Next check in {CHECK_INTERVAL_SECONDS} seconds...\n")
                    for i in range(CHECK_INTERVAL_SECONDS):
                        if not running:
                            break
                        time.sleep(1)

        except KeyboardInterrupt:
            print(f"\nMonitoring stopped by user")
        except Exception as e:
            print(f"Critical error: {e}")
        finally:
            print("Closing browser...")
            browser.close()
            print(f"Session complete. Total posts captured: {total_captured}")

if __name__ == "__main__":
    try:
        real_time_social_media_monitor()
    except KeyboardInterrupt:
        print("\nEnhanced monitoring system shut down")
    finally:
        sys.exit(0)