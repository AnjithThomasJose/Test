"""
Cache Debug Script - Verify cache key generation and lookup

This script helps debug why cache isn't hitting by:
1. Sending a request
2. Checking logs for CACHE_MISS/CACHE_HIT
3. Sending identical request
4. Analyzing cache key fingerprints
"""

import requests
import time
import json
from pathlib import Path
import logging

# Setup logging for this script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

BASE_URL = "http://localhost:8000"

def read_recent_logs(log_file: str = "logs/main.log", lines: int = 100):
    """Read recent log lines"""
    try:
        log_path = Path(log_file)
        if not log_path.exists():
            log.error(f"Log file not found: {log_file}")
            return []
        
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.readlines()[-lines:]
    except Exception as e:
        log.error(f"Error reading logs: {e}")
        return []

def find_cache_events(lines, uid):
    """Find cache-related events for a specific UID"""
    cache_events = []
    for line in lines:
        if uid in line and any(x in line for x in ['CACHE_HIT', 'CACHE_MISS', 'CACHE_STORE', 'COALESCE']):
            cache_events.append(line.strip())
    return cache_events

def test_cache_with_logs():
    """Test cache and show log analysis"""
    log.info("=" * 80)
    log.info("CACHE DEBUG TEST")
    log.info("=" * 80)
    
    uid = f"debug_test_{int(time.time())}"
    
    payload = {
        "uid": uid,
        "tenant_id": "debug_tenant",
        "callback_url": "https://example.com/callback",
        "resume_url": "https://storage.googleapis.com/test-bucket/sample.pdf"
    }
    
    log.info(f"\nTest UID: {uid}")
    log.info(f"Payload: {json.dumps(payload, indent=2)}")
    
    # Request 1
    log.info("\n" + "=" * 80)
    log.info("REQUEST 1 (Should be CACHE_MISS)")
    log.info("=" * 80)
    
    start1 = time.time()
    resp1 = requests.patch(f"{BASE_URL}/analyze-resume-callback", json=payload)
    time1 = time.time() - start1
    
    log.info(f"Status: {resp1.status_code}")
    log.info(f"Time: {time1:.3f}s")
    log.info(f"Response: {json.dumps(resp1.json(), indent=2)}")
    
    # Wait for processing
    log.info("\nWaiting 5 seconds for processing to complete...")
    time.sleep(5)
    
    # Check logs after first request
    log.info("\n" + "=" * 80)
    log.info("LOG ANALYSIS - After Request 1")
    log.info("=" * 80)
    
    logs1 = read_recent_logs(lines=200)
    events1 = find_cache_events(logs1, uid)
    
    if events1:
        for event in events1:
            log.info(f"  {event}")
    else:
        log.warning("⚠️  No cache events found - check if logs are being written")
    
    # Request 2
    log.info("\n" + "=" * 80)
    log.info("REQUEST 2 (Should be CACHE_HIT)")
    log.info("=" * 80)
    
    start2 = time.time()
    resp2 = requests.patch(f"{BASE_URL}/analyze-resume-callback", json=payload)
    time2 = time.time() - start2
    
    log.info(f"Status: {resp2.status_code}")
    log.info(f"Time: {time2:.3f}s")
    log.info(f"Response: {json.dumps(resp2.json(), indent=2)}")
    
    # Check logs after second request
    log.info("\n" + "=" * 80)
    log.info("LOG ANALYSIS - After Request 2")
    log.info("=" * 80)
    
    logs2 = read_recent_logs(lines=200)
    events2 = find_cache_events(logs2, uid)
    
    # Show only NEW events (not in events1)
    new_events = [e for e in events2 if e not in events1]
    
    if new_events:
        for event in new_events:
            log.info(f"  {event}")
    else:
        log.warning("⚠️  No new cache events - this is suspicious!")
    
    # Analysis
    log.info("\n" + "=" * 80)
    log.info("ANALYSIS")
    log.info("=" * 80)
    
    speedup = time1 / time2 if time2 > 0 else 0
    log.info(f"Speedup: {speedup:.2f}x")
    
    # Check if we see expected cache patterns
    has_cache_miss = any("CACHE_MISS" in e for e in events1)
    has_cache_store = any("CACHE_STORE" in e for e in events1)
    has_cache_hit = any("CACHE_HIT" in e for e in new_events)
    
    log.info(f"\nCache Events Detected:")
    log.info(f"  CACHE_MISS (Request 1): {'✅' if has_cache_miss else '❌'}")
    log.info(f"  CACHE_STORE (Request 1): {'✅' if has_cache_store else '❌'}")
    log.info(f"  CACHE_HIT (Request 2): {'✅' if has_cache_hit else '❌'}")
    
    if has_cache_hit and speedup > 5:
        log.info(f"\n✅ CACHE IS WORKING CORRECTLY!")
    elif has_cache_hit:
        log.warning(f"\n⚠️  Cache hit detected but speedup is low ({speedup:.2f}x)")
        log.warning(f"    This might be due to callback processing overhead")
    else:
        log.error(f"\n❌ CACHE IS NOT WORKING")
        log.error(f"\nPossible issues:")
        log.error(f"  1. Resume URL validation failing")
        log.error(f"  2. Cache key fingerprint mismatch")
        log.error(f"  3. Cache being cleared between requests")
        log.error(f"  4. URL sanitization changing the key")
        
        # Check for URL validation issues
        url_events = [e for e in events1 if "sanitize" in e.lower() or "invalid" in e.lower()]
        if url_events:
            log.warning(f"\n  URL Validation Issues Found:")
            for e in url_events:
                log.warning(f"    {e}")
    
    # Check metrics endpoint
    log.info("\n" + "=" * 80)
    log.info("METRICS CHECK")
    log.info("=" * 80)
    
    try:
        metrics = requests.get(f"{BASE_URL}/metrics").json()
        cache_hit_rate = metrics.get('cache_hit_rate', 0)
        method_split = metrics.get('method_split', {})
        
        log.info(f"Cache Hit Rate: {cache_hit_rate:.2%}")
        log.info(f"Method Split: {json.dumps(method_split, indent=2)}")
    except Exception as e:
        log.error(f"Failed to get metrics: {e}")

if __name__ == "__main__":
    # Check if server is running
    try:
        resp = requests.get(f"{BASE_URL}/health", timeout=5)
        if resp.status_code == 200:
            log.info("✅ Server is running, starting cache test")
            test_cache_with_logs()
        else:
            log.error(f"❌ Server returned status {resp.status_code}")
    except requests.exceptions.ConnectionError:
        log.error("❌ Cannot connect to server")
        log.info("\nStart the server first:")
        log.info("  uvicorn app:app --reload --port 8000")