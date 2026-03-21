#!/usr/bin/env python3
"""
Test script for daily re-ranking functionality.

This script provides multiple ways to test the daily re-ranking:
1. Direct function call (fastest)
2. Via API endpoint
3. Check scheduler status
"""

import asyncio
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.agents.ranker import daily_rerank_all_jds_handler
from utils.job_scheduler import job_scheduler
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

async def test_direct_call():
    """Test 1: Direct function call (fastest way to test)"""
    print("\n" + "="*80)
    print("TEST 1: Direct Function Call")
    print("="*80)
    
    try:
        result = await daily_rerank_all_jds_handler()
        print(f"\n✅ Test completed!")
        print(f"Status: {result.get('status')}")
        print(f"Results: {result.get('results', {})}")
        return result
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

async def test_scheduler_status():
    """Test 2: Check scheduler status and jobs"""
    print("\n" + "="*80)
    print("TEST 2: Scheduler Status")
    print("="*80)
    
    try:
        jobs = job_scheduler.get_scheduled_jobs()
        print(f"\n📋 Found {len(jobs)} scheduled jobs:")
        
        for job in jobs:
            if job.get('job_type') == 'daily_rerank_all_jds':
                print(f"\n🔄 Daily Re-ranking Job:")
                print(f"   Job ID: {job.get('job_id')}")
                print(f"   Status: {'Active' if job.get('is_active') else 'Paused'}")
                print(f"   Schedule: {job.get('schedule_type')} - {job.get('schedule_config')}")
                print(f"   Last Run: {job.get('last_run', 'Never')}")
                print(f"   Next Run: {job.get('next_run', 'Not scheduled')}")
                print(f"   Description: {job.get('metadata', {}).get('description', 'N/A')}")
        
        return jobs
    except Exception as e:
        print(f"\n❌ Failed to get scheduler status: {e}")
        import traceback
        traceback.print_exc()
        return None

async def test_manual_trigger():
    """Test 3: Manually trigger via scheduler"""
    print("\n" + "="*80)
    print("TEST 3: Manual Trigger via Scheduler")
    print("="*80)
    
    try:
        # Get the daily rerank job
        jobs = job_scheduler.get_scheduled_jobs()
        rerank_job = None
        
        for job in jobs:
            if job.get('job_type') == 'daily_rerank_all_jds':
                rerank_job = job
                break
        
        if not rerank_job:
            print("❌ Daily re-ranking job not found in scheduler")
            return None
        
        print(f"🔄 Triggering job: {rerank_job.get('job_id')}")
        
        # Manually trigger the job
        await job_scheduler._run_job(rerank_job.get('job_id'), rerank_job)
        
        print("✅ Job triggered successfully")
        return True
        
    except Exception as e:
        print(f"\n❌ Failed to trigger job: {e}")
        import traceback
        traceback.print_exc()
        return None

async def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("🧪 DAILY RE-RANKING TEST SUITE")
    print("="*80)
    
    # Test 1: Direct call (recommended for quick testing)
    print("\n📝 Running Test 1: Direct Function Call...")
    result1 = await test_direct_call()
    
    # Test 2: Check scheduler
    print("\n📝 Running Test 2: Scheduler Status...")
    result2 = await test_scheduler_status()
    
    # Test 3: Manual trigger (optional)
    print("\n📝 Running Test 3: Manual Trigger via Scheduler...")
    result3 = await test_manual_trigger()
    
    print("\n" + "="*80)
    print("✅ ALL TESTS COMPLETED")
    print("="*80)
    
    print("\n💡 Tips:")
    print("   - Use Test 1 (direct call) for fastest testing")
    print("   - Use Test 2 to check when the job will run next")
    print("   - Use Test 3 to test the full scheduler integration")
    print("   - Or use the API endpoint: POST /test/daily-rerank")

if __name__ == "__main__":
    asyncio.run(main())


