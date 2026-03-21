#!/usr/bin/env python3
"""
Script to check KB auto-population daily logs.
Shows whether the auto-population ran on specific dates and provides statistics.
"""

import sys
import os
import json
from datetime import date, datetime, timedelta
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.kb_auto_populator import get_kb_auto_populator
import argparse

def format_summary(summary: dict):
    """Format summary for display"""
    print("\n" + "="*80)
    print(f"📅 KB Auto-Population Summary for {summary['date']}")
    print("="*80)
    
    if not summary.get("ran", False):
        print("❌ No runs recorded for this date")
        return
    
    print(f"✅ Ran: Yes ({summary['total_runs']} run(s))")
    print(f"\n📊 Statistics:")
    print(f"   Topics processed: {summary['total_topics_processed']}")
    print(f"   Topics successful: {summary['total_topics_successful']}")
    print(f"   Topics failed: {summary['total_topics_failed']}")
    print(f"   Total materials added: {summary['total_materials_added']}")
    
    print(f"\n📚 Materials by type:")
    for material_type, count in summary['materials_by_type'].items():
        if count > 0:
            print(f"   {material_type.capitalize()}: {count}")
    
    if summary.get('runs'):
        print(f"\n🔄 Run Details:")
        for i, run in enumerate(summary['runs'], 1):
            print(f"\n   Run {i}:")
            print(f"      Time: {run.get('timestamp', 'N/A')}")
            print(f"      Status: {run.get('status', 'unknown')}")
            if 'stats' in run:
                stats = run['stats']
                print(f"      Topics: {stats.get('topics_processed', 0)} processed, "
                      f"{stats.get('topics_successful', 0)} successful, "
                      f"{stats.get('topics_failed', 0)} failed")
                print(f"      Materials added: {stats.get('total_materials_added', 0)}")
    
    print("="*80 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Check KB auto-population daily logs")
    parser.add_argument(
        "--date",
        type=str,
        help="Date to check (YYYY-MM-DD format, defaults to today)",
        default=None
    )
    parser.add_argument(
        "--days",
        type=int,
        help="Number of recent days to check (default: 1)",
        default=1
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available log files"
    )
    
    args = parser.parse_args()
    
    populator = get_kb_auto_populator()
    
    if args.list:
        # List all log files from ChromaDB
        try:
            from chroma import _get_collection
            collection = _get_collection("kb_population_logs")
            
            # Get all unique dates
            all_results = collection.get()
            if not all_results.get("ids") or not all_results["ids"]:
                print("📭 No log entries found in ChromaDB. Auto-population may not have run yet.")
                return
            
            # Group by date
            dates_data = {}
            metadatas = all_results.get("metadatas", [])
            for i, metadata in enumerate(metadatas):
                date_str = metadata.get("date", "Unknown")
                if date_str not in dates_data:
                    dates_data[date_str] = {
                        "runs": 0,
                        "materials": 0
                    }
                dates_data[date_str]["runs"] += 1
                dates_data[date_str]["materials"] += metadata.get("total_materials_added", 0)
            
            # Sort by date (newest first)
            sorted_dates = sorted(dates_data.items(), reverse=True)
            
            print(f"\n📋 Found {len(sorted_dates)} day(s) with log entries:\n")
            for date_str, data in sorted_dates:
                print(f"   {date_str}: {data['runs']} run(s), {data['materials']} materials added")
            print()
            
        except Exception as e:
            print(f"❌ Error reading from ChromaDB: {e}")
            # Fallback to JSON files
            logs_dir = Path("logs/kb_population")
            if not logs_dir.exists():
                print("❌ No logs directory found. Auto-population may not have run yet.")
                return
            
            log_files = sorted(logs_dir.glob("kb_population_*.json"), reverse=True)
            if not log_files:
                print("📭 No log files found. Auto-population may not have run yet.")
                return
            
            print(f"\n📋 Found {len(log_files)} log file(s) (from JSON backup):\n")
            for log_file in log_files:
                try:
                    with open(log_file, 'r') as f:
                        data = json.load(f)
                        date_str = data.get('date', 'Unknown')
                        runs = len(data.get('runs', []))
                        summary = data.get('summary', {})
                        materials = summary.get('total_materials_added', 0)
                        print(f"   {date_str}: {runs} run(s), {materials} materials added")
                except Exception as e:
                    print(f"   {log_file.name}: Error reading ({e})")
            print()
        return
    
    # Check specific date or date range
    if args.date:
        try:
            target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
        except ValueError:
            print(f"❌ Invalid date format: {args.date}. Use YYYY-MM-DD format.")
            return
    else:
        target_date = date.today()
    
    # Check date range
    for i in range(args.days):
        check_date = target_date - timedelta(days=i)
        summary = populator.get_daily_summary(check_date)
        format_summary(summary)

if __name__ == "__main__":
    main()

