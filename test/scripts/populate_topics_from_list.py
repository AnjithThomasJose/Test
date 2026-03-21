#!/usr/bin/env python3
"""
Populate KB with courses and materials for a specific list of topics using APIs.

This script:
1. Takes a list of topics
2. For each topic, uses auto-discovery to find courses, books, papers, videos, tutorials
3. Adds discovered materials to KB

Usage:
    python scripts/populate_topics_from_list.py [--limit N] [--material-types course,book,paper,video,tutorial] [--max-per-topic N]
"""

import sys
import os
import argparse
import asyncio
import logging
from typing import List, Dict, Any

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.market_and_course_recommender import _check_and_discover_materials
from core.logging_helpers import AgentLogger, create_log_context

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Topics list from user
TOPICS = [
    "2d animation", "2d games", "3d animation", "3d character", "3d games", "3d modeling",
    "5 pillars", "abnormal psychology", "active directory", "actix", "adobe illustrator", "adobe photoshop",
    "advanced english", "advanced german", "advanced javascript", "advanced topics", "advocacy", "after effects",
    "agile", "agile tools", "ai", "ai basics", "ai for robotics", "ai reasoning",
    "algorithms", "analysis methods", "android", "angular", "animation", "animation basics",
    "animation principles", "animations", "anonymity", "api", "api design", "api gateway",
    "apis", "ar filters", "ar/vr development", "aria roles", "arm", "arm cortex",
    "arrays", "artificial intelligence", "async", "async api", "async programming", "async/await",
    "attack types", "audit", "audits", "auth", "automation", "automl",
    "aws", "aws architecture", "aws iot", "aws lambda", "aws services", "azure",
    "azure devops", "azure services", "backend", "basic chinese", "basic cloud concepts", "basic spanish",
    "behavioral economics", "behavioral science", "bem", "best practices", "big data", "blender",
    "blockchain", "blueprints", "board of directors", "bootstrap", "bot framework", "branding",
    "buffer overflows", "business analysis", "business analytics", "business basics", "business chinese", "business english",
    "business ethics", "business french", "business german", "business law", "business spanish", "business vocabulary",
    "c++", "caching", "canva", "career development", "ccna", "ccna prep",
    "certification prep", "cfa", "cfd", "chai", "chaining", "character animation",
    "chinese", "chinese for business", "ci/cd", "cia triad", "cidr", "cipp/e",
    "circuit analysis", "circuit design", "circuit theory", "cisco", "cisco iot", "cisco sales",
    "classes", "classical animation", "cli tools", "closures", "cloud basics", "cloud concepts",
    "cloud exploitation", "cloud functions", "cloud fundamentals", "cloud iot", "cloud ml", "cloud security",
    "cloud threat models", "cloudformation", "cnns", "cognitive services", "collaboration", "collections",
    "color correction", "color theory", "community health", "compliance", "components", "composition",
    "comptia server+", "computer forensics", "computer networking", "computer science", "concurrency", "connected devices",
    "consensus", "consistency", "container orchestration", "container technology", "containerization", "containers",
    "contract law", "control systems", "cooling systems", "copyright", "corporate governance", "corporate strategy",
    "cortex-m3", "cortex-m4", "cpt", "crm", "crm administration", "cross-platform",
    "cryptography", "css", "css optimization", "css3", "cyber attacks", "cyber law",
    "cyber threats", "cybersecurity", "dapp development", "dapps", "dart", "dashboards",
    "data", "data analysis", "data center", "data center operations", "data centers", "data privacy",
    "data protection", "data querying", "data science", "data structures", "data visualization", "davinci resolve",
    "decision making", "deep learning", "defender", "demo skills", "design basics", "design theory",
    "devops", "digital body language", "digital design", "digital experience", "digital forensics", "digital sales",
    "direct connect", "discovery", "distributed computing", "distributed systems", "django", "dl best practices",
    "docker", "docker containers", "dom", "drf", "dynamodb", "economics",
    "edge ai", "electrical controls", "electrical engineering", "electronics", "embedded ai", "embedded design",
    "embedded programming", "embedded systems", "emerging markets", "employment law", "encryption", "engines",
    "english for esl", "enterprise", "entity framework", "entrepreneurship", "enumeration", "erc-721",
    "es6", "es6+", "esl", "essential english", "ethereum", "ethics",
    "event handling", "event loop", "events", "evidence handling", "executors", "experience design",
    "exploitation", "exploitation basics", "expo", "express", "expressroute", "facilities",
    "factory", "fault analysis", "fault tolerance", "financial analysis", "fintech", "firebase",
    "firewalls", "first aid", "flask api", "flexbox", "fluid mechanics", "flutter",
    "forensic imaging", "foundations", "founder skills", "fpga", "fpga development", "frame-by-frame",
    "french", "french for business", "french for professionals", "frontend development", "full stack development", "functions",
    "game design", "game development", "ganache", "gcp", "gcp ml tools", "gdpr",
    "genai basics", "german", "german at work", "german for business", "github actions", "glue",
    "go", "google cardboard", "google cloud", "goroutines", "governance", "graphic design",
    "graphs", "grid", "gui", "hacking labs", "hands-on learning", "hardware",
    "hashing", "hcpcs", "hdl", "hdl programming", "health", "health basics",
    "health economics", "health policy", "healthcare", "healthcare policy", "hr analytics", "hr strategy",
    "html", "html5", "http", "hubspot", "hubspot certification", "hvac",
    "hyperledger", "i/o", "iac", "iam", "iam bypass", "identity",
    "identity design", "ids/ips", "illustration", "immutability", "inbound sales", "inference",
    "infrastructure", "innovation", "input/output", "intellectual property", "interfaces", "intermediate spanish",
    "investment management", "ios", "iot", "iot architecture", "iot deployment", "iot fundamentals",
    "ip addressing", "java", "javascript", "jenkins", "jest", "jetpack compose",
    "jira", "jquery", "jvm", "jwt", "k8s", "kali linux",
    "kanban", "kms", "kotlin", "kubernetes", "lambda", "lambdas",
    "law", "leadership", "legal", "legal tech", "linux", "linux hardening",
    "localization", "logical reasoning", "logo design", "machine learning", "macronutrients", "malware",
    "malware defense", "mandarin", "market entry", "maya", "medical coding", "memory",
    "mental health", "mental health ambassador", "mental wellness", "meta ar", "metadata", "metrics",
    "microcontrollers", "microservices", "mindfulness", "mining", "mixins", "ml",
    "ml basics", "ml optimization", "ml theory", "ml with python", "ml workflow", "mobile",
    "mobile design", "mobile vr", "mocha", "mocks", "modeling", "modern javascript",
    "modern ui", "modern web development", "modifiers", "modules", "mongodb", "monitoring",
    "monopoly theory", "moralis api", "motion design", "multi-tenancy", "multiplatform", "multithreading",
    "mutexes", "mvc", "nesting", "net/http", "netlify", "netmiko",
    "network attacks", "network basics", "network defense", "network design", "networking", "networking fundamentals",
    "neural networks", "neural networks basics", "next.js", "nft minting", "nfts", "node",
    "node.js", "nodes", "numerical methods", "nutrition", "nutrition certification", "nutrition science",
    "oop", "packet analysis", "passwords", "patents", "path planning", "pcb design",
    "pcb layout", "people analytics", "permissions", "photo editing", "photovoltaics", "physics",
    "pipeline", "pipelines", "pivoting", "plc", "pointers", "polymorphism",
    "postgresql", "pow", "power system protection", "presales", "presentation skills", "privacy awareness",
    "privacy certification", "privilege escalation", "probability", "procedure coding", "product demo", "product development",
    "professional certificate", "professional communication", "professional english", "professional french", "professional german", "professional spanish",
    "profiling", "programming", "project tracking", "projects", "promises", "proof-of-work",
    "protection systems", "psychological disorders", "psychology", "psychology basics", "pub/sub", "public health",
    "python", "python ai", "python blockchain", "python ml", "pytorch", "quantum algorithms",
    "quantum basics", "quantum computers", "quantum computing", "quantum engineering", "quantum mechanics", "quantum programming",
    "quantum science", "quantum theory", "queries", "questioning techniques", "rails", "rbac",
    "react", "react native", "react ui", "react.js", "reactivity", "reasoning",
    "reconnaissance", "recursion", "redshift", "redux", "reentrancy", "regression",
    "relational design", "relays", "reliability", "remix", "reporting", "repos",
    "resilience", "responsive design", "responsive ui", "rest", "risk management", "rnns",
    "robot basics", "robot operating system", "robot programming", "robotics", "robotics software", "ros",
    "ros1", "ros2", "routing", "rpc", "ruby", "rust",
    "sales", "sales basics", "sales certification", "sales communication", "sales demo", "sales engineer",
    "sales engineering", "sales essentials", "sales foundations", "sales questions", "sales skills", "sales software",
    "salesforce", "salesforce administrator", "salesforce certification", "sass", "scalability", "scanning",
    "scripting", "scripts", "scrum", "sdl", "search", "secure architecture",
    "secure routing", "security architecture", "security basics", "security controls", "security law", "security patterns",
    "security posture", "segmentation", "selenium", "server administration", "server basics", "server certification",
    "server management", "serverless", "services", "shared code", "shared responsibility", "shell scripting",
    "siem", "simulation", "skills", "sklearn", "slam", "smart contracts",
    "smart pointers", "snapshots", "social ar", "social entrepreneurship", "sockets", "solar energy",
    "solar systems", "solidity", "solidity syntax", "solution consulting", "solutions design", "spanish",
    "spanish certificate", "spanish culture", "spanish for beginners", "spanish for business", "spanish for professionals", "spanish language",
    "spark ar", "spring", "spring boot", "sql", "sqs", "sre",
    "ssh security", "startups", "state", "static generation", "statistics for ds", "stl",
    "storytelling", "strategy", "stress management", "subnetting", "svelte", "swift",
    "system administration", "system design", "systemverilog", "tableau", "tailwind", "talent management",
    "tech startups", "technical sales", "technology law", "templates", "tensorflow", "tensorflow/keras",
    "terraform", "testbench", "testing", "threads", "threat modeling", "threats",
    "tinyml", "traditional animation", "transit gateway", "truffle", "typescript", "typography",
    "ui components", "ui/ux", "uikit", "understanding", "unity", "universal components",
    "unreal engine 5", "user journey", "validation", "variables", "vector graphics", "verification",
    "vhdl", "video editing", "video training", "virtual sales", "virtualization", "vnet",
    "vpc", "vpn", "vpns", "vr apps", "vr design", "vr development",
    "vr games", "vue", "vue.js", "vulnerabilities", "vulnerability scanning", "wallets",
    "wcag", "web design", "web development", "web exploits", "web hacking", "web3.js",
    "wellness", "windows server", "wipo", "workload isolation", "workplace wellness", "xgboost",
    "xr development", "yaml pipelines", "yc methodology", "zoho", "zoho crm"
]


async def populate_topics(
    topics: List[str],
    material_types: List[str] = None,
    max_per_topic: int = 5,
    limit: int = None,
    log_context: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Populate KB with materials for given topics.
    
    Args:
        topics: List of topics to populate
        material_types: Types of materials to discover (default: all)
        max_per_topic: Maximum materials to discover per topic
        limit: Maximum number of topics to process (None = all)
        log_context: Logging context
        
    Returns:
        Dict with statistics about population
    """
    if material_types is None:
        material_types = ["course", "book", "paper", "video", "tutorial"]
    
    if limit:
        topics = topics[:limit]
    
    log_context = log_context or create_log_context("populate_topics", tenant_id="system")
    
    stats = {
        "topics_processed": 0,
        "topics_successful": 0,
        "topics_failed": 0,
        "total_materials_added": 0,
        "materials_by_type": {mt: 0 for mt in material_types}
    }
    
    logger.info(f"🚀 Starting population for {len(topics)} topics...")
    logger.info(f"📚 Material types: {', '.join(material_types)}")
    logger.info(f"📊 Max per topic: {max_per_topic}")
    
    for i, topic in enumerate(topics, 1):
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"📝 Processing topic {i}/{len(topics)}: {topic}")
            logger.info(f"{'='*60}")
            
            # Discover materials for this topic
            discovered = await _check_and_discover_materials(
                topic=topic,
                material_types=material_types,
                min_results_required=2,  # Lower threshold to discover more
                min_similarity_threshold=0.3,  # Lower threshold for broader discovery
                max_results_per_type=max_per_topic,
                log_context=log_context
            )
            
            if discovered:
                # Count materials by type
                for material in discovered:
                    material_type = material.get("type", "unknown")
                    if material_type in stats["materials_by_type"]:
                        stats["materials_by_type"][material_type] += 1
                    stats["total_materials_added"] += 1
                
                stats["topics_successful"] += 1
                logger.info(f"✅ Topic '{topic}': Added {len(discovered)} materials")
            else:
                logger.warning(f"⚠️  Topic '{topic}': No materials discovered")
                stats["topics_failed"] += 1
            
            stats["topics_processed"] += 1
            
            # Progress update every 10 topics
            if i % 10 == 0:
                logger.info(f"\n📊 Progress: {i}/{len(topics)} topics processed")
                logger.info(f"   ✅ Successful: {stats['topics_successful']}")
                logger.info(f"   ❌ Failed: {stats['topics_failed']}")
                logger.info(f"   📚 Total materials: {stats['total_materials_added']}")
            
            # Small delay to avoid rate limiting
            await asyncio.sleep(0.5)
            
        except Exception as e:
            logger.error(f"❌ Error processing topic '{topic}': {e}", exc_info=True)
            stats["topics_failed"] += 1
            stats["topics_processed"] += 1
            continue
    
    return stats


async def main_async():
    """Main async function"""
    parser = argparse.ArgumentParser(description="Populate KB with materials for topics using APIs")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of topics to process (default: all)"
    )
    parser.add_argument(
        "--material-types",
        type=str,
        default="course,book,paper,video,tutorial",
        help="Comma-separated list of material types (default: course,book,paper,video,tutorial)"
    )
    parser.add_argument(
        "--max-per-topic",
        type=int,
        default=5,
        help="Maximum materials to discover per topic (default: 5)"
    )
    parser.add_argument(
        "--topics-file",
        type=str,
        default=None,
        help="Path to file with topics (one per line). If not provided, uses built-in list."
    )
    
    args = parser.parse_args()
    
    # Parse material types
    material_types = [mt.strip() for mt in args.material_types.split(",")]
    
    # Get topics
    if args.topics_file and os.path.exists(args.topics_file):
        logger.info(f"📄 Loading topics from file: {args.topics_file}")
        with open(args.topics_file, 'r', encoding='utf-8') as f:
            topics = [line.strip() for line in f if line.strip()]
    else:
        topics = TOPICS
        logger.info(f"📋 Using built-in topics list ({len(topics)} topics)")
    
    # Create log context
    log_context = create_log_context("populate_topics", tenant_id="system")
    
    # Run population
    stats = await populate_topics(
        topics=topics,
        material_types=material_types,
        max_per_topic=args.max_per_topic,
        limit=args.limit,
        log_context=log_context
    )
    
    # Print final summary
    logger.info("\n" + "="*60)
    logger.info("📊 FINAL SUMMARY")
    logger.info("="*60)
    logger.info(f"Topics processed: {stats['topics_processed']}")
    logger.info(f"Topics successful: {stats['topics_successful']}")
    logger.info(f"Topics failed: {stats['topics_failed']}")
    logger.info(f"Total materials added: {stats['total_materials_added']}")
    logger.info("\nMaterials by type:")
    for material_type, count in stats['materials_by_type'].items():
        logger.info(f"  {material_type}: {count}")
    logger.info("="*60)


def main():
    """Main entry point"""
    try:
        asyncio.run(main_async())
        return 0
    except KeyboardInterrupt:
        logger.info("\n⚠️  Interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())


"""
Populate KB with courses and materials for a specific list of topics using APIs.

This script:
1. Takes a list of topics
2. For each topic, uses auto-discovery to find courses, books, papers, videos, tutorials
3. Adds discovered materials to KB

Usage:
    python scripts/populate_topics_from_list.py [--limit N] [--material-types course,book,paper,video,tutorial] [--max-per-topic N]
"""

import sys
import os
import argparse
import asyncio
import logging
from typing import List, Dict, Any

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.market_and_course_recommender import _check_and_discover_materials
from core.logging_helpers import AgentLogger, create_log_context

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Topics list from user
TOPICS = [
    "2d animation", "2d games", "3d animation", "3d character", "3d games", "3d modeling",
    "5 pillars", "abnormal psychology", "active directory", "actix", "adobe illustrator", "adobe photoshop",
    "advanced english", "advanced german", "advanced javascript", "advanced topics", "advocacy", "after effects",
    "agile", "agile tools", "ai", "ai basics", "ai for robotics", "ai reasoning",
    "algorithms", "analysis methods", "android", "angular", "animation", "animation basics",
    "animation principles", "animations", "anonymity", "api", "api design", "api gateway",
    "apis", "ar filters", "ar/vr development", "aria roles", "arm", "arm cortex",
    "arrays", "artificial intelligence", "async", "async api", "async programming", "async/await",
    "attack types", "audit", "audits", "auth", "automation", "automl",
    "aws", "aws architecture", "aws iot", "aws lambda", "aws services", "azure",
    "azure devops", "azure services", "backend", "basic chinese", "basic cloud concepts", "basic spanish",
    "behavioral economics", "behavioral science", "bem", "best practices", "big data", "blender",
    "blockchain", "blueprints", "board of directors", "bootstrap", "bot framework", "branding",
    "buffer overflows", "business analysis", "business analytics", "business basics", "business chinese", "business english",
    "business ethics", "business french", "business german", "business law", "business spanish", "business vocabulary",
    "c++", "caching", "canva", "career development", "ccna", "ccna prep",
    "certification prep", "cfa", "cfd", "chai", "chaining", "character animation",
    "chinese", "chinese for business", "ci/cd", "cia triad", "cidr", "cipp/e",
    "circuit analysis", "circuit design", "circuit theory", "cisco", "cisco iot", "cisco sales",
    "classes", "classical animation", "cli tools", "closures", "cloud basics", "cloud concepts",
    "cloud exploitation", "cloud functions", "cloud fundamentals", "cloud iot", "cloud ml", "cloud security",
    "cloud threat models", "cloudformation", "cnns", "cognitive services", "collaboration", "collections",
    "color correction", "color theory", "community health", "compliance", "components", "composition",
    "comptia server+", "computer forensics", "computer networking", "computer science", "concurrency", "connected devices",
    "consensus", "consistency", "container orchestration", "container technology", "containerization", "containers",
    "contract law", "control systems", "cooling systems", "copyright", "corporate governance", "corporate strategy",
    "cortex-m3", "cortex-m4", "cpt", "crm", "crm administration", "cross-platform",
    "cryptography", "css", "css optimization", "css3", "cyber attacks", "cyber law",
    "cyber threats", "cybersecurity", "dapp development", "dapps", "dart", "dashboards",
    "data", "data analysis", "data center", "data center operations", "data centers", "data privacy",
    "data protection", "data querying", "data science", "data structures", "data visualization", "davinci resolve",
    "decision making", "deep learning", "defender", "demo skills", "design basics", "design theory",
    "devops", "digital body language", "digital design", "digital experience", "digital forensics", "digital sales",
    "direct connect", "discovery", "distributed computing", "distributed systems", "django", "dl best practices",
    "docker", "docker containers", "dom", "drf", "dynamodb", "economics",
    "edge ai", "electrical controls", "electrical engineering", "electronics", "embedded ai", "embedded design",
    "embedded programming", "embedded systems", "emerging markets", "employment law", "encryption", "engines",
    "english for esl", "enterprise", "entity framework", "entrepreneurship", "enumeration", "erc-721",
    "es6", "es6+", "esl", "essential english", "ethereum", "ethics",
    "event handling", "event loop", "events", "evidence handling", "executors", "experience design",
    "exploitation", "exploitation basics", "expo", "express", "expressroute", "facilities",
    "factory", "fault analysis", "fault tolerance", "financial analysis", "fintech", "firebase",
    "firewalls", "first aid", "flask api", "flexbox", "fluid mechanics", "flutter",
    "forensic imaging", "foundations", "founder skills", "fpga", "fpga development", "frame-by-frame",
    "french", "french for business", "french for professionals", "frontend development", "full stack development", "functions",
    "game design", "game development", "ganache", "gcp", "gcp ml tools", "gdpr",
    "genai basics", "german", "german at work", "german for business", "github actions", "glue",
    "go", "google cardboard", "google cloud", "goroutines", "governance", "graphic design",
    "graphs", "grid", "gui", "hacking labs", "hands-on learning", "hardware",
    "hashing", "hcpcs", "hdl", "hdl programming", "health", "health basics",
    "health economics", "health policy", "healthcare", "healthcare policy", "hr analytics", "hr strategy",
    "html", "html5", "http", "hubspot", "hubspot certification", "hvac",
    "hyperledger", "i/o", "iac", "iam", "iam bypass", "identity",
    "identity design", "ids/ips", "illustration", "immutability", "inbound sales", "inference",
    "infrastructure", "innovation", "input/output", "intellectual property", "interfaces", "intermediate spanish",
    "investment management", "ios", "iot", "iot architecture", "iot deployment", "iot fundamentals",
    "ip addressing", "java", "javascript", "jenkins", "jest", "jetpack compose",
    "jira", "jquery", "jvm", "jwt", "k8s", "kali linux",
    "kanban", "kms", "kotlin", "kubernetes", "lambda", "lambdas",
    "law", "leadership", "legal", "legal tech", "linux", "linux hardening",
    "localization", "logical reasoning", "logo design", "machine learning", "macronutrients", "malware",
    "malware defense", "mandarin", "market entry", "maya", "medical coding", "memory",
    "mental health", "mental health ambassador", "mental wellness", "meta ar", "metadata", "metrics",
    "microcontrollers", "microservices", "mindfulness", "mining", "mixins", "ml",
    "ml basics", "ml optimization", "ml theory", "ml with python", "ml workflow", "mobile",
    "mobile design", "mobile vr", "mocha", "mocks", "modeling", "modern javascript",
    "modern ui", "modern web development", "modifiers", "modules", "mongodb", "monitoring",
    "monopoly theory", "moralis api", "motion design", "multi-tenancy", "multiplatform", "multithreading",
    "mutexes", "mvc", "nesting", "net/http", "netlify", "netmiko",
    "network attacks", "network basics", "network defense", "network design", "networking", "networking fundamentals",
    "neural networks", "neural networks basics", "next.js", "nft minting", "nfts", "node",
    "node.js", "nodes", "numerical methods", "nutrition", "nutrition certification", "nutrition science",
    "oop", "packet analysis", "passwords", "patents", "path planning", "pcb design",
    "pcb layout", "people analytics", "permissions", "photo editing", "photovoltaics", "physics",
    "pipeline", "pipelines", "pivoting", "plc", "pointers", "polymorphism",
    "postgresql", "pow", "power system protection", "presales", "presentation skills", "privacy awareness",
    "privacy certification", "privilege escalation", "probability", "procedure coding", "product demo", "product development",
    "professional certificate", "professional communication", "professional english", "professional french", "professional german", "professional spanish",
    "profiling", "programming", "project tracking", "projects", "promises", "proof-of-work",
    "protection systems", "psychological disorders", "psychology", "psychology basics", "pub/sub", "public health",
    "python", "python ai", "python blockchain", "python ml", "pytorch", "quantum algorithms",
    "quantum basics", "quantum computers", "quantum computing", "quantum engineering", "quantum mechanics", "quantum programming",
    "quantum science", "quantum theory", "queries", "questioning techniques", "rails", "rbac",
    "react", "react native", "react ui", "react.js", "reactivity", "reasoning",
    "reconnaissance", "recursion", "redshift", "redux", "reentrancy", "regression",
    "relational design", "relays", "reliability", "remix", "reporting", "repos",
    "resilience", "responsive design", "responsive ui", "rest", "risk management", "rnns",
    "robot basics", "robot operating system", "robot programming", "robotics", "robotics software", "ros",
    "ros1", "ros2", "routing", "rpc", "ruby", "rust",
    "sales", "sales basics", "sales certification", "sales communication", "sales demo", "sales engineer",
    "sales engineering", "sales essentials", "sales foundations", "sales questions", "sales skills", "sales software",
    "salesforce", "salesforce administrator", "salesforce certification", "sass", "scalability", "scanning",
    "scripting", "scripts", "scrum", "sdl", "search", "secure architecture",
    "secure routing", "security architecture", "security basics", "security controls", "security law", "security patterns",
    "security posture", "segmentation", "selenium", "server administration", "server basics", "server certification",
    "server management", "serverless", "services", "shared code", "shared responsibility", "shell scripting",
    "siem", "simulation", "skills", "sklearn", "slam", "smart contracts",
    "smart pointers", "snapshots", "social ar", "social entrepreneurship", "sockets", "solar energy",
    "solar systems", "solidity", "solidity syntax", "solution consulting", "solutions design", "spanish",
    "spanish certificate", "spanish culture", "spanish for beginners", "spanish for business", "spanish for professionals", "spanish language",
    "spark ar", "spring", "spring boot", "sql", "sqs", "sre",
    "ssh security", "startups", "state", "static generation", "statistics for ds", "stl",
    "storytelling", "strategy", "stress management", "subnetting", "svelte", "swift",
    "system administration", "system design", "systemverilog", "tableau", "tailwind", "talent management",
    "tech startups", "technical sales", "technology law", "templates", "tensorflow", "tensorflow/keras",
    "terraform", "testbench", "testing", "threads", "threat modeling", "threats",
    "tinyml", "traditional animation", "transit gateway", "truffle", "typescript", "typography",
    "ui components", "ui/ux", "uikit", "understanding", "unity", "universal components",
    "unreal engine 5", "user journey", "validation", "variables", "vector graphics", "verification",
    "vhdl", "video editing", "video training", "virtual sales", "virtualization", "vnet",
    "vpc", "vpn", "vpns", "vr apps", "vr design", "vr development",
    "vr games", "vue", "vue.js", "vulnerabilities", "vulnerability scanning", "wallets",
    "wcag", "web design", "web development", "web exploits", "web hacking", "web3.js",
    "wellness", "windows server", "wipo", "workload isolation", "workplace wellness", "xgboost",
    "xr development", "yaml pipelines", "yc methodology", "zoho", "zoho crm"
]


async def populate_topics(
    topics: List[str],
    material_types: List[str] = None,
    max_per_topic: int = 5,
    limit: int = None,
    log_context: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Populate KB with materials for given topics.
    
    Args:
        topics: List of topics to populate
        material_types: Types of materials to discover (default: all)
        max_per_topic: Maximum materials to discover per topic
        limit: Maximum number of topics to process (None = all)
        log_context: Logging context
        
    Returns:
        Dict with statistics about population
    """
    if material_types is None:
        material_types = ["course", "book", "paper", "video", "tutorial"]
    
    if limit:
        topics = topics[:limit]
    
    log_context = log_context or create_log_context("populate_topics", tenant_id="system")
    
    stats = {
        "topics_processed": 0,
        "topics_successful": 0,
        "topics_failed": 0,
        "total_materials_added": 0,
        "materials_by_type": {mt: 0 for mt in material_types}
    }
    
    logger.info(f"🚀 Starting population for {len(topics)} topics...")
    logger.info(f"📚 Material types: {', '.join(material_types)}")
    logger.info(f"📊 Max per topic: {max_per_topic}")
    
    for i, topic in enumerate(topics, 1):
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"📝 Processing topic {i}/{len(topics)}: {topic}")
            logger.info(f"{'='*60}")
            
            # Discover materials for this topic
            discovered = await _check_and_discover_materials(
                topic=topic,
                material_types=material_types,
                min_results_required=2,  # Lower threshold to discover more
                min_similarity_threshold=0.3,  # Lower threshold for broader discovery
                max_results_per_type=max_per_topic,
                log_context=log_context
            )
            
            if discovered:
                # Count materials by type
                for material in discovered:
                    material_type = material.get("type", "unknown")
                    if material_type in stats["materials_by_type"]:
                        stats["materials_by_type"][material_type] += 1
                    stats["total_materials_added"] += 1
                
                stats["topics_successful"] += 1
                logger.info(f"✅ Topic '{topic}': Added {len(discovered)} materials")
            else:
                logger.warning(f"⚠️  Topic '{topic}': No materials discovered")
                stats["topics_failed"] += 1
            
            stats["topics_processed"] += 1
            
            # Progress update every 10 topics
            if i % 10 == 0:
                logger.info(f"\n📊 Progress: {i}/{len(topics)} topics processed")
                logger.info(f"   ✅ Successful: {stats['topics_successful']}")
                logger.info(f"   ❌ Failed: {stats['topics_failed']}")
                logger.info(f"   📚 Total materials: {stats['total_materials_added']}")
            
            # Small delay to avoid rate limiting
            await asyncio.sleep(0.5)
            
        except Exception as e:
            logger.error(f"❌ Error processing topic '{topic}': {e}", exc_info=True)
            stats["topics_failed"] += 1
            stats["topics_processed"] += 1
            continue
    
    return stats


async def main_async():
    """Main async function"""
    parser = argparse.ArgumentParser(description="Populate KB with materials for topics using APIs")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of topics to process (default: all)"
    )
    parser.add_argument(
        "--material-types",
        type=str,
        default="course,book,paper,video,tutorial",
        help="Comma-separated list of material types (default: course,book,paper,video,tutorial)"
    )
    parser.add_argument(
        "--max-per-topic",
        type=int,
        default=5,
        help="Maximum materials to discover per topic (default: 5)"
    )
    parser.add_argument(
        "--topics-file",
        type=str,
        default=None,
        help="Path to file with topics (one per line). If not provided, uses built-in list."
    )
    
    args = parser.parse_args()
    
    # Parse material types
    material_types = [mt.strip() for mt in args.material_types.split(",")]
    
    # Get topics
    if args.topics_file and os.path.exists(args.topics_file):
        logger.info(f"📄 Loading topics from file: {args.topics_file}")
        with open(args.topics_file, 'r', encoding='utf-8') as f:
            topics = [line.strip() for line in f if line.strip()]
    else:
        topics = TOPICS
        logger.info(f"📋 Using built-in topics list ({len(topics)} topics)")
    
    # Create log context
    log_context = create_log_context("populate_topics", tenant_id="system")
    
    # Run population
    stats = await populate_topics(
        topics=topics,
        material_types=material_types,
        max_per_topic=args.max_per_topic,
        limit=args.limit,
        log_context=log_context
    )
    
    # Print final summary
    logger.info("\n" + "="*60)
    logger.info("📊 FINAL SUMMARY")
    logger.info("="*60)
    logger.info(f"Topics processed: {stats['topics_processed']}")
    logger.info(f"Topics successful: {stats['topics_successful']}")
    logger.info(f"Topics failed: {stats['topics_failed']}")
    logger.info(f"Total materials added: {stats['total_materials_added']}")
    logger.info("\nMaterials by type:")
    for material_type, count in stats['materials_by_type'].items():
        logger.info(f"  {material_type}: {count}")
    logger.info("="*60)


def main():
    """Main entry point"""
    try:
        asyncio.run(main_async())
        return 0
    except KeyboardInterrupt:
        logger.info("\n⚠️  Interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())


"""
Populate KB with courses and materials for a specific list of topics using APIs.

This script:
1. Takes a list of topics
2. For each topic, uses auto-discovery to find courses, books, papers, videos, tutorials
3. Adds discovered materials to KB

Usage:
    python scripts/populate_topics_from_list.py [--limit N] [--material-types course,book,paper,video,tutorial] [--max-per-topic N]
"""

import sys
import os
import argparse
import asyncio
import logging
from typing import List, Dict, Any

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.market_and_course_recommender import _check_and_discover_materials
from core.logging_helpers import AgentLogger, create_log_context

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Topics list from user
TOPICS = [
    "2d animation", "2d games", "3d animation", "3d character", "3d games", "3d modeling",
    "5 pillars", "abnormal psychology", "active directory", "actix", "adobe illustrator", "adobe photoshop",
    "advanced english", "advanced german", "advanced javascript", "advanced topics", "advocacy", "after effects",
    "agile", "agile tools", "ai", "ai basics", "ai for robotics", "ai reasoning",
    "algorithms", "analysis methods", "android", "angular", "animation", "animation basics",
    "animation principles", "animations", "anonymity", "api", "api design", "api gateway",
    "apis", "ar filters", "ar/vr development", "aria roles", "arm", "arm cortex",
    "arrays", "artificial intelligence", "async", "async api", "async programming", "async/await",
    "attack types", "audit", "audits", "auth", "automation", "automl",
    "aws", "aws architecture", "aws iot", "aws lambda", "aws services", "azure",
    "azure devops", "azure services", "backend", "basic chinese", "basic cloud concepts", "basic spanish",
    "behavioral economics", "behavioral science", "bem", "best practices", "big data", "blender",
    "blockchain", "blueprints", "board of directors", "bootstrap", "bot framework", "branding",
    "buffer overflows", "business analysis", "business analytics", "business basics", "business chinese", "business english",
    "business ethics", "business french", "business german", "business law", "business spanish", "business vocabulary",
    "c++", "caching", "canva", "career development", "ccna", "ccna prep",
    "certification prep", "cfa", "cfd", "chai", "chaining", "character animation",
    "chinese", "chinese for business", "ci/cd", "cia triad", "cidr", "cipp/e",
    "circuit analysis", "circuit design", "circuit theory", "cisco", "cisco iot", "cisco sales",
    "classes", "classical animation", "cli tools", "closures", "cloud basics", "cloud concepts",
    "cloud exploitation", "cloud functions", "cloud fundamentals", "cloud iot", "cloud ml", "cloud security",
    "cloud threat models", "cloudformation", "cnns", "cognitive services", "collaboration", "collections",
    "color correction", "color theory", "community health", "compliance", "components", "composition",
    "comptia server+", "computer forensics", "computer networking", "computer science", "concurrency", "connected devices",
    "consensus", "consistency", "container orchestration", "container technology", "containerization", "containers",
    "contract law", "control systems", "cooling systems", "copyright", "corporate governance", "corporate strategy",
    "cortex-m3", "cortex-m4", "cpt", "crm", "crm administration", "cross-platform",
    "cryptography", "css", "css optimization", "css3", "cyber attacks", "cyber law",
    "cyber threats", "cybersecurity", "dapp development", "dapps", "dart", "dashboards",
    "data", "data analysis", "data center", "data center operations", "data centers", "data privacy",
    "data protection", "data querying", "data science", "data structures", "data visualization", "davinci resolve",
    "decision making", "deep learning", "defender", "demo skills", "design basics", "design theory",
    "devops", "digital body language", "digital design", "digital experience", "digital forensics", "digital sales",
    "direct connect", "discovery", "distributed computing", "distributed systems", "django", "dl best practices",
    "docker", "docker containers", "dom", "drf", "dynamodb", "economics",
    "edge ai", "electrical controls", "electrical engineering", "electronics", "embedded ai", "embedded design",
    "embedded programming", "embedded systems", "emerging markets", "employment law", "encryption", "engines",
    "english for esl", "enterprise", "entity framework", "entrepreneurship", "enumeration", "erc-721",
    "es6", "es6+", "esl", "essential english", "ethereum", "ethics",
    "event handling", "event loop", "events", "evidence handling", "executors", "experience design",
    "exploitation", "exploitation basics", "expo", "express", "expressroute", "facilities",
    "factory", "fault analysis", "fault tolerance", "financial analysis", "fintech", "firebase",
    "firewalls", "first aid", "flask api", "flexbox", "fluid mechanics", "flutter",
    "forensic imaging", "foundations", "founder skills", "fpga", "fpga development", "frame-by-frame",
    "french", "french for business", "french for professionals", "frontend development", "full stack development", "functions",
    "game design", "game development", "ganache", "gcp", "gcp ml tools", "gdpr",
    "genai basics", "german", "german at work", "german for business", "github actions", "glue",
    "go", "google cardboard", "google cloud", "goroutines", "governance", "graphic design",
    "graphs", "grid", "gui", "hacking labs", "hands-on learning", "hardware",
    "hashing", "hcpcs", "hdl", "hdl programming", "health", "health basics",
    "health economics", "health policy", "healthcare", "healthcare policy", "hr analytics", "hr strategy",
    "html", "html5", "http", "hubspot", "hubspot certification", "hvac",
    "hyperledger", "i/o", "iac", "iam", "iam bypass", "identity",
    "identity design", "ids/ips", "illustration", "immutability", "inbound sales", "inference",
    "infrastructure", "innovation", "input/output", "intellectual property", "interfaces", "intermediate spanish",
    "investment management", "ios", "iot", "iot architecture", "iot deployment", "iot fundamentals",
    "ip addressing", "java", "javascript", "jenkins", "jest", "jetpack compose",
    "jira", "jquery", "jvm", "jwt", "k8s", "kali linux",
    "kanban", "kms", "kotlin", "kubernetes", "lambda", "lambdas",
    "law", "leadership", "legal", "legal tech", "linux", "linux hardening",
    "localization", "logical reasoning", "logo design", "machine learning", "macronutrients", "malware",
    "malware defense", "mandarin", "market entry", "maya", "medical coding", "memory",
    "mental health", "mental health ambassador", "mental wellness", "meta ar", "metadata", "metrics",
    "microcontrollers", "microservices", "mindfulness", "mining", "mixins", "ml",
    "ml basics", "ml optimization", "ml theory", "ml with python", "ml workflow", "mobile",
    "mobile design", "mobile vr", "mocha", "mocks", "modeling", "modern javascript",
    "modern ui", "modern web development", "modifiers", "modules", "mongodb", "monitoring",
    "monopoly theory", "moralis api", "motion design", "multi-tenancy", "multiplatform", "multithreading",
    "mutexes", "mvc", "nesting", "net/http", "netlify", "netmiko",
    "network attacks", "network basics", "network defense", "network design", "networking", "networking fundamentals",
    "neural networks", "neural networks basics", "next.js", "nft minting", "nfts", "node",
    "node.js", "nodes", "numerical methods", "nutrition", "nutrition certification", "nutrition science",
    "oop", "packet analysis", "passwords", "patents", "path planning", "pcb design",
    "pcb layout", "people analytics", "permissions", "photo editing", "photovoltaics", "physics",
    "pipeline", "pipelines", "pivoting", "plc", "pointers", "polymorphism",
    "postgresql", "pow", "power system protection", "presales", "presentation skills", "privacy awareness",
    "privacy certification", "privilege escalation", "probability", "procedure coding", "product demo", "product development",
    "professional certificate", "professional communication", "professional english", "professional french", "professional german", "professional spanish",
    "profiling", "programming", "project tracking", "projects", "promises", "proof-of-work",
    "protection systems", "psychological disorders", "psychology", "psychology basics", "pub/sub", "public health",
    "python", "python ai", "python blockchain", "python ml", "pytorch", "quantum algorithms",
    "quantum basics", "quantum computers", "quantum computing", "quantum engineering", "quantum mechanics", "quantum programming",
    "quantum science", "quantum theory", "queries", "questioning techniques", "rails", "rbac",
    "react", "react native", "react ui", "react.js", "reactivity", "reasoning",
    "reconnaissance", "recursion", "redshift", "redux", "reentrancy", "regression",
    "relational design", "relays", "reliability", "remix", "reporting", "repos",
    "resilience", "responsive design", "responsive ui", "rest", "risk management", "rnns",
    "robot basics", "robot operating system", "robot programming", "robotics", "robotics software", "ros",
    "ros1", "ros2", "routing", "rpc", "ruby", "rust",
    "sales", "sales basics", "sales certification", "sales communication", "sales demo", "sales engineer",
    "sales engineering", "sales essentials", "sales foundations", "sales questions", "sales skills", "sales software",
    "salesforce", "salesforce administrator", "salesforce certification", "sass", "scalability", "scanning",
    "scripting", "scripts", "scrum", "sdl", "search", "secure architecture",
    "secure routing", "security architecture", "security basics", "security controls", "security law", "security patterns",
    "security posture", "segmentation", "selenium", "server administration", "server basics", "server certification",
    "server management", "serverless", "services", "shared code", "shared responsibility", "shell scripting",
    "siem", "simulation", "skills", "sklearn", "slam", "smart contracts",
    "smart pointers", "snapshots", "social ar", "social entrepreneurship", "sockets", "solar energy",
    "solar systems", "solidity", "solidity syntax", "solution consulting", "solutions design", "spanish",
    "spanish certificate", "spanish culture", "spanish for beginners", "spanish for business", "spanish for professionals", "spanish language",
    "spark ar", "spring", "spring boot", "sql", "sqs", "sre",
    "ssh security", "startups", "state", "static generation", "statistics for ds", "stl",
    "storytelling", "strategy", "stress management", "subnetting", "svelte", "swift",
    "system administration", "system design", "systemverilog", "tableau", "tailwind", "talent management",
    "tech startups", "technical sales", "technology law", "templates", "tensorflow", "tensorflow/keras",
    "terraform", "testbench", "testing", "threads", "threat modeling", "threats",
    "tinyml", "traditional animation", "transit gateway", "truffle", "typescript", "typography",
    "ui components", "ui/ux", "uikit", "understanding", "unity", "universal components",
    "unreal engine 5", "user journey", "validation", "variables", "vector graphics", "verification",
    "vhdl", "video editing", "video training", "virtual sales", "virtualization", "vnet",
    "vpc", "vpn", "vpns", "vr apps", "vr design", "vr development",
    "vr games", "vue", "vue.js", "vulnerabilities", "vulnerability scanning", "wallets",
    "wcag", "web design", "web development", "web exploits", "web hacking", "web3.js",
    "wellness", "windows server", "wipo", "workload isolation", "workplace wellness", "xgboost",
    "xr development", "yaml pipelines", "yc methodology", "zoho", "zoho crm"
]


async def populate_topics(
    topics: List[str],
    material_types: List[str] = None,
    max_per_topic: int = 5,
    limit: int = None,
    log_context: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Populate KB with materials for given topics.
    
    Args:
        topics: List of topics to populate
        material_types: Types of materials to discover (default: all)
        max_per_topic: Maximum materials to discover per topic
        limit: Maximum number of topics to process (None = all)
        log_context: Logging context
        
    Returns:
        Dict with statistics about population
    """
    if material_types is None:
        material_types = ["course", "book", "paper", "video", "tutorial"]
    
    if limit:
        topics = topics[:limit]
    
    log_context = log_context or create_log_context("populate_topics", tenant_id="system")
    
    stats = {
        "topics_processed": 0,
        "topics_successful": 0,
        "topics_failed": 0,
        "total_materials_added": 0,
        "materials_by_type": {mt: 0 for mt in material_types}
    }
    
    logger.info(f"🚀 Starting population for {len(topics)} topics...")
    logger.info(f"📚 Material types: {', '.join(material_types)}")
    logger.info(f"📊 Max per topic: {max_per_topic}")
    
    for i, topic in enumerate(topics, 1):
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"📝 Processing topic {i}/{len(topics)}: {topic}")
            logger.info(f"{'='*60}")
            
            # Discover materials for this topic
            discovered = await _check_and_discover_materials(
                topic=topic,
                material_types=material_types,
                min_results_required=2,  # Lower threshold to discover more
                min_similarity_threshold=0.3,  # Lower threshold for broader discovery
                max_results_per_type=max_per_topic,
                log_context=log_context
            )
            
            if discovered:
                # Count materials by type
                for material in discovered:
                    material_type = material.get("type", "unknown")
                    if material_type in stats["materials_by_type"]:
                        stats["materials_by_type"][material_type] += 1
                    stats["total_materials_added"] += 1
                
                stats["topics_successful"] += 1
                logger.info(f"✅ Topic '{topic}': Added {len(discovered)} materials")
            else:
                logger.warning(f"⚠️  Topic '{topic}': No materials discovered")
                stats["topics_failed"] += 1
            
            stats["topics_processed"] += 1
            
            # Progress update every 10 topics
            if i % 10 == 0:
                logger.info(f"\n📊 Progress: {i}/{len(topics)} topics processed")
                logger.info(f"   ✅ Successful: {stats['topics_successful']}")
                logger.info(f"   ❌ Failed: {stats['topics_failed']}")
                logger.info(f"   📚 Total materials: {stats['total_materials_added']}")
            
            # Small delay to avoid rate limiting
            await asyncio.sleep(0.5)
            
        except Exception as e:
            logger.error(f"❌ Error processing topic '{topic}': {e}", exc_info=True)
            stats["topics_failed"] += 1
            stats["topics_processed"] += 1
            continue
    
    return stats


async def main_async():
    """Main async function"""
    parser = argparse.ArgumentParser(description="Populate KB with materials for topics using APIs")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of topics to process (default: all)"
    )
    parser.add_argument(
        "--material-types",
        type=str,
        default="course,book,paper,video,tutorial",
        help="Comma-separated list of material types (default: course,book,paper,video,tutorial)"
    )
    parser.add_argument(
        "--max-per-topic",
        type=int,
        default=5,
        help="Maximum materials to discover per topic (default: 5)"
    )
    parser.add_argument(
        "--topics-file",
        type=str,
        default=None,
        help="Path to file with topics (one per line). If not provided, uses built-in list."
    )
    
    args = parser.parse_args()
    
    # Parse material types
    material_types = [mt.strip() for mt in args.material_types.split(",")]
    
    # Get topics
    if args.topics_file and os.path.exists(args.topics_file):
        logger.info(f"📄 Loading topics from file: {args.topics_file}")
        with open(args.topics_file, 'r', encoding='utf-8') as f:
            topics = [line.strip() for line in f if line.strip()]
    else:
        topics = TOPICS
        logger.info(f"📋 Using built-in topics list ({len(topics)} topics)")
    
    # Create log context
    log_context = create_log_context("populate_topics", tenant_id="system")
    
    # Run population
    stats = await populate_topics(
        topics=topics,
        material_types=material_types,
        max_per_topic=args.max_per_topic,
        limit=args.limit,
        log_context=log_context
    )
    
    # Print final summary
    logger.info("\n" + "="*60)
    logger.info("📊 FINAL SUMMARY")
    logger.info("="*60)
    logger.info(f"Topics processed: {stats['topics_processed']}")
    logger.info(f"Topics successful: {stats['topics_successful']}")
    logger.info(f"Topics failed: {stats['topics_failed']}")
    logger.info(f"Total materials added: {stats['total_materials_added']}")
    logger.info("\nMaterials by type:")
    for material_type, count in stats['materials_by_type'].items():
        logger.info(f"  {material_type}: {count}")
    logger.info("="*60)


def main():
    """Main entry point"""
    try:
        asyncio.run(main_async())
        return 0
    except KeyboardInterrupt:
        logger.info("\n⚠️  Interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())


"""
Populate KB with courses and materials for a specific list of topics using APIs.

This script:
1. Takes a list of topics
2. For each topic, uses auto-discovery to find courses, books, papers, videos, tutorials
3. Adds discovered materials to KB

Usage:
    python scripts/populate_topics_from_list.py [--limit N] [--material-types course,book,paper,video,tutorial] [--max-per-topic N]
"""

import sys
import os
import argparse
import asyncio
import logging
from typing import List, Dict, Any

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.market_and_course_recommender import _check_and_discover_materials
from core.logging_helpers import AgentLogger, create_log_context

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Topics list from user
TOPICS = [
    "2d animation", "2d games", "3d animation", "3d character", "3d games", "3d modeling",
    "5 pillars", "abnormal psychology", "active directory", "actix", "adobe illustrator", "adobe photoshop",
    "advanced english", "advanced german", "advanced javascript", "advanced topics", "advocacy", "after effects",
    "agile", "agile tools", "ai", "ai basics", "ai for robotics", "ai reasoning",
    "algorithms", "analysis methods", "android", "angular", "animation", "animation basics",
    "animation principles", "animations", "anonymity", "api", "api design", "api gateway",
    "apis", "ar filters", "ar/vr development", "aria roles", "arm", "arm cortex",
    "arrays", "artificial intelligence", "async", "async api", "async programming", "async/await",
    "attack types", "audit", "audits", "auth", "automation", "automl",
    "aws", "aws architecture", "aws iot", "aws lambda", "aws services", "azure",
    "azure devops", "azure services", "backend", "basic chinese", "basic cloud concepts", "basic spanish",
    "behavioral economics", "behavioral science", "bem", "best practices", "big data", "blender",
    "blockchain", "blueprints", "board of directors", "bootstrap", "bot framework", "branding",
    "buffer overflows", "business analysis", "business analytics", "business basics", "business chinese", "business english",
    "business ethics", "business french", "business german", "business law", "business spanish", "business vocabulary",
    "c++", "caching", "canva", "career development", "ccna", "ccna prep",
    "certification prep", "cfa", "cfd", "chai", "chaining", "character animation",
    "chinese", "chinese for business", "ci/cd", "cia triad", "cidr", "cipp/e",
    "circuit analysis", "circuit design", "circuit theory", "cisco", "cisco iot", "cisco sales",
    "classes", "classical animation", "cli tools", "closures", "cloud basics", "cloud concepts",
    "cloud exploitation", "cloud functions", "cloud fundamentals", "cloud iot", "cloud ml", "cloud security",
    "cloud threat models", "cloudformation", "cnns", "cognitive services", "collaboration", "collections",
    "color correction", "color theory", "community health", "compliance", "components", "composition",
    "comptia server+", "computer forensics", "computer networking", "computer science", "concurrency", "connected devices",
    "consensus", "consistency", "container orchestration", "container technology", "containerization", "containers",
    "contract law", "control systems", "cooling systems", "copyright", "corporate governance", "corporate strategy",
    "cortex-m3", "cortex-m4", "cpt", "crm", "crm administration", "cross-platform",
    "cryptography", "css", "css optimization", "css3", "cyber attacks", "cyber law",
    "cyber threats", "cybersecurity", "dapp development", "dapps", "dart", "dashboards",
    "data", "data analysis", "data center", "data center operations", "data centers", "data privacy",
    "data protection", "data querying", "data science", "data structures", "data visualization", "davinci resolve",
    "decision making", "deep learning", "defender", "demo skills", "design basics", "design theory",
    "devops", "digital body language", "digital design", "digital experience", "digital forensics", "digital sales",
    "direct connect", "discovery", "distributed computing", "distributed systems", "django", "dl best practices",
    "docker", "docker containers", "dom", "drf", "dynamodb", "economics",
    "edge ai", "electrical controls", "electrical engineering", "electronics", "embedded ai", "embedded design",
    "embedded programming", "embedded systems", "emerging markets", "employment law", "encryption", "engines",
    "english for esl", "enterprise", "entity framework", "entrepreneurship", "enumeration", "erc-721",
    "es6", "es6+", "esl", "essential english", "ethereum", "ethics",
    "event handling", "event loop", "events", "evidence handling", "executors", "experience design",
    "exploitation", "exploitation basics", "expo", "express", "expressroute", "facilities",
    "factory", "fault analysis", "fault tolerance", "financial analysis", "fintech", "firebase",
    "firewalls", "first aid", "flask api", "flexbox", "fluid mechanics", "flutter",
    "forensic imaging", "foundations", "founder skills", "fpga", "fpga development", "frame-by-frame",
    "french", "french for business", "french for professionals", "frontend development", "full stack development", "functions",
    "game design", "game development", "ganache", "gcp", "gcp ml tools", "gdpr",
    "genai basics", "german", "german at work", "german for business", "github actions", "glue",
    "go", "google cardboard", "google cloud", "goroutines", "governance", "graphic design",
    "graphs", "grid", "gui", "hacking labs", "hands-on learning", "hardware",
    "hashing", "hcpcs", "hdl", "hdl programming", "health", "health basics",
    "health economics", "health policy", "healthcare", "healthcare policy", "hr analytics", "hr strategy",
    "html", "html5", "http", "hubspot", "hubspot certification", "hvac",
    "hyperledger", "i/o", "iac", "iam", "iam bypass", "identity",
    "identity design", "ids/ips", "illustration", "immutability", "inbound sales", "inference",
    "infrastructure", "innovation", "input/output", "intellectual property", "interfaces", "intermediate spanish",
    "investment management", "ios", "iot", "iot architecture", "iot deployment", "iot fundamentals",
    "ip addressing", "java", "javascript", "jenkins", "jest", "jetpack compose",
    "jira", "jquery", "jvm", "jwt", "k8s", "kali linux",
    "kanban", "kms", "kotlin", "kubernetes", "lambda", "lambdas",
    "law", "leadership", "legal", "legal tech", "linux", "linux hardening",
    "localization", "logical reasoning", "logo design", "machine learning", "macronutrients", "malware",
    "malware defense", "mandarin", "market entry", "maya", "medical coding", "memory",
    "mental health", "mental health ambassador", "mental wellness", "meta ar", "metadata", "metrics",
    "microcontrollers", "microservices", "mindfulness", "mining", "mixins", "ml",
    "ml basics", "ml optimization", "ml theory", "ml with python", "ml workflow", "mobile",
    "mobile design", "mobile vr", "mocha", "mocks", "modeling", "modern javascript",
    "modern ui", "modern web development", "modifiers", "modules", "mongodb", "monitoring",
    "monopoly theory", "moralis api", "motion design", "multi-tenancy", "multiplatform", "multithreading",
    "mutexes", "mvc", "nesting", "net/http", "netlify", "netmiko",
    "network attacks", "network basics", "network defense", "network design", "networking", "networking fundamentals",
    "neural networks", "neural networks basics", "next.js", "nft minting", "nfts", "node",
    "node.js", "nodes", "numerical methods", "nutrition", "nutrition certification", "nutrition science",
    "oop", "packet analysis", "passwords", "patents", "path planning", "pcb design",
    "pcb layout", "people analytics", "permissions", "photo editing", "photovoltaics", "physics",
    "pipeline", "pipelines", "pivoting", "plc", "pointers", "polymorphism",
    "postgresql", "pow", "power system protection", "presales", "presentation skills", "privacy awareness",
    "privacy certification", "privilege escalation", "probability", "procedure coding", "product demo", "product development",
    "professional certificate", "professional communication", "professional english", "professional french", "professional german", "professional spanish",
    "profiling", "programming", "project tracking", "projects", "promises", "proof-of-work",
    "protection systems", "psychological disorders", "psychology", "psychology basics", "pub/sub", "public health",
    "python", "python ai", "python blockchain", "python ml", "pytorch", "quantum algorithms",
    "quantum basics", "quantum computers", "quantum computing", "quantum engineering", "quantum mechanics", "quantum programming",
    "quantum science", "quantum theory", "queries", "questioning techniques", "rails", "rbac",
    "react", "react native", "react ui", "react.js", "reactivity", "reasoning",
    "reconnaissance", "recursion", "redshift", "redux", "reentrancy", "regression",
    "relational design", "relays", "reliability", "remix", "reporting", "repos",
    "resilience", "responsive design", "responsive ui", "rest", "risk management", "rnns",
    "robot basics", "robot operating system", "robot programming", "robotics", "robotics software", "ros",
    "ros1", "ros2", "routing", "rpc", "ruby", "rust",
    "sales", "sales basics", "sales certification", "sales communication", "sales demo", "sales engineer",
    "sales engineering", "sales essentials", "sales foundations", "sales questions", "sales skills", "sales software",
    "salesforce", "salesforce administrator", "salesforce certification", "sass", "scalability", "scanning",
    "scripting", "scripts", "scrum", "sdl", "search", "secure architecture",
    "secure routing", "security architecture", "security basics", "security controls", "security law", "security patterns",
    "security posture", "segmentation", "selenium", "server administration", "server basics", "server certification",
    "server management", "serverless", "services", "shared code", "shared responsibility", "shell scripting",
    "siem", "simulation", "skills", "sklearn", "slam", "smart contracts",
    "smart pointers", "snapshots", "social ar", "social entrepreneurship", "sockets", "solar energy",
    "solar systems", "solidity", "solidity syntax", "solution consulting", "solutions design", "spanish",
    "spanish certificate", "spanish culture", "spanish for beginners", "spanish for business", "spanish for professionals", "spanish language",
    "spark ar", "spring", "spring boot", "sql", "sqs", "sre",
    "ssh security", "startups", "state", "static generation", "statistics for ds", "stl",
    "storytelling", "strategy", "stress management", "subnetting", "svelte", "swift",
    "system administration", "system design", "systemverilog", "tableau", "tailwind", "talent management",
    "tech startups", "technical sales", "technology law", "templates", "tensorflow", "tensorflow/keras",
    "terraform", "testbench", "testing", "threads", "threat modeling", "threats",
    "tinyml", "traditional animation", "transit gateway", "truffle", "typescript", "typography",
    "ui components", "ui/ux", "uikit", "understanding", "unity", "universal components",
    "unreal engine 5", "user journey", "validation", "variables", "vector graphics", "verification",
    "vhdl", "video editing", "video training", "virtual sales", "virtualization", "vnet",
    "vpc", "vpn", "vpns", "vr apps", "vr design", "vr development",
    "vr games", "vue", "vue.js", "vulnerabilities", "vulnerability scanning", "wallets",
    "wcag", "web design", "web development", "web exploits", "web hacking", "web3.js",
    "wellness", "windows server", "wipo", "workload isolation", "workplace wellness", "xgboost",
    "xr development", "yaml pipelines", "yc methodology", "zoho", "zoho crm"
]


async def populate_topics(
    topics: List[str],
    material_types: List[str] = None,
    max_per_topic: int = 5,
    limit: int = None,
    log_context: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Populate KB with materials for given topics.
    
    Args:
        topics: List of topics to populate
        material_types: Types of materials to discover (default: all)
        max_per_topic: Maximum materials to discover per topic
        limit: Maximum number of topics to process (None = all)
        log_context: Logging context
        
    Returns:
        Dict with statistics about population
    """
    if material_types is None:
        material_types = ["course", "book", "paper", "video", "tutorial"]
    
    if limit:
        topics = topics[:limit]
    
    log_context = log_context or create_log_context("populate_topics", tenant_id="system")
    
    stats = {
        "topics_processed": 0,
        "topics_successful": 0,
        "topics_failed": 0,
        "total_materials_added": 0,
        "materials_by_type": {mt: 0 for mt in material_types}
    }
    
    logger.info(f"🚀 Starting population for {len(topics)} topics...")
    logger.info(f"📚 Material types: {', '.join(material_types)}")
    logger.info(f"📊 Max per topic: {max_per_topic}")
    
    for i, topic in enumerate(topics, 1):
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"📝 Processing topic {i}/{len(topics)}: {topic}")
            logger.info(f"{'='*60}")
            
            # Discover materials for this topic
            discovered = await _check_and_discover_materials(
                topic=topic,
                material_types=material_types,
                min_results_required=2,  # Lower threshold to discover more
                min_similarity_threshold=0.3,  # Lower threshold for broader discovery
                max_results_per_type=max_per_topic,
                log_context=log_context
            )
            
            if discovered:
                # Count materials by type
                for material in discovered:
                    material_type = material.get("type", "unknown")
                    if material_type in stats["materials_by_type"]:
                        stats["materials_by_type"][material_type] += 1
                    stats["total_materials_added"] += 1
                
                stats["topics_successful"] += 1
                logger.info(f"✅ Topic '{topic}': Added {len(discovered)} materials")
            else:
                logger.warning(f"⚠️  Topic '{topic}': No materials discovered")
                stats["topics_failed"] += 1
            
            stats["topics_processed"] += 1
            
            # Progress update every 10 topics
            if i % 10 == 0:
                logger.info(f"\n📊 Progress: {i}/{len(topics)} topics processed")
                logger.info(f"   ✅ Successful: {stats['topics_successful']}")
                logger.info(f"   ❌ Failed: {stats['topics_failed']}")
                logger.info(f"   📚 Total materials: {stats['total_materials_added']}")
            
            # Small delay to avoid rate limiting
            await asyncio.sleep(0.5)
            
        except Exception as e:
            logger.error(f"❌ Error processing topic '{topic}': {e}", exc_info=True)
            stats["topics_failed"] += 1
            stats["topics_processed"] += 1
            continue
    
    return stats


async def main_async():
    """Main async function"""
    parser = argparse.ArgumentParser(description="Populate KB with materials for topics using APIs")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of topics to process (default: all)"
    )
    parser.add_argument(
        "--material-types",
        type=str,
        default="course,book,paper,video,tutorial",
        help="Comma-separated list of material types (default: course,book,paper,video,tutorial)"
    )
    parser.add_argument(
        "--max-per-topic",
        type=int,
        default=5,
        help="Maximum materials to discover per topic (default: 5)"
    )
    parser.add_argument(
        "--topics-file",
        type=str,
        default=None,
        help="Path to file with topics (one per line). If not provided, uses built-in list."
    )
    
    args = parser.parse_args()
    
    # Parse material types
    material_types = [mt.strip() for mt in args.material_types.split(",")]
    
    # Get topics
    if args.topics_file and os.path.exists(args.topics_file):
        logger.info(f"📄 Loading topics from file: {args.topics_file}")
        with open(args.topics_file, 'r', encoding='utf-8') as f:
            topics = [line.strip() for line in f if line.strip()]
    else:
        topics = TOPICS
        logger.info(f"📋 Using built-in topics list ({len(topics)} topics)")
    
    # Create log context
    log_context = create_log_context("populate_topics", tenant_id="system")
    
    # Run population
    stats = await populate_topics(
        topics=topics,
        material_types=material_types,
        max_per_topic=args.max_per_topic,
        limit=args.limit,
        log_context=log_context
    )
    
    # Print final summary
    logger.info("\n" + "="*60)
    logger.info("📊 FINAL SUMMARY")
    logger.info("="*60)
    logger.info(f"Topics processed: {stats['topics_processed']}")
    logger.info(f"Topics successful: {stats['topics_successful']}")
    logger.info(f"Topics failed: {stats['topics_failed']}")
    logger.info(f"Total materials added: {stats['total_materials_added']}")
    logger.info("\nMaterials by type:")
    for material_type, count in stats['materials_by_type'].items():
        logger.info(f"  {material_type}: {count}")
    logger.info("="*60)


def main():
    """Main entry point"""
    try:
        asyncio.run(main_async())
        return 0
    except KeyboardInterrupt:
        logger.info("\n⚠️  Interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())



