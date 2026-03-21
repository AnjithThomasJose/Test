"""
Versioned Prompt Store for KAFIN Agents

This module provides comprehensive prompt management with:
- Version control and history tracking
- Diff generation and rollback capabilities
- A/B testing support
- Performance tracking per version
- Centralized prompt repository
"""

import json
import hashlib
import difflib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

class PromptType(Enum):
    """Types of prompts"""
    ASSESSMENT_GENERATION = "assessment_generation"
    ASSESSMENT_EVALUATION = "assessment_evaluation"
    RESUME_ANALYSIS = "resume_analysis"
    CODING = "coding"
    INTERVIEW = "interview"
    REPORT_GENERATION = "report_generation"
    SKILL_EXTRACTION = "skill_extraction"
    EDUCATION_PARSING = "education_parsing"
    EXPERIENCE_PARSING = "experience_parsing"
    GENERIC = "generic"

class PromptStatus(Enum):
    """Prompt version status"""
    DRAFT = "draft"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    TESTING = "testing"

@dataclass
class PromptVersion:
    """Represents a version of a prompt"""
    version: str
    content: str
    prompt_type: PromptType
    created_at: datetime
    author: str
    description: str
    tags: List[str] = field(default_factory=list)
    hash: str = ""
    status: PromptStatus = PromptStatus.DRAFT
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.hash:
            self.hash = hashlib.sha256(self.content.encode()).hexdigest()[:16]

@dataclass
class PromptDiff:
    """Represents differences between prompt versions"""
    version1: str
    version2: str
    added_lines: List[str]
    removed_lines: List[str]
    modified_lines: List[tuple]
    similarity_score: float
    created_at: datetime

class PromptStore:
    """Centralized prompt store with versioning"""
    
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path or os.path.join(os.path.dirname(__file__), "prompt_store")
        self.prompts: Dict[str, Dict[str, PromptVersion]] = {}
        self.current_versions: Dict[str, str] = {}
        self.prompt_types: Dict[str, PromptType] = {}
        self.performance_history: Dict[str, List[Dict[str, Any]]] = {}
        
        # Ensure storage directory exists
        Path(self.storage_path).mkdir(parents=True, exist_ok=True)
        
        # Load existing prompts
        self._load_prompts()
    
    def add_prompt(self, name: str, content: str, prompt_type: Union[PromptType, str],
                   version: str = "v1.0", author: str = "system", 
                   description: str = "", tags: List[str] = None,
                   status: PromptStatus = PromptStatus.DRAFT,
                   metadata: Dict[str, Any] = None) -> str:
        """Add new prompt version with automatic versioning"""
        if isinstance(prompt_type, str):
            try:
                prompt_type = PromptType(prompt_type)
            except ValueError:
                prompt_type = PromptType.GENERIC
        
        # Generate version if not provided
        if version == "v1.0" and name in self.prompts:
            version = self._generate_next_version(name)
        
        prompt_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        
        if name not in self.prompts:
            self.prompts[name] = {}
        
        prompt_version = PromptVersion(
            version=version,
            content=content,
            prompt_type=prompt_type,
            created_at=datetime.now(),
            author=author,
            description=description,
            tags=tags or [],
            hash=prompt_hash,
            status=status,
            metadata=metadata or {}
        )
        
        self.prompts[name][version] = prompt_version
        self.prompt_types[name] = prompt_type
        
        # Set as current version if it's the first version or explicitly active
        if version not in self.current_versions or status == PromptStatus.ACTIVE:
            self.current_versions[name] = version
        
        # Save to disk
        self._save_prompts()
        
        log.info(f"Added prompt '{name}' version '{version}' by {author}")
        return version
    
    def get_prompt(self, name: str, version: Optional[str] = None) -> str:
        """Get prompt content with version support"""
        if name not in self.prompts:
            raise ValueError(f"Prompt '{name}' not found")
        
        target_version = version or self.current_versions.get(name)
        if not target_version:
            raise ValueError(f"No current version set for prompt '{name}'")
        
        if target_version not in self.prompts[name]:
            raise ValueError(f"Version '{target_version}' not found for prompt '{name}'")
        
        return self.prompts[name][target_version].content
    
    def get_prompt_version(self, name: str, version: Optional[str] = None) -> PromptVersion:
        """Get full prompt version object"""
        if name not in self.prompts:
            raise ValueError(f"Prompt '{name}' not found")
        
        target_version = version or self.current_versions.get(name)
        if not target_version:
            raise ValueError(f"No current version set for prompt '{name}'")
        
        if target_version not in self.prompts[name]:
            raise ValueError(f"Version '{target_version}' not found for prompt '{name}'")
        
        return self.prompts[name][target_version]
    
    def get_prompt_diff(self, name: str, version1: str, version2: str) -> PromptDiff:
        """Get diff between prompt versions"""
        if name not in self.prompts:
            raise ValueError(f"Prompt '{name}' not found")
        
        if version1 not in self.prompts[name] or version2 not in self.prompts[name]:
            raise ValueError(f"One or both versions not found for prompt '{name}'")
        
        content1 = self.prompts[name][version1].content
        content2 = self.prompts[name][version2].content
        
        lines1 = content1.splitlines()
        lines2 = content2.splitlines()
        
        # Generate diff
        diff = list(difflib.unified_diff(lines1, lines2, lineterm=''))
        
        added_lines = []
        removed_lines = []
        modified_lines = []
        
        for line in diff:
            if line.startswith('+') and not line.startswith('+++'):
                added_lines.append(line[1:])
            elif line.startswith('-') and not line.startswith('---'):
                removed_lines.append(line[1:])
            elif line.startswith('@@'):
                # Parse hunk headers for modified lines
                continue
        
        # Calculate similarity score
        similarity_score = difflib.SequenceMatcher(None, content1, content2).ratio()
        
        return PromptDiff(
            version1=version1,
            version2=version2,
            added_lines=added_lines,
            removed_lines=removed_lines,
            modified_lines=modified_lines,
            similarity_score=similarity_score,
            created_at=datetime.now()
        )
    
    def rollback_prompt(self, name: str, target_version: str) -> bool:
        """Rollback to previous prompt version"""
        if name not in self.prompts:
            log.error(f"Cannot rollback: prompt '{name}' not found")
            return False
        
        if target_version not in self.prompts[name]:
            log.error(f"Cannot rollback: version '{target_version}' not found for prompt '{name}'")
            return False
        
        old_version = self.current_versions.get(name)
        self.current_versions[name] = target_version
        
        # Save to disk
        self._save_prompts()
        
        log.info(f"Rolled back prompt '{name}' from '{old_version}' to '{target_version}'")
        return True
    
    def set_active_version(self, name: str, version: str) -> bool:
        """Set active version for a prompt"""
        if name not in self.prompts:
            log.error(f"Cannot set active version: prompt '{name}' not found")
            return False
        
        if version not in self.prompts[name]:
            log.error(f"Cannot set active version: version '{version}' not found for prompt '{name}'")
            return False
        
        self.current_versions[name] = version
        self.prompts[name][version].status = PromptStatus.ACTIVE
        
        # Save to disk
        self._save_prompts()
        
        log.info(f"Set active version '{version}' for prompt '{name}'")
        return True
    
    def list_prompts(self, prompt_type: Optional[PromptType] = None) -> List[Dict[str, Any]]:
        """List all prompts, optionally filtered by type"""
        prompts = []
        for name, versions in self.prompts.items():
            if prompt_type and self.prompt_types.get(name) != prompt_type:
                continue
            
            current_version = self.current_versions.get(name)
            latest_version = max(versions.keys(), key=lambda v: versions[v].created_at)
            
            prompts.append({
                "name": name,
                "type": self.prompt_types.get(name, PromptType.GENERIC).value,
                "current_version": current_version,
                "latest_version": latest_version,
                "total_versions": len(versions),
                "status": versions[current_version].status.value if current_version else "unknown",
                "created_at": versions[latest_version].created_at.isoformat(),
                "author": versions[latest_version].author,
                "description": versions[latest_version].description,
                "tags": versions[latest_version].tags
            })
        
        return sorted(prompts, key=lambda x: x["created_at"], reverse=True)
    
    def list_versions(self, name: str) -> List[Dict[str, Any]]:
        """List all versions of a specific prompt"""
        if name not in self.prompts:
            return []
        
        versions = []
        for version, prompt_version in self.prompts[name].items():
            versions.append({
                "version": version,
                "status": prompt_version.status.value,
                "created_at": prompt_version.created_at.isoformat(),
                "author": prompt_version.author,
                "description": prompt_version.description,
                "hash": prompt_version.hash,
                "tags": prompt_version.tags,
                "is_current": version == self.current_versions.get(name),
                "performance_metrics": prompt_version.performance_metrics
            })
        
        return sorted(versions, key=lambda x: x["created_at"], reverse=True)
    
    def update_performance_metrics(self, name: str, version: str, 
                                 metrics: Dict[str, Any]):
        """Update performance metrics for a prompt version"""
        if name not in self.prompts or version not in self.prompts[name]:
            log.error(f"Cannot update metrics: prompt '{name}' version '{version}' not found")
            return
        
        prompt_version = self.prompts[name][version]
        prompt_version.performance_metrics.update(metrics)
        
        # Track in performance history
        if name not in self.performance_history:
            self.performance_history[name] = []
        
        self.performance_history[name].append({
            "version": version,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics
        })
        
        # Save to disk
        self._save_prompts()
        
        log.debug(f"Updated performance metrics for '{name}' version '{version}'")
    
    def get_performance_history(self, name: str, version: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get performance history for a prompt"""
        if name not in self.performance_history:
            return []
        
        history = self.performance_history[name]
        if version:
            history = [h for h in history if h["version"] == version]
        
        return sorted(history, key=lambda x: x["timestamp"], reverse=True)
    
    def search_prompts(self, query: str, search_content: bool = True) -> List[Dict[str, Any]]:
        """Search prompts by name, description, or content"""
        results = []
        query_lower = query.lower()
        
        for name, versions in self.prompts.items():
            current_version = self.current_versions.get(name)
            if not current_version:
                continue
            
            prompt_version = versions[current_version]
            
            # Search in name
            if query_lower in name.lower():
                results.append({
                    "name": name,
                    "version": current_version,
                    "match_type": "name",
                    "match_text": name,
                    "description": prompt_version.description,
                    "tags": prompt_version.tags
                })
                continue
            
            # Search in description
            if query_lower in prompt_version.description.lower():
                results.append({
                    "name": name,
                    "version": current_version,
                    "match_type": "description",
                    "match_text": prompt_version.description,
                    "description": prompt_version.description,
                    "tags": prompt_version.tags
                })
                continue
            
            # Search in content
            if search_content and query_lower in prompt_version.content.lower():
                # Find matching lines
                lines = prompt_version.content.splitlines()
                matching_lines = [line for line in lines if query_lower in line.lower()]
                
                results.append({
                    "name": name,
                    "version": current_version,
                    "match_type": "content",
                    "match_text": matching_lines[0] if matching_lines else "",
                    "description": prompt_version.description,
                    "tags": prompt_version.tags
                })
        
        return results
    
    def _generate_next_version(self, name: str) -> str:
        """Generate next version number for a prompt"""
        if name not in self.prompts:
            return "v1.0"
        
        versions = list(self.prompts[name].keys())
        version_numbers = []
        
        for version in versions:
            try:
                if version.startswith('v'):
                    num = float(version[1:])
                    version_numbers.append(num)
            except ValueError:
                continue
        
        if not version_numbers:
            return "v1.0"
        
        next_version = max(version_numbers) + 0.1
        return f"v{next_version:.1f}"
    
    def _save_prompts(self):
        """Save prompts to disk"""
        try:
            data = {
                "prompts": {},
                "current_versions": self.current_versions,
                "prompt_types": {name: pt.value for name, pt in self.prompt_types.items()},
                "performance_history": self.performance_history
            }
            
            # Convert prompts to serializable format
            for name, versions in self.prompts.items():
                data["prompts"][name] = {}
                for version, prompt_version in versions.items():
                    data["prompts"][name][version] = {
                        "version": prompt_version.version,
                        "content": prompt_version.content,
                        "prompt_type": prompt_version.prompt_type.value,
                        "created_at": prompt_version.created_at.isoformat(),
                        "author": prompt_version.author,
                        "description": prompt_version.description,
                        "tags": prompt_version.tags,
                        "hash": prompt_version.hash,
                        "status": prompt_version.status.value,
                        "performance_metrics": prompt_version.performance_metrics,
                        "metadata": prompt_version.metadata
                    }
            
            file_path = os.path.join(self.storage_path, "prompt_store.json")
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            log.error(f"Failed to save prompts: {e}")
    
    def _load_prompts(self):
        """Load prompts from disk"""
        try:
            file_path = os.path.join(self.storage_path, "prompt_store.json")
            if not os.path.exists(file_path):
                return
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Load current versions
            self.current_versions = data.get("current_versions", {})
            
            # Load prompt types
            self.prompt_types = {
                name: PromptType(pt) for name, pt in data.get("prompt_types", {}).items()
            }
            
            # Load performance history
            self.performance_history = data.get("performance_history", {})
            
            # Load prompts
            for name, versions in data.get("prompts", {}).items():
                self.prompts[name] = {}
                for version, prompt_data in versions.items():
                    prompt_version = PromptVersion(
                        version=prompt_data["version"],
                        content=prompt_data["content"],
                        prompt_type=PromptType(prompt_data["prompt_type"]),
                        created_at=datetime.fromisoformat(prompt_data["created_at"]),
                        author=prompt_data["author"],
                        description=prompt_data["description"],
                        tags=prompt_data["tags"],
                        hash=prompt_data["hash"],
                        status=PromptStatus(prompt_data["status"]),
                        performance_metrics=prompt_data["performance_metrics"],
                        metadata=prompt_data["metadata"]
                    )
                    self.prompts[name][version] = prompt_version
            
            log.info(f"Loaded {len(self.prompts)} prompts from storage")
            
        except Exception as e:
            log.error(f"Failed to load prompts: {e}")
    
    def export_prompts(self, output_path: str, format: str = "json"):
        """Export all prompts to a file"""
        try:
            if format == "json":
                data = {
                    "prompts": {},
                    "current_versions": self.current_versions,
                    "exported_at": datetime.now().isoformat()
                }
                
                for name, versions in self.prompts.items():
                    data["prompts"][name] = {}
                    for version, prompt_version in versions.items():
                        data["prompts"][name][version] = {
                            "content": prompt_version.content,
                            "prompt_type": prompt_version.prompt_type.value,
                            "created_at": prompt_version.created_at.isoformat(),
                            "author": prompt_version.author,
                            "description": prompt_version.description,
                            "tags": prompt_version.tags,
                            "status": prompt_version.status.value
                        }
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            
            log.info(f"Exported prompts to {output_path}")
            
        except Exception as e:
            log.error(f"Failed to export prompts: {e}")
    
    def import_prompts(self, input_path: str):
        """Import prompts from a file"""
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            imported_count = 0
            for name, versions in data.get("prompts", {}).items():
                for version, prompt_data in versions.items():
                    self.add_prompt(
                        name=name,
                        content=prompt_data["content"],
                        prompt_type=prompt_data.get("prompt_type", "generic"),
                        version=version,
                        author=prompt_data.get("author", "imported"),
                        description=prompt_data.get("description", ""),
                        tags=prompt_data.get("tags", []),
                        status=PromptStatus(prompt_data.get("status", "draft"))
                    )
                    imported_count += 1
            
            log.info(f"Imported {imported_count} prompt versions from {input_path}")
            
        except Exception as e:
            log.error(f"Failed to import prompts: {e}")

# Global instance
prompt_store = PromptStore()

# Convenience functions
def get_prompt(name: str, version: Optional[str] = None) -> str:
    """Get prompt content"""
    return prompt_store.get_prompt(name, version)

def add_prompt(name: str, content: str, prompt_type: Union[PromptType, str],
               version: str = "v1.0", author: str = "system", 
               description: str = "", tags: List[str] = None) -> str:
    """Add new prompt version"""
    return prompt_store.add_prompt(name, content, prompt_type, version, author, description, tags)

def rollback_prompt(name: str, target_version: str) -> bool:
    """Rollback to previous prompt version"""
    return prompt_store.rollback_prompt(name, target_version)

def list_prompts(prompt_type: Optional[PromptType] = None) -> List[Dict[str, Any]]:
    """List all prompts"""
    return prompt_store.list_prompts(prompt_type)

def search_prompts(query: str, search_content: bool = True) -> List[Dict[str, Any]]:
    """Search prompts"""
    return prompt_store.search_prompts(query, search_content)

# Seed a default prompt for step rewriting if not present
try:
    prompt_store.get_prompt("report_steps_rewrite")
except Exception:
    default_rewrite_prompt = (
        "You are enhancing a short plan of learning/improvement steps.\n"
        "Rules:\n"
        "- Keep 4-6 steps, concise and actionable.\n"
        "- No external links unless explicitly provided in resources.\n"
        "- Avoid duplicates. Include concrete examples or milestones.\n"
        "Context: {context_json}.\n"
        "Templates to refine (keep structure, improve clarity):\n{templates_block}\n"
        "Return only a numbered list of steps."
    )
    prompt_store.add_prompt(
        name="report_steps_rewrite",
        content=default_rewrite_prompt,
        prompt_type=PromptType.REPORT_GENERATION,
        version="v1.0",
        author="system",
        description="Rewrite/enrich steps with constraints",
        tags=["rewrite", "report", "steps"],
        status=PromptStatus.ACTIVE,
    )
