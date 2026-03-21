"""
Task queue with prioritization and deadline-based expiration.
Supports P0/P1/P2 priorities and automatic deadline enforcement.
"""
import asyncio
import time
from enum import IntEnum
from dataclasses import dataclass, field
from typing import Callable, Any, Dict, Optional, List
from collections import deque
import heapq

class TaskPriority(IntEnum):
    """Task priority levels (higher number = higher priority)."""
    P2 = 0  # Low priority
    P1 = 1  # Medium priority
    P0 = 2  # High priority

@dataclass
class Task:
    """Represents a task in the queue."""
    task_id: str
    func: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.P1
    deadline: Optional[float] = None  # Unix timestamp
    created_at: float = field(default_factory=time.time)
    retries: int = 0
    max_retries: int = 3
    
    def __lt__(self, other):
        """Priority queue ordering: higher priority first, then earlier deadline."""
        if self.priority != other.priority:
            return self.priority > other.priority
        if self.deadline and other.deadline:
            return self.deadline < other.deadline
        if self.deadline:
            return True
        if other.deadline:
            return False
        return self.created_at < other.created_at
    
    def is_expired(self) -> bool:
        """Check if task has expired."""
        if self.deadline is None:
            return False
        return time.time() > self.deadline

class TaskQueue:
    """Priority queue for tasks with deadline-based expiration."""
    
    def __init__(self):
        self._queue: List[Task] = []
        self._task_map: Dict[str, Task] = {}
        self._lock = asyncio.Lock()
        self._processing = False
    
    async def enqueue(
        self,
        task_id: str,
        func: Callable,
        priority: TaskPriority = TaskPriority.P1,
        deadline_seconds: Optional[float] = None,
        max_retries: int = 3,
        *args,
        **kwargs
    ) -> str:
        """Add a task to the queue."""
        deadline = None
        if deadline_seconds:
            deadline = time.time() + deadline_seconds
        
        task = Task(
            task_id=task_id,
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            deadline=deadline,
            max_retries=max_retries
        )
        
        async with self._lock:
            heapq.heappush(self._queue, task)
            self._task_map[task_id] = task
        
        return task_id
    
    async def dequeue(self) -> Optional[Task]:
        """Get the next task from the queue, removing expired ones."""
        async with self._lock:
            while self._queue:
                task = heapq.heappop(self._queue)
                
                # Remove from map
                if task.task_id in self._task_map:
                    del self._task_map[task.task_id]
                
                # Skip expired tasks
                if task.is_expired():
                    continue
                
                return task
            return None
    
    async def remove(self, task_id: str) -> bool:
        """Remove a task from the queue."""
        async with self._lock:
            if task_id in self._task_map:
                task = self._task_map[task_id]
                # Mark as expired so it's skipped on dequeue
                task.deadline = 0
                del self._task_map[task_id]
                return True
            return False
    
    async def size(self) -> int:
        """Get the current queue size."""
        async with self._lock:
            return len(self._queue)
    
    async def clear_expired(self) -> int:
        """Remove all expired tasks from the queue."""
        async with self._lock:
            expired_count = 0
            new_queue = []
            for task in self._queue:
                if not task.is_expired() and task.task_id in self._task_map:
                    heapq.heappush(new_queue, task)
                else:
                    if task.task_id in self._task_map:
                        del self._task_map[task.task_id]
                    expired_count += 1
            self._queue = new_queue
            return expired_count

# Global task queue instance
_global_task_queue: Optional[TaskQueue] = None
_queue_lock = asyncio.Lock()

async def get_task_queue() -> TaskQueue:
    """Get or create the global task queue."""
    global _global_task_queue
    if _global_task_queue is None:
        async with _queue_lock:
            if _global_task_queue is None:
                _global_task_queue = TaskQueue()
    return _global_task_queue

async def execute_with_priorities(
    tasks: List[Dict[str, Any]],
    max_concurrency: int = 6,
    short_circuit_confidence: float = 0.8
) -> List[Any]:
    """
    Execute tasks with priority ordering and concurrency control.
    
    Args:
        tasks: List of task dicts with keys: task_id, func, priority, deadline_seconds, args, kwargs
        max_concurrency: Maximum concurrent tasks
        short_circuit_confidence: If a task returns confidence >= this, skip remaining tasks
    
    Returns:
        List of task results
    """
    queue = await get_task_queue()
    semaphore = asyncio.Semaphore(max_concurrency)
    results = {}
    
    # Enqueue all tasks
    for task_data in tasks:
        await queue.enqueue(
            task_id=task_data.get("task_id", f"task_{len(results)}"),
            func=task_data["func"],
            priority=task_data.get("priority", TaskPriority.P1),
            deadline_seconds=task_data.get("deadline_seconds"),
            max_retries=task_data.get("max_retries", 3),
            *task_data.get("args", ()),
            **task_data.get("kwargs", {})
        )
    
    async def execute_task(task: Task):
        """Execute a single task with retry logic."""
        async with semaphore:
            for attempt in range(task.max_retries + 1):
                try:
                    if asyncio.iscoroutinefunction(task.func):
                        result = await task.func(*task.args, **task.kwargs)
                    else:
                        result = task.func(*task.args, **task.kwargs)
                    
                    results[task.task_id] = {
                        "success": True,
                        "result": result,
                        "attempts": attempt + 1
                    }
                    
                    # Check for short-circuit condition
                    if isinstance(result, dict):
                        confidence = result.get("confidence_score", result.get("confidence", 0.0))
                        if confidence >= short_circuit_confidence:
                            return True  # Signal to short-circuit
                    
                    return False
                    
                except Exception as e:
                    if attempt == task.max_retries:
                        results[task.task_id] = {
                            "success": False,
                            "error": str(e),
                            "attempts": attempt + 1
                        }
                        return False
                    await asyncio.sleep(0.1 * (2 ** attempt))  # Exponential backoff
    
    # Process tasks
    pending_tasks = []
    while True:
        task = await queue.dequeue()
        if task is None:
            break
        
        # Check if we should short-circuit
        should_short_circuit = False
        for result in results.values():
            if result.get("success") and isinstance(result.get("result"), dict):
                confidence = result["result"].get("confidence_score", result["result"].get("confidence", 0.0))
                if confidence >= short_circuit_confidence:
                    should_short_circuit = True
                    break
        
        if should_short_circuit:
            # Cancel remaining tasks
            await queue.clear_expired()
            break
        
        pending_tasks.append(execute_task(task))
    
    # Wait for all pending tasks
    if pending_tasks:
        await asyncio.gather(*pending_tasks, return_exceptions=True)
    
    return [results.get(f"task_{i}", {"success": False, "error": "Not executed"}) 
            for i in range(len(tasks))]

def can_short_circuit(state: Dict[str, Any], confidence_threshold: float = 0.8) -> bool:
    """Check if we can short-circuit based on current state confidence."""
    confidence = state.get("confidence_score", state.get("confidence", 0.0))
    return confidence >= confidence_threshold






