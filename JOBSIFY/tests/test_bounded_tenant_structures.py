"""Concurrent stress for BoundedTenantDict and TenantAwareCache LRU structures."""
import threading

from core.supervisor_agent import BoundedTenantDict, TenantAwareCache


def test_bounded_tenant_dict_respects_max_entries():
    d = BoundedTenantDict(max_entries=10, ttl_seconds=3600)
    created = []

    def factory(i):
        def _f():
            created.append(i)
            return f"v{i}"

        return _f

    for i in range(25):
        d.get_or_create(f"k{i}", factory(i))
    assert len(d) <= 10


def test_bounded_tenant_dict_concurrent_get_or_create():
    d = BoundedTenantDict(max_entries=50, ttl_seconds=3600)
    errors = []

    def worker(wid: int):
        try:
            for i in range(100):
                key = f"t{wid}_{i % 30}"
                d.get_or_create(key, lambda k=key: {"id": k})
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not errors
    assert len(d) <= 50


def test_tenant_aware_cache_lru_eviction():
    c = TenantAwareCache(max_entries=5, ttl_minutes=60)
    for i in range(10):
        c.set("tenant", i, uid=str(i))
    # At most 5 entries stored
    hits = sum(1 for i in range(10) if c.get("tenant", uid=str(i)) is not None)
    assert hits <= 5


def test_tenant_aware_cache_concurrent_set_get():
    c = TenantAwareCache(max_entries=100, ttl_minutes=60)
    errors = []

    def run():
        try:
            for i in range(200):
                c.set("t", i, k=i % 40)
                c.get("t", k=i % 40)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=run) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert not errors
