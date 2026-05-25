"""
⚠️  INTENTIONALLY MESSY MODULE — for demo purposes only.

This module simulates the kind of code that tends to accumulate technical debt:
  - High cyclomatic complexity (deep nesting, many branches)
  - No docstrings
  - High fan-in (imported by both app.py and services.py)
  - Many imports (coupled to many external things)
  - Functions that do too many things
  - Global mutable state
  - Inconsistent error handling

GraphGuard should flag this module as structurally risky.
This is NOT production-quality code — it exists purely as a demo target.
"""

import json
import logging
import os
import re
import sys
import time
from typing import Any

import models
import utils

_CACHE: dict[str, Any] = {}
_RETRY_LIMIT = 5
_logger = logging.getLogger("buggy_module")


def process_order_data(raw_data, user_id, product_ids, quantities, discount, flags):
    errors = []
    results = []
    total = 0
    retries = 0
    while retries < _RETRY_LIMIT:
        try:
            if raw_data is None:
                errors.append("null data")
                break
            if not isinstance(raw_data, dict):
                if isinstance(raw_data, str):
                    raw_data = json.loads(raw_data)
                elif isinstance(raw_data, bytes):
                    raw_data = json.loads(raw_data.decode())
                else:
                    errors.append("bad type")
                    retries += 1
                    continue
            if not user_id:
                errors.append("missing user")
            else:
                if user_id in _CACHE:
                    user = _CACHE[user_id]
                else:
                    user = models.User(user_id, raw_data.get("email", ""), raw_data.get("name", ""))
                    _CACHE[user_id] = user
                if not user.is_active:
                    errors.append("inactive user")
                    break
            if len(product_ids) != len(quantities):
                errors.append("mismatched lists")
                break
            for i, pid in enumerate(product_ids):
                qty = quantities[i]
                if qty <= 0:
                    errors.append(f"bad qty {qty} for {pid}")
                    continue
                prod = models.Product(pid, raw_data.get("name", pid), 0.0, qty * 2, "unknown")
                if discount and isinstance(discount, (int, float)) and 0 < discount < 100:
                    price = prod.apply_discount(discount)
                else:
                    if flags and "admin_override" in flags:
                        price = 0.0
                    elif flags and "no_discount" in flags:
                        price = prod.price
                    else:
                        price = prod.price
                subtotal = price * qty
                total += subtotal
                results.append({"product_id": pid, "qty": qty, "subtotal": subtotal})
            break
        except json.JSONDecodeError as exc:
            errors.append(f"json error: {exc}")
            retries += 1
            time.sleep(0.01 * retries)
        except Exception as exc:
            errors.append(f"unknown error: {exc}")
            break
    return {"results": results, "total": total, "errors": errors}


def validate_and_transform(payload, schema=None, strict=False, transform_fn=None, extra_keys=None):
    if payload is None:
        return None
    out = {}
    if schema:
        for key, expected_type in schema.items():
            val = payload.get(key)
            if val is None:
                if strict:
                    raise ValueError(f"Missing required key: {key}")
                continue
            if not isinstance(val, expected_type):
                try:
                    val = expected_type(val)
                except (ValueError, TypeError):
                    if strict:
                        raise
            out[key] = val
    else:
        out = dict(payload)
    if extra_keys:
        for k in extra_keys:
            if k not in out:
                out[k] = None
    if transform_fn:
        try:
            out = transform_fn(out)
        except Exception as e:
            _logger.warning(f"transform_fn failed: {e}")
    return out


def compute_risk_score(node_data, weights=None, normalize=True, clip=True):
    if weights is None:
        weights = {"complexity": 0.3, "fan_in": 0.4, "betweenness": 0.3}
    score = 0.0
    for key, w in weights.items():
        val = node_data.get(key, 0)
        if val is None:
            val = 0
        if isinstance(val, str):
            try:
                val = float(val)
            except ValueError:
                val = 0
        score += w * val
    if normalize:
        total_weight = sum(weights.values())
        if total_weight > 0:
            score /= total_weight
    if clip:
        score = utils.clamp(score, 0.0, 1.0)
    return score


def batch_process_files(file_paths, processor_fn, parallel=False, max_workers=4, timeout=30):
    results = {}
    errors = {}
    if not file_paths:
        return results, errors
    if parallel:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(processor_fn, fp): fp for fp in file_paths}
            for future in concurrent.futures.as_completed(futures, timeout=timeout):
                fp = futures[future]
                try:
                    results[fp] = future.result()
                except Exception as exc:
                    errors[fp] = str(exc)
    else:
        for fp in file_paths:
            try:
                if not os.path.exists(fp):
                    errors[fp] = "file not found"
                    continue
                results[fp] = processor_fn(fp)
            except PermissionError as exc:
                errors[fp] = f"permission denied: {exc}"
            except OSError as exc:
                errors[fp] = f"os error: {exc}"
            except Exception as exc:
                errors[fp] = f"unexpected: {exc}"
    return results, errors


def _internal_cache_flush(key_pattern=None, max_age=None, dry_run=False):
    removed = []
    if key_pattern:
        pattern = re.compile(key_pattern)
        keys_to_remove = [k for k in list(_CACHE.keys()) if pattern.match(k)]
    else:
        keys_to_remove = list(_CACHE.keys())
    if max_age is not None:
        now = time.time()
        keys_to_remove = [
            k for k in keys_to_remove
            if isinstance(_CACHE[k], dict) and now - _CACHE[k].get("_ts", now) > max_age
        ]
    for k in keys_to_remove:
        if not dry_run:
            del _CACHE[k]
        removed.append(k)
    return removed


def get_system_info():
    return {
        "platform": sys.platform,
        "python": sys.version,
        "pid": os.getpid(),
        "env_vars": {k: v for k, v in os.environ.items() if k.startswith("GRAPHGUARD_")},
        "cache_size": len(_CACHE),
    }
