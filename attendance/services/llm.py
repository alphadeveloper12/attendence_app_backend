"""Thin OpenAI wrapper used by the analytics views.

Design contract:
    - Single entry point per use case (parse_question, polish_summary).
    - ALWAYS safe to call — if AI is disabled OR the API errors out, returns
      None and the caller falls back to the heuristic engine. Never raises
      out into the view layer.
    - Reads OPENAI_API_KEY + AI_* settings on every call (no module-level
      client init) so toggling AI_ENABLED takes effect on the next request.
    - Logs every call to the `attendance` logger so cost/usage is observable.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Optional

from django.conf import settings

log = logging.getLogger('attendance')


def ai_is_enabled() -> bool:
    """True when the kill-switch is on AND a key is configured."""
    return bool(getattr(settings, 'AI_ENABLED', False)
                and getattr(settings, 'OPENAI_API_KEY', ''))


def _client():
    """Lazy import so a missing `openai` package doesn't blow up imports.
    Returns the OpenAI client or None if unavailable."""
    if not ai_is_enabled():
        return None
    try:
        from openai import OpenAI  # type: ignore
    except ImportError:
        log.warning('AI enabled but `openai` package not installed — install with: pip install openai')
        return None
    try:
        return OpenAI(
            api_key=settings.OPENAI_API_KEY,
            timeout=getattr(settings, 'AI_REQUEST_TIMEOUT', 12),
        )
    except Exception as e:  # noqa: BLE001
        log.warning('OpenAI client init failed: %s', e)
        return None


def _chat_json(system: str, user: str, *, max_tokens: Optional[int] = None,
               feature: str = 'unknown') -> Optional[Dict[str, Any]]:
    """Call gpt-4o-mini in JSON mode. Returns parsed dict on success, None on
    any error. Logs latency and rough token usage for cost-monitoring."""
    client = _client()
    if client is None:
        return None
    model = getattr(settings, 'AI_MODEL', 'gpt-4o-mini')
    max_t = max_tokens or getattr(settings, 'AI_MAX_TOKENS', 400)
    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model=model,
            response_format={'type': 'json_object'},
            messages=[
                {'role': 'system', 'content': system},
                {'role': 'user',   'content': user},
            ],
            max_tokens=max_t,
            temperature=0.2,
        )
        elapsed_ms = int((time.time() - t0) * 1000)
        usage = getattr(resp, 'usage', None)
        in_tok = getattr(usage, 'prompt_tokens', None)
        out_tok = getattr(usage, 'completion_tokens', None)
        log.info('[AI/%s] model=%s elapsed=%dms tokens=%s/%s',
                 feature, model, elapsed_ms, in_tok, out_tok)
        content = resp.choices[0].message.content or ''
        return json.loads(content)
    except json.JSONDecodeError as e:
        log.warning('[AI/%s] JSON decode failed: %s — raw: %r',
                    feature, e, locals().get('content', '')[:240])
        return None
    except Exception as e:  # noqa: BLE001
        log.warning('[AI/%s] call failed (%s) — heuristic fallback used',
                    feature, type(e).__name__)
        return None


def _chat_text(system: str, user: str, *, max_tokens: Optional[int] = None,
               feature: str = 'unknown') -> Optional[str]:
    """Plain-text variant for prose rewrites (executive summary)."""
    client = _client()
    if client is None:
        return None
    model = getattr(settings, 'AI_MODEL', 'gpt-4o-mini')
    max_t = max_tokens or getattr(settings, 'AI_MAX_TOKENS', 400)
    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {'role': 'system', 'content': system},
                {'role': 'user',   'content': user},
            ],
            max_tokens=max_t,
            temperature=0.4,
        )
        elapsed_ms = int((time.time() - t0) * 1000)
        usage = getattr(resp, 'usage', None)
        in_tok = getattr(usage, 'prompt_tokens', None)
        out_tok = getattr(usage, 'completion_tokens', None)
        log.info('[AI/%s] model=%s elapsed=%dms tokens=%s/%s',
                 feature, model, elapsed_ms, in_tok, out_tok)
        return (resp.choices[0].message.content or '').strip() or None
    except Exception as e:  # noqa: BLE001
        log.warning('[AI/%s] call failed (%s) — heuristic fallback used',
                    feature, type(e).__name__)
        return None


# ---------------------------------------------------------------------------
# Public API used by the views
# ---------------------------------------------------------------------------

def parse_question(question: str, *, sites: list, positions: list,
                   departments: list) -> Optional[Dict[str, Any]]:
    """LLM-powered NL → structured query parser.

    Returns a dict mirroring the keyword parser's `as_understood` shape so the
    downstream queryset-builder stays unchanged:
        {
          'intent': 'count' | 'list',
          'status': 'Active' | 'Leave' | ...  | null,
          'attendance': 'present' | 'absent' | 'late' | null,
          'site': '<one of sites>' | null,
          'position': '<one of positions>' | null,
          'department': '<one of departments>' | null,
          'time_window': 'today' | 'yesterday' | 'this_week' | 'last_week' | 'this_month' | null
        }
    Returns None on any failure — caller MUST fall back to the keyword parser.
    """
    if not ai_is_enabled():
        return None

    # Cap the context to keep prompts small — 50 each is plenty for matching.
    sites_s = ', '.join(sites[:50])
    positions_s = ', '.join(positions[:80])
    departments_s = ', '.join(departments[:50])

    system = (
        "You translate workforce questions into a structured query. "
        "Reply ONLY with JSON matching exactly this schema (every key required, "
        "use null when absent):\n"
        '{"intent":"count|list",'
        '"status":"Active|Leave|Resigned|Terminated|No Renewal|Absconding|null",'
        '"attendance":"present|absent|late|null",'
        '"site":"<exact name from sites list>|null",'
        '"position":"<exact name from positions list>|null",'
        '"department":"<exact name from departments list>|null",'
        '"time_window":"today|yesterday|this_week|last_week|this_month|null"}\n'
        "Names MUST match the provided lists EXACTLY (case + spelling). "
        "If a name isn't in the list, use null — don't invent."
    )
    user = (
        f"Question: {question}\n\n"
        f"Sites: {sites_s}\n"
        f"Positions: {positions_s}\n"
        f"Departments: {departments_s}"
    )

    out = _chat_json(system, user, max_tokens=200, feature='ask')
    if not isinstance(out, dict):
        return None

    # Defensive normalization — turn the LLM's null/string into Python None
    def _clean(v):
        if v in (None, '', 'null', 'None'):
            return None
        return v

    return {k: _clean(out.get(k)) for k in
            ('intent', 'status', 'attendance', 'site', 'position', 'department', 'time_window')}


def polish_summary(bullets: list) -> Optional[list]:
    """Rewrite the heuristic bullets into smoother prose, keeping the metric
    numbers + severity intact.

    Input  bullets: [{icon, severity, metric, delta, text}, ...]
    Output bullets: same shape with `text` rewritten. Returns None on failure
    (caller keeps the original bullets).
    """
    if not ai_is_enabled() or not bullets:
        return None

    system = (
        "You are a workforce analytics editor. Rewrite each bullet so it reads "
        "like a senior ops manager dictated it — concise, direct, no hedging, "
        "no marketing fluff. Keep every NUMBER and PROPER NOUN intact. "
        "Maximum 18 words per bullet. Reply ONLY with JSON: "
        '{"bullets":[{"text":"..."},{"text":"..."}, ...]} '
        "with the SAME number of items in the SAME order as the input."
    )
    user = json.dumps({'bullets': [{'text': b.get('text', '')} for b in bullets]})

    out = _chat_json(system, user, max_tokens=400, feature='summary')
    if not isinstance(out, dict):
        return None
    rewritten = out.get('bullets') or []
    if len(rewritten) != len(bullets):
        return None

    polished = []
    for orig, new in zip(bullets, rewritten):
        clean = (new.get('text') or '').strip() if isinstance(new, dict) else ''
        polished.append({**orig, 'text': clean or orig.get('text', '')})
    return polished
