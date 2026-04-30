import json
import logging
import re
from typing import Dict, List

import requests

from config.settings import (
    NEWS_API_KEY,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    PERPLEXITY_API_KEY,
    PERPLEXITY_MODEL,
)
from src.retrieval.news_fetcher import fetch_verified_news

logger = logging.getLogger(__name__)


class LiveFactChecker:
    """Live evidence helper using NewsAPI + OpenAI (Perplexity optional)."""

    def __init__(self):
        self.perplexity_key = PERPLEXITY_API_KEY
        self.openai_key = OPENAI_API_KEY

    def _build_newsapi_evidence(self, query: str, max_items: int = 6) -> Dict:
        articles = fetch_verified_news(query, page_size=max_items)
        evidence_items = []
        citations = []
        for article in articles[:max_items]:
            title = (article.get("title") or "").strip()
            desc = (article.get("description") or "").strip()
            source = (article.get("source") or "Unknown source").strip()
            url = (article.get("url") or "").strip()
            if not title and not desc:
                continue
            evidence_items.append(
                {
                    "title": title,
                    "description": desc,
                    "source": source,
                    "url": url,
                }
            )
            if url:
                citations.append(url)
        return {
            "available": len(evidence_items) > 0,
            "error": "" if evidence_items else "No supporting articles found.",
            "articles": evidence_items,
            "citations": citations,
            "newsapi_configured": bool(NEWS_API_KEY),
        }

    def _call_perplexity(self, query: str) -> Dict:
        if not self.perplexity_key:
            return {
                "available": False,
                "error": "PERPLEXITY_API_KEY is missing.",
                "verdict": "Unavailable",
                "summary": "",
                "citations": [],
            }

        url = "https://api.perplexity.ai/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.perplexity_key}",
            "Content-Type": "application/json",
        }
        prompt = (
            "You are a real-time fact-check assistant. "
            "Assess whether the claim is likely true, false, mixed, or insufficient evidence. "
            "Return strict JSON with keys: verdict, confidence, summary, key_points. "
            "confidence must be 0-1 number. key_points must be a short list of strings. "
            f"Claim: {query}"
        )
        payload = {
            "model": PERPLEXITY_MODEL,
            "temperature": 0.1,
            "messages": [
                {"role": "system", "content": "Provide factual answers with citations."},
                {"role": "user", "content": prompt},
            ],
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=20)
            response.raise_for_status()
            data = response.json()

            content = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
            citations = data.get("citations") or []

            parsed = {
                "verdict": "Insufficient Evidence",
                "confidence": 0.5,
                "summary": content,
                "key_points": [],
            }
            try:
                # Try to parse if the model returned JSON.
                parsed_json = json.loads(content)
                if isinstance(parsed_json, dict):
                    parsed.update(
                        {
                            "verdict": str(
                                parsed_json.get("verdict", parsed["verdict"])
                            ),
                            "confidence": float(
                                parsed_json.get("confidence", parsed["confidence"])
                            ),
                            "summary": str(
                                parsed_json.get("summary", parsed["summary"])
                            ),
                            "key_points": parsed_json.get("key_points", []),
                        }
                    )
            except (json.JSONDecodeError, ValueError, TypeError):
                # Keep text fallback.
                pass

            return {
                "available": True,
                "error": "",
                "verdict": parsed["verdict"],
                "confidence": max(0.0, min(1.0, float(parsed["confidence"]))),
                "summary": parsed["summary"],
                "key_points": parsed["key_points"][:5]
                if isinstance(parsed["key_points"], list)
                else [],
                "citations": citations[:8],
            }
        except requests.RequestException as ex:
            logger.error("Perplexity request failed: %s", ex)
            return {
                "available": False,
                "error": f"Perplexity request failed: {ex}",
                "verdict": "Unavailable",
                "confidence": 0.0,
                "summary": "",
                "key_points": [],
                "citations": [],
            }

    def _call_openai_refiner(
        self,
        query: str,
        ml_label: str,
        ml_confidence: float,
        evidence: Dict,
        ppl_evidence: Dict,
    ) -> Dict:
        if not self.openai_key:
            return {
                "available": False,
                "error": "OPENAI_API_KEY is missing.",
                "refined_summary": "",
            }

        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.openai_key}",
            "Content-Type": "application/json",
        }
        source_lines = []
        for idx, a in enumerate(evidence.get("articles", []), start=1):
            source_lines.append(
                f"{idx}. [{a.get('source')}] {a.get('title')} | {a.get('description')} | {a.get('url')}"
            )
        sources_block = "\n".join(source_lines) if source_lines else "No sources available."

        user_prompt = (
            "You are a fact-check assistant. Based on the claim and any sources below, give a short assessment.\n\n"
            f"Claim: {query}\n\n"
            f"Our ML model said: {ml_label} (confidence {ml_confidence:.2f}).\n\n"
            f"Sources (if any):\n{sources_block}\n\n"
            "Reply with a brief paragraph (2-4 sentences) assessing whether the claim is likely true, false, mixed, or unclear. "
            "Then optionally add a line 'Key points:' followed by 2-4 bullet points. "
            "If you cannot verify, say so. Be neutral and cite uncertainty when needed."
        )
        payload = {
            "model": OPENAI_MODEL,
            "temperature": 0.2,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a neutral editor. Avoid sensational language. "
                        "Do not fabricate facts."
                    ),
                },
                {"role": "user", "content": user_prompt},
            ],
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            content = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
            if not content:
                return {
                    "available": False,
                    "error": "Empty response from OpenAI.",
                    "verdict": "Unavailable",
                    "confidence": 0.0,
                    "summary": "",
                    "key_points": [],
                    "refined_summary": "",
                }
            # Extract verdict from first line or sentence
            verdict = "Insufficient Evidence"
            for v in ("Likely True", "Likely False", "Mixed", "Unclear", "True", "False"):
                if v.lower() in content[:200].lower():
                    verdict = v
                    break
            # Optional: parse "Key points:" section into list
            key_pts = []
            if "key points" in content.lower():
                try:
                    after = content.lower().split("key points")[-1].strip(": \n")
                    parts = re.split(r"\n\s*[•\-*]\s*", after)
                    key_pts = [p.strip() for p in parts[:5] if p.strip() and len(p.strip()) > 15]
                except Exception:
                    pass
            if not key_pts:
                key_pts = [content[:200] + ("..." if len(content) > 200 else "")]

            return {
                "available": True,
                "error": "",
                "verdict": verdict,
                "confidence": 0.6,
                "summary": content,
                "key_points": key_pts[:5],
                "refined_summary": content,
            }
        except requests.RequestException as ex:
            logger.error("OpenAI request failed: %s", ex)
            return {
                "available": False,
                "error": f"OpenAI request failed: {ex}",
                "verdict": "Unavailable",
                "confidence": 0.0,
                "summary": "",
                "key_points": [],
                "refined_summary": "",
            }

    def analyze(self, query: str, ml_result: Dict) -> Dict:
        evidence = self._build_newsapi_evidence(query)
        ppl_evidence = self._call_perplexity(query) if self.perplexity_key else {
            "available": False,
            "error": "Perplexity is optional and not configured.",
            "verdict": "Unavailable",
            "confidence": 0.0,
            "summary": "",
            "key_points": [],
            "citations": [],
        }
        refiner = self._call_openai_refiner(
            query=query,
            ml_label=ml_result.get("label", "Unknown"),
            ml_confidence=float(ml_result.get("confidence", 0.0)),
            evidence=evidence,
            ppl_evidence=ppl_evidence,
        )
        return {
            "evidence": evidence,
            "perplexity": ppl_evidence,
            "openai": refiner,
        }


def check_contradiction(claim: str, articles: List[Dict], openai_key: str = None) -> str:
    """
    Ask OpenAI whether any of the given source snippets contradict the claim.
    Returns one sentence (e.g. a quoted contradiction or "No clear contradiction found.").
    This is a novel feature: most fact-checkers don't explicitly surface contradictions.
    """
    if not openai_key:
        return ""
    if not articles or not claim or not claim.strip():
        return ""

    snippets = []
    for a in articles[:5]:
        title = (a.get("title") or "").strip()
        desc = (a.get("description") or "").strip()
        source = (a.get("source") or "Source").strip()
        if title or desc:
            snippets.append(f"[{source}] {title}. {desc}".strip())

    if not snippets:
        return ""

    text_block = "\n".join(snippets)[:2000]
    prompt = (
        "Claim to check: \"\"\"%s\"\"\"\n\n"
        "Source snippets from trusted outlets:\n\n%s\n\n"
        "In exactly one short sentence, does any of the above contradict the claim? "
        "If yes, quote the contradiction. If no, say: No clear contradiction found."
    ) % (claim.strip()[:500], text_block)

    try:
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {openai_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": OPENAI_MODEL,
                "temperature": 0.1,
                "messages": [
                    {"role": "system", "content": "Answer in one sentence only. Be neutral."},
                    {"role": "user", "content": prompt},
                ],
            },
            timeout=15,
        )
        resp.raise_for_status()
        content = (
            resp.json()
            .get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
        return content if content else ""
    except Exception as e:
        logger.warning("Contradiction check failed: %s", e)
        return ""
