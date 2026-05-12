"""Playwright skeleton for stack-level web tests.

Phase 2 ships only a smoke-shaped fixture; real user-flow tests land in
Phase 5/6. The fixture is a no-op when ``playwright`` isn't installed so
collecting tests on a bare CI runner doesn't error — the actual web tests
are gated by ``@pytest.mark.web_e2e``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


@pytest.fixture
async def browser_context(stack: Any) -> AsyncIterator[Any]:
    """Yield a Playwright browser context pointed at ``stack``'s control plane.

    Skips cleanly if Playwright (and its browsers) aren't installed —
    Phase 2 doesn't want to make ``make test-new`` depend on heavyweight
    browser infra.
    """
    pytest.importorskip("playwright.async_api")
    from playwright.async_api import async_playwright

    async with async_playwright() as pw:
        try:
            browser = await pw.chromium.launch(headless=True)
        except Exception as exc:
            pytest.skip(f"playwright chromium not installed: {exc}")
        context = await browser.new_context()
        try:
            yield context
        finally:
            await context.close()
            await browser.close()


__all__ = ["browser_context"]
