"""
Providers package (training + inference).

Goal:
- Keep provider implementations in ONE place (`src/providers/...`).
- Keep pipeline/orchestration code provider-agnostic via interfaces and factories.

Subpackages:
- `src.providers.training`: training provider interfaces + factory
- `src.providers.inference`: inference provider interfaces + factory
- `src.providers.<provider>.training`: provider-specific training implementation
- `src.providers.<provider>.inference`: provider-specific inference implementation (if available)
- SSH client lives in `src.utils.ssh_client`
"""
