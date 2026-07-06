"""
Central OpenAI client factory — supports BOTH a normal OpenAI API key and an
Azure / EY OpenAI endpoint.

Selection rule (mirrors the FiscalPlatform FiscalKernelFactory.cs):
  * OPENAI_ENDPOINT set   → Azure OpenAI  (deployment = model name, uses
                            OPENAI_API_VERSION).  This is the EY "eyq" fabric case.
  * OPENAI_ENDPOINT unset → normal OpenAI  (api.openai.com).
Both modes read the key from OPENAI_API_KEY, so the same code path serves a
personal OpenAI key at home and the EY Azure endpoint at work — only the env
vars differ.

Environment variables
─────────────────────
  OPENAI_API_KEY      (required)  the key — personal OpenAI OR the EY key
  OPENAI_ENDPOINT     (optional)  EY/Azure resource URL → switches to Azure mode
  OPENAI_API_VERSION  (optional)  Azure api-version         (default below)
  OPENAI_CHAT_MODEL   (optional)  chat deployment/model     (default gpt-4o-mini)
  OPENAI_QA_MODEL     (optional)  legacy chat override      (takes precedence)
  OPENAI_EMBED_MODEL  (optional)  embedding DEPLOYMENT name for Azure mode
  OPENAI_CA_BUNDLE / ZSCALER_CERT (optional) corporate-proxy CA cert (e.g. EY
                       Zscaler) so the outbound HTTPS calls verify.
"""
from __future__ import annotations

import os

_DEFAULT_API_VERSION = "2024-02-15-preview"


# ── one-time environment bootstrap (dotenv + corporate CA bundle) ─────────────
def _load_dotenv_once() -> None:
    try:
        from dotenv import load_dotenv  # optional dependency
        load_dotenv()
    except Exception:
        pass


def _configure_ssl() -> None:
    """Point the SSL stack at a CA bundle that includes the corporate proxy root
    (e.g. EY Zscaler) so OpenAI HTTPS calls verify behind the MITM proxy.
    No-op unless OPENAI_CA_BUNDLE or ZSCALER_CERT is provided."""
    bundle = os.environ.get("OPENAI_CA_BUNDLE")
    cert = os.environ.get("ZSCALER_CERT")
    if not bundle and cert and os.path.exists(cert):
        try:
            import certifi
            combined = os.path.join(os.path.dirname(os.path.abspath(cert)), "combined_bundle.pem")
            if not os.path.exists(combined):
                with open(certifi.where(), "rb") as f:
                    base = f.read()
                with open(cert, "rb") as f:
                    extra = f.read()
                with open(combined, "wb") as f:
                    f.write(base + b"\n" + extra)
            bundle = combined
        except Exception:
            bundle = None
    if bundle and os.path.exists(bundle):
        os.environ.setdefault("SSL_CERT_FILE", bundle)
        os.environ.setdefault("REQUESTS_CA_BUNDLE", bundle)
        os.environ.setdefault("CURL_CA_BUNDLE", bundle)


_load_dotenv_once()
_configure_ssl()


# ── capability / mode helpers ─────────────────────────────────────────────────
def azure_mode() -> bool:
    """True when an Azure/EY endpoint is configured."""
    return bool(os.environ.get("OPENAI_ENDPOINT"))


def openai_configured() -> bool:
    """True when a key is present (either provider)."""
    return bool(os.environ.get("OPENAI_API_KEY"))


def provider() -> str:
    if not openai_configured():
        return "none"
    return "azure" if azure_mode() else "openai"


# ── client + model resolution ─────────────────────────────────────────────────
def get_client():
    """Return an OpenAI or AzureOpenAI client based on the environment."""
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not configured")
    endpoint = os.environ.get("OPENAI_ENDPOINT")
    if endpoint:
        from openai import AzureOpenAI
        return AzureOpenAI(
            api_key=key,
            azure_endpoint=endpoint,
            api_version=os.environ.get("OPENAI_API_VERSION", _DEFAULT_API_VERSION),
        )
    from openai import OpenAI
    return OpenAI(api_key=key)


def chat_model(default: str = "gpt-4o-mini") -> str:
    """Resolve the chat model / Azure deployment name.
    OPENAI_QA_MODEL keeps the chunker's historical override precedence; on the EY
    box OPENAI_CHAT_MODEL (e.g. 'gpt-4o') supplies the deployment name."""
    return (os.environ.get("OPENAI_QA_MODEL")
            or os.environ.get("OPENAI_CHAT_MODEL")
            or default)


def embed_deployment(logical_model: str) -> str:
    """In Azure mode the API `model` argument is the DEPLOYMENT name, which may
    differ from the OpenAI model id; OPENAI_EMBED_MODEL supplies it.  In normal
    mode the logical model id is returned unchanged."""
    if azure_mode():
        return os.environ.get("OPENAI_EMBED_MODEL", logical_model)
    return logical_model
