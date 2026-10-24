"""Real provider connectivity checks with explicit failure classifications."""

from time import perf_counter

from pydantic import BaseModel

from st_agent.deepseek import (
    DeepSeekAuthenticationError,
    DeepSeekClient,
    DeepSeekError,
    DeepSeekMalformedResponse,
    DeepSeekProviderPaused,
)
from st_agent.intent_runtime import IntentInterpreter
from st_agent.planning import TaskOperation


class PreflightReport(BaseModel):
    passed: bool = False
    secret_configured: bool
    model: str
    configuration_source: str
    chat_response_id: str | None = None
    intent_response_id: str | None = None
    error_code: str | None = None
    elapsed_seconds: float = 0


def run_provider_preflight(settings, *, client=None) -> PreflightReport:
    report = PreflightReport(
        secret_configured=bool(settings.deepseek_api_key),
        model=settings.deepseek_model,
        configuration_source=settings.configuration_source,
    )
    if not report.secret_configured:
        report.error_code = "provider.secret_missing"
        return report
    owned = client is None
    if owned:
        client = DeepSeekClient(
            settings.deepseek_api_key.get_secret_value(),
            base_url=settings.deepseek_base_url,
            model=settings.deepseek_model,
        )
    started = perf_counter()
    try:
        chat = client.complete([{"role": "user", "content": "Reply with OK."}], [])
        if not chat.content:
            raise DeepSeekMalformedResponse("Empty chat response")
        report.chat_response_id = chat.response_id
        interpretation = IntentInterpreter(client).interpret("请检查当前数据集的基本结构。")
        if interpretation.intent.operation is not TaskOperation.INSPECT:
            raise DeepSeekMalformedResponse("Unexpected preflight intent")
        report.intent_response_id = interpretation.response_id
        report.passed = True
    except DeepSeekAuthenticationError:
        report.error_code = "provider.authentication_failed"
    except DeepSeekMalformedResponse:
        report.error_code = "provider.tool_calling_invalid"
    except DeepSeekProviderPaused:
        report.error_code = "provider.unavailable"
    except DeepSeekError:
        report.error_code = "provider.request_rejected"
    finally:
        report.elapsed_seconds = perf_counter() - started
        if owned:
            client._client.close()
    return report
