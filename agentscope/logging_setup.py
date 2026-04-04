import logging
import sys
import structlog

from agentscope.otel import current_trace_ids

_CONFIGURED = False


def _add_trace_context(_logger, _method_name, event_dict):
    trace_ids = current_trace_ids()
    if trace_ids.get("trace_id"):
        event_dict.update(trace_ids)
    return event_dict


def _configure_once(log_level: str = "INFO") -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            _add_trace_context,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level),
    )
    _CONFIGURED = True


def configure_logging(
    run_id: str | None = None,
    *,
    component: str = "agentscope",
    log_level: str = "INFO",
    **bindings,
) -> structlog.BoundLogger:
    _configure_once(log_level=log_level)
    payload = {"component": component, **bindings}
    if run_id is not None:
        payload["run_id"] = run_id
    return structlog.get_logger(component).bind(**payload)
