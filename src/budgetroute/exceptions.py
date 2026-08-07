"""Domain-specific exceptions with safe, actionable messages."""


class BudgetRouteError(Exception):
    """Base class for expected BudgetRoute failures."""


class ConfigurationError(BudgetRouteError):
    """Configuration is invalid or cannot be resolved."""


class BackendError(BudgetRouteError):
    """A generation backend failed or is unavailable."""


class RetrievalError(BudgetRouteError):
    """A corpus or retrieval index could not be used."""


class ArtifactError(BudgetRouteError):
    """Experiment artifacts are incomplete or invalid."""


class RouterTrainingError(BudgetRouteError):
    """A learned router could not be trained or loaded."""


class OverloadError(BudgetRouteError):
    """The bounded inference scheduler cannot safely accept more work."""


class RequestDeadlineError(BudgetRouteError):
    """A request exceeded its configured queue or execution deadline."""
