"""SageMaker MLOps package for workflow orchestration and model building.

This package provides high-level orchestration capabilities for SageMaker workflows,
including pipeline definitions, step implementations, and model building utilities.

The MLOps package sits at the top of the dependency hierarchy and can import from:
- sagemaker.core (foundation primitives) -- a hard requirement
- sagemaker.train (training functionality) -- the ``train`` extra
- sagemaker.serve (serving functionality) -- the ``serve`` extra

Pipeline orchestration needs only ``sagemaker.core``. The two extras are resolved lazily so
that ``import sagemaker.mlops`` works -- and stays small -- without them.

Key components:
- workflow: Pipeline and step orchestration
- model_builder: Model building and orchestration

Example usage:
    from sagemaker.mlops import ModelBuilder
    from sagemaker.mlops.workflow import Pipeline, TrainingStep
"""

from __future__ import absolute_import

__version__ = "0.1.0"

# Workflow submodule is available via:
#   from sagemaker.mlops import workflow
#   from sagemaker.mlops.workflow import Pipeline, TrainingStep, etc.

__all__ = [
    "ModelBuilder",
    "workflow",  # Submodule
]


def __getattr__(name):
    """Resolve ``ModelBuilder`` on first access rather than at import time.

    ``ModelBuilder`` lives in ``sagemaker-serve``, an optional extra. Importing it eagerly
    here would make ``import sagemaker.mlops`` -- and therefore every pipeline definition --
    require the whole serving stack.
    """
    if name == "ModelBuilder":
        from sagemaker.mlops._optional_deps import require_model_builder

        return require_model_builder()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
