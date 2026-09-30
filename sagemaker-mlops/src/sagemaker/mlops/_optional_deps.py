# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
"""Accessors for the handful of MLOps features that need ``sagemaker-train``/``-serve``.

Pipeline orchestration itself needs neither package, so they are declared as extras rather
than hard requirements — installing ``sagemaker-mlops`` should not drag in ``torch`` and the
CUDA stack. The few call sites that genuinely need a ``ModelTrainer`` or ``ModelBuilder``
resolve it through here so a missing extra produces an actionable message instead of a bare
``ModuleNotFoundError`` deep in an import chain.
"""

from __future__ import absolute_import

_INSTALL_HINT = (
    "{feature} requires the '{extra}' extra, which is not installed. Install it with:\n"
    "    pip install 'sagemaker-mlops[{extra}]'\n"
    "It is kept optional because it pulls in large dependencies that pipeline "
    "orchestration does not otherwise need."
)


def require_model_trainer():
    """Return ``sagemaker.train.ModelTrainer``, or raise if the ``train`` extra is missing.

    Returns:
        type: The ``ModelTrainer`` class.

    Raises:
        ImportError: If ``sagemaker-train`` is not installed.
    """
    try:
        from sagemaker.train.model_trainer import ModelTrainer

        return ModelTrainer
    except ImportError as e:
        raise ImportError(
            _INSTALL_HINT.format(feature="ModelTrainer", extra="train")
        ) from e


def require_model_builder():
    """Return ``sagemaker.serve.ModelBuilder``, or raise if the ``serve`` extra is missing.

    Returns:
        type: The ``ModelBuilder`` class.

    Raises:
        ImportError: If ``sagemaker-serve`` is not installed.
    """
    try:
        from sagemaker.serve.model_builder import ModelBuilder

        return ModelBuilder
    except ImportError as e:
        raise ImportError(
            _INSTALL_HINT.format(feature="ModelBuilder", extra="serve")
        ) from e
