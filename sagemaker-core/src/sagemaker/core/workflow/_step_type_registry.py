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
"""Registry letting higher layers announce step types that Core must recognise.

Core's remote-function machinery has to special-case ``DelayedReturn``, which is defined
in ``sagemaker-mlops``. Importing it from Core would invert the package layering and make
``sagemaker-core`` unusable without ``sagemaker-mlops`` installed. Instead MLOps pushes the
class down here at import time and Core reads it back, so the dependency arrow only ever
points from MLOps to Core.

``sagemaker.mlops.workflow.function_step`` calls :func:`register_delayed_return_type` at
module scope; anything that can hold a ``DelayedReturn`` has necessarily imported that
module first, so by the time Core looks the class up it is registered.
"""

from __future__ import absolute_import

from typing import Callable, List, Optional, Type

_delayed_return_type: Optional[type] = None
_on_register_callbacks: List[Callable[[type], None]] = []


def register_delayed_return_type(cls: Type) -> None:
    """Register the concrete ``DelayedReturn`` class with Core.

    Args:
        cls (Type): The ``sagemaker.mlops.workflow.function_step.DelayedReturn`` class.
    """
    global _delayed_return_type  # pylint: disable=global-statement
    _delayed_return_type = cls
    for callback in _on_register_callbacks:
        callback(cls)


def on_delayed_return_registered(callback: Callable[[type], None]) -> None:
    """Run ``callback`` when ``DelayedReturn`` is registered, or now if it already was.

    Core modules that build lookup tables at import time cannot simply read the registry,
    because MLOps may be imported after them. They subscribe here instead, so the table is
    populated whichever order the two packages load in.

    Args:
        callback (Callable[[type], None]): Invoked with the ``DelayedReturn`` class.
    """
    _on_register_callbacks.append(callback)
    if _delayed_return_type is not None:
        callback(_delayed_return_type)


def get_delayed_return_type() -> Optional[type]:
    """Return the registered ``DelayedReturn`` class, or ``None`` if MLOps is not loaded.

    Returns:
        Optional[type]: The registered class, or ``None`` when ``sagemaker-mlops`` is not
        installed or its ``function_step`` module has not been imported.
    """
    return _delayed_return_type


def is_delayed_return(obj) -> bool:
    """Return whether ``obj`` is a ``DelayedReturn``, tolerating MLOps being absent.

    Args:
        obj: Any object.

    Returns:
        bool: ``True`` only if MLOps registered its class and ``obj`` is an instance of it.
    """
    cls = _delayed_return_type
    return cls is not None and isinstance(obj, cls)
