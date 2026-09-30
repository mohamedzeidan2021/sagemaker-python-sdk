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
"""Workflow utilities that are aware of MLOps step types.

These helpers dispatch on concrete step classes (``ProcessingStep``, ``TrainingStep``,
``StepCollection``) which live in this package. They used to live in
``sagemaker.core.workflow.utilities``, which forced Core to import MLOps at runtime and
inverted the package layering. The generic, step-agnostic hashing helpers they build on
remain in Core and are re-exported here for convenience.
"""

from __future__ import absolute_import

from typing import List, Sequence, Union

from sagemaker.core.helper.pipeline_variable import RequestType
from sagemaker.core.workflow.entities import Entity
from sagemaker.core.workflow.utilities import (
    get_processing_code_hash,
    get_processing_dependencies,
    get_training_code_hash,
    hash_object,
)

from sagemaker.mlops.workflow.step_collections import StepCollection
from sagemaker.mlops.workflow.steps import ProcessingStep, TrainingStep

__all__ = [
    "get_code_hash",
    "get_config_hash",
    "list_to_request",
]


def list_to_request(entities: Sequence[Union[Entity, StepCollection]]) -> List[RequestType]:
    """Get the request structure for list of entities.

    Args:
        entities (Sequence[Entity]): A list of entities.
    Returns:
        list: A request structure for a workflow service call.
    """
    request_dicts = []
    for entity in entities:
        if isinstance(entity, Entity):
            request_dicts.append(entity.to_request())
        elif isinstance(entity, StepCollection):
            request_dicts.extend(entity.request_dicts())
    return request_dicts


def get_code_hash(step: Entity) -> str:
    """Get the hash of the code artifact(s) for the given step

    Args:
        step (Entity): A pipeline step object (Entity type because Step causes circular import)
    Returns:
        str: A hash string representing the unique code artifact(s) for the step
    """
    if isinstance(step, ProcessingStep) and step.step_args:
        kwargs = step.step_args.func_kwargs
        source_dir = kwargs.get("source_dir")
        submit_class = kwargs.get("submit_class")
        dependencies = get_processing_dependencies(
            [
                kwargs.get("dependencies"),
                kwargs.get("submit_py_files"),
                [submit_class] if submit_class else None,
                kwargs.get("submit_jars"),
                kwargs.get("submit_files"),
            ]
        )
        code = kwargs.get("submit_app") or kwargs.get("code")

        return get_processing_code_hash(code, source_dir, dependencies)

    if isinstance(step, TrainingStep) and step.step_args:
        model_trainer = step.step_args.func_args[0]
        source_code = model_trainer.source_code
        if source_code:
            source_dir = source_code.source_dir
            requirements = source_code.requirements
            entry_point = source_code.entry_script
            return get_training_code_hash(entry_point, source_dir, requirements)
    return None


def get_config_hash(step: Entity):
    """Get the hash of the config artifact(s) for the given step

    Args:
        step (Entity): A pipeline step object (Entity type because Step causes circular import)
    Returns:
        str: A hash string representing the unique config artifact(s) for the step
    """
    if isinstance(step, ProcessingStep) and step.step_args:
        config = step.step_args.func_kwargs.get("configuration")
        if config:
            return hash_object(config)
    return None
