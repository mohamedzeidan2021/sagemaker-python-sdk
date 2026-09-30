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
"""Guard the dependency layering between the four distributions.

``sagemaker-core`` is the foundation and must not reach upward into ``sagemaker-train``,
``sagemaker-serve`` or ``sagemaker-mlops``. ``sagemaker-mlops`` must not import ``train`` or
``serve`` at module scope, because both are optional extras -- ``sagemaker-serve`` in
particular pulls ``torch`` and the CUDA stack, which is what made v3 too large to run MLOps
orchestration inside a Lambda (aws/sagemaker-python-sdk#5813, #5531).

These are static checks over the source tree, so they run without the optional extras
installed and fail on the import statement rather than on a missing wheel.
"""

from __future__ import absolute_import

import ast
import os

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DISTRIBUTIONS = {
    "sagemaker-core": "core",
    "sagemaker-train": "train",
    "sagemaker-serve": "serve",
    "sagemaker-mlops": "mlops",
}

#: ``(importer, imported)`` pairs that may not appear at module scope anywhere.
#: ``core`` sits at the bottom, so nothing below it may be imported eagerly by it.
FORBIDDEN_AT_MODULE_SCOPE = {
    ("core", "train"),
    ("core", "serve"),
    ("core", "mlops"),
    ("mlops", "train"),
    ("mlops", "serve"),
}

#: Deprecated back-compat shims in Core that still delegate upward from inside a function.
#: They are opt-in (callers get a ``DeprecationWarning``) and are slated for removal; no new
#: entries should be added here.
ALLOWED_FUNCTION_SCOPE_EXCEPTIONS = {
    ("core", "mlops", "sagemaker-core/src/sagemaker/core/workflow/utilities.py"),
}


def _iter_source_files():
    """Yield ``(distribution_short_name, repo_relative_path, parsed_ast)`` for every module."""
    for dist, short in DISTRIBUTIONS.items():
        src_root = os.path.join(REPO_ROOT, dist, "src")
        for dirpath, dirnames, filenames in os.walk(src_root):
            dirnames[:] = [
                d for d in dirnames if d != "__pycache__" and not d.endswith(".egg-info")
            ]
            for filename in filenames:
                if not filename.endswith(".py"):
                    continue
                path = os.path.join(dirpath, filename)
                with open(path, encoding="utf-8") as f:
                    try:
                        tree = ast.parse(f.read())
                    except SyntaxError:  # pragma: no cover - generated shapes files
                        continue
                yield short, os.path.relpath(path, REPO_ROOT), tree


def _import_scopes(tree):
    """Map ``id(import_node)`` to the scope it appears in.

    Returns:
        dict: ``"module"``, ``"typecheck"`` (inside ``if TYPE_CHECKING:``), ``"try"`` or
        ``"func"`` for every import node in the tree.
    """
    scopes = {}

    def visit(node, scope):
        for child in ast.iter_child_nodes(node):
            child_scope = scope
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                child_scope = "func"
            elif (
                isinstance(child, ast.If)
                and scope == "module"
                and "TYPE_CHECKING" in ast.unparse(child.test)
            ):
                child_scope = "typecheck"
            elif isinstance(child, ast.Try) and scope == "module":
                child_scope = "try"
            if isinstance(child, (ast.Import, ast.ImportFrom)):
                scopes[id(child)] = scope
            visit(child, child_scope)

    visit(tree, "module")
    return scopes


def _cross_distribution_imports():
    """Yield ``(importer, imported, path, lineno, module, scope)`` for every cross-dist import."""
    for short, relpath, tree in _iter_source_files():
        scopes = _import_scopes(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and not node.level:
                modules = [node.module or ""]
            else:
                continue
            for module in modules:
                parts = module.split(".")
                if len(parts) < 2 or parts[0] != "sagemaker":
                    continue
                target = parts[1]
                if target not in DISTRIBUTIONS.values() or target == short:
                    continue
                yield short, target, relpath, node.lineno, module, scopes.get(id(node), "module")


@pytest.mark.parametrize("importer,imported", sorted(FORBIDDEN_AT_MODULE_SCOPE))
def test_no_module_scope_import(importer, imported):
    """A forbidden pair must not be imported at module scope (``TYPE_CHECKING`` is fine)."""
    violations = [
        f"{path}:{lineno} imports {module}"
        for actual_importer, actual_imported, path, lineno, module, scope in (
            _cross_distribution_imports()
        )
        if (actual_importer, actual_imported) == (importer, imported) and scope == "module"
    ]
    assert not violations, (
        f"sagemaker-{importer} must not import sagemaker.{imported} at module scope; "
        f"sagemaker-{imported} is a lower layer or an optional extra. Use a function-scope "
        f"import behind sagemaker.mlops._optional_deps, a TYPE_CHECKING guard, or the "
        f"registry in sagemaker.core.workflow._step_type_registry. Violations:\n  "
        + "\n  ".join(violations)
    )


def test_core_never_imports_train_or_serve_at_all():
    """Core must not reach into ``train``/``serve`` in any scope, not even lazily."""
    violations = [
        f"{path}:{lineno} imports {module} (scope: {scope})"
        for importer, imported, path, lineno, module, scope in _cross_distribution_imports()
        if importer == "core" and imported in ("train", "serve")
    ]
    assert not violations, (
        "sagemaker-core is the foundation layer and must not depend on sagemaker-train or "
        "sagemaker-serve in any way. Violations:\n  " + "\n  ".join(violations)
    )


def test_core_upward_imports_are_only_known_deprecated_shims():
    """Any remaining lazy ``core -> mlops`` import must be an accounted-for shim."""
    unexpected = [
        f"{path}:{lineno} imports {module}"
        for importer, imported, path, lineno, module, scope in _cross_distribution_imports()
        if importer == "core"
        and imported == "mlops"
        and scope not in ("typecheck",)
        and (importer, imported, path) not in ALLOWED_FUNCTION_SCOPE_EXCEPTIONS
    ]
    assert not unexpected, (
        "New upward dependency from sagemaker-core into sagemaker-mlops. Invert it with "
        "sagemaker.core.workflow._step_type_registry rather than adding an exception. "
        "Violations:\n  " + "\n  ".join(unexpected)
    )
