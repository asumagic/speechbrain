"""
Module importing related utilities.

Author
 * Sylvain de Langen 2024
"""

from types import ModuleType
import importlib
import sys
import os
from typing import Optional, List
import warnings


class LazyModule(ModuleType):
    """Defines a module type that lazily imports the target module, thus
    exposing
    defined without immediately importing the target module needlessly."""

    # TODO: args

    def __init__(
        self,
        name: str,
        target: str,
        package: str,
    ):
        super().__init__(name)
        self.target = target
        self.lazy_module = None
        self.package = package

    def _ensure_module(self) -> ModuleType:
        """Ensures that the target module is imported and available as
        `self.lazy_module`, also returning it."""

        if self.lazy_module is None:
            try:
                self.lazy_module = importlib.import_module(f".{self.target}", self.package)
            except Exception as e:
                raise ImportError(f"Lazy import of {repr(self)} failed") from e

        return self.lazy_module

    def __repr__(self) -> str:
        return f"LazyModule(package={self.package}, target={self.target}, loaded={self.lazy_module is not None})"

    def __getattr__(self, attr):
        # NOTE: exceptions here get eaten and not displayed
        return getattr(self._ensure_module(), attr)


class DeprecatedModuleRedirect(LazyModule):
    """Defines a module type that lazily imports the target module using
    :class:`~LazyModule`, but logging a deprecation warning when the import
    is actually being performed.

    This is only the module type itself; if you want to define a redirection,
    use :func:`~deprecated_redirect` instead.

    Arguments
    ---------
    old_import : str
        Old module import path e.g. `mypackage.myoldmodule`
    new_import : str
        New module import path e.g. `mypackage.mynewcoolmodule.mycoolsubmodule`
    extra_reason : str, optional
        If specified, extra text to attach to the warning for clarification
        (e.g. justifying why the move has occurred, or additional problems to
        look out for).
    """

    def __init__(
        self,
        old_import: str,
        new_import: str,
        package: str,
        extra_reason: Optional[str] = None,
    ):
        super().__init__(name=old_import, target=new_import, package=package)
        self.old_import = old_import
        self.extra_reason = extra_reason

    def _redirection_warn(self):
        """Emits the warning for the redirection (with the extra reason if
        provided)."""

        warning_text = (
            f"Module '{self.old_import}' was deprecated, redirecting to "
            f"'{self.target}'. Please update your script."
        )

        if self.extra_reason is not None:
            warning_text += f" {self.extra_reason}"

        # NOTE: we are not using DeprecationWarning because this gets ignored by
        # default, even though we consider the warning to be rather important
        # in the context of SB

        warnings.warn(
            warning_text,
            # category=DeprecationWarning,
            stacklevel=4,  # _ensure_module <- __getattr__ <- python <- user
        )

    def _ensure_module(self) -> ModuleType:
        if self.lazy_module is None:
            self._redirection_warn()

        return super()._ensure_module()

def find_imports(file_path: str, find_subpackages: bool = False) -> List[str]:
    """Returns a list of importable scripts in the same module as the specified
    file. e.g. if you have `foo/__init__.py` and `foo/bar.py`, then
    `files_in_module("foo/__init__.py")` then the result will be `["bar"]`.

    Not recursive; this is only for a given module.

    Arguments
    ---------
    file_path : str
        Path of the file to navigate the directory of. Typically the
        `__init__.py` path this is called from, using `__file__`.
    find_subpackages : bool
        Whether we should find the subpackages as well.
    """

    imports = []

    module_dir = os.path.dirname(file_path)

    for filename in os.listdir(module_dir):
        if filename.startswith("__"):
            continue

        if filename.endswith(".py"):
            imports.append(filename[:-3])

        if find_subpackages and os.path.isdir(
            os.path.join(module_dir, filename)
        ):
            imports.append(filename)

    return imports


def lazy_export_all(
    init_file_path: str, package: str, export_subpackages: bool = False
):
    """Makes all scripts under a module lazily importable merely by accessing
    them; e.g. `foo/bar.py` could be accessed with `foo.bar.some_func()`.

    Arguments
    ---------
    init_file_path : str
        Path of the `__init__.py` file, usually determined with `__file__` from
        there.
    package : str
        The relevant package, usually determined with `__name__` from the
        `__init__.py`.
    export_subpackages : bool
        Whether we should make the subpackages (subdirectories) available
        directly as well.
    """

    for name in find_imports(
        init_file_path, find_subpackages=export_subpackages
    ):
        # already imported for real (e.g. utils.importutils itself)
        if hasattr(sys.modules[package], name):
            continue

        setattr(sys.modules[package], name, LazyModule(name, name, package))


def deprecated_redirect(
    old_import: str, new_import: str, package: str, extra_reason: Optional[str] = None
) -> None:
    """Patches the module list to add a lazy redirection from `old_import` to
    `new_import`, emitting a `DeprecationWarning` when imported.

    Arguments
    ---------
    old_import : str
        Old module import path e.g. `mypackage.myoldmodule`
    new_import : str
        New module import path e.g. `mypackage.mynewcoolmodule.mycoolsubmodule`
    extra_reason : str, optional
        If specified, extra text to attach to the warning for clarification
        (e.g. justifying why the move has occurred, or additional problems to
        look out for).
    """

    assert not hasattr(sys.modules[package], old_import), f"Failed to create redirection {old_import}->{new_import} for package {package}, because it was already set to {sys.modules[old_import]}"

    setattr(sys.modules[package], old_import, DeprecatedModuleRedirect(
        old_import, new_import, package, extra_reason=extra_reason
    ))
