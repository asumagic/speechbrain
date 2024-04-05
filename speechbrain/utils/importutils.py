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
) -> List[str]:
    """Returns a function that a package's `__getattr__` should get assigned to.
    This makes all scripts under a module lazily importable merely by accessing
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

    known_imports = find_imports(
        init_file_path, find_subpackages=export_subpackages
    )
    print(f"from {package} discovered {known_imports}")

    def _getter(name):
        """`__getattr__`-compatible function being returned"""

        print(f"trying to import {package}.{name}")

        if name in known_imports:
            return importlib.import_module(f".{name}", package)

        raise ModuleNotFoundError(
            f"module '{package}' has no attribute '{name}'"
        )

    return _getter


class LegacyModuleRedirect(ModuleType):
    """Defines a module type that lazily imports the target module (and warns
    about the deprecation when this happens), thus allowing deprecated
    redirections to be defined without immediately importing the target module
    needlessly.

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
        extra_reason: Optional[str] = None,
    ):
        super().__init__(old_import)
        self.old_import = old_import
        self.new_import = new_import
        self.extra_reason = extra_reason
        self.lazy_module = None

    def _redirection_warn(self):
        """Emits the warning for the redirection (with the extra reason if
        provided)."""

        warning_text = (
            f"Module '{self.old_import}' was deprecated, redirecting to "
            f"'{self.new_import}'. Please update your script."
        )

        if self.extra_reason is not None:
            warning_text += f" {self.extra_reason}"

        # NOTE: we are not using DeprecationWarning because this gets ignored by
        # default, even though we consider the warning to be rather important
        # in the context of SB

        warnings.warn(
            warning_text,
            # category=DeprecationWarning,
            stacklevel=3,
        )

    def __getattr__(self, attr):
        # NOTE: exceptions here get eaten and not displayed

        if self.lazy_module is None:
            self._redirection_warn()
            self.lazy_module = importlib.import_module(self.new_import)

        return getattr(self.lazy_module, attr)


def deprecated_redirect(
    old_import: str, new_import: str, extra_reason: Optional[str] = None
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

    sys.modules[old_import] = LegacyModuleRedirect(
        old_import, new_import, extra_reason=extra_reason
    )
