"""Various model or function wrapping functionality.

Authors
* Sylvain de Langen 2024
"""

from typing import Any, Callable, Dict, List, Optional


class ArgumentMapper:
    """Wraps a function or callable, renaming or removing arguments passed
    through it in a configurable way.

    When calling the argument mapper, positional arguments are forwarded as-is.
    Keyword arguments are forward as-is, renamed, or removed.

    This wrapper is useful to enable simple argument filtering in hyperpyyaml.

    Arguments
    ---------
    function: Callable
        Function to be called when the mapper is called.
    remove_keys: List[str], optional
        List of keyword argument names that should be filtered out before the
        call to `function`.
        Useful when some keyword arguments are not supported by `function`.
    key_mappings: Dict[str, Optional[str]], optional
        Map of keyword argument names to different names, e.g. `{"foo": "bar"}`
        would rename the keyword argument `foo` to `bar`.
        Can also accept mapping to `None`, which would remove the keyword
        argument, similar to `remove_keys`.
    """

    def __init__(
        self,
        function: Callable,
        remove_keys: List[str] = [],
        key_mappings: Dict[str, Optional[str]] = {},
    ):
        self.function = function

        self.key_mappings = {}
        self.key_mappings.update(key_mappings)
        self.key_mappings.update((key, None) for key in remove_keys)

    def transform_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Transforms keyword arguments according to the mappings provided
        during initialization.

        Arguments
        ---------
        kwargs: Dict[str, Any]
            Dictionary representing keyword arguments, some of which may be
            removed or renamed according to the mappings provided during
            initialization.

        Returns
        -------
        Dict[str, Any]
            Filtered keyword arguments.
        """

        return {
            self.key_mappings.get(name, name): value
            for name, value in kwargs.items()
            if self.key_mappings.get(name, name) is not None
        }

    def __call__(
        self,
        *args,
        **kwargs,
    ) -> Any:
        """Calls the `function` provided during initialization.

        Arguments
        ---------
        *args
            Positional arguments. Unconditionally passed as-is to `function`.
        **kwargs
            Keyword arguments. Keys may be passed as-is, renamed or removed.
            See :class:`~ArgumentMapper`.

        Returns
        -------
        Any
            Return value of the called function. Can be `None`, including the
            case where the function does not return a value.
        """
        return self.function(*args, **self.transform_kwargs(**kwargs))
