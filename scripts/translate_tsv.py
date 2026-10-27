#!/usr/bin/env python3
"""Compatibility imports for the renamed TranslateGemma 12B driver."""

from importlib import import_module

_driver = import_module(
    ".translate_translategemma-12b", __package__
) if __package__ else import_module("translate_translategemma-12b")

# Preserve shared helpers imported by the Qwen driver, training, and older callers.
globals().update({name: value for name, value in vars(_driver).items()
                  if not name.startswith("_")})

if __name__ == "__main__":
    _driver.main()
