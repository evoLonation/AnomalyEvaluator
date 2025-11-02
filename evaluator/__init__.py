from jaxtyping import install_import_hook
with install_import_hook("evaluator", "beartype.beartype"):
    import evaluator

from beartype.claw import beartype_this_package
beartype_this_package()