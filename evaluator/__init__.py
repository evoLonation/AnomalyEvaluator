from beartype import BeartypeConf
from beartype.claw import beartype_this_package
beartype_this_package()
# beartype_this_package(conf=BeartypeConf(is_debug=True))

# from jaxtyping import install_import_hook
# install_import_hook("evaluator.clip", typechecker=None)