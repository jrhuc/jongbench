import os
import sys

# The libriichi extension and the vendored Mortal files expect `import libriichi`
# at top level; the .so lives inside this package directory.
_pkg_dir = os.path.dirname(os.path.abspath(__file__))
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)
