#!/bin/bash

"$PYTHON" setup.py install
"$PYTHON" -c "import dpyr; print(dpyr.__version__)" > dpyr/.version
