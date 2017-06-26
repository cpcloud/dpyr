"%PYTHON%" setup.py install
if errorlevel 1 exit 1

"%PYTHON%" -c "import dpyr; print(dpyr.__version__)" > __conda_version__.txt
