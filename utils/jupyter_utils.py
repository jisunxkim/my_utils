def restart_kernel():
    import IPython
    app = IPython.Application.instance()
    app.kernel.do_shutdown(True)

def add_parent_path():
    import os
    import sys
    sys.path.insert(1, os.path.abspath(os.path.join(os.getcwd(), '..')))
