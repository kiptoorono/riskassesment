import multiprocessing
import os

# Base directory configuration
base_dir = os.path.dirname(os.path.abspath(__file__))

# Server socket configuration
bind = "0.0.0.0:8000"
backlog = 2048

# Worker processes
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "sync"
worker_connections = 1000
timeout = 30
keepalive = 2

# Process naming
proc_name = "climate-risk-app"

# Logging
accesslog = "-"  # stdout
errorlog = "-"   # stderr
loglevel = "info"

# SSL (if needed)
# keyfile = "/path/to/keyfile"
# certfile = "/path/to/certfile"

# App specific
reload = True
preload_app = True
chdir = base_dir

# Security
limit_request_line = 4096
limit_request_fields = 100
limit_request_field_size = 8190