import multiprocessing
import os

# Configuración para Render (plan gratuito tiene 512MB RAM)
workers = 1  # Solo 1 worker para no exceder memoria
threads = 2  # 2 threads por worker
worker_class = 'sync'
worker_connections = 1000
timeout = 300  # 5 minutos - modelos grandes tardan en cargar
keepalive = 5
max_requests = 100
max_requests_jitter = 10

# Logging
accesslog = '-'
errorlog = '-'
loglevel = 'info'

# Preload app para cargar modelos una sola vez
preload_app = True

bind = f"0.0.0.0:{os.environ.get('PORT', 5000)}"