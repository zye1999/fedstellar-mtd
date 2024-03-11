#!/bin/bash

# Print commands and their arguments as they are executed (debugging)
set -x

# Print in console debug messages
echo "Starting services..."

# Iniciar Nginx en primer plano en un subshell para que el script continúe
nginx &

# Change directory to where app.py is located
FEDSTELLAR_FRONTEND_DIR=/fedstellar/fedstellar/frontend
cd $FEDSTELLAR_FRONTEND_DIR

# Iniciar Gunicorn
DEBUG=$FEDSTELLAR_DEBUG
echo "DEBUG: $DEBUG"
if [ "$DEBUG" = "True" ]; then
    echo "Starting Gunicorn in debug mode..."
    # Include PYTHONUNBUFFERED=1 to avoid buffering of stdout and stderr
    export PYTHONUNBUFFERED=1
    gunicorn --worker-class eventlet --workers 1 --bind unix:/tmp/fedstellar.sock --access-logfile $SERVER_LOG --error-logfile $SERVER_LOG  --reload --reload-extra-file $FEDSTELLAR_FRONTEND_DIR --capture-output --log-level debug app:app &
else
    echo "Starting Gunicorn in production mode..."
    gunicorn --worker-class eventlet --workers 1 --bind unix:/tmp/fedstellar.sock --access-logfile $SERVER_LOG app:app &
fi

# Iniciar TensorBoard
tensorboard --host 0.0.0.0 --port 8080 --logdir $FEDSTELLAR_LOGS_DIR --window_title "Fedstellar Statistics" --reload_interval 30 --max_reload_threads 10 --reload_multifile true &

tail -f /dev/null
