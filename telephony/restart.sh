#!/bin/bash
# Restart Telephony Service on port 8084

cd /home/info/Acengage-VA-G2.5/telephony

# Stop existing service
echo "Stopping service on port 8084..."
PID=$(lsof -i :8084 2>/dev/null | grep python3 | awk '{print $2}' | head -1)
if [ -n "$PID" ]; then
    kill -9 $PID 2>/dev/null
    sleep 2
    echo "✅ Stopped process $PID"
else
    echo "ℹ️  No service running on port 8084"
fi

# Wait for port to be free
sleep 1

# Start service
echo "Starting service..."
source ../venv/bin/activate
nohup python3 -u main.py > /dev/null 2>&1 &
sleep 3

# Verify it's running
NEW_PID=$(lsof -i :8084 2>/dev/null | grep python3 | awk '{print $2}' | head -1)
if [ -n "$NEW_PID" ]; then
    echo "✅ Service restarted successfully!"
    echo "   PID: $NEW_PID"
    echo "   Port: 8084"
    echo "   Endpoint: ws://0.0.0.0:8084/ws"
    echo ""
    echo "Recent logs:"
    tail -5 telephony.log
else
    echo "❌ Service failed to start"
    echo "Check logs: tail -20 telephony.log"
fi
