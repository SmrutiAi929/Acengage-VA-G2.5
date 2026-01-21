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

