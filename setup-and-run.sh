#!/bin/bash

# Setup script for running the original Google Gemini Live API demo

echo "========================================="
echo "  Gemini Live API - Original Google Demo"
echo "========================================="
echo ""

# Set custom gcloud config directory to avoid permission issues
export CLOUDSDK_CONFIG=/Users/gulshan/.gcloud
export PATH="/opt/homebrew/share/google-cloud-sdk/bin:$PATH"

# Create config directory if it doesn't exist
mkdir -p $CLOUDSDK_CONFIG

echo "Step 1: Authenticating with Google Cloud..."
echo ""
echo "A browser window will open. Please:"
echo "1. Sign in with your Google account"
echo "2. Allow the requested permissions"
echo "3. Copy the verification code and paste it back here"
echo ""

# Authenticate
gcloud auth application-default login

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Authentication failed!"
    echo ""
    echo "Alternative: You can also use gcloud init and set up a project:"
    echo "  gcloud init"
    exit 1
fi

echo ""
echo "✅ Authentication successful!"
echo ""
echo "Step 2: Installing Python dependencies..."
pip3 install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies"
    exit 1
fi

echo ""
echo "✅ Dependencies installed!"
echo ""
echo "Step 3: Starting the server..."
echo ""
echo "Server will be available at:"
echo "  📱 Web Interface:   http://localhost:8000"
echo "  🔌 WebSocket Proxy: ws://localhost:8080"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the server
python3 server.py

