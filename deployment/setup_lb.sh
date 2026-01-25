#!/bin/bash

# Define paths
PROJECT_DIR="/home/ubuntu/attendence_app_backend"
SYSTEMD_DIR="/etc/systemd/system"
NGINX_DIR="/etc/nginx/sites-available"

echo "Setting up Load Balancer..."

# 1. Copy Systemd Service Template
echo "Copying systemd service..."
cp deployment/attendance@.service $SYSTEMD_DIR/

# 2a. Stop Old Service (if exists)
echo "Stopping old single-process service..."
systemctl stop attendance
systemctl disable attendance

# 2b. Reload Systemd
echo "Reloading systemd..."
systemctl daemon-reload

# 3. Enable and Start 4 Workers (Ports 8000-8003)
echo "Starting 4 worker nodes..."
for PORT in {8000..8003}; do
    echo "Starting attendance@$PORT..."
    systemctl enable attendance@$PORT
    systemctl restart attendance@$PORT
done

# 4. Configure Nginx
echo "Configuring Nginx..."
cp deployment/nginx_lb.conf $NGINX_DIR/attendance_lb
ln -sf $NGINX_DIR/attendance_lb /etc/nginx/sites-enabled/

# 5. Test and Restart Nginx
echo "Testing Nginx config..."
nginx -t
if [ $? -eq 0 ]; then
    echo "Nginx config OK. Restarting Nginx..."
    systemctl restart nginx
    echo "Deployment Complete! Load Balancer is active."
else
    echo "Nginx config failed! Please check logs."
    exit 1
fi
