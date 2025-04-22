#!/bin/bash

echo "Cleaning up old processes..."

sudo fuser -k 9000/tcp || true
sudo fuser -k 9001/tcp || true
sudo fuser -k 9002/tcp || true
sudo fuser -k 9003/tcp || true
sudo fuser -k 9004/tcp || true

sudo sync
sudo sysctl -w vm.drop_caches=3

sleep 5

echo "Done. Ready to start new run."

function wait_for_port() {
  IP=$1
  PORT=$2
  while ! nc -z $IP $PORT; do
    echo "Waiting for $IP:$PORT..."
    sleep 1
  done
  echo "$IP:$PORT is ready!"
}

echo "Starting device servers..."
python3 device_server.py 0 localhost 9000 &
PID0=$!
python3 device_server.py 1 localhost 9001 &
PID1=$!
python3 device_server.py 2 localhost 9002 &
PID2=$!
python3 device_server.py 3 localhost 9003 &
PID3=$!

wait_for_port localhost 9000
wait_for_port localhost 9001
wait_for_port localhost 9002
wait_for_port localhost 9003
wait_for_port 192.168.1.44 9004  # Replace with Pi IP

echo "Running client..."
python3 client.py

sleep 2

echo "Shutting down servers..."
kill $PID0
kill $PID1
kill $PID2
kill $PID3

echo "All done."
