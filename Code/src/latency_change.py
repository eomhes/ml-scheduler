#!/usr/bin/python

import os
import time

#latency = {0, 5, 10, 15, 20}

#for l in (len(latency)):
#	os.system("tc qdisc add dev eth0 root netem delay" latency(l)"ms")
#	sleep(20)
#	os.system("tc qdisc del dev eth0 root")

os.system("tc qdisc add dev eth0 root netem delay 0ms")
time.sleep(40)
os.system("tc qdisc del dev eth0 root")
os.system("tc qdisc add dev eth0 root netem delay 8ms")
time.sleep(40)
os.system("tc qdisc del dev eth0 root")
os.system("tc qdisc add dev eth0 root netem delay 2ms")
time.sleep(40)
os.system("tc qdisc del dev eth0 root")
os.system("tc qdisc add dev eth0 root netem delay 10ms")
time.sleep(40)
os.system("tc qdisc del dev eth0 root")
os.system("tc qdisc add dev eth0 root netem delay 20ms")
time.sleep(40)
os.system("tc qdisc del dev eth0 root")
