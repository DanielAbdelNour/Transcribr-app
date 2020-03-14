#!/bin/bash

SWAPFILE=/var/swapfile
SWAP_MEGABYTES=4096

if [ -f $SWAPFILE ]; then
	echo "Swapfile $SWAPFILE found, assuming already setup"
	exit;
fi

/bin/dd if=/dev/zero of=$SWAPFILE bs=128M count=32
/bin/chmod 600 $SWAPFILE
/sbin/mkswap $SWAPFILE
/sbin/swapon $SWAPFILE
echo /sbin/swapon -s