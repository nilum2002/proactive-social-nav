 #### 1. On Jetson Nano:

    # Create a permanent static profile named 'direct-eth'
    sudo nmcli connection add type ethernet con-name direct-eth ifname eth0 ip4 192.168.2.2/24 gw4 192.168.2.1
    
    # Configure link-local / static mode so it never auto-disconnects when DHCP fails
    sudo nmcli connection modify direct-eth ipv4.method manual
    sudo nmcli connection modify direct-eth ipv4.addresses 192.168.2.2/24
    
    # Activate the connection
    sudo nmcli connection up direct-eth
    ──────
  #### 2. On Raspberry Pi 4:

    # Create permanent static profile on RPi
    sudo nmcli connection add type ethernet con-name direct-eth ifname eth0 ip4 192.168.2.1/24
    
    # Set manual mode
    sudo nmcli connection modify direct-eth ipv4.method manual
    sudo nmcli connection modify direct-eth ipv4.addresses 192.168.2.1/24
    
    # Activate connection
    sudo nmcli connection up direct-eth