# Mac-to-Mac File Sharing Setup

  ## Prerequisites
  - Both Macs connected to same network (ethernet/wifi)
  - Know the target Mac's IP address

  ## Find Target Mac's IP Address
  On the target Mac, run:
  ```bash
  ifconfig en0 | grep inet
  ```
  Look for the inet line - that's your IP address (e.g., 192.168.100.2)

  Enable File Sharing on Target Mac

  1. System Settings → General → Sharing
  2. Turn on File Sharing
  3. Note which folders are being shared

  Connect from Source Mac

  1. Finder → Go → Connect to Server (or Cmd+K)
  2. Enter: smb://[TARGET_IP] (e.g., smb://192.168.100.2)
  3. Enter username/password for target Mac
  4. Select which volumes/folders to mount
  5. Target Mac folders now appear in Finder sidebar

  Usage

  - Drag files between machines via Finder
  - Target Mac appears as network drive
  - Access persists until you disconnect or restart

  Troubleshooting

  - Ensure File Sharing is enabled on target Mac
  - Verify both machines on same network
  - Check firewall settings if connection fails
