#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   source ./wsl_proxy.sh
# Optional:
#   export WSL_PROXY_PORT=7890
#   export WSL_PROXY_HOST=172.19.160.1

if ! grep -qi microsoft /proc/version 2>/dev/null; then
  echo "[wsl_proxy] Not running in WSL, skip."
  return 0 2>/dev/null || exit 0
fi

proxy_host="${WSL_PROXY_HOST:-$(ip route 2>/dev/null | awk '/^default/ {print $3; exit}')}"
proxy_port="${WSL_PROXY_PORT:-7890}"

if [[ -z "${proxy_host}" ]]; then
  echo "[wsl_proxy] Cannot detect Windows host IP. Set WSL_PROXY_HOST manually."
  return 1 2>/dev/null || exit 1
fi

export http_proxy="http://${proxy_host}:${proxy_port}"
export https_proxy="http://${proxy_host}:${proxy_port}"
export HTTP_PROXY="${http_proxy}"
export HTTPS_PROXY="${https_proxy}"
export all_proxy="socks5://${proxy_host}:${proxy_port}"
export ALL_PROXY="${all_proxy}"
export no_proxy="localhost,127.0.0.1,::1"
export NO_PROXY="${no_proxy}"

echo "[wsl_proxy] Enabled"
echo "[wsl_proxy] http_proxy=${http_proxy}"
