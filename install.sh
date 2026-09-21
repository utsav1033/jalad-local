#!/bin/sh
# curl -fsSL https://raw.githubusercontent.com/utsav1033/jalad-local/main/install.sh | sh
set -eu

REPO="utsav1033/jalad-local"
BIN="tokenpati"

os=$(uname -s)
arch=$(uname -m)
case "$os-$arch" in
  Darwin-arm64)            target="aarch64-apple-darwin" ;;
  Darwin-x86_64)           target="x86_64-apple-darwin" ;;
  Linux-x86_64)            target="x86_64-unknown-linux-gnu" ;;
  Linux-aarch64|Linux-arm64) target="aarch64-unknown-linux-gnu" ;;
  *) echo "no prebuilt binary for $os $arch. try: cargo install $BIN" >&2; exit 1 ;;
esac

url="https://github.com/$REPO/releases/latest/download/$BIN-$target.tar.gz"
dir="${TOKENPATI_INSTALL_DIR:-$HOME/.local/bin}"
mkdir -p "$dir"

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
echo "downloading $url"
curl -fsSL "$url" | tar -xz -C "$tmp"
install -m 755 "$tmp/$BIN" "$dir/$BIN"
echo "installed $dir/$BIN"

case ":$PATH:" in
  *":$dir:"*) ;;
  *) echo "add it to your PATH:  export PATH=\"$dir:\$PATH\"" ;;
esac
echo "run: $BIN"
