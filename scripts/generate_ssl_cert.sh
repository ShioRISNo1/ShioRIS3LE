#!/bin/bash

# SSL Certificate Generation Script for ShioRIS3
# This script generates a self-signed SSL certificate for HTTPS support
#
# Usage:
#   ./generate_ssl_cert.sh [additional_ip1] [additional_ip2] ...
#
# Example:
#   ./generate_ssl_cert.sh 192.168.10.178 192.168.1.100

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CERT_DIR="$SCRIPT_DIR/../ssl_certs"
CERT_FILE="$CERT_DIR/server.crt"
KEY_FILE="$CERT_DIR/server.key"
DAYS_VALID=365

echo "ShioRIS3 SSL Certificate Generator"
echo "==================================="
echo ""

# Create certificate directory if it doesn't exist
mkdir -p "$CERT_DIR"

# Get local IP address
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    LOCAL_IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo "localhost")
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    LOCAL_IP=$(hostname -I | awk '{print $1}' || echo "localhost")
else
    # Windows/Other
    LOCAL_IP="localhost"
fi

echo "Detected local IP: $LOCAL_IP"

# Additional IP addresses from command line arguments
ADDITIONAL_IPS=("$@")
if [ ${#ADDITIONAL_IPS[@]} -gt 0 ]; then
    echo "Additional IPs: ${ADDITIONAL_IPS[*]}"
fi
echo ""

# Check if certificates already exist
if [ -f "$CERT_FILE" ] && [ -f "$KEY_FILE" ]; then
    echo "WARNING: Certificates already exist:"
    echo "  Certificate: $CERT_FILE"
    echo "  Private Key: $KEY_FILE"
    echo ""
    read -p "Do you want to regenerate them? (y/N): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Keeping existing certificates."
        exit 0
    fi
    echo "Regenerating certificates..."
fi

# Create OpenSSL config file with SAN (Subject Alternative Names)
CONFIG_FILE="$CERT_DIR/openssl.cnf"

# Build alt_names section dynamically
ALT_NAMES="DNS.1 = localhost
DNS.2 = *.local
IP.1 = 127.0.0.1
IP.2 = ::1
IP.3 = $LOCAL_IP"

# Add additional IP addresses
IP_INDEX=4
for EXTRA_IP in "${ADDITIONAL_IPS[@]}"; do
    ALT_NAMES="$ALT_NAMES
IP.$IP_INDEX = $EXTRA_IP"
    IP_INDEX=$((IP_INDEX + 1))
done

cat > "$CONFIG_FILE" << EOF
[req]
default_bits = 2048
prompt = no
default_md = sha256
distinguished_name = dn
x509_extensions = v3_req

[dn]
C = JP
ST = Tokyo
L = Tokyo
O = ShioRIS3
OU = Medical Imaging
CN = ShioRIS3 WebServer
emailAddress = admin@shioriis3.local

[v3_req]
subjectAltName = @alt_names
basicConstraints = CA:FALSE
keyUsage = digitalSignature, keyEncipherment
extendedKeyUsage = serverAuth

[alt_names]
$ALT_NAMES
EOF

echo "Generating SSL certificate..."
echo ""

# Generate private key and certificate
openssl req -x509 -nodes -days $DAYS_VALID \
    -newkey rsa:2048 \
    -keyout "$KEY_FILE" \
    -out "$CERT_FILE" \
    -config "$CONFIG_FILE"

# Set appropriate permissions
chmod 600 "$KEY_FILE"
chmod 644 "$CERT_FILE"

# Remove temporary config file
rm "$CONFIG_FILE"

echo ""
echo "✓ SSL Certificate generated successfully!"
echo ""
echo "Certificate location: $CERT_FILE"
echo "Private key location: $KEY_FILE"
echo "Valid for: $DAYS_VALID days"
echo ""
echo "Included IP addresses:"
echo "  - 127.0.0.1 (localhost IPv4)"
echo "  - ::1 (localhost IPv6)"
echo "  - $LOCAL_IP (auto-detected)"
for EXTRA_IP in "${ADDITIONAL_IPS[@]}"; do
    echo "  - $EXTRA_IP (manually added)"
done
echo ""
echo "==================================="
echo "Vision Pro Setup Instructions:"
echo "==================================="
echo ""
echo "1. Start ShioRIS3 with HTTPS enabled"
if [ ${#ADDITIONAL_IPS[@]} -gt 0 ]; then
    echo "2. On Vision Pro Safari, navigate to one of:"
    echo "   - https://$LOCAL_IP:8443"
    for EXTRA_IP in "${ADDITIONAL_IPS[@]}"; do
        echo "   - https://$EXTRA_IP:8443"
    done
else
    echo "2. On Vision Pro Safari, navigate to: https://$LOCAL_IP:8443"
fi
echo "3. You will see a certificate warning"
echo "4. Tap 'Show Details' → 'visit this website'"
echo "5. Enter your device passcode to trust the certificate"
echo ""
echo "For detailed setup, see: VISIONPRO_SETUP.md"
echo ""
