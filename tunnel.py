"""
Creates a public tunnel to your local Streamlit app.
Run this in a second terminal while streamlit is running:
    python tunnel.py
"""
from pyngrok import ngrok

# Create a tunnel to port 8501
public_url = ngrok.connect(8501)
print(f"\n{'='*50}")
print(f"Share this URL with your colleagues:")
print(f"{public_url}")
print(f"{'='*50}")
print("\nKeep this terminal open while sharing.")
print("Press Ctrl+C to stop the tunnel.")

try:
    input()
except KeyboardInterrupt:
    ngrok.disconnect(public_url)
    print("\nTunnel closed.")
