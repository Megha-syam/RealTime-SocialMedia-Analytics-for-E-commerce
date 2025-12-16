import sys
import os

# Add the backend directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uvicorn

# Importing `app` here is optional for some use-cases; uvicorn reload requires an import string
# so we'll start uvicorn using the import string "main:app" which enables reload support.
if __name__ == "__main__":
    try:
        print("Starting Real-Time Social Media Analytics Backend...")
        print("Backend will be available at: http://127.0.0.1:8000")
        print("API documentation at: http://127.0.0.1:8000/docs")

        # Use import string so uvicorn can enable reload/workers correctly
        # Bind to 127.0.0.1 by default to avoid potential firewall/proxy issues on some Windows setups.
        uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

    except Exception as exc:
        # Print a helpful message and re-raise so you can see the full traceback
        print("Failed to start uvicorn server:", exc)
        raise
