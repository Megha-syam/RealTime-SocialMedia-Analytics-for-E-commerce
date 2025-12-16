"""
Startup script for the real-time Real-Time Social Media Analytics for E-Commerce Trends system
Run this to start both the FastAPI backend and real-time data monitoring
"""
"""
Startup script for the real-time Real-Time Social Media Analytics for E-Commerce Trends system
Run this to start both the FastAPI backend and real-time data monitoring
"""

import asyncio
import subprocess
import sys
import os
from pathlib import Path

async def start_backend_server():
    """Start the FastAPI backend server"""
    try:
        backend_path = Path(__file__).parent.parent / "backend"
        os.chdir(backend_path)
        
        # Start uvicorn server
        process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", 
            "main:app", 
            "--host", "0.0.0.0", 
            "--port", "8000", 
            "--reload"
        ])
        
        print("✅ FastAPI backend server started on http://localhost:8000")
        return process
        
    except Exception as e:
        print(f"❌ Error starting backend server: {e}")
        return None

async def start_data_monitoring():
    """Start the real-time data monitoring system"""
    try:
        from backend.realtime_data_manager import start_data_monitoring
        print("✅ Starting real-time data monitoring system...")
        await start_data_monitoring()
        
    except Exception as e:
        print(f"❌ Error starting data monitoring: {e}")

async def main():
    """Main startup function"""
    print("🚀 Starting Real-Time Social Media Analytics real-time system...")
        print("🚀 Starting Real-Time Social Media Analytics Real-time System...")
    print("=" * 50)
    
    # Start backend server
    backend_process = await start_backend_server()
    
    if backend_process:
        try:
            # Start data monitoring in the background
            await start_data_monitoring()
            
        except KeyboardInterrupt:
            print("\n🛑 Shutting down Real-Time Social Media Analytics system...")
                print("\n🛑 Shutting down Real-Time Social Media Analytics system...")
            if backend_process:
                backend_process.terminate()
                backend_process.wait()
            print("✅ System shutdown complete")
        
        except Exception as e:
            print(f"❌ System error: {e}")
            if backend_process:
                backend_process.terminate()
                backend_process.wait()

if __name__ == "__main__":
    asyncio.run(main())
