from datetime import datetime

def log(msg):
    """Печать с timestamp"""
    print(f"[{datetime.now():%H:%M:%S}] {msg}")