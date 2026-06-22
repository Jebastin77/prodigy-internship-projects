"""
PRODIGY_GA_Task02 - Image Generation with Pre-trained Models
Entry point: runs the Django development server.
Usage: python app.py
"""

import os
import sys
import subprocess

def main():
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "image_gen_project.settings")
    # Run migrations first
    subprocess.run([sys.executable, "manage.py", "migrate", "--run-syncdb"], check=True)
    # Start dev server
    subprocess.run([sys.executable, "manage.py", "runserver", "127.0.0.1:8000"], check=True)

if __name__ == "__main__":
    main()
