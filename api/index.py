"""
Vercel serverless function entry point.
This file is the main entry point for Vercel serverless functions.
"""

import os
import sys

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the simplified app for Vercel deployment
from vercel_app import app

# Make the handler available to Vercel
handler = app