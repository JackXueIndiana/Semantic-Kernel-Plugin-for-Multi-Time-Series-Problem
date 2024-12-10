from flask import Flask, request, jsonify
import requests
from bs4 import BeautifulSoup
import asyncio
import xml.etree.ElementTree as ET
import sys
import io
import light_plugin
from sk_agent_dep_inj_web import query

from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/v0/phLow', methods=['POST'])
def pHLow():
    try:
        # Get JSON data from the request body
        data = request.get_json()

        # Extract a specific element from the JSON data
        # For example, extracting the 'name' element
        device_id = data.get('DeviceId')
    
        if device_id is None:
            return jsonify({'error': 'DeviceId not found'}), 400
        
        # Use Semantic Kernel agent/dependency injection (DI) to identify the low pH ROA
        summary = asyncio.run(query(device_id)).content
        
        if summary is None:
            return jsonify({'error': 'No summary found'}), 400
        else:
            print(f"summary: {summary}")
            summary = summary.partition("<body>")[2].partition("</body>")[0]
            if summary is None or len(summary.strip()) == 0:
                return jsonify({'Retry': 'Call the API again in a few seconds please.'}), 200
            print(f"summary: {summary}")
            return summary, 200
    except Exception as e:
        if str(e).startswith("\'str\' object has no attribute \'content\'"):
            return jsonify({'Retry': 'Call the API with a valid Asset Name (aka. Device Id).'}), 200
        else:
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
