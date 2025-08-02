"""
API interface for the Simple Decision AI.

This module can be run with a WSGI server like Gunicorn.
Example: `gunicorn src.interfaces.api:app`
"""
from flask import Flask, request, jsonify
from core.decision_maker import DecisionMaker

app = Flask(__name__)
decision_maker = None

def get_decision_maker():
    """Initializes and returns a single DecisionMaker instance."""
    global decision_maker
    if decision_maker is None:
        decision_maker = DecisionMaker()
        decision_maker.initialize()
    return decision_maker

@app.route('/decide', methods=['POST'])
def decide():
    """
    API endpoint to make a decision.
    Expects a JSON body with a "text" key.
    """
    if not request.json or 'text' not in request.json:
        return jsonify({"error": "Invalid request, 'text' key is required."}), 400

    text = request.json['text']
    
    try:
        maker = get_decision_maker()
        result = maker.decide(text)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # For local testing only
    app.run(debug=True, port=5000)
