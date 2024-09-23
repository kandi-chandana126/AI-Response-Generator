from flask import Flask, request, jsonify
import modules

app = Flask(__name__)

https_match = r'^https:\/\/[^\s"]+$'

@app.route('/process', methods=['POST'])
def process_data():
    try:
        data = request.json
        query = data.get('input', '')
        sites_required=data.get('sites_required', '')

        gemini_response= modules.main(query, https_match, sites_required)

        #ensuring response is in JSON format.
        response = {
            "result": gemini_response,
        }
        return jsonify(response)
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
