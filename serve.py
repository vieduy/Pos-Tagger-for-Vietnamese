from flask import Flask, request, render_template
from src.HMM import main

global model
model = None

# Khởi tạo flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'any string works here'
app.static_folder = 'static'


@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    text = request.form.get('text')
    if text:
        results = main(text)
        return render_template('index.html', results=results, text=text)
    return render_template('index.html')


if __name__ == "__main__":
    print("App run!")
    # Load model
    app.run(debug=False, threaded=False)
