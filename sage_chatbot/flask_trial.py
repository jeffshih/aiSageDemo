from flask import Flask, request, abort 

app = Flask(__name__) 

@app.route('/')
def homepage():
    return 'Hello'

@app.route('/test')
def test_page():
    return 'Test page'

if __name__=='__main__':
    app.run()
