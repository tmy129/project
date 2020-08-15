from flask import Flask
import logging
import pickle

logging.basicConfig(filename="api.log",level=logging.DEBUG,format='%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')

app = Flask(__name__)

@app.route('/run')
def run():
    app.logger.info('app running')
    model = pickle.load(open('model.pkl','rb'))
    values = model.get_coherence()
    return ('model coherence is {}'.format(values))

@app.route('/')
def init():
    return ('running, please use /run to get model result')

if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0', port=8888)