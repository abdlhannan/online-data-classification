from skmultiflow.data import FileStream
from skmultiflow.trees import HoeffdingTree
from skmultiflow.evaluation import EvaluatePrequential
from skmultiflow.bayes import NaiveBayes
from sklearn.neural_network import MLPClassifier

def HT_online(dataset, WINDOW):
    stream = FileStream("./"+dataset+'.csv')
    stream.prepare_for_use()
    ht = HoeffdingTree()
    # 3. Setup the evaluator
    evaluator = EvaluatePrequential(show_plot=True, pretrain_size=WINDOW,
                                    max_samples=10000)
    evaluator.evaluate(stream=stream, model=ht)
def Bayes_online(dataset, WINDOW):
    stream = FileStream("./"+dataset+'.csv')
    stream.prepare_for_use()
    bayes = NaiveBayes()
    # 3. Setup the evaluator
    evaluator = EvaluatePrequential(show_plot=True, pretrain_size=WINDOW,
                                    max_samples=10000,
                                    output_file='results.csv')
    evaluator.evaluate(stream=stream, model=bayes)
def neuralnetwork_online(dataset, WINDOW):
    stream = FileStream("./"+dataset+'.csv')
    stream.prepare_for_use()
    nn = MLPClassifier(hidden_layer_sizes=(200, 200, 200, 200))
    # 3. Setup the evaluator
    evaluator = EvaluatePrequential(show_plot=True, pretrain_size=WINDOW,
                                    max_samples=10000,
                                    output_file='results.csv')
    evaluator.evaluate(stream=stream, model=nn)

WINDOW = 3000
datasets = ['RBF Dataset','RBF Dataset 10', 'RBF Dataset 70']
dataset = datasets[0]
for dataset in datasets:
    print('HT_online loading....')
    HT_online(dataset, WINDOW)
    print('bayes_online loading....')
    Bayes_online(dataset, WINDOW)
    print('neuralnetwork_online loading....')
    neuralnetwork_online(dataset, WINDOW)