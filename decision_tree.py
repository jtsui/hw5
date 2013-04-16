import math
import numpy
import random
import scipy
import scipy.io
from operator import itemgetter
import progressbar


def pbar(size):
    bar = progressbar.ProgressBar(maxval=size,
                                  widgets=[progressbar.Bar('=', '[', ']'),
                                           ' ', progressbar.Percentage(),
                                           ' ', progressbar.ETA(),
                                           ' ', progressbar.Counter(),
                                           '/%s' % size])
    return bar


class DecisionTree:

    def __init__(self, xtrain, ytrain, entropy=None, T=0.01, X=10):
        if entropy is None:
            spam = float(sum(ytrain)) / len(ytrain)
            if spam == 0 or spam == 1:
                entropy = 0
            else:
                entropy = spam * math.log(spam)
                entropy += (1 - spam) * math.log(1 - spam)
                entropy = entropy * -1.0
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.entropy = entropy
        self.left_child = None
        self.right_child = None
        self.is_spam = None

        entropies = []
        bar = pbar(len(xtrain))
        bar.start()
        for i in xrange(len(xtrain)):
            for j in xrange(len(xtrain[i])):
                left, right, e = self.get_entropy(j, xtrain[i][j])
                entropies.append((j, xtrain[i][j], left, right, e))
            bar.update(i)
        bar.finish()
        self.feat, self.val, l_e, r_e, n_e = min(entropies, key=itemgetter(4))

        print n_e

        if (n_e - self.entropy) < T or len(xtrain) < X:
            self.is_spam = self.get_majority_label()
        else:
            lxtrain, lytrain, rxtrain, rytrain = self.splitData(xtrain, ytrain)
            self.left_child = DecisionTree(lxtrain, lytrain, l_e)
            self.right_child = DecisionTree(rxtrain, rytrain, r_e)

    def get_majority_label(self):
        ones = len(self.ytrain[self.ytrain == 1])
        zeros = len(self.ytrain[self.ytrain == 0])
        if ones > zeros:
            return True
        elif zeros > ones:
            return False
        else:
            return random.choice((True, False))

    def split(self, feat, val):
        return self.xtrain[:, feat] < val, self.xtrain[:, feat] >= val

    def splitData(self):
        left_indices, right_indices = self.split(self.feat, self.val)
        return (self.xtrain[numpy.array(left_indices)],
                self.ytrain[numpy.array(left_indices)],
                self.xtrain[numpy.array(right_indices)],
                self.ytrain[numpy.array(right_indices)])

    def child_entropy(self, indices):
        labels = self.ytrain[indices]
        if len(labels) == 0:
            return 0
        spam = float(sum(labels)) / len(labels)
        if spam == 0 or spam == 1:
            entropy = 0
        else:
            entropy = spam * math.log(spam)
            entropy += (1 - spam) * math.log(1 - spam)
            entropy = entropy * -1.0
        return entropy

    def get_entropy(self, feat, val):
        '''
        Calculates entropy: H(x) = Summation( P(x) * log(P(x)) ) for all x
        '''
        left_indices, right_indices = self.split(feat, val)
        left_entropy = self.child_entropy(left_indices)
        right_entropy = self.child_entropy(right_indices)
        entropy = left_entropy * len(left_indices) / len(self.xtrain)
        entropy += right_entropy * len(right_indices) / len(self.xtrain)
        return left_entropy, right_entropy, entropy

    def classify(self, sample):
        '''
        evaluate the sample, returning True for spam and False for ham
        '''
        if self.left_child is not None and self.right_child is not None:
            sample_val = sample[self.feat]
            if sample_val < self.val:
                return self.left_child.classify(sample)
            else:
                return self.right_child.classify(sample)
        else:
            return self.is_spam


def main():
    data = scipy.io.loadmat('spamData.mat')
    xtrain = data['Xtrain']
    ytrain = data['ytrain']
    xtest = data['Xtest']
    ytest = data['ytest']
    tree = DecisionTree(xtrain, ytrain)
    error = 0
    for i in xrange(len(xtest)):
        sample = xtest[i]
        if tree.classify(sample) != ytest[i]:
            error += 1
    print 'Error rate %s' % error / len(xtest)


if __name__ == "__main__":
    main()
