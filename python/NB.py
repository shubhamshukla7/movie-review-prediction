# NLP Programming Assignment #3
# NaiveBayes
# 2012

#
# The area for you to implement is marked with TODO!
# Generally, you should not need to touch things *not* marked TODO
#
# Remember that when you submit your code, it is not run from the command line
# and your main() will *not* be run. To be safest, restrict your changes to
# addExample() and classify() and anything you further invoke from there.
#

from __future__ import division
import sys
import getopt
import os
import math
import collections
from sets import Set


class NaiveBayes:
  class TrainSplit:
    """Represents a set of training/testing data. self.train is a list of Examples, as is self.test.
    """
    def __init__(self):
      self.train = []
      self.test = []

  class Example:
    """Represents a document with a label. klass is 'pos' or 'neg' by convention.
       words is a list of strings.
    """
    def __init__(self):
      self.klass = ''
      self.words = []

  def __init__(self):
    """NaiveBayes initialization"""
    self.FILTER_STOP_WORDS = False
    self.stopList = set(self.readFile('../data/english.stop'))
    self.numFolds = 10

    self.doc_count = 0
    self.sentiment_to_word_frequency_map = {'pos': collections.defaultdict(lambda: 0), 'neg': collections.defaultdict(lambda: 0)};
    self.V = Set()
    self.document_in_each_class = {'pos' : 0, 'neg' : 0}
    self.words_in_each_class = {'pos' : 0, 'neg' : 0}

  #############################################################################
  # TODO TODO TODO TODO TODO

  def classify(self, words):
    """ TODO
      'words' is a list of words to classify. Return 'pos' or 'neg' classification.
    """
    score_pos = 0.0
    score_neg = 0.0

    # log prior for positive and negative
    log_prior_pos = math.log(self.document_in_each_class['pos']/self.doc_count)
    log_prior_neg = math.log(self.document_in_each_class['neg']/self.doc_count)

    log_likelihood_pos = 0.0
    log_likelihood_neg = 0.0
    for word in words:
      # If word is not in vocabulary - use the 'special' likelihood for unknown word
      if word not in self.V:
	log_likelihood_pos += math.log(1) - math.log(self.document_in_each_class['pos'] + len(self.V) + 1)
	log_likelihood_neg += math.log(1) - math.log(self.document_in_each_class['neg'] + len(self.V) + 1)
      else:
        log_likelihood_pos += math.log(self.sentiment_to_word_frequency_map['pos'][word] + 1) - math.log(self.words_in_each_class['pos'] + len(self.V))
        log_likelihood_neg += math.log(self.sentiment_to_word_frequency_map['neg'][word] + 1) - math.log(self.words_in_each_class['neg'] + len(self.V))
      
    # combine prior and likelihood.
    score_pos += log_prior_pos + log_likelihood_pos
    score_neg += log_prior_neg + log_likelihood_neg

    if(score_pos > score_neg):
      return 'pos'
    else:
      return 'neg'

  def addExample(self, klass, words):
    """
     * TODO
     * Train your model on an example document with label klass ('pos' or 'neg') and
     * words, a list of strings.
     * You should store whatever data structures you use for your classifier
     * in the NaiveBayes class.
     * Returns nothing
    """
    self.doc_count += 1
    self.document_in_each_class[klass] += 1

    # Clip words to single frequency
    s = Set()
    for word in words:
      s.add(word)

    for word in s:
      self.sentiment_to_word_frequency_map[klass][word] += 1
      self.V.add(word)
      self.words_in_each_class[klass] += 1

  def filterStopWords(self, words):
    """
    * TODO
    * Filters stop words found in self.stopList.
    """
    filtered = []
    for word in words:
      if not word in self.stopList and word.strip() != '':
	filtered.append(word);
    return filtered

  # TODO TODO TODO TODO TODO
  #############################################################################


  def readFile(self, fileName):
    """
     * Code for reading a file.  you probably don't want to modify anything here,
     * unless you don't like the way we segment files.
    """
    contents = []
    f = open(fileName)
    for line in f:
      contents.append(line)
    f.close()
    result = self.segmentWords('\n'.join(contents))
    return result


  def segmentWords(self, s):
    """
     * Splits lines on whitespace for file reading
    """
    return s.split()


  def trainSplit(self, trainDir):
    """Takes in a trainDir, returns one TrainSplit with train set."""
    split = self.TrainSplit()
    posTrainFileNames = os.listdir('%s/pos/' % trainDir)
    negTrainFileNames = os.listdir('%s/neg/' % trainDir)
    for fileName in posTrainFileNames:
      example = self.Example()
      example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
      example.klass = 'pos'
      split.train.append(example)
    for fileName in negTrainFileNames:
      example = self.Example()
      example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
      example.klass = 'neg'
      split.train.append(example)
    return split

  def train(self, split):
    for example in split.train:
      words = example.words
      if self.FILTER_STOP_WORDS:
        words =  self.filterStopWords(words)
      self.addExample(example.klass, words)

  def crossValidationSplits(self, trainDir):
    """Returns a lsit of TrainSplits corresponding to the cross validation splits."""
    splits = []
    posTrainFileNames = os.listdir('%s/pos/' % trainDir)
    negTrainFileNames = os.listdir('%s/neg/' % trainDir)
    #for fileName in trainFileNames:
    for fold in range(0, self.numFolds):
      split = self.TrainSplit()
      for fileName in posTrainFileNames:
        example = self.Example()
        example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
        example.klass = 'pos'
        if fileName[2] == str(fold):
          split.test.append(example)
        else:
          split.train.append(example)
      for fileName in negTrainFileNames:
        example = self.Example()
        example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
        example.klass = 'neg'
        if fileName[2] == str(fold):
          split.test.append(example)
        else:
          split.train.append(example)
      splits.append(split)
    return splits


  def test(self, split):
    """Returns a list of labels for split.test."""
    labels = []
    for example in split.test:
      words = example.words
      if self.FILTER_STOP_WORDS:
        words =  self.filterStopWords(words)
      guess = self.classify(words)
      labels.append(guess)
    return labels

  def buildSplits(self, args):
    """Builds the splits for training/testing"""
    trainData = []
    testData = []
    splits = []
    trainDir = args[0]
    if len(args) == 1:
      print '[INFO]\tPerforming %d-fold cross-validation on data set:\t%s' % (self.numFolds, trainDir)

      posTrainFileNames = os.listdir('%s/pos/' % trainDir)
      negTrainFileNames = os.listdir('%s/neg/' % trainDir)
      for fold in range(0, self.numFolds):
        split = self.TrainSplit()
        for fileName in posTrainFileNames:
          example = self.Example()
          example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
          example.klass = 'pos'
          if fileName[2] == str(fold):
            split.test.append(example)
          else:
            split.train.append(example)
        for fileName in negTrainFileNames:
          example = self.Example()
          example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
          example.klass = 'neg'
          if fileName[2] == str(fold):
            split.test.append(example)
          else:
            split.train.append(example)
        splits.append(split)
    elif len(args) == 2:
      split = self.TrainSplit()
      testDir = args[1]
      print '[INFO]\tTraining on data set:\t%s testing on data set:\t%s' % (trainDir, testDir)
      posTrainFileNames = os.listdir('%s/pos/' % trainDir)
      negTrainFileNames = os.listdir('%s/neg/' % trainDir)
      for fileName in posTrainFileNames:
        example = self.Example()
        example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
        example.klass = 'pos'
        split.train.append(example)
      for fileName in negTrainFileNames:
        example = self.Example()
        example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
        example.klass = 'neg'
        split.train.append(example)

      posTestFileNames = os.listdir('%s/pos/' % testDir)
      negTestFileNames = os.listdir('%s/neg/' % testDir)
      for fileName in posTestFileNames:
        example = self.Example()
        example.words = self.readFile('%s/pos/%s' % (testDir, fileName))
        example.klass = 'pos'
        split.test.append(example)
      for fileName in negTestFileNames:
        example = self.Example()
        example.words = self.readFile('%s/neg/%s' % (testDir, fileName))
        example.klass = 'neg'
        split.test.append(example)
      splits.append(split)
    return splits

def main():
  nb = NaiveBayes()

  # default parameters: no stop word filtering, and
  # training/testing on ../data/imdb1
  if len(sys.argv) < 2:
      options = [('','')]
      args = ['../data/imdb1/']
  else:
      (options, args) = getopt.getopt(sys.argv[1:], 'f')
  if ('-f','') in options:
    nb.FILTER_STOP_WORDS = True

  splits = nb.buildSplits(args)
  avgAccuracy = 0.0
  fold = 0
  for split in splits:
    classifier = NaiveBayes()
    accuracy = 0.0
    for example in split.train:
      words = example.words
      if nb.FILTER_STOP_WORDS:
        words =  classifier.filterStopWords(words)
      classifier.addExample(example.klass, words)
    for example in split.test:
      words = example.words
      if nb.FILTER_STOP_WORDS:
        words =  classifier.filterStopWords(words)
      guess = classifier.classify(words)
      if example.klass == guess:
        accuracy += 1.0

    accuracy = accuracy / len(split.test)
    avgAccuracy += accuracy
    print '[INFO]\tFold %d Accuracy: %f' % (fold, accuracy)
    fold += 1
  avgAccuracy = avgAccuracy / fold
  print '[INFO]\tAccuracy: %f' % avgAccuracy

if __name__ == "__main__":
    main()
