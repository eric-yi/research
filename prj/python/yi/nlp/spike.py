#!/usr/bin/env python
# -*- coding:utf-8 -*-

import nltk

# nltk.download()

def run_on_book():
    # nltk.download('gutenberg')
    # nltk.download('genesis')
    # nltk.download('inaugural')
    import nltk.book as book
    print(book)
    print(book.text1)
    search = book.text1.concordance('china')
    print(search)
    search = book.text1.similar('china')
    print(search)
    f = nltk.FreqDist(book.text1)
    print(f)
    # f.plot(50)
    f.plot(50, cumulative=True)

def run_on_reuters():
    nltk.download('reuters')
    from nltk.corpus import reuters
    print(reuters)
    print(reuters.fileids())
    search = reuters.concordance('china')
    print(search)

def run_condition_frequence():
    from nltk.corpus import brown
    cfg = nltk.ConditionalFreqDist((genre, word)
                                   for genre in brown.categories()
                                   for word in brown.words(categories=genre))
    print(cfg)
    genre_word = [(genre, word)
                  for genre in ['news', 'romance']
                  for word in brown.words(categories=genre)
                  ]
    print(len(genre_word))


def run_web_spiker():
    import feedparser
    l = feedparser.parse('https://languagelog.ldc.upenn.edu/nll/?feed=atom')
    print(l.entries)


def run_gram():
    from nltk.corpus import brown
    sents = brown.tagged_sents(categories='news')
    tags = nltk.UnigramTagger(sents)
    print(tags)
    s = brown.sents(categories='news')
    print(s[2007])
    r = tags.tag(s[2007])
    print(r)

def run_math():
    import math
    p = 0.7
    print(math.log(p, 2))
    print(math.log(1, 2))
    entropy = p * math.log(p, 2)
    print(entropy)


# run_on_book()
# run_condition_frequence()
# run_web_spiker()
#run_gram()
run_math()
