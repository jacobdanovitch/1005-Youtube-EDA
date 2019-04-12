from nltk.tokenize import TweetTokenizer
tt = TweetTokenizer()

def ngrams(w, n=3):
    return [(tuple(w[i:i+n-1]), w[i+n-1]) for i in range(len(w)-n+1)]

def tokenize(series):
    return series.str.lower().replace(r'[^A-Za-z\s+]', '', regex=True).apply(tt.tokenize)

def make_trigrams(tokenized):
    return [y for x in tokenized.apply(ngrams) for y in x]


def most_similar_word(idx, model, n=10):
    vec = model[idx]
    return nearest_neighbors(vec, model, n)
    
def nearest_neighbors(vec, model, n):
    # negation sorts the list ascending
    sim = (-model.embeddings.weight.mul(vec).sum(dim=1)).argsort(dim=0)

    # for ref: tensor.item() returns val of one-ele tensor
    # sim[0] will be original word
    return list(map(model.vocab.index2word.get, sim.tolist()))[:n]
  

def next_word(ctx, model, n=5):
    inputs = model.vocab.sent2tensor(ctx)
    outputs = model(inputs)
    sims = (-outputs[0]).argsort()

    print(f"{' '.join(ctx)} =>\n-----")
    for i in range(n):
        print(model.vocab.index2word[sims[i].item()])