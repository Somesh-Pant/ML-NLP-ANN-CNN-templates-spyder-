import spacy

nlp = spacy.load('en_core_web_lg')


tokens = nlp(u"car airplane wing")
for token in tokens:
    print(token.text, token.vector_norm)
    

from scipy import spatial
cos_sim = lambda vec1, vec2:1 - spatial.distance.cosine(vec1, vec2)

ship = nlp.vocab['ship'].vector
water = nlp.vocab['water'].vector
air = nlp.vocab['air'].vector

new_vec = ship - water + air


computed_sim = []

for word in nlp.vocab:
    if word.has_vector:
        if word.is_lower:
            if word.is_alpha:
                sim = cos_sim(new_vec , word.vector)
                computed_sim.append((word,sim))
                
computed_sim = sorted(computed_sim, key = lambda item:-item[1])
print([w[0].text for w in computed_sim[:10]])

def vector_math(a,b,c):
    for word in nlp.vocab:
        if word == a:
            a = word.vector
        elif word == b:
            b = word.vector
        elif word == c:
            c = word.vector
    
    new_vec = a - b + c
    
    computed_sim = []

    for word in nlp.vocab:
        if word.has_vector:
            if word.is_lower:
                if word.is_alpha:
                    sim = cos_sim(new_vec , word.vector)
                    computed_sim.append((word,sim))
                
    computed_sim = sorted(computed_sim, key = lambda item:-item[1])
    print([w[0].text for w in computed_sim[:10]])
    

from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()

review = 'This movie was the worst movie EVER!!!!'

sid.polarity_scores(review)

def review_rating(string):
    score = sid.polarity_scores(review)
    compound = score['compound']
    if compound > 0:
        sc = "Positive"
    elif compound < 0:
        sc = "Negative"
    else:
        sc = "Neutral"
    print(sc)
    
