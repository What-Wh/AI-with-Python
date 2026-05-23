from matplotlib import pyplot as plt
import pandas as pd
from textblob import TextBlob


df = pd.read_csv('./assets/product-details_captured-texts_2026-05-16_10-07-26_019e2f9c-0eea-7b18-a9ea-7a8401b87e39.csv')
text = df['newText']

blob1 = TextBlob(text[0])
blob2 = TextBlob(text[1])
blob3 = TextBlob(text[2])

print(f"Text 1: {text[0]}")
print("Sentiment:", blob1.sentiment)

print(f"Text 2: {text[1]}")
print("Sentiment:", blob2.sentiment)

print(f"Text 3: {text[2]}")
print("Sentiment:", blob3.sentiment)

YN = 0
YP = 0
YNeg = 0
if(blob1.polarity < 0):
    YN += 1 
elif blob1.polarity < 0.4:
    YNeg += 1
else: 
    YP += 1

if(blob2.polarity < 0):
    YN += 1 
elif blob2.polarity < 0.4:
    YNeg += 1
else: 
    YP += 1

if(blob3.polarity < 0):
    YN += 1 
elif blob3.polarity < 0.4:
    YNeg += 1
else: 
    YP += 1

X = ['Negative', 'Nitral', 'Positive']
Y = [blob1.sentiment.subjectivity, blob2.sentiment.subjectivity, blob3.sentiment.subjectivity]

plt.figure(figsize=(10,6))
plt.bar(X[0], YN)
plt.bar(X[1], YNeg)
plt.bar(X[2], YP)
plt.title("polarity / subjectivity")
plt.show()