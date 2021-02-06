import pandas as pd

#clean news_articles csv
df= pd.read_csv('news_articles.csv')
df=df[pd.notnull(df.text_without_stopwords)]
df=df[df.language!='german']
df.drop(['author','published','title_without_stopwords','main_img_url','title','text','language','site_url','type'], axis = 1, inplace = True)

df.to_csv(r'clean_news_articles.csv')
