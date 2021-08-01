import re
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import pandas as pd


def getHTML(url):
    try:
        r = requests.get(url)
        # r.raise_for_status()
        # r.encoding = 'utf-8'#r.apparent_encoding
        return r.text
    except:
        return ""


def getContent(url):
    html = getHTML(url)
    soup = BeautifulSoup(html, 'html.parser')
    # title = soup.select('div.mbtitle')
    paras_tmp = soup.select('div')
    paras = paras_tmp
    return paras


# def saveFile(text):
#     f=open('novel.txt','w')
#     for t in text:
#         if len(t) > 0:
#             f.writelines(t.get_text() + "\n\n")
#     f.close()


def download_text(text):
    f = []
    for t in text:
        if len(t) > 0:
            f.append(t.get_text())
    return f


def main():
    text = getContent(url)
    return download_text(text)

# the 11th and 20th are the position of the summary and the document
# url = 'https://edit.tosdr.org/points/5001'
# a = main()
# display(a[11].strip(), a[20].strip())

if __name__ == '__main__':
    summary = []
    content = []
    for i in tqdm(range(601,24230)):
        try:
            url = 'https://edit.tosdr.org/points/'+str(i)
            a=main()
            summary.append(a[11].strip('\n ""')) # remove the noise around the strings
            content.append(a[21].strip('\n ""'))
        except:
            pass
    data = pd.DataFrame({'Content':content,'Summary':summary})
    data.to_csv('legal_dataset_all.csv', index=False)