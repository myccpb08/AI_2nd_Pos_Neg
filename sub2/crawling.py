from urllib.request import urlopen
from bs4 import BeautifulSoup
from datetime import datetime

pn = 6

def get_movie_links():
        home_html = urlopen("https://movie.naver.com")
        bsObject = BeautifulSoup(home_html, "html.parser")

        movie_links = []
        for a in bsObject.body.find_all('a', {"class":"_select_title_anchor"}):
                temp = a.get('href')
                temp = temp.replace('basic', 'point')
                movie_links.append("https://movie.naver.com"+temp+"&type=after&onlyActualPointYn=Y")
        return movie_links

def get_reple_link(url):
        resp = urlopen(url)
        html = BeautifulSoup(resp, 'html.parser')
        div = html.find('div', {"class": "ifr_module2"})
        reple_link = "https://movie.naver.com" + div.find('iframe').get('src')
        return reple_link

def get_good_reples(url):
        reples = []
        for i in range(1,pn):
                pageNum = i
                resp = urlopen(url + "&order=highest&page=" + str(pageNum))
                html = BeautifulSoup(resp, 'html.parser')

                score_result = html.find('div', {'class':'score_result'})
                lis = score_result.findAll('li')

                for li in lis:
                        score = li.find('em').getText()
                        reple = li.find('p').getText()
                        if score == '9' or score == '10':
                                reples.append([1, reple.lstrip("관람객").strip(), 1])
        return reples

def get_bad_reples(url):
        reples = []
        for i in range(1,pn):
                pageNum = i
                resp = urlopen(url + "&order=lowest&page=" + str(pageNum))
                html = BeautifulSoup(resp, 'html.parser')

                score_result = html.find('div', {'class':'score_result'})
                lis = score_result.findAll('li')

                for li in lis:
                        score = li.find('em').getText()
                        reple = li.find('p').getText()
                        if score == '1' or score == '2':
                                reples.append([1, reple.lstrip("관람객").strip(), 0])
        return reples
    
    
movie_links = get_movie_links()

reples = []
for i in range(len(movie_links)):
        reple_link = get_reple_link(movie_links[i])
        good_reples = get_good_reples(reple_link)
        bad_reples = get_bad_reples(reple_link)
        reples += good_reples + bad_reples

f = open("naver_reple.txt", 'a', encoding='UTF-8')
for i in range(len(reples)):
        data = str(reples[i][0]) + "\t" + reples[i][1] + "\t" + str(reples[i][2]) + "\n"
        f.write(data)
f.close()

