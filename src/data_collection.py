import os
import re
from pathlib import Path
from urllib.request import Request, urlopen

import mlrun
from bs4 import BeautifulSoup, Tag

ARTICLE_TOKEN = "Article: "
HEADER_TOKEN = "Subject: "


def normalize(s: str) -> str:
    """
    Remove newline and tab characters from string
    """
    return s.replace("\n", "").replace("\t", "")


def mark_header_tags(soup: BeautifulSoup):
    """
    Adding header token and article token prefixes to all headers in html, in order to parse the text later easily.

    :param soup: BeautifulSoup object of the html file
    """
    nodes = soup.find_all(re.compile("^h[1-6]$"))
    # Tagging headers in html to identify in text files:
    if nodes:
        content_type = type(nodes[0].contents[0])
        nodes[0].string = content_type(
            ARTICLE_TOKEN + normalize(str(nodes[0].contents[0]))
        )
        for node in nodes[1:]:
            if node.string:
                content_type = type(node.contents[0])
                if content_type == Tag:
                    node.string = HEADER_TOKEN + normalize(node.string)
                else:
                    node.string = content_type(HEADER_TOKEN + str(node.contents[0]))


def get_html_as_string(url: str, mark_headers: bool) -> str:
    """
    Retrieve text from html URL.

    :param url:             html URL
    :param mark_headers:    Whether to add article and header prefixes to headers to text

    :returns:                html text content
    """
    # read html source:
    req = Request(url=url, headers={"User-Agent": "Mozilla/5.0"})
    web_html_content = urlopen(req).read().decode("utf-8")
    soup = BeautifulSoup(web_html_content, features="html.parser")
    if mark_headers:
        mark_header_tags(soup)
    return soup.get_text()


@mlrun.handler(outputs=["html-as-text-files:directory"])
def collect_html_to_text_files(urls_file: str, mark_headers=True) -> str:
    """
    Retrieve all html text content from URLs as text files.

    :param urls_file:       html URLs file
    :param mark_headers:    Whether to add article and header prefixes to headers to text

    :returns:  the directory name that contains all the content text files.
    """
    directory = "html_as_text_files"
    os.makedirs(directory, exist_ok=True)
    # Writing html files as text files:
    with open(urls_file, "r") as f:
        urls = f.readlines()
    for url in urls:
        url = url.replace("\n", "")
        page_name = Path(url).name
        with open(f"{directory}/{page_name}.txt", "w") as f:
            f.write(get_html_as_string(url, mark_headers))
    return directory
