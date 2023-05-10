#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      rePUBLIC
#
# Created:     06-04-2023
# Copyright:   (c) rePUBLIC 2023
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import requests
from bs4 import BeautifulSoup
import random
def get_wallpapers():
    # Define the URL of the website
    url = "https://wallhaven.cc//"

    # Send a GET request to the website
    response = requests.get(url)

    # Parse the HTML content of the response using BeautifulSoup
    soup = BeautifulSoup(response.content, "html.parser")

    # Find all the image elements on the page
    image_elements = soup.find_all("img")

    # Extract the src attribute of each image element and store it in a list
    image_urls = [img["src"] for img in image_elements]

    # Return the list of image URLs
    return image_urls
